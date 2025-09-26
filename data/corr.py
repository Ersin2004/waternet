#!/usr/bin/env python3
# analyze_multicollinearity.py
#
# Usage examples:
#   # 1) Use a directory of per-parameter CSVs (semicolon-delimited)
#   python analyze_multicollinearity.py --input-dir ./per_param_csvs --out-dir ./ml_diag
#
#   # 2) Use a single merged yearly CSV
#   python analyze_multicollinearity.py --merged ./merged_yearly.csv --out-dir ./ml_diag
#
#   # Optional knobs:
#   #   --key-cols year,monsterident   # row index keys to align datasets
#   #   --restrict-years 2000 2009     # filter to year range (inclusive)
#   #   --standardize                  # z-score the predictors before VIF/corr
#   #   --min-nonnull 10               # drop columns with <10 non-null values
#
# Outputs in out-dir:
#   - predictors_wide.csv               (final wide table used)
#   - correlation_matrix.csv
#   - correlation_heatmap.png
#   - vif.csv                           (Variance Inflation Factors)
#
# Notes:
# - Handles decimal commas and qualifiers like "<0.01".
# - If mixing units for the same compound across files, please normalize upstream.
# - Heatmap uses matplotlib only, with automatic date/year tick handling removed.

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Parsing helpers ----------

def parse_number(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s[0] in "<>~≈≲≳":
        s = s[1:].strip()
    s = s.replace(",", ".")
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None

def parse_year_from_datum(s: Optional[str]) -> Optional[int]:
    if not s or len(s) < 4 or not s[:4].isdigit():
        return None
    return int(s[:4])

def read_semicolon_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        rows = [row for row in reader]
        return header, rows

def norm(s: str) -> str:
    return (s or "").strip().lower().replace("_", " ").replace("-", " ")

def find_idx(header: List[str], *cands: str) -> Optional[int]:
    m = {norm(c): i for i, c in enumerate(header)}
    for c in cands:
        if norm(c) in m:
            return m[norm(c)]
    return None

# ---------- Loaders for two schemas ----------

def load_per_parameter_file(path: str) -> pd.DataFrame:
    """
    Expect columns: datum, meetwaarde, fewsparameternaam or fewsparameter, optionally locatie x/y, eenheid
    Returns DataFrame with columns: id, year, value
    Where id is from file name (without .csv)
    """
    header, rows = read_semicolon_csv(path)
    if not header:
        return pd.DataFrame(columns=["id","year","value"])

    i_datum = find_idx(header, "datum")
    i_val   = find_idx(header, "meetwaarde")
    if i_datum is None or i_val is None:
        # Not a per-parameter schema
        return pd.DataFrame(columns=["id","year","value"])

    file_id = os.path.splitext(os.path.basename(path))[0]

    data = []
    for r in rows:
        if not r:
            continue
        if i_datum >= len(r) or i_val >= len(r):
            continue
        y = parse_year_from_datum(r[i_datum])
        v = parse_number(r[i_val])
        if y is None or v is None:
            continue
        data.append((file_id, y, v))
    return pd.DataFrame(data, columns=["id","year","value"])

def load_merged_yearly_file(path: str) -> pd.DataFrame:
    """
    Expect columns: id;year;value;...
    """
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, sep=";", encoding="latin1")
    # Keep only necessary cols
    cols = {c.lower(): c for c in df.columns}
    need = []
    for c in ["id","year","value"]:
        if c not in cols:
            return pd.DataFrame(columns=["id","year","value"])
        need.append(cols[c])
    out = df[need].rename(columns={cols["id"]: "id", cols["year"]: "year", cols["value"]: "value"})
    # Coerce value numeric
    if out["value"].dtype == object:
        out["value"] = out["value"].apply(lambda x: parse_number(str(x)))
    return out.dropna(subset=["id","year","value"])

# ---------- Build wide predictors ----------

def build_wide_matrix(frames: List[pd.DataFrame], key_cols: List[str]) -> pd.DataFrame:
    """
    frames contain columns ['id','year','value'] (id = predictor name)
    key_cols is a subset of available columns used as the row index, e.g., ['year'] or ['year','monsterident'].
    We pivot to wide: index = key_cols, columns = id, values = mean(value)
    """
    df = pd.concat(frames, ignore_index=True)
    # Aggregate in case duplicates within same (key,id)
    # First, ensure key columns exist in df; we only have 'year' from loaders.
    # If user asked for extra keys (like monsterident), we won't have them unless merged file carried them.
    missing = [k for k in key_cols if k not in df.columns]
    if missing:
        # Fall back to 'year' only
        key_cols = ["year"]

    gp = df.groupby(key_cols + ["id"], dropna=True)["value"].mean().reset_index()
    wide = gp.pivot_table(index=key_cols, columns="id", values="value", aggfunc="mean")
    wide = wide.sort_index(axis=1)
    return wide

# ---------- VIF ----------

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Variance Inflation Factor per column.
    VIF_j = 1 / (1 - R^2_j) where R^2_j is from regressing column j on the others.
    Implemented with numpy lstsq to avoid heavy deps; drops columns with <2 non-null overlap.
    """
    # Drop all-NA and constant columns
    X_num = X.copy()
    # Keep only columns with >= 2 distinct non-null values
    valid_cols = []
    for c in X_num.columns:
        s = X_num[c].dropna()
        if s.nunique() >= 2:
            valid_cols.append(c)
    X_num = X_num[valid_cols]

    vifs = []
    cols = list(X_num.columns)
    for j, col in enumerate(cols):
        y = X_num[col]
        X_others = X_num.drop(columns=[col])
        # Align on rows where both y and all predictors are non-null
        Z = pd.concat([y, X_others], axis=1).dropna()
        if Z.shape[0] < max(10, X_others.shape[1] + 2):
            vifs.append((col, np.nan))
            continue
        yv = Z.iloc[:, 0].to_numpy(dtype=float)
        Xm = Z.iloc[:, 1:].to_numpy(dtype=float)
        # Add intercept
        ones = np.ones((Xm.shape[0], 1))
        Xm_i = np.concatenate([ones, Xm], axis=1)
        # OLS via lstsq
        beta, *_ = np.linalg.lstsq(Xm_i, yv, rcond=None)
        y_hat = Xm_i @ beta
        ss_res = float(np.sum((yv - y_hat) ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        if ss_tot <= 0:
            vifs.append((col, np.nan))
            continue
        r2 = 1.0 - ss_res / ss_tot
        if r2 >= 1.0:
            vifs.append((col, np.inf))
        else:
            vifs.append((col, 1.0 / (1.0 - r2)))
    return pd.DataFrame(vifs, columns=["variable", "VIF"]).sort_values("VIF", ascending=False)

# ---------- Plot heatmap (matplotlib only) ----------

def plot_corr_heatmap(corr: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots()
    im = ax.imshow(corr.to_numpy(), aspect='auto', interpolation='nearest')
    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_xticklabels(list(corr.columns), rotation=90)
    ax.set_yticklabels(list(corr.index))
    ax.set_title("Correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------- Main CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Evaluate multicollinearity across compound/organism datasets.")
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--input-dir", help="Directory of per-parameter CSVs (semicolon-delimited)")
    m.add_argument("--merged", help="A merged yearly CSV (id;year;value;...)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--key-cols", default="year", help="Comma-separated key columns to align on (default: year)")
    ap.add_argument("--restrict-years", nargs=2, type=int, metavar=("START","END"),
                    help="Restrict to inclusive year range (e.g., 2000 2009)")
    ap.add_argument("--standardize", action="store_true",
                    help="Z-score variables before VIF/correlation")
    ap.add_argument("--min-nonnull", type=int, default=10,
                    help="Drop predictor columns with < this many non-null values (default: 10)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    key_cols = [k.strip() for k in args.key_cols.split(",") if k.strip()]

    frames = []
    if args.input_dir:
        paths = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
        if not paths:
            print(f"[WARN] No CSVs in {args.input_dir}", file=sys.stderr)
        for p in paths:
            df = load_per_parameter_file(p)
            if df.empty:
                continue
            frames.append(df)
    else:
        df = load_merged_yearly_file(args.merged)
        if not df.empty:
            frames.append(df)

    if not frames:
        sys.exit("No usable data found.")

    wide = build_wide_matrix(frames, key_cols=key_cols)

    # Restrict by year if requested (only works if 'year' is in the index)
    if args.restrict_years and "year" in wide.index.names:
        y0, y1 = args.restrict_years
        # Build a boolean mask over MultiIndex or Index
        if isinstance(wide.index, pd.MultiIndex):
            # Find position of 'year' level
            try:
                year_pos = wide.index.names.index("year")
            except ValueError:
                year_pos = None
            if year_pos is not None:
                idx_mask = [(y0 <= idx[year_pos] <= y1) for idx in wide.index]
                wide = wide.loc[idx_mask]
        else:
            # simple Index named 'year'
            try:
                years = wide.index.astype(int)
                wide = wide[(years >= y0) & (years <= y1)]
            except Exception:
                pass

    # Drop columns that are too sparse
    keep_cols = [c for c in wide.columns if wide[c].notna().sum() >= args.min_nonnull]
    wide = wide[keep_cols]

    # Optionally standardize (z-score by column)
    X = wide.copy()
    if args.standardize:
        for c in X.columns:
            s = X[c]
            mu = s.mean(skipna=True)
            sd = s.std(skipna=True, ddof=1)
            if pd.notna(sd) and sd > 0:
                X[c] = (s - mu) / sd

    # Save the table used for diagnostics
    wide_out = os.path.join(args.out_dir, "predictors_wide.csv")
    X.to_csv(wide_out, sep=";", encoding="utf-8")
    print(f"[OK] wrote {wide_out}")

    # Correlation matrix
    corr = X.corr(method="pearson", min_periods=2)
    corr_out = os.path.join(args.out_dir, "correlation_matrix.csv")
    corr.to_csv(corr_out, sep=";", encoding="utf-8")
    print(f"[OK] wrote {corr_out}")

    # Heatmap
    heatmap_out = os.path.join(args.out_dir, "correlation_heatmap.png")
    plot_corr_heatmap(corr, heatmap_out)
    print(f"[OK] wrote {heatmap_out}")

    # VIFs
    vif_df = compute_vif(X)
    vif_out = os.path.join(args.out_dir, "vif.csv")
    vif_df.to_csv(vif_out, sep=";", index=False, encoding="utf-8")
    print(f"[OK] wrote {vif_out}")

    # Small console summary
    bad = vif_df[vif_df["VIF"] >= 10]
    if not bad.empty:
        print("[INFO] High multicollinearity (VIF ≥ 10) detected for:\n", bad)

if __name__ == "__main__":
    main()


# # Using a directory of per-parameter files
# python corr.py --input-dir /Users/Skye/Downloads/waternet/data/out/chem --out-dir /Users/Skye/Downloads/waternet/data/resultscorr