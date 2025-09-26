import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import csv
import os


def read_semicolon_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, sep=";", dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, sep=";", dtype=str, encoding="latin-1")


def read_optional_reference(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return read_semicolon_csv(path)
    return None


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        new_col = col.strip().lower().replace(" ", "_")
        new_col = new_col.replace("/", "_").replace("\\", "_")
        renamed[col] = new_col
    return df.rename(columns=renamed)


def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df


def to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            # Zorg dat we veilig kunnen vervangen, ook als kolom geen string-dtype heeft
            series = df[col]
            series = series.astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(series, errors="coerce")
    return df


def to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df)
    df = strip_string_columns(df)
    # Zet lege strings in ALLE tekstkolommen naar NaN
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace({"": np.nan})

    df = to_datetime(df, "datum")

    numeric_candidates = [
        "xcoormonster",
        "ycoormonster",
        "locatie_x",
        "locatie_y",
        "locatie_z",
        "locatie_referentievlakzcoord",
        "meetwaarde",
        "afronding",
    ]
    df = to_numeric(df, numeric_candidates)

    subset = [c for c in [
        "monsterident",
        "datum",
        "fewsparameter",
        "biotaxonnaam",
        "twn_naam",
        "meetwaarde",
        "locatiecode",
    ] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset)
    else:
        df = df.drop_duplicates()

    # Verwijder rijen die volledig leeg zijn
    df = df.dropna(how="all")

    # Verwijder rijen zonder meetwaarde (veel lege records met ,,,)
    if "meetwaarde" in df.columns:
        df = df[~df["meetwaarde"].isna()]

    # Verwijder volledig lege kolommen
    df = df.dropna(axis=1, how="all")

    return df


def main() -> None:
    root = Path(__file__).resolve().parent
    src = root / "data/waternetFEWSdata/HB_sampled50locations.csv"
    if not src.exists():
        print(f"Bestand niet gevonden: {src}", file=sys.stderr)
        sys.exit(1)

    df_raw = read_semicolon_csv(src)
    df_clean = clean_dataframe(df_raw.copy())

    rows_raw = int(len(df_raw))
    rows_clean_pre_drop = int(len(df_clean))

    # Drop kolommen met zeer hoge missings (bv. > 95% leeg)
    missing_fraction = df_clean.isna().mean()
    high_missing_cols = missing_fraction[missing_fraction > 0.95].index.tolist()
    if high_missing_cols:
        df_clean = df_clean.drop(columns=high_missing_cols)

    # Write compact subset with core fields to reduce empties
    core_columns = [
        "monsterident",
        "datum",
        "locatiecode",
        "xcoormonster",
        "ycoormonster",
        "fewsparameter",
        "fewsparameternaam",
        "parametercode",
        "grootheid",
        "biotaxonnaam",
        "twn_naam",
        "wna_nederlandse_soortnaam",
        "meetwaarde",
        "eenheid",
        "limietsymbool",
        "compartiment",
    ]
    core_columns = [c for c in core_columns if c in df_clean.columns]
    if core_columns:
        df_compact = df_clean[core_columns].dropna(how="all")
        out_compact = root / "data/out/HB_sampled50locations_clean_compact.csv"
        df_compact.to_csv(out_compact, index=False)
    else:
        out_compact = None

    print("Cleaning done:")
    print(f"- Input:  {src}")
    if out_compact is not None:
        print(f"- Output (compact): {out_compact}")
    else:
        print("- Output (compact): skipped (no core columns found)")
    # Summary JSON writing disabled by request

    # Console statistieken
    print("\nStatistics:")
    print(f"- Rows (raw): {rows_raw}")
    print(f"- Rows (after basic clean, pre column-drop): {rows_clean_pre_drop}")
    print(f"- Rows (final): {len(df_clean)}")
    print(f"- Columns (final): {len(df_clean.columns)}")
    if high_missing_cols:
        print(f"- Dropped columns (>95% missing): {len(high_missing_cols)} -> {', '.join(high_missing_cols[:10])}{' ...' if len(high_missing_cols) > 10 else ''}")
    else:
        print("- No columns dropped by missing threshold")

    # Top missings per kolom
    if not missing_fraction.empty:
        top_missing = missing_fraction.sort_values(ascending=False).head(10)
        print("- Top 10 columns by missing fraction:")
        for c, v in top_missing.items():
            print(f"  · {c}: {v:.2%}")

    # Datumbereik
    if "datum" in df_clean.columns and pd.api.types.is_datetime64_any_dtype(df_clean["datum"]):
        dmin = pd.to_datetime(df_clean["datum"], errors="coerce").min()
        dmax = pd.to_datetime(df_clean["datum"], errors="coerce").max()
        print(f"- Date range: {dmin} to {dmax}")

    # Top parameters en eenheden
    pref_param = "fewsparameternaam" if "fewsparameternaam" in df_clean.columns else ("fewsparameter" if "fewsparameter" in df_clean.columns else None)
    if pref_param:
        vc = df_clean[pref_param].value_counts(dropna=True).head(10)
        print(f"- Top 10 {pref_param}:")
        for k, v in vc.items():
            print(f"  · {k}: {v}")
    if "eenheid" in df_clean.columns:
        vc = df_clean["eenheid"].value_counts(dropna=True).head(10)
        print("- Top 10 units:")
        for k, v in vc.items():
            print(f"  · {k}: {v}")

    # Filtering against external reference disabled by request

    # -------- Also create per-parameter CSVs (combined logic) --------
    # This mirrors the behavior of data/clean.py (splitter) on the raw semicolon FEWS file
    try:
        output_dir = root / "data/out/bio"
        output_dir.mkdir(parents=True, exist_ok=True)

        # First pass: count per fewsparameter (only 2000s dates)
        def in_2000s(date_str: str) -> bool:
            return len(date_str) >= 4 and date_str[:2] == "20"

        counts: dict[str, int] = {}
        with open(src, "r", encoding="utf-8-sig", newline="") as f_in:
            reader = csv.reader(f_in, delimiter=';')
            try:
                header = next(reader)
            except StopIteration:
                header = []
            norm = [h.strip().lower() for h in header]
            idx = {col: i for i, col in enumerate(norm)}
            i_param = idx.get("fewsparameter")
            i_date = idx.get("datum")
            if i_param is not None and i_date is not None:
                for row in reader:
                    if not row or i_param >= len(row) or i_date >= len(row):
                        continue
                    if not in_2000s(row[i_date]):
                        continue
                    param_val = row[i_param]
                    counts[param_val] = counts.get(param_val, 0) + 1

        min_count = 50  # default threshold; adjust if needed
        allowed = {p for p, c in counts.items() if c >= min_count}

        # Second pass: write per-parameter files (and others_below_threshold)
        OUTPUT_COLUMNS = [
            "monsterident",
            "datum",
            "fewsparametercategorie",
            "fewsparameternaam",
            "meetwaarde",
            "eenheid",
            "x_location",
            "y_location",
        ]

        def must_get(local_idx: dict[str, int], col: str) -> int:
            key = col.lower()
            if key not in local_idx:
                raise KeyError(f"Missing required column in header: {col}")
            return local_idx[key]

        def sanitize_filename(name: str) -> str:
            safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in (name or "empty"))
            if safe in {".", ".."}:
                safe = "__"
            return safe[:120]

        writers: dict[str, csv.writer] = {}
        files: dict[str, any] = {}
        others_path = output_dir / "others_below_threshold.csv"
        others_file = open(others_path, "w", encoding="utf-8", newline="")
        others_writer = csv.writer(others_file, delimiter=';')
        others_writer.writerow(OUTPUT_COLUMNS)

        try:
            with open(src, "r", encoding="utf-8-sig", newline="") as f_in2:
                reader2 = csv.reader(f_in2, delimiter=';')
                header2 = next(reader2)
                norm2 = [h.strip().lower() for h in header2]
                idx2 = {col: i for i, col in enumerate(norm2)}
                # map FEWS location to requested output x/y names
                idx_map = {
                    "monsterident": must_get(idx2, "monsterident"),
                    "datum": must_get(idx2, "datum"),
                    "fewsparameter": must_get(idx2, "fewsparameter"),
                    "fewsparametercategorie": must_get(idx2, "fewsparametercategorie"),
                    "fewsparameternaam": must_get(idx2, "fewsparameternaam"),
                    "meetwaarde": must_get(idx2, "meetwaarde"),
                    "eenheid": must_get(idx2, "eenheid"),
                    # FEWS uses either "locatie x"/"locatie y" or similar; try both fallbacks
                }
                i_x = idx2.get("locatie x") if idx2.get("locatie x") is not None else idx2.get("xcoormonster")
                i_y = idx2.get("locatie y") if idx2.get("locatie y") is not None else idx2.get("ycoormonster")
                if i_x is None or i_y is None:
                    # if still missing, default to empty values later
                    i_x = -1
                    i_y = -1

                def ensure_writer(param_value: str) -> csv.writer:
                    if param_value in writers:
                        return writers[param_value]
                    fname = sanitize_filename(param_value) + ".csv"
                    fh = open(output_dir / fname, "w", encoding="utf-8", newline="")
                    w = csv.writer(fh, delimiter=';')
                    w.writerow(OUTPUT_COLUMNS)
                    writers[param_value] = w
                    files[param_value] = fh
                    return w

                for row in reader2:
                    if not row:
                        continue
                    try:
                        dval = row[idx_map["datum"]]
                    except Exception:
                        continue
                    if not in_2000s(dval):
                        continue
                    if any(idx_map[k] >= len(row) for k in [
                        "monsterident","datum","fewsparameter","fewsparametercategorie","fewsparameternaam","meetwaarde","eenheid"
                    ]):
                        continue
                    param_val = row[idx_map["fewsparameter"]]
                    out_row = [
                        row[idx_map["monsterident"]],
                        dval,
                        row[idx_map["fewsparametercategorie"]],
                        row[idx_map["fewsparameternaam"]],
                        row[idx_map["meetwaarde"]],
                        row[idx_map["eenheid"]],
                        (row[i_x] if 0 <= i_x < len(row) else ""),
                        (row[i_y] if 0 <= i_y < len(row) else ""),
                    ]
                    if param_val in allowed:
                        w = ensure_writer(param_val)
                        w.writerow(out_row)
                    else:
                        others_writer.writerow(out_row)
        finally:
            for fh in files.values():
                try:
                    fh.close()
                except Exception:
                    pass
            try:
                others_file.close()
            except Exception:
                pass
        print(f"- Parameter splits: wrote to {output_dir} (threshold: {min_count})")
    except Exception as e:
        print(f"- Parameter splits skipped due to error: {e}")


if __name__ == "__main__":
    main()