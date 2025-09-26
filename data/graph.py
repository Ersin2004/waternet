#!/usr/bin/env python3
# plot_folder_timeseries.py
#
# Usage:
#   python plot_folder_timeseries.py <input_dir> <output_dir> [--show]
#
# - Expects each CSV to have at least: datum;meetwaarde
# - Delimiter is ';'
# - Saves a PNG per CSV into <output_dir> (same basename + .png)
# - --show will also display the plots interactively
#
# No seaborn. One chart per figure. No explicit colors/styles.

import argparse
import csv
import os
import sys
import glob
from datetime import datetime
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REQUIRED_COLS = ["datum", "meetwaarde"]  # case-insensitive

def parse_args():
    ap = argparse.ArgumentParser(description="Plot one time-series graph per CSV in a folder.")
    ap.add_argument("input_dir", help="Directory containing the per-parameter CSV files")
    ap.add_argument("output_dir", help="Directory to write PNG plots")
    ap.add_argument("--show", action="store_true", help="Also display plots interactively")
    return ap.parse_args()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_header_build_index(header_row):
    norm = [h.strip().lower() for h in header_row]
    idx = {col: i for i, col in enumerate(norm)}
    for c in REQUIRED_COLS:
        if c not in idx:
            raise ValueError(f"Missing required column '{c}' in CSV header.")
    # Optional columns for titles/labels if present
    opt = {}
    for name in ["fewsparameter", "fewsparameternaam", "fewsparametercategorie", "eenheid"]:
        if name in idx:
            opt[name] = idx[name]
    return idx, opt

def parse_date(s: str):
    s = s.strip()
    if not s:
        return None
    # Fast path: YYYY[-MM[-DD]] [HH:MM]
    # Try common formats
    fmts = ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    # Fallback: if first 4 are digits, try only year-month-day substr
    if len(s) >= 10 and s[:4].isdigit():
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None
    return None

def parse_numeric(s: str):
    """
    Convert 'meetwaarde' to float:
    - strip whitespace
    - remove leading '<' or '>' qualifiers
    - convert decimal comma to dot
    - return None if not finite
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # Remove qualifiers like '<' or '>' (keep the number)
    if s[0] in "<>~≈≲≳":
        s = s[1:].strip()
    # Replace decimal comma
    s = s.replace(",", ".")
    try:
        val = float(s)
        if math.isfinite(val):
            return val
        return None
    except ValueError:
        return None

def title_from_meta(basename, first_row, opt_idx):
    """
    Prefer a meaningful title using fewsparameternaam or fewsparameter,
    falling back to file name.
    """
    title_parts = []
    if "fewsparameternaam" in opt_idx:
        title_parts.append(first_row[opt_idx["fewsparameternaam"]].strip())
    elif "fewsparameter" in opt_idx:
        title_parts.append(first_row[opt_idx["fewsparameter"]].strip())

    if "fewsparametercategorie" in opt_idx:
        cat = first_row[opt_idx["fewsparametercategorie"]].strip()
        if cat:
            title_parts.append(f"({cat})")

    title = " ".join([p for p in title_parts if p]) if title_parts else basename
    return title or basename

def y_label_from_meta(first_row, opt_idx):
    if "eenheid" in opt_idx:
        unit = first_row[opt_idx["eenheid"]].strip()
        if unit:
            return f"meetwaarde [{unit}]"
    return "meetwaarde"

def plot_file(csv_path, out_dir):
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    dates = []
    vals = []
    meta_title_row = None

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        try:
            header = next(reader)
        except StopIteration:
            print(f"[WARN] Empty file: {csv_path}")
            return

        try:
            idx, opt_idx = read_header_build_index(header)
        except ValueError as e:
            print(f"[WARN] {csv_path}: {e}")
            return

        i_datum = idx["datum"]
        i_meetwaarde = idx["meetwaarde"]

        for row_i, row in enumerate(reader, start=2):
            if not row:
                continue
            # Keep a row around for title labels if we haven't yet
            if meta_title_row is None:
                meta_title_row = row

            if i_datum >= len(row) or i_meetwaarde >= len(row):
                continue
            dt = parse_date(row[i_datum])
            v = parse_numeric(row[i_meetwaarde])
            if dt is None or v is None:
                continue
            dates.append(dt)
            vals.append(v)

    if not dates:
        print(f"[INFO] No plottable rows in: {csv_path}")
        return

    # Sort by time (in case input isn't ordered)
    order = np.argsort(np.array(dates, dtype="datetime64[ns]"))
    dates = [dates[i] for i in order]
    vals  = [vals[i]  for i in order]

    # Create the plot (one graph per file)
    plt.figure()
    plt.plot_date(dates, vals, linestyle='solid', marker=None)  # no explicit colors
    plt.gcf().autofmt_xdate()

    # Configure x-axis date formatting
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_xlabel("datum")

    # Titles/labels
    title = title_from_meta(basename, meta_title_row or [], opt_idx)
    ax.set_title(title)
    ax.set_ylabel(y_label_from_meta(meta_title_row or [], opt_idx))

    # Save
    out_path = os.path.join(out_dir, basename + ".png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Wrote {out_path}")

def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not csv_files:
        print(f"[WARN] No CSV files found in {args.input_dir}")
        return

    for p in csv_files:
        try:
            plot_file(p, args.output_dir)
        except Exception as e:
            print(f"[ERROR] Failed to process {p}: {e}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()


