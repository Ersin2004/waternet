#!/usr/bin/env python3
# split_fews_by_parameter.py
# Usage:
#   python split_fews_by_parameter.py <input.csv> <output_dir> <min_count_X> [--no-others]
#
# Notes:
# - Input is semicolon-separated; quotes are supported via csv module.
# - Creates one CSV per fewsparameter (filename-safe).
# - Columns in outputs: monsterident;datum;fewsparametercategorie;fewsparameternaam;meetwaarde;eenheid
# - Rows with parameters below threshold are skipped, unless --no-others is omitted (default writes others_below_threshold.csv).

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

OUTPUT_COLUMNS = [
    "monsterident",
    "datum",
    "fewsparametercategorie",
    "fewsparameternaam",
    "meetwaarde",
    "eenheid",
]

def sanitize_filename(name: str) -> str:
    # Replace anything not alnum/_/- with '_'; trim length and avoid reserved names
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", name or "empty")
    if safe in {".", ".."}:
        safe = "__"
    return safe[:120]  # keep it short-ish

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_header_build_index(reader) -> dict:
    try:
        header = next(reader)
    except StopIteration:
        sys.exit("Input is empty.")
    # normalize: strip & lowercase
    norm = [h.strip().lower() for h in header]
    idx = {col: i for i, col in enumerate(norm)}
    return idx, header  # return original header too, in case you need it

def must_get(idx: dict, col: str) -> int:
    key = col.lower()
    if key not in idx:
        sys.exit(f"Missing required column in header: {col}")
    return idx[key]

def open_csv_reader(path):
    # utf-8-sig to gracefully handle BOM, newline='' is recommended for csv
    return open(path, "r", encoding="utf-8-sig", newline="")

def open_csv_writer(path):
    return open(path, "w", encoding="utf-8", newline="")

def in_2000s(d: str) -> bool:
    return len(d) >= 4 and d[:2] == "20"


def main():
    ap = argparse.ArgumentParser(description="Split FEWS CSV into per-parameter files with selected columns.")
    ap.add_argument("input_csv", help="Path to the semicolon-delimited input CSV")
    ap.add_argument("output_dir", help="Directory to place output CSV files")
    ap.add_argument("min_count", type=int, help="Minimum number of rows per parameter to keep (>= X)")
    ap.add_argument("--no-others", action="store_true", help="Do not write others_below_threshold.csv for below-threshold rows")
    args = ap.parse_args()

    if args.min_count < 0:
        sys.exit("min_count must be non-negative.")

    ensure_dir(args.output_dir)

    # -------- Pass 1: count occurrences of fewsparameter --------
    counts = defaultdict(int)
    with open_csv_reader(args.input_csv) as f:
        reader = csv.reader(f, delimiter=';')
        idx, _ = read_header_build_index(reader)

        i_fewsparameter = must_get(idx, "fewsparameter")
        i_datum        = must_get(idx, "datum")  # NEW


        for row in reader:
            if not row:
                continue
            if not in_2000s(row[i_datum]):       # NEW filter
                continue
            if i_fewsparameter >= len(row):
                continue  # malformed/short row
            counts[row[i_fewsparameter]] += 1

    allowed = {param for param, cnt in counts.items() if cnt >= args.min_count}

    # -------- Pass 2: write per-parameter CSVs --------
    writers = {}  # param -> csv writer
    files = {}    # param -> file handle (to close later)

    others_writer = None
    others_file = None
    rows_written = 0
    rows_skipped = 0

    try:
        if not args.no_others:
            others_path = os.path.join(args.output_dir, "others_below_threshold.csv")
            others_file = open_csv_writer(others_path)
            others_writer = csv.writer(others_file, delimiter=';')
            others_writer.writerow(OUTPUT_COLUMNS)

        with open_csv_reader(args.input_csv) as f:
            reader = csv.reader(f, delimiter=';')
            idx, _ = read_header_build_index(reader)

            # Resolve required indices (case-insensitive)
            col_indices = {
                col: must_get(idx, col) for col in
                ["monsterident", "datum", "fewsparameter", "fewsparametercategorie", "fewsparameternaam", "meetwaarde", "eenheid"]
            }

            def ensure_param_writer(param_value: str):
                if param_value in writers:
                    return writers[param_value]
                fname = sanitize_filename(param_value) + ".csv"
                path = os.path.join(args.output_dir, fname)
                fh = open_csv_writer(path)
                w = csv.writer(fh, delimiter=';')
                w.writerow(OUTPUT_COLUMNS)
                writers[param_value] = w
                files[param_value] = fh
                return w

            for row in reader:
                if not row:
                    continue
                # guard against short/malformed rows
                needed_max_idx = max(col_indices.values())
                if needed_max_idx >= len(row):
                    rows_skipped += 1
                    continue

                if not in_2000s(row[col_indices["datum"]]):
                    rows_skipped += 1
                    continue

                param_val = row[col_indices["fewsparameter"]]
                out_vals = [
                    row[col_indices["monsterident"]],
                    row[col_indices["datum"]],
                    row[col_indices["fewsparametercategorie"]],
                    row[col_indices["fewsparameternaam"]],
                    row[col_indices["meetwaarde"]],
                    row[col_indices["eenheid"]],
                ]

                if param_val in allowed:
                    w = ensure_param_writer(param_val)
                    w.writerow(out_vals)
                    rows_written += 1
                else:
                    if others_writer is not None:
                        others_writer.writerow(out_vals)
                    rows_skipped += 1
    finally:
        # Close all files
        for fh in files.values():
            try:
                fh.close()
            except Exception:
                pass
        if others_file:
            try:
                others_file.close()
            except Exception:
                pass

    sys.stderr.write(
        f"Done. Wrote {rows_written} rows across {len(writers)} parameter files. "
        f"Skipped {rows_skipped} rows "
        f"({'saved to others_below_threshold.csv' if others_writer else 'not written'}).\n"
    )

if __name__ == "__main__":
    main()
