#!/usr/bin/env python3
# merge_yearly_averages.py
#
# Usage:
#   python merge_yearly_averages.py <input_dir> <output_csv>
#
# Reads all *.csv in <input_dir> and writes a single semicolon-delimited CSV with:
#   id;year;value;eenheid;x_avg;y_avg
#
# Assumes required columns per row:
#   - datum (YYYY-MM-DD [HH:MM[:SS]]), used for year
#   - meetwaarde (numeric; supports decimal comma, <, > qualifiers)
# Optional columns (used when present):
#   - eenheid
#   - fewsparameternaam (preferred as id), or fewsparameter
#   - locatie x, locatie y (averaged per year)
#
# Notes:
# - Rows with non-parsable year or meetwaarde are skipped.
# - If units vary within a (id,year) group, the most frequent string wins.

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict, Counter

# ---------- Helpers ----------

def norm(s: str) -> str:
    """Normalize header names: lowercase, strip, collapse separators."""
    return (s or "").strip().lower().replace("_", " ").replace("-", " ")

def find_idx(header_map, *candidates):
    """Find first existing column index among candidate names (case-insensitive)."""
    for cand in candidates:
        k = norm(cand)
        if k in header_map:
            return header_map[k]
    return None

def parse_year(d: str):
    """Grab the 4-digit year from start of 'datum' (expects YYYY-...)."""
    if not d or len(d) < 4 or not d[:4].isdigit():
        return None
    y = int(d[:4])
    # If you want to restrict to certain ranges, enforce here.
    return y

def parse_number(s: str):
    """Parse meetwaarde / coords: strip '<'/'>' etc., support decimal comma."""
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
        if math.isfinite(v):
            return v
        return None
    except ValueError:
        return None

# ---------- Aggregator ----------

class Agg:
    __slots__ = ("sum_v","cnt_v","units","sum_x","cnt_x","sum_y","cnt_y")
    def __init__(self):
        self.sum_v = 0.0
        self.cnt_v = 0
        self.units = Counter()
        self.sum_x = 0.0
        self.cnt_x = 0
        self.sum_y = 0.0
        self.cnt_y = 0

    def add(self, v, unit, x, y):
        if v is not None:
            self.sum_v += v
            self.cnt_v += 1
        if unit:
            self.units[unit] += 1
        if x is not None:
            self.sum_x += x
            self.cnt_x += 1
        if y is not None:
            self.sum_y += y
            self.cnt_y += 1

    def result(self):
        value = (self.sum_v / self.cnt_v) if self.cnt_v else None
        eenheid = self.units.most_common(1)[0][0] if self.units else ""
        x_avg = (self.sum_x / self.cnt_x) if self.cnt_x else None
        y_avg = (self.sum_y / self.cnt_y) if self.cnt_y else None
        return value, eenheid, x_avg, y_avg

# ---------- Main ----------

def process_folder(input_dir, output_csv):
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        print(f"[WARN] No CSV files found in {input_dir}", file=sys.stderr)

    aggs = defaultdict(Agg)

    for path in files:
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f, delimiter=";")
                try:
                    header = next(reader)
                except StopIteration:
                    print(f"[WARN] Empty file skipped: {path}", file=sys.stderr)
                    continue

                header_map = {norm(col): i for i, col in enumerate(header)}

                i_datum = find_idx(header_map, "datum")
                i_meet  = find_idx(header_map, "meetwaarde")
                if i_datum is None or i_meet is None:
                    print(f"[WARN] Missing required columns in {path}; skipping.", file=sys.stderr)
                    continue

                # Optional columns
                i_unit  = find_idx(header_map, "eenheid")
                i_name  = find_idx(header_map, "fewsparameternaam")
                i_param = find_idx(header_map, "fewsparameter")
                # Coordinate columns: accept several spellings
                i_x = find_idx(header_map, "x_location")
                i_y = find_idx(header_map, "y_location")

                # Derive a default id from filename (basename) if no name fields exist
                default_id = os.path.splitext(os.path.basename(path))[0]

                for row in reader:
                    if not row:
                        continue
                    if i_datum >= len(row) or i_meet >= len(row):
                        continue

                    year = parse_year(row[i_datum])
                    if year is None:
                        continue

                    val = parse_number(row[i_meet])
                    if val is None:
                        # Non-numeric measurement; skip row
                        continue

                    unit = row[i_unit].strip() if (i_unit is not None and i_unit < len(row)) else ""

                    # Determine substance id: prefer fewsparameternaam, then fewsparameter, else filename
                    if i_name is not None and i_name < len(row) and row[i_name].strip():
                        sid = row[i_name].strip()
                    elif i_param is not None and i_param < len(row) and row[i_param].strip():
                        sid = row[i_param].strip()
                    else:
                        sid = default_id

                    # Coordinates (optional)
                    x = parse_number(row[i_x]) if (i_x is not None and i_x < len(row)) else None
                    y = parse_number(row[i_y]) if (i_y is not None and i_y < len(row)) else None

                    aggs[(sid, year)].add(val, unit, x, y)
        except Exception as e:
            print(f"[ERROR] Failed {path}: {e}", file=sys.stderr)

    # Write merged CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out, delimiter=";")
        w.writerow(["id", "year", "value", "eenheid", "x_avg", "y_avg"])
        for (sid, year), agg in sorted(aggs.items(), key=lambda kv: (kv[0][0].lower(), kv[0][1])):
            value, unit, x_avg, y_avg = agg.result()
            # Skip groups without any numeric values (shouldn't happen due to checks)
            if value is None:
                continue
            # Format floats; leave empty if None
            def fmt(v):
                return "" if v is None else f"{v:.6g}"
            w.writerow([sid, year, fmt(value), unit, fmt(x_avg), fmt(y_avg)])

    print(f"[OK] Wrote {output_csv}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Merge per-parameter CSVs into yearly averages.")
    ap.add_argument("input_dir", help="Folder with input CSV files")
    ap.add_argument("output_csv", help="Output merged CSV path")
    args = ap.parse_args()
    process_folder(args.input_dir, args.output_csv)

if __name__ == "__main__":
    main()
