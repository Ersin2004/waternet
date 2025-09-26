#!/usr/bin/env python3
# merge_yearly_averages.py
#
# Usage:
#   python merge_yearly_averages.py <input_dir> <output_csv>
#
# Output columns:
#   id;param_name;year;value;eenheid;x_avg;y_avg
#
# - id: taken from filename (basename without .csv)
# - param_name: fewsparameternaam (preferred) or fewsparameter from the file
# - year: extracted from 'datum' (YYYY-*)
# - value: mean of meetwaarde for (id, year)
# - eenheid: most frequent unit string for (id, year)
# - x_avg, y_avg: mean of coordinates for (id, year) if available

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict, Counter

def norm(s: str) -> str:
    return (s or "").strip().lower().replace("_", " ").replace("-", " ")

def find_idx(header_map, *candidates):
    for cand in candidates:
        k = norm(cand)
        if k in header_map:
            return header_map[k]
    return None

def parse_year(d: str):
    if not d or len(d) < 4 or not d[:4].isdigit():
        return None
    return int(d[:4])

def parse_number(s: str):
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

class Agg:
    __slots__ = ("sum_v","cnt_v","units","sum_x","cnt_x","sum_y","cnt_y","param_names")
    def __init__(self):
        self.sum_v = 0.0
        self.cnt_v = 0
        self.units = Counter()
        self.sum_x = 0.0
        self.cnt_x = 0
        self.sum_y = 0.0
        self.cnt_y = 0
        self.param_names = Counter()  # track names seen for this (file_id, year)

    def add(self, v, unit, x, y, param_name):
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
        if param_name:
            self.param_names[param_name] += 1

    def result(self):
        value = (self.sum_v / self.cnt_v) if self.cnt_v else None
        eenheid = self.units.most_common(1)[0][0] if self.units else ""
        x_avg = (self.sum_x / self.cnt_x) if self.cnt_x else None
        y_avg = (self.sum_y / self.cnt_y) if self.cnt_y else None
        pname = self.param_names.most_common(1)[0][0] if self.param_names else ""
        return value, eenheid, x_avg, y_avg, pname

def process_folder(input_dir, output_csv):
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        print(f"[WARN] No CSV files found in {input_dir}", file=sys.stderr)

    # key: (file_id, year)
    aggs = defaultdict(Agg)

    for path in files:
        file_id = os.path.splitext(os.path.basename(path))[0]  # <-- ID from filename

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
                i_meet  = find_idx(header_map, "meetwaarde")   # you referenced this line
                if i_datum is None or i_meet is None:
                    print(f"[WARN] Missing required columns in {path}; skipping.", file=sys.stderr)
                    continue

                i_unit  = find_idx(header_map, "eenheid")
                i_name  = find_idx(header_map, "fewsparameternaam")
                i_param = find_idx(header_map, "fewsparameter")

                i_x = find_idx(header_map, "x_location")
                i_y = find_idx(header_map, "y_location")

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
                        continue

                    unit = row[i_unit].strip() if (i_unit is not None and i_unit < len(row)) else ""

                    # New: keep the param name from the CSV in a separate column
                    if i_name is not None and i_name < len(row) and row[i_name].strip():
                        param_name = row[i_name].strip()
                    elif i_param is not None and i_param < len(row) and row[i_param].strip():
                        param_name = row[i_param].strip()
                    else:
                        param_name = ""

                    x = parse_number(row[i_x]) if (i_x is not None and i_x < len(row)) else None
                    y = parse_number(row[i_y]) if (i_y is not None and i_y < len(row)) else None

                    aggs[(file_id, year)].add(val, unit, x, y, param_name)
        except Exception as e:
            print(f"[ERROR] Failed {path}: {e}", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out, delimiter=";")
        # Added param_name column
        w.writerow(["id", "param_name", "year", "value", "eenheid", "x_avg", "y_avg"])
        for (fid, year), agg in sorted(aggs.items(), key=lambda kv: (kv[0][0].lower(), kv[0][1])):
            value, unit, x_avg, y_avg, pname = agg.result()
            if value is None:
                continue
            def fmt(v):
                return "" if v is None else f"{v:.6g}"
            w.writerow([fid, pname, year, fmt(value), unit, fmt(x_avg), fmt(y_avg)])

    print(f"[OK] Wrote {output_csv}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Merge per-parameter CSVs into yearly averages (ID from filename).")
    ap.add_argument("input_dir", help="Folder with input CSV files")
    ap.add_argument("output_csv", help="Output merged CSV path")
    args = ap.parse_args()
    process_folder(args.input_dir, args.output_csv)

if __name__ == "__main__":
    main()
