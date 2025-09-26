import os
import glob

# Resolve path to project root from this file (frontend/src/test.py -> project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Absolute glob path to the chem CSVs
file_path = os.path.join(PROJECT_ROOT, "data", "out", "chem")

# Resolve glob and print summary (avoid os.path.exists on patterns)
matches = glob.glob(file_path)
print("found:", len(matches) > 0)
print("count:", len(matches))
for p in matches[:5]:
    print(p)
# ././data/out/chem/*.csv

import os
print(os.path.exists(file_path))
