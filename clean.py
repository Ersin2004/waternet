import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    main()