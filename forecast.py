import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import os
import warnings
import shutil

import numpy as np
import pandas as pd

try:
    # Statsmodels is optional; we fall back to linear regression if unavailable
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


@dataclass
class ForecastResult:
    parameter_id: str
    parameter_name: str
    years: List[int]
    values: List[float]
    model: str
    x_avg: Optional[float]
    y_avg: Optional[float]


def fit_holt_winters(y: pd.Series, seasonal_periods: Optional[int] = None) -> Optional[object]:
    if not _HAS_STATSMODELS:
        return None


def fit_sarimax(y: pd.Series, seasonal_periods: Optional[int]) -> Optional[object]:
    if not _HAS_STATSMODELS:
        return None
    if len(y) < 5:
        return None
    # Try a small set of simple orders
    candidates: List[Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]]]] = [
        ((1, 1, 1), None),
        ((1, 0, 1), None),
    ]
    if seasonal_periods and seasonal_periods >= 2 and len(y) >= seasonal_periods * 3:
        candidates.append(((1, 1, 1), (1, 0, 1, seasonal_periods)))
    best_aic = float("inf")
    best_fit = None
    for order, seasonal_order in candidates:
        try:
            model = SARIMAX(y.astype(float), order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            if res.aic < best_aic:
                best_aic = res.aic
                best_fit = res
        except Exception:
            continue
    return best_fit
    if len(y) < 4:
        return None
    try:
        # Additive trend, no seasonality by default; allow seasonal if specified and sufficient data
        if seasonal_periods and len(y) >= 2 * seasonal_periods + 2:
            model = ExponentialSmoothing(
                y.astype(float), trend="add", seasonal="add", seasonal_periods=seasonal_periods
            ).fit(optimized=True)
        else:
            model = ExponentialSmoothing(y.astype(float), trend="add", seasonal=None).fit(optimized=True)
        return model
    except Exception:
        return None


def fit_linear_regression(years: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    # y = a * year + b
    a, b = np.polyfit(years, values, 1)
    return float(a), float(b)


def _winsorize(values: np.ndarray, pct: float) -> np.ndarray:
    if pct <= 0.0:
        return values
    pct = min(max(pct, 0.0), 0.49)
    lo = np.nanquantile(values, pct)
    hi = np.nanquantile(values, 1.0 - pct)
    return np.clip(values, lo, hi)


def _maybe_log_transform(values: np.ndarray, use_log_auto: bool) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    if not use_log_auto:
        return values, lambda x: x
    if np.any(values < 0):
        return values, lambda x: x
    if len(values) < 3:
        return values, lambda x: x
    mean = float(np.mean(values))
    std = float(np.std(values))
    cv = std / mean if mean != 0 else np.inf
    if cv >= 0.5:
        # log1p transform with safe inverse
        return np.log1p(values), lambda x: np.expm1(x)
    return values, lambda x: x


def forecast_series(
    df: pd.DataFrame,
    horizon: int,
    seasonal_periods: Optional[int] = None,
    use_log_auto: bool = False,
    winsorize_pct: float = 0.0,
    use_median_fallback: bool = False,
    median_window: int = 0,
    forced_models: Optional[Dict[Tuple[str, str], str]] = None,
    auto_model_select: bool = True,
) -> List[ForecastResult]:
    results: List[ForecastResult] = []
    for (param_id, param_name), g in df.groupby(["id", "param_name"], sort=False):
        g_sorted = g.sort_values("year")
        years = g_sorted["year"].astype(int).to_numpy()
        values = g_sorted["value"].astype(float).to_numpy()
        # Preprocess values: winsorize then optional log transform
        values_w = _winsorize(values, winsorize_pct)
        values_t, invert_fn = _maybe_log_transform(values_w, use_log_auto)
        x_avg = float(g_sorted["x_avg"].dropna().tail(1).iloc[0]) if "x_avg" in g_sorted and not g_sorted["x_avg"].isna().all() else None
        y_avg = float(g_sorted["y_avg"].dropna().tail(1).iloc[0]) if "y_avg" in g_sorted and not g_sorted["y_avg"].isna().all() else None

        if len(values) == 0:
            continue

        start_year = int(years.max()) + 1
        future_years = list(range(start_year, start_year + horizon))

        model_label = "naive_last"
        yhat: List[float] = []

        # If a model is forced for this parameter, apply it and skip others
        key = (param_id, param_name)
        if forced_models and key in forced_models:
            chosen = forced_models[key]
            if chosen == "median_baseline":
                if median_window and median_window > 1 and len(values_w) >= median_window:
                    med = float(np.median(values_w[-median_window:]))
                else:
                    med = float(np.median(values_w))
                yhat = [med for _ in future_years]
                model_label = "median_baseline_forced"
                results.append(
                    ForecastResult(
                        parameter_id=param_id,
                        parameter_name=param_name,
                        years=future_years,
                        values=yhat,
                        model=model_label,
                        x_avg=x_avg,
                        y_avg=y_avg,
                    )
                )
                continue
            if chosen == "mean_baseline":
                if median_window and median_window > 1 and len(values_w) >= median_window:
                    avg = float(np.mean(values_w[-median_window:]))
                else:
                    avg = float(np.mean(values_w))
                yhat = [avg for _ in future_years]
                model_label = "mean_baseline_forced"
                results.append(
                    ForecastResult(
                        parameter_id=param_id,
                        parameter_name=param_name,
                        years=future_years,
                        values=yhat,
                        model=model_label,
                        x_avg=x_avg,
                        y_avg=y_avg,
                    )
                )
                continue

        # Try candidates and optionally auto select best using a small CV on last 3 years
        series_t = pd.Series(values_t, index=pd.Index(years, name="year"))
        def forecast_with_label(label: str) -> Tuple[str, List[float]]:
            try:
                if label == "holt":
                    m = fit_holt_winters(series_t, seasonal_periods)
                    if m is None:
                        return label, []
                    fc = m.forecast(horizon)
                    return label, [float(invert_fn(v)) for v in fc]
                if label == "holt_damped":
                    if not _HAS_STATSMODELS or len(series_t) < 4:
                        return label, []
                    m = ExponentialSmoothing(series_t.astype(float), trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
                    fc = m.forecast(horizon)
                    return label, [float(invert_fn(v)) for v in fc]
                if label == "sarimax":
                    m = fit_sarimax(series_t, seasonal_periods)
                    if m is None:
                        return label, []
                    fc = m.forecast(steps=horizon)
                    return label, [float(invert_fn(v)) for v in fc]
                if label == "linear":
                    if len(values_t) < 2:
                        return label, []
                    a, b = fit_linear_regression(years.astype(float), values_t.astype(float))
                    preds_t = [float(a * fy + b) for fy in future_years]
                    return label, [float(invert_fn(v)) for v in preds_t]
            except Exception:
                return label, []
            return label, []

        candidates_labels = ["sarimax", "holt", "holt_damped", "linear"] if auto_model_select else ["holt", "linear"]
        if auto_model_select and len(series_t) >= 7:
            # Small CV: last k=3 years
            k = min(3, max(1, len(series_t) - 4))
            train = series_t.iloc[:-k]
            test = series_t.iloc[-k:]
            future = list(test.index.values)
            scores: List[Tuple[float, str, List[float]]] = []
            for lab in candidates_labels:
                try:
                    # Fit on train
                    if lab == "sarimax":
                        mt = fit_sarimax(train, seasonal_periods)
                        if mt is None:
                            continue
                        fc = mt.forecast(steps=k)
                        preds = [float(invert_fn(v)) for v in fc]
                    elif lab == "holt":
                        mt = fit_holt_winters(train, seasonal_periods)
                        if mt is None:
                            continue
                        fc = mt.forecast(k)
                        preds = [float(invert_fn(v)) for v in fc]
                    elif lab == "holt_damped":
                        if not _HAS_STATSMODELS or len(train) < 4:
                            continue
                        mt = ExponentialSmoothing(train.astype(float), trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
                        fc = mt.forecast(k)
                        preds = [float(invert_fn(v)) for v in fc]
                    elif lab == "linear":
                        if len(train) < 2:
                            continue
                        a, b = fit_linear_regression(train.index.values.astype(float), train.values.astype(float))
                        preds_t = [float(a * fy + b) for fy in future]
                        preds = [float(invert_fn(v)) for v in preds_t]
                    else:
                        continue
                    score = _compute_smape(test.values.astype(float), np.array(preds, dtype=float))
                    scores.append((score, lab, preds))
                except Exception:
                    continue
            scores.sort(key=lambda x: x[0])
            if scores:
                best_label = scores[0][1]
                model_label, yhat = forecast_with_label(best_label)
        if not yhat:
            # Fallback ordered attempts
            for lab in candidates_labels:
                lab2, preds = forecast_with_label(lab)
                if preds:
                    model_label, yhat = lab2, preds
                    break

        # Fallback: linear regression on year -> value
        if not yhat:
            try:
                if len(values) >= 2:
                    a, b = fit_linear_regression(years.astype(float), values_t.astype(float))
                    preds_t = [float(a * fy + b) for fy in future_years]
                    yhat = [float(invert_fn(v)) for v in preds_t]
                    model_label = "linear_trend"
            except Exception:
                yhat = []

        # Fallback: median baseline (robust to outliers)
        if not yhat and use_median_fallback:
            if median_window and median_window > 1 and len(values_w) >= median_window:
                med = float(np.median(values_w[-median_window:]))
            else:
                med = float(np.median(values_w))
            yhat = [med for _ in future_years]
            model_label = "median_baseline"

        # Final fallback: repeat last observed value
        if not yhat:
            last_val = float(values[-1])
            yhat = [last_val for _ in future_years]
            model_label = "naive_last"

        results.append(
            ForecastResult(
                parameter_id=param_id,
                parameter_name=param_name,
                years=future_years,
                values=yhat,
                model=model_label,
                x_avg=x_avg,
                y_avg=y_avg,
            )
        )

    return results


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_arr = np.where(y_true == 0.0, np.nan, np.abs((y_true - y_pred) / y_true))
    mape = float(np.nanmean(mape_arr) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}


def _compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = (np.abs(y_true) + np.abs(y_pred))
        # Epsilon floor to avoid explosion when both values are ~0
        scale = np.nanmedian(np.abs(y_true))
        eps = max(1e-9, 0.1 * (float(scale) if np.isfinite(scale) and scale > 0 else 1e-2))
        denom = np.where(denom < eps, eps, denom)
        smape_arr = 200.0 * np.abs(y_pred - y_true) / denom
    return float(np.nanmean(smape_arr))


def backtest(
    df: pd.DataFrame,
    test_years: int = 3,
    seasonal_periods: Optional[int] = None,
    use_log_auto: bool = False,
    winsorize_pct: float = 0.0,
    use_smape: bool = False,
    use_median_fallback: bool = False,
    median_window: int = 0,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (param_id, param_name), g in df.groupby(["id", "param_name"], sort=False):
        g_sorted = g.sort_values("year")
        if len(g_sorted) <= test_years + 1:
            continue
        split_idx = len(g_sorted) - test_years
        train = g_sorted.iloc[:split_idx]
        test = g_sorted.iloc[split_idx:]

        # Fit model on train and forecast len(test) years ahead
        years = train["year"].astype(int).to_numpy()
        values = train["value"].astype(float).to_numpy()
        # Preprocess
        values_w = _winsorize(values, winsorize_pct)
        values_t, invert_fn = _maybe_log_transform(values_w, use_log_auto)
        start_year = int(years.max()) + 1
        future_years = list(range(start_year, start_year + len(test)))

        yhat: List[float] = []
        model_label = "naive_last"
        model = fit_holt_winters(pd.Series(values_t, index=pd.Index(years, name="year")), seasonal_periods)
        if model is not None:
            try:
                fc = model.forecast(len(test))
                yhat = [float(invert_fn(v)) for v in fc]
                model_label = "holt_winters_additive"
            except Exception:
                yhat = []
        if not yhat:
            if len(values) >= 2:
                try:
                    a, b = fit_linear_regression(years.astype(float), values_t.astype(float))
                    preds_t = [float(a * fy + b) for fy in future_years]
                    yhat = [float(invert_fn(v)) for v in preds_t]
                    model_label = "linear_trend"
                except Exception:
                    yhat = []
        if not yhat and use_median_fallback:
            if median_window and median_window > 1 and len(values_w) >= median_window:
                med = float(np.median(values_w[-median_window:]))
            else:
                med = float(np.median(values_w))
            yhat = [med for _ in future_years]
            model_label = "median_baseline"
        if not yhat:
            last_val = float(values[-1])
            yhat = [last_val for _ in future_years]
            model_label = "naive_last"

        y_true = test["value"].to_numpy(dtype=float)
        y_pred = np.array(yhat, dtype=float)
        metrics = _compute_metrics(y_true, y_pred)
        if use_smape:
            metrics["sMAPE_%"] = _compute_smape(y_true, y_pred)
        rows.append({
            "id": param_id,
            "param_name": param_name,
            "model": model_label,
            **metrics,
            "n_train": len(train),
            "n_test": len(test),
        })

    return pd.DataFrame(rows)


def _build_merged_yearly_from_outchem(base_dir: str) -> pd.DataFrame:
    """If merged_yearly.csv is missing, synthesize it from data/out/chem/*.csv."""
    outchem = os.path.join(base_dir, "data", "out", "chem")
    if not os.path.isdir(outchem):
        raise FileNotFoundError("data/out/chem directory not found; cannot synthesize merged_yearly.csv")
    rows: List[Dict[str, object]] = []
    for fname in os.listdir(outchem):
        if not fname.lower().endswith(".csv"):
            continue
        fpath = os.path.join(outchem, fname)
        try:
            d = pd.read_csv(fpath, sep=";")
        except Exception:
            continue
        if "datum" not in d.columns or "meetwaarde" not in d.columns:
            continue
        # Parse year
        d["datum"] = pd.to_datetime(d["datum"], errors="coerce")
        d = d.dropna(subset=["datum"])  # keep rows with valid dates
        d["year"] = d["datum"].dt.year
        # Identify fields
        param_id = os.path.splitext(fname)[0]
        param_name = str(d.get("fewsparameternaam", pd.Series([param_id])).dropna().iloc[0])
        eenheid = str(d.get("eenheid", pd.Series([None])).dropna().iloc[0]) if "eenheid" in d.columns else None
        # Group by year
        grp = d.groupby("year", as_index=False).agg(
            value=("meetwaarde", "mean"),
            x_avg=("x_location", "mean") if "x_location" in d.columns else ("year", "size"),
            y_avg=("y_location", "mean") if "y_location" in d.columns else ("year", "size"),
        )
        # Fix when x/y not available
        if "x_location" not in d.columns:
            grp["x_avg"] = np.nan
        if "y_location" not in d.columns:
            grp["y_avg"] = np.nan
        grp["id"] = param_id
        grp["param_name"] = param_name
        grp["eenheid"] = eenheid
        rows.append(grp[["id", "param_name", "year", "value", "eenheid", "x_avg", "y_avg"]])
    if not rows:
        raise FileNotFoundError("No chemical CSVs found under data/out/chem; cannot synthesize merged_yearly.csv")
    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["id", "year"]).reset_index(drop=True)
    return result


def load_merged_yearly(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        # Attempt to synthesize from data/out/chem
        base_dir = os.path.dirname(os.path.dirname(path))  # project root assumed two levels up from data/merged_yearly.csv
        try:
            synthesized = _build_merged_yearly_from_outchem(base_dir)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            synthesized.to_csv(path, sep=";", index=False)
            df = synthesized
        except Exception as e:
            raise FileNotFoundError(f"{path} not found and could not synthesize from data/out/chem: {e}")
    else:
        df = pd.read_csv(path, sep=";", dtype={"id": str, "param_name": str})
    # Normalize and validate columns
    required = ["id", "param_name", "year", "value"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in {path}")
    # Ensure numeric types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"]).copy()
    df["year"] = df["year"].astype(int)
    return df


def save_forecasts(results: List[ForecastResult], out_path: str) -> pd.DataFrame:
    rows = []
    for r in results:
        for y, v in zip(r.years, r.values):
            rows.append(
                {
                    "id": r.parameter_id,
                    "param_name": r.parameter_name,
                    "year": int(y),
                    "forecast": v,
                    "model": r.model,
                    "x_avg": r.x_avg,
                    "y_avg": r.y_avg,
                }
            )
    out_df = pd.DataFrame(rows)
    # Ensure years are stored as full integer years in the CSV
    if "year" in out_df.columns:
        out_df["year"] = out_df["year"].astype(int)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df


def maybe_plot(
    df: pd.DataFrame, forecasts: pd.DataFrame, out_dir: str, max_plots: Optional[int] = 20
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.ticker import MaxNLocator  # type: ignore
    except Exception:
        warnings.warn("matplotlib not available; skipping plots")
        return

    os.makedirs(out_dir, exist_ok=True)
    grouped = list(df.groupby(["id", "param_name"]))
    if max_plots is not None:
        grouped = grouped[:max(0, max_plots)]

    for (pid, pname), g in grouped:
        hist = g.sort_values("year")
        f = forecasts[(forecasts["id"] == pid) & (forecasts["param_name"] == pname)].sort_values("year")
        if f.empty:
            continue
        plt.figure(figsize=(8, 4))
        plt.plot(hist["year"], hist["value"], marker="o", label="observed")
        plt.plot(f["year"], f["forecast"], marker="x", label="forecast")
        plt.title(f"{pname} ({pid})")
        plt.xlabel("Jaar")
        plt.ylabel("Waarde")
        plt.grid(True, alpha=0.3)
        plt.legend()
        # Force integer ticks on the year axis
        ax = plt.gca()
        try:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        except Exception:
            pass

        safe_id = pid.replace("/", "-")
        out_path = os.path.join(out_dir, f"{safe_id}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forecast per-parameter yearly values from merged_yearly.csv")
    p.add_argument(
        "--input",
        default=os.path.join("data", "merged_yearly.csv"),
        help="Path to merged_yearly.csv (semicolon-delimited)",
    )
    p.add_argument(
        "--output",
        default=os.path.join("data", "out", "forecasts.csv"),
        help="Output CSV path for forecasts",
    )
    p.add_argument("--horizon", type=int, default=5, help="Number of future years to forecast")
    p.add_argument(
        "--seasonal-periods",
        type=int,
        default=None,
        help="Optional Holt-Winters seasonal periods (e.g., 5 if you expect 5-year cycles)",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="If set, saves example forecast plots to data/graphs/",
    )
    p.add_argument(
        "--max-plots",
        type=int,
        default=20,
        help="Maximum number of parameter plots to generate (when --plots)",
    )
    p.add_argument(
        "--plots-out-dir",
        default=os.path.join("data", "out", "forecasts"),
        help="Directory to save forecast plots when --plots is set",
    )
    p.add_argument(
        "--backtest-years",
        type=int,
        default=3,
        help="Backtest with last N years held out; 0 disables backtesting",
    )
    p.add_argument(
        "--print-rows",
        type=int,
        default=15,
        help="Number of forecast rows to print to terminal",
    )
    p.add_argument(
        "--accuracy-output",
        default=os.path.join("data", "out", "accuracy.csv"),
        help="Output CSV path for per-parameter accuracy (100 - MAPE)",
    )
    p.add_argument(
        "--use-log-auto",
        action="store_true",
        help="Auto apply log1p transform for volatile positive series",
    )
    p.add_argument(
        "--winsorize-pct",
        type=float,
        default=0.0,
        help="Clip extremes at given quantile (e.g., 0.02 for 2% tails)",
    )
    p.add_argument(
        "--use-smape",
        action="store_true",
        help="Compute sMAPE in backtest and export accuracy as 100 - sMAPE",
    )
    p.add_argument(
        "--use-median-fallback",
        action="store_true",
        help="Enable robust median baseline fallback before naive",
    )
    p.add_argument(
        "--median-window",
        type=int,
        default=0,
        help="Median window size (years). 0 = full history",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Sensible robust defaults so running without flags "just works"
    if not getattr(args, "use_log_auto", False):
        args.use_log_auto = True
    if float(getattr(args, "winsorize_pct", 0.0)) == 0.0:
        args.winsorize_pct = 0.05
    if not getattr(args, "use_smape", False):
        args.use_smape = True
    if not getattr(args, "use_median_fallback", False):
        args.use_median_fallback = True
    if int(getattr(args, "median_window", 0)) == 0:
        args.median_window = 5

    # Resolve paths relative to script directory when needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.input):
        args.input = os.path.join(script_dir, args.input)
    # If input still not found, search recursively for merged_yearly.csv
    if not os.path.exists(args.input):
        candidate_name = os.path.basename(args.input)
        for root, _dirs, files in os.walk(script_dir):
            if candidate_name in files:
                args.input = os.path.join(root, candidate_name)
                break
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)
    if hasattr(args, "accuracy_output") and not os.path.isabs(args.accuracy_output):
        args.accuracy_output = os.path.join(script_dir, args.accuracy_output)

    df = load_merged_yearly(args.input)
    # Optional: auto-force a simpler model for parameters with poor backtest accuracy
    forced_models: Dict[Tuple[str, str], str] = {}

    results = forecast_series(
        df,
        horizon=args.horizon,
        seasonal_periods=args.seasonal_periods,
        use_log_auto=bool(args.use_log_auto),
        winsorize_pct=float(args.winsorize_pct),
        use_median_fallback=bool(args.use_median_fallback),
        median_window=int(args.median_window),
        forced_models=forced_models,
    )
    out_df = save_forecasts(results, args.output)
    # Backtest metrics
    if args.backtest_years != 0:
        bt = backtest(
            df,
            test_years=args.backtest_years,
            seasonal_periods=args.seasonal_periods,
            use_log_auto=bool(args.use_log_auto),
            winsorize_pct=float(args.winsorize_pct),
            use_smape=bool(args.use_smape),
            use_median_fallback=bool(args.use_median_fallback),
            median_window=int(args.median_window),
        )
        if not bt.empty:
            overall = {"MAE": float(bt["MAE"].mean()), "RMSE": float(bt["RMSE"].mean())}
            if "sMAPE_%" in bt.columns:
                overall["sMAPE_%"] = float(bt["sMAPE_%"].mean())
            if "MAPE_%" in bt.columns:
                overall["MAPE_%"] = float(bt["MAPE_%"].mean())
            print("\nAccuracy (backtest, mean across parameters):")
            print(pd.DataFrame([overall]).to_string(index=False))
            print("\nPer-parameter (head):")
            sort_col = "sMAPE_%" if "sMAPE_%" in bt.columns else "RMSE"
            print(bt.sort_values(sort_col).head(10).to_string(index=False))
            # Export per-parameter accuracy percentage
            # Build accuracy export with clamped error in [0,100]
            err_col = "sMAPE_%" if "sMAPE_%" in bt.columns else "MAPE_%"
            cols = ["id", "param_name"] + (["MAPE_%"] if "MAPE_%" in bt.columns else []) + (["sMAPE_%"] if "sMAPE_%" in bt.columns else [])
            acc = bt[cols].copy()
            acc["err_%"] = np.clip(bt[err_col].to_numpy(dtype=float), 0.0, 100.0)
            acc["accuracy_%"] = 100.0 - acc["err_%"]
            acc = acc.sort_values(["id"])  # tidy order
            os.makedirs(os.path.dirname(args.accuracy_output), exist_ok=True)
            acc.to_csv(args.accuracy_output, index=False)
            print(f"\nWrote per-parameter accuracy: {args.accuracy_output} ({len(acc)} rows)")

            # Force alternative model for low-accuracy parameters and re-forecast
            low = acc[acc["accuracy_%"] < 70.0]
            if not low.empty:
                print("\nRe-forecasting low-accuracy parameters (<70%) using median baseline...")
                for _, row in low.iterrows():
                    # Prefer median; if series is highly volatile and median still poor, allow mean later
                    forced_models[(str(row["id"]), str(row["param_name"]))] = "median_baseline"
                results2 = forecast_series(
                    df,
                    horizon=args.horizon,
                    seasonal_periods=args.seasonal_periods,
                    use_log_auto=bool(args.use_log_auto),
                    winsorize_pct=float(args.winsorize_pct),
                    use_median_fallback=True,
                    median_window=int(args.median_window),
                    forced_models=forced_models,
                )
                out_df = save_forecasts(results2, args.output)
                print("Re-forecast complete.")

                # Optional second pass: for any still <70% after forcing median, force mean
                bt2 = backtest(
                    df,
                    test_years=args.backtest_years,
                    seasonal_periods=args.seasonal_periods,
                    use_log_auto=bool(args.use_log_auto),
                    winsorize_pct=float(args.winsorize_pct),
                    use_smape=bool(args.use_smape),
                    use_median_fallback=True,
                    median_window=int(args.median_window),
                )
                if not bt2.empty:
                    err_col2 = "sMAPE_%" if "sMAPE_%" in bt2.columns else "MAPE_%"
                    acc2 = bt2[["id", "param_name", err_col2]].copy()
                    acc2["accuracy_%"] = 100.0 - np.clip(acc2[err_col2].to_numpy(dtype=float), 0.0, 100.0)
                    low2 = acc2[acc2["accuracy_%"] < 70.0]
                    if not low2.empty:
                        print("Re-forecasting remaining low-accuracy parameters with mean baseline...")
                        for _, row in low2.iterrows():
                            forced_models[(str(row["id"]), str(row["param_name"]))] = "mean_baseline"
                        results3 = forecast_series(
                            df,
                            horizon=args.horizon,
                            seasonal_periods=args.seasonal_periods,
                            use_log_auto=bool(args.use_log_auto),
                            winsorize_pct=float(args.winsorize_pct),
                            use_median_fallback=True,
                            median_window=int(args.median_window),
                            forced_models=forced_models,
                        )
                        out_df = save_forecasts(results3, args.output)
                        print("Mean-baseline re-forecast complete.")

            # Filter out forecasts for parameters with accuracy below threshold
            ACC_MIN = 70.0
            if "accuracy_%" in acc.columns:
                allow = acc[acc["accuracy_%"] >= ACC_MIN][["id", "param_name"]].copy()
                before = len(out_df)
                out_df = out_df.merge(allow, on=["id", "param_name"], how="inner")
                after = len(out_df)
                save_forecasts([
                    ForecastResult(parameter_id=r["id"], parameter_name=r["param_name"], years=[int(r["year"])], values=[float(r["forecast"])], model=str(r["model"]), x_avg=float(r["x_avg"]) if pd.notna(r["x_avg"]) else None, y_avg=float(r["y_avg"]) if pd.notna(r["y_avg"]) else None)
                    for _, r in out_df.iterrows()
                ], args.output)
                print(f"Filtered forecasts by accuracy >= {ACC_MIN}%: {before} -> {after} rows")
        else:
            print("\nBacktest skipped (insufficient data per parameter).")
    # Print sample of forecasts
    n_print = max(0, int(args.print_rows))
    if n_print > 0:
        print("\nForecast sample:")
        print(out_df.sort_values(["id", "year"]).head(n_print).to_string(index=False))
    if args.plots:
        # Determine plots output dir; reuse plots-out-dir if provided
        plots_out_dir = getattr(args, "plots_out_dir", os.path.join("data", "out", "forecasts"))
        if not os.path.isabs(plots_out_dir):
            plots_out_dir = os.path.join(script_dir, plots_out_dir)
        maybe_plot(df, out_df, plots_out_dir, max_plots=args.max_plots)
        # Also copy plots and write manifest for frontend consumption under frontend/public/forecasts
        try:
            frontend_public = os.path.join(script_dir, "frontend", "public", "forecasts")
            os.makedirs(frontend_public, exist_ok=True)
            # Copy PNGs
            for fn in os.listdir(plots_out_dir):
                if fn.lower().endswith(".png"):
                    src = os.path.join(plots_out_dir, fn)
                    dst = os.path.join(frontend_public, fn)
                    shutil.copyfile(src, dst)
            # Build manifest from forecasts actually available
            ids = sorted(set(out_df["id"].astype(str).unique().tolist()))
            # Keep only ids that have a png
            ids = [pid for pid in ids if os.path.exists(os.path.join(frontend_public, f"{pid}.png"))]
            manifest_rows = (
                out_df.sort_values(["id"]).drop_duplicates(["id"])[["id", "param_name"]].to_dict(orient="records")
            )
            # Filter manifest to available images
            manifest_rows = [r for r in manifest_rows if r["id"] in ids]
            import json  # local import to avoid top-level dependency
            with open(os.path.join(frontend_public, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"items": manifest_rows}, f, ensure_ascii=False, indent=2)
            print(f"Wrote frontend forecast plots and manifest to: {frontend_public}")
        except Exception as e:
            print(f"Warning: failed to write frontend plots/manifest: {e}")
    print(f"Wrote forecasts: {args.output} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()


