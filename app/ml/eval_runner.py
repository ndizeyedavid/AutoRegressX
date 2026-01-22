from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from app.ml.paths import ensure_runs_dir


def _emit(event: str, payload: dict) -> None:
    line = json.dumps({"event": event, **payload}, ensure_ascii=False)
    print(line, flush=True)


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _plot_prediction_only(y_pred: np.ndarray, plots_dir: Path) -> None:
    plt.figure(figsize=(6.4, 4.0))
    sns.set_style("whitegrid")
    sns.histplot(y_pred, bins=30, kde=True)
    plt.title("Prediction Distribution")
    plt.xlabel("Predicted")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "pred_distribution.png", dpi=160)
    plt.close()


def _plot_with_target(y_true: np.ndarray, y_pred: np.ndarray, plots_dir: Path) -> None:
    residuals = y_true - y_pred

    # Parity
    plt.figure(figsize=(5.2, 5.2))
    sns.set_style("whitegrid")
    sns.scatterplot(x=y_true, y=y_pred, s=22, alpha=0.7)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], color="#fb7185", linewidth=1.5)
    plt.title("Parity Plot (True vs Predicted)")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(plots_dir / "parity.png", dpi=160)
    plt.close()

    # Residuals
    plt.figure(figsize=(6.4, 4.8))
    sns.set_style("whitegrid")
    sns.scatterplot(x=y_pred, y=residuals, s=22, alpha=0.7)
    plt.axhline(0.0, color="#fb7185", linewidth=1.5)
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(plots_dir / "residuals.png", dpi=160)
    plt.close()

    # Residual distribution
    plt.figure(figsize=(6.4, 4.0))
    sns.set_style("whitegrid")
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "residual_distribution.png", dpi=160)
    plt.close()

    _plot_prediction_only(y_pred, plots_dir)


def run(model_dir: str, csv_path: str, max_rows: int = 100) -> int:
    start_all = time.perf_counter()

    mdir = Path(model_dir).expanduser().resolve()
    if not mdir.exists() or not mdir.is_dir():
        _emit("error", {"message": "Model directory not found", "model_dir": str(mdir)})
        return 2

    model_path = mdir / "model.joblib"
    schema_path = mdir / "schema.json"

    if not model_path.exists() or not schema_path.exists():
        _emit(
            "error",
            {
                "message": "Model directory must contain model.joblib and schema.json",
                "model": str(model_path),
                "schema": str(schema_path),
            },
        )
        return 2

    _emit("log", {"level": "INFO", "message": "Loading model + schema"})
    try:
        model = joblib.load(model_path)
    except Exception as e:
        _emit("error", {"message": f"Failed to load model.joblib: {e}"})
        return 2

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as e:
        _emit("error", {"message": f"Failed to read schema.json: {e}"})
        return 2

    target = str(schema.get("target", "")).strip() or None
    feature_cols = schema.get("feature_columns", [])
    if not isinstance(feature_cols, list) or not feature_cols:
        _emit("error", {"message": "schema.json missing feature_columns"})
        return 2

    p = Path(csv_path).expanduser().resolve()
    if not p.exists() or p.suffix.lower() != ".csv":
        _emit("error", {"message": "CSV path is invalid", "csv": str(p)})
        return 2

    _emit("log", {"level": "INFO", "message": "Loading dataset"})
    try:
        df = pd.read_csv(p)
    except Exception as e:
        _emit("error", {"message": f"Failed to read CSV: {e}"})
        return 2

    if len(df) > int(max_rows):
        _emit(
            "log",
            {
                "level": "WARN",
                "message": f"Dataset has {len(df):,} rows; evaluating only first {int(max_rows):,} rows.",
            },
        )
        df = df.head(int(max_rows)).copy()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        _emit(
            "error",
            {
                "message": f"Dataset is missing required feature columns: {', '.join(map(str, missing[:10]))}{'...' if len(missing) > 10 else ''}",
            },
        )
        return 2

    X = df[feature_cols].copy()

    has_target = bool(target) and (target in df.columns)
    y_true = None
    if has_target and target is not None:
        y_true = pd.to_numeric(df[target], errors="coerce")
        if y_true.isna().any():
            bad = int(y_true.isna().sum())
            _emit("log", {"level": "WARN", "message": f"Target has {bad} non-numeric values; dropping those rows."})
            mask = ~y_true.isna()
            X = X.loc[mask].copy()
            y_true = y_true.loc[mask]

    runs_dir = ensure_runs_dir()
    eval_dir = runs_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _emit("run_started", {"run_dir": str(eval_dir)})

    _emit("log", {"level": "INFO", "message": "Running predictions"})
    try:
        y_pred = model.predict(X)
    except Exception as e:
        _emit("error", {"message": f"Prediction failed: {e}"})
        return 2

    y_pred = np.asarray(y_pred).reshape(-1)

    metrics: dict[str, float | None] = {"r2": None, "mae": None, "rmse": None}
    if y_true is not None:
        y_arr = y_true.to_numpy()
        metrics["r2"] = float(r2_score(y_arr, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_arr, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_arr, y_pred)))

    eval_summary = {
        "model_dir": str(mdir),
        "csv_path": str(p),
        "target": target,
        "target_present": bool(y_true is not None),
        "n_rows": int(len(X)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    _save_json(eval_dir / "eval_metrics.json", eval_summary)

    out_df = pd.DataFrame({"y_pred": y_pred})
    if y_true is not None:
        out_df.insert(0, "y_true", y_true.to_numpy())
    out_df.to_csv(eval_dir / "eval_predictions.csv", index=False)

    try:
        if y_true is not None:
            _plot_with_target(y_true.to_numpy(), y_pred, plots_dir)
        else:
            _plot_prediction_only(y_pred, plots_dir)
    except Exception as e:
        _emit("log", {"level": "WARN", "message": f"Plot generation failed: {e}"})

    total_elapsed = max(0.0, time.perf_counter() - start_all)
    _emit(
        "run_finished",
        {
            "run_dir": str(eval_dir),
            "seconds": float(total_elapsed),
            "target_present": bool(y_true is not None),
            "n_rows": int(len(X)),
            "metrics": metrics,
        },
    )
    return 0


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-rows", type=int, default=100)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    return run(args.model_dir, args.csv, max_rows=args.max_rows)


if __name__ == "__main__":
    raise SystemExit(main())
