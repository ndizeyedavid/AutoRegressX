from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import joblib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from app.ml.paths import ensure_runs_dir


MODEL_SPECS: list[tuple[str, object]] = [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge()),
    (
        "Random Forest",
        RandomForestRegressor(
            n_estimators=120,
            max_depth=22,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
    ("SVR", SVR()),
    ("KNN Regression", KNeighborsRegressor()),
]


def _emit(event: str, payload: dict) -> None:
    line = json.dumps({"event": event, **payload}, ensure_ascii=False)
    print(line, flush=True)


def _build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    # Attempt to coerce numeric-looking object columns (currency, commas, etc.) into real numeric.
    X = X.copy()
    for c in X.columns:
        if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]):
            s = X[c].astype(str)
            s = s.str.replace(",", "", regex=False)
            s = s.str.replace("$", "", regex=False)
            s = s.str.replace(" ", "", regex=False)
            coerced = pd.to_numeric(s, errors="coerce")
            # If almost all values convert, treat it as numeric.
            non_null = int(X[c].notna().sum())
            ok = int(coerced.notna().sum())
            if non_null > 0 and (ok / non_null) >= 0.98:
                X[c] = coerced

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Guard against huge one-hot expansions.
    kept_cat: list[str] = []
    dropped_cat: list[str] = []
    for c in categorical_cols:
        try:
            nunique = int(X[c].nunique(dropna=True))
        except Exception:
            nunique = 0
        if nunique > 80:
            dropped_cat.append(c)
        else:
            kept_cat.append(c)
    categorical_cols = kept_cat

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    # Attach diagnostics to help with logging (without expanding function signature too much).
    preprocessor._autoregressex_dropped_cat = dropped_cat  # type: ignore[attr-defined]

    return preprocessor, numeric_cols, categorical_cols


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _plot_model_comparison(results: dict[str, dict], plots_dir: Path) -> None:
    names = list(results.keys())
    r2s = [results[n]["r2"] for n in names]

    plt.figure(figsize=(9, 4.6))
    sns.set_style("whitegrid")
    ax = sns.barplot(x=names, y=r2s, palette="viridis")
    ax.set_title("Model Comparison (R²)")
    ax.set_xlabel("Model")
    ax.set_ylabel("R²")
    ax.set_ylim(min(-1.0, float(min(r2s)) - 0.05), 1.0)
    plt.xticks(rotation=20, ha="right")

    for i, v in enumerate(r2s):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison_r2.png", dpi=160)
    plt.close()


def _plot_best_performance(y_true: np.ndarray, y_pred: np.ndarray, plots_dir: Path, best_name: str) -> None:
    residuals = y_true - y_pred

    # Parity
    plt.figure(figsize=(5.2, 5.2))
    sns.set_style("whitegrid")
    sns.scatterplot(x=y_true, y=y_pred, s=22, alpha=0.7)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], color="#fb7185", linewidth=1.5)
    plt.title(f"Parity Plot (Best: {best_name})")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(plots_dir / "best_parity.png", dpi=160)
    plt.close()

    # Residuals
    plt.figure(figsize=(6.4, 4.8))
    sns.set_style("whitegrid")
    sns.scatterplot(x=y_pred, y=residuals, s=22, alpha=0.7)
    plt.axhline(0.0, color="#fb7185", linewidth=1.5)
    plt.title(f"Residuals vs Predicted (Best: {best_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(plots_dir / "best_residuals.png", dpi=160)
    plt.close()

    # Residual distribution
    plt.figure(figsize=(6.4, 4.0))
    sns.set_style("whitegrid")
    sns.histplot(residuals, bins=30, kde=True)
    plt.title(f"Residual Distribution (Best: {best_name})")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "best_residual_distribution.png", dpi=160)
    plt.close()


def run(csv_path: str, target: str, seed: int = 42, test_size: float = 0.2) -> int:
    start_all = time.perf_counter()

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

    if target not in df.columns:
        _emit("error", {"message": "Target column not found", "target": target})
        return 2

    df = df.dropna(axis=0, how="all")

    y_raw = df[target]
    y = pd.to_numeric(y_raw, errors="coerce")
    if y.isna().all():
        examples = [str(v) for v in y_raw.dropna().astype(str).unique().tolist()[:5]]
        example_txt = ", ".join(examples) if examples else "(no non-null values)"
        _emit(
            "error",
            {
                "message": (
                    "Regression requires a numeric target column. "
                    f"The selected target '{target}' appears non-numeric (e.g. {example_txt}). "
                    "Choose a numeric target column."
                )
            },
        )
        return 2

    if y.isna().any():
        bad = int(y.isna().sum())
        _emit(
            "log",
            {
                "level": "WARN",
                "message": f"Target contains {bad} non-numeric values; dropping those rows.",
            },
        )
        mask = ~y.isna()
        df = df.loc[mask].copy()
        y = y.loc[mask]

    X = df.drop(columns=[target])

    # Drop completely empty columns
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        _emit("error", {"message": "No feature columns available after cleaning"})
        return 2

    _emit(
        "log",
        {
            "level": "INFO",
            "message": f"Splitting data (test_size={test_size:.2f}, seed={seed})",
        },
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    preprocessor, numeric_cols, categorical_cols = _build_preprocessor(X_train)
    dropped_cat = getattr(preprocessor, "_autoregressex_dropped_cat", [])
    if dropped_cat:
        _emit(
            "log",
            {
                "level": "WARN",
                "message": f"Dropping high-cardinality categorical columns: {', '.join(map(str, dropped_cat[:8]))}{'...' if len(dropped_cat) > 8 else ''}",
            },
        )

    _emit(
        "log",
        {
            "level": "INFO",
            "message": f"Features: {X_train.shape[1]} columns (numeric={len(numeric_cols)}, categorical={len(categorical_cols)})",
        },
    )

    runs_dir = ensure_runs_dir()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _emit("run_started", {"run_dir": str(run_dir)})

    schema = {
        "csv_path": str(p),
        "target": target,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "feature_columns": list(map(str, X.columns.tolist())),
        "numeric_columns": list(map(str, numeric_cols)),
        "categorical_columns": list(map(str, categorical_cols)),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "seed": int(seed),
        "test_size": float(test_size),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_json(run_dir / "schema.json", schema)

    results: dict[str, dict] = {}
    best_name: str | None = None
    best_r2 = -float("inf")
    best_pipeline: Pipeline | None = None
    best_y_pred: np.ndarray | None = None

    for name, estimator in MODEL_SPECS:
        _emit("model_started", {"name": name})
        _emit("log", {"level": "INFO", "message": f"Training {name}"})

        pipeline = Pipeline(steps=[("prep", preprocessor), ("model", estimator)])
        t0 = time.perf_counter()

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
        except Exception as e:
            _emit("log", {"level": "ERROR", "message": f"{name} failed: {e}"})
            continue

        elapsed = max(0.0, time.perf_counter() - t0)

        r2 = float(r2_score(y_val, y_pred))
        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

        results[name] = {"r2": r2, "mae": mae, "rmse": rmse, "seconds": float(elapsed)}
        _emit(
            "model_finished",
            {"name": name, "r2": r2, "mae": mae, "rmse": rmse, "seconds": float(elapsed)},
        )

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipeline = pipeline
            best_y_pred = np.asarray(y_pred)

    if not results or best_pipeline is None or best_name is None or best_y_pred is None:
        _emit("error", {"message": "Training failed for all models"})
        return 2

    _emit("log", {"level": "SUCCESS", "message": f"Best model: {best_name} (R²={best_r2:.3f})"})

    # Persist best model pipeline
    joblib.dump(best_pipeline, run_dir / "model.joblib")

    # Save metrics
    metrics = {
        "best_model": best_name,
        "best_r2": float(best_r2),
        "per_model": results,
    }
    _save_json(run_dir / "metrics.json", metrics)

    # Save validation predictions
    pd.DataFrame({"y_true": y_val.to_numpy(), "y_pred": best_y_pred}).to_csv(
        run_dir / "val_predictions.csv", index=False
    )

    # Plots
    try:
        _plot_model_comparison(results, plots_dir)
        _plot_best_performance(y_val.to_numpy(), best_y_pred, plots_dir, best_name)
    except Exception as e:
        _emit("log", {"level": "WARN", "message": f"Plot generation failed: {e}"})

    total_elapsed = max(0.0, time.perf_counter() - start_all)
    _emit(
        "run_finished",
        {
            "run_dir": str(run_dir),
            "best_model": best_name,
            "best_r2": float(best_r2),
            "seconds": float(total_elapsed),
        },
    )
    return 0


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    return run(args.csv, args.target, seed=args.seed, test_size=args.test_size)


if __name__ == "__main__":
    raise SystemExit(main())
