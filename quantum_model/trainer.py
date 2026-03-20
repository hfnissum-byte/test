from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

from .dataset import build_training_arrays
from .hybrid_model import HybridModelConfig, HybridQuantumInspiredEstimator


def train_quantum_model(
    pattern_csv_path: str,
    out_dir: str,
    symbols: Optional[Sequence[str]] = None,
    timeframes: Optional[Sequence[str]] = None,
    horizon: str = "1h",
    max_rows: int = 50_000,
    seed: int = 42,
    cfg: Optional[HybridModelConfig] = None,
) -> dict[str, float]:
    # Step 1 correctness: time-correct holdout with 24h embargo.
    embargo_ms = 24 * 60 * 60 * 1000
    val_fraction = 0.2

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if cfg is None:
        cfg = HybridModelConfig(seed=seed)

    arrays = build_training_arrays(
        pattern_csv_path=pattern_csv_path,
        symbols=symbols,
        timeframes=timeframes,
        horizon=horizon,
        max_rows=max_rows,
        seed=seed,
        embedding_dim=cfg.embedding_dim,
    )

    X = arrays.X
    y_dir = arrays.y_dir
    y_ret = arrays.y_return
    timestamps = arrays.timestamps

    order = np.argsort(timestamps)
    X = X[order]
    y_dir = y_dir[order]
    timestamps = timestamps[order]
    if y_ret is not None:
        y_ret = y_ret[order]

    n = len(timestamps)
    if n < 10:
        raise ValueError(f"Not enough samples for time split: n={n}")

    n_train_end = int(np.floor(n * (1.0 - val_fraction)))
    n_train_end = max(1, min(n_train_end, n - 1))
    val_start_ts = int(timestamps[n_train_end])
    train_cutoff_ts = int(val_start_ts - embargo_ms)

    train_mask = timestamps <= train_cutoff_ts
    val_mask = timestamps >= val_start_ts

    if int(np.sum(train_mask)) < 5 or int(np.sum(val_mask)) < 5:
        raise ValueError(
            f"Time split produced too few samples (train={int(np.sum(train_mask))}, val={int(np.sum(val_mask))}, "
            f"val_start_ts={val_start_ts}, embargo_ms={embargo_ms})"
        )

    X_train = X[train_mask]
    y_dir_train = y_dir[train_mask]
    X_val = X[val_mask]
    y_dir_val = y_dir[val_mask]

    if y_ret is not None:
        y_ret_train = y_ret[train_mask]
        y_ret_val = y_ret[val_mask]
    else:
        y_ret_train = None
        y_ret_val = None

    est = HybridQuantumInspiredEstimator(cfg=cfg)
    est.fit(
        X_train,
        y_dir_train,
        y_return=y_ret_train,
        X_val=X_val,
        y_dir_val=y_dir_val,
        y_return_val=y_ret_val,
    )

    # Validation metrics
    proba = est.predict_proba(X_val)
    y_pred = (proba >= 0.5).astype(int)
    metrics: dict[str, float] = {
        "val_accuracy": float(accuracy_score(y_dir_val, y_pred)),
    }
    # ROC-AUC requires both classes in the val split.
    if len(np.unique(y_dir_val)) == 2:
        metrics["val_roc_auc"] = float(roc_auc_score(y_dir_val, proba))

    if y_ret_val is not None:
        ret_pred = est.predict_expected_return(X_val)
        metrics["val_return_mae"] = float(mean_absolute_error(y_ret_val, ret_pred))

    # Save model + minimal metadata for auditing.
    model_path = out_path / "quantum_model.joblib"
    est.save(model_path)

    meta_path = out_path / "quantum_model_meta.json"
    meta = {
        "pattern_csv_path": pattern_csv_path,
        "horizon": horizon,
        "symbols": list(symbols) if symbols else None,
        "timeframes": list(timeframes) if timeframes else None,
        "cfg": cfg.__dict__,
        "metrics": metrics,
        "num_samples": int(len(X)),
        "backend": est.backend,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return metrics

