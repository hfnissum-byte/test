from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .features import build_sequence_dataset
from .transformer_lstm import ClassicalSequenceModelConfig, ClassicalSequenceEstimator


def train_classical_model(
    pattern_csv_path: str,
    out_dir: str,
    *,
    symbols: Optional[Sequence[str]] = None,
    timeframes: Optional[Sequence[str]] = None,
    horizon: str = "1h",
    seq_len: int = 12,
    stride: int = 1,
    max_labeled_rows: int = 25_000,
    max_windows: Optional[int] = 50_000,
    seed: int = 42,
) -> dict[str, float]:
    # Step 1 correctness: time-correct holdout with 24h embargo.
    embargo_ms = 24 * 60 * 60 * 1000
    val_fraction = 0.2

    """
    Train the Transformer+LSTM hybrid baseline.

    In runtimes without torch, this uses the sklearn fallback (flattened
    embedding sequences) but keeps the same estimator API.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = build_sequence_dataset(
        pattern_csv_path=pattern_csv_path,
        symbols=symbols,
        timeframes=timeframes,
        seq_len=seq_len,
        horizon=horizon,
        max_labeled_rows=max_labeled_rows,
        seed=seed,
        stride=stride,
    )

    X_seq = dataset.X_seq
    y_dir = dataset.y_dir
    y_ret = dataset.y_return
    ts_end = dataset.ts_end

    if max_windows is not None and len(X_seq) > max_windows:
        order = np.argsort(ts_end)
        keep = order[-int(max_windows) :]
        X_seq = X_seq[keep]
        y_dir = y_dir[keep]
        ts_end = ts_end[keep]
        if y_ret is not None:
            y_ret = y_ret[keep]

    cfg = ClassicalSequenceModelConfig(embedding_dim=X_seq.shape[-1], seq_len=seq_len, seed=seed)
    est = ClassicalSequenceEstimator(cfg=cfg)

    # Time-based split (single holdout) with embargo to reduce horizon overlap leakage.
    order = np.argsort(ts_end)
    X_seq = X_seq[order]
    y_dir = y_dir[order]
    ts_end = ts_end[order]
    if y_ret is not None:
        y_ret = y_ret[order]

    n = len(ts_end)
    if n < 10:
        raise ValueError(f"Not enough windows for time split: n={n}")

    n_train_end = int(np.floor(n * (1.0 - val_fraction)))
    n_train_end = max(1, min(n_train_end, n - 1))
    val_start_ts = int(ts_end[n_train_end])
    train_cutoff_ts = int(val_start_ts - embargo_ms)

    train_mask = ts_end <= train_cutoff_ts
    val_mask = ts_end >= val_start_ts

    if int(np.sum(train_mask)) < 5 or int(np.sum(val_mask)) < 5:
        raise ValueError(
            f"Time split produced too few samples (train={int(np.sum(train_mask))}, val={int(np.sum(val_mask))}, "
            f"val_start_ts={val_start_ts}, embargo_ms={embargo_ms})"
        )

    X_train = X_seq[train_mask]
    y_dir_train = y_dir[train_mask]
    X_val = X_seq[val_mask]
    y_dir_val = y_dir[val_mask]

    if y_ret is not None:
        y_ret_train = y_ret[train_mask]
        y_ret_val = y_ret[val_mask]
    else:
        y_ret_train = None
        y_ret_val = None

    metrics = est.fit(
        X_train,
        y_dir_train,
        y_return=y_ret_train,
        X_val_seq=X_val,
        y_dir_val=y_dir_val,
        y_return_val=y_ret_val,
    )

    # Save
    model_path = out_path / "classical_model.joblib"
    est.save(model_path)

    meta_path = out_path / "classical_model_meta.json"
    meta = {
        "pattern_csv_path": pattern_csv_path,
        "horizon": horizon,
        "symbols": list(symbols) if symbols else None,
        "timeframes": list(timeframes) if timeframes else None,
        "seq_len": seq_len,
        "stride": stride,
        "max_labeled_rows": max_labeled_rows,
        "max_windows": max_windows,
        "cfg": asdict(cfg),
        "metrics": metrics,
        "num_windows": int(len(X_seq)),
        "backend": est.backend,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return metrics

