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

    if max_windows is not None and len(X_seq) > max_windows:
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(X_seq), size=max_windows, replace=False)
        X_seq = X_seq[idxs]
        y_dir = y_dir[idxs]
        if y_ret is not None:
            y_ret = y_ret[idxs]

    cfg = ClassicalSequenceModelConfig(embedding_dim=X_seq.shape[-1], seq_len=seq_len, seed=seed)
    est = ClassicalSequenceEstimator(cfg=cfg)
    metrics = est.fit(X_seq, y_dir, y_return=y_ret)

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

