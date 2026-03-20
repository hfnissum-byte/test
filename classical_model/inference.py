from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .features import build_sequence_dataset
from .transformer_lstm import ClassicalSequenceEstimator, ClassicalSequenceModelConfig


def load_model(model_path: str | Path) -> ClassicalSequenceEstimator:
    return ClassicalSequenceEstimator.load(Path(model_path))


def predict_proba(model: ClassicalSequenceEstimator, X_seq: np.ndarray) -> np.ndarray:
    return model.predict_proba(X_seq)


def predict_expected_return(model: ClassicalSequenceEstimator, X_seq: np.ndarray) -> np.ndarray:
    return model.predict_expected_return(X_seq)


def build_and_predict_from_pattern_csv(
    model: ClassicalSequenceEstimator,
    pattern_csv_path: str,
    *,
    symbols: Optional[Sequence[str]] = None,
    timeframes: Optional[Sequence[str]] = None,
    horizon: str = "1h",
    seq_len: Optional[int] = None,
    stride: int = 1,
    max_labeled_rows: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience: build sequences from pattern CSV then run inference.
    """
    if seq_len is None:
        seq_len = model.cfg.seq_len

    dataset = build_sequence_dataset(
        pattern_csv_path,
        symbols=symbols,
        timeframes=timeframes,
        seq_len=int(seq_len),
        horizon=horizon,
        max_labeled_rows=max_labeled_rows,
        seed=42,
        stride=stride,
    )
    proba = model.predict_proba(dataset.X_seq)
    return proba, dataset.X_seq

