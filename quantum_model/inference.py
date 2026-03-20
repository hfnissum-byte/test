from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .hybrid_model import HybridModelConfig, HybridQuantumInspiredEstimator


def load_model(model_path: str | Path) -> HybridQuantumInspiredEstimator:
    return HybridQuantumInspiredEstimator.load(Path(model_path))


def _parse_embedding(value: object) -> list[float]:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    if not isinstance(parsed, list):
        raise ValueError("embedding parse did not produce a list")
    return [float(x) for x in parsed]


def load_embeddings_from_pattern_csv(
    pattern_csv_path: str,
    symbols: Optional[Sequence[str]] = None,
    timeframes: Optional[Sequence[str]] = None,
    max_rows: int = 1000,
) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(pattern_csv_path)
    if symbols:
        df = df[df["symbol"].isin(list(symbols))]
    if timeframes:
        df = df[df["timeframe"].isin(list(timeframes))]

    if len(df) == 0:
        raise ValueError("No rows found in pattern CSV after filtering.")

    df = df.tail(max_rows).copy()
    embeddings = [_parse_embedding(v) for v in df["embedding"].to_list()]
    X = np.asarray(embeddings, dtype=np.float32)
    return X, df


def predict_proba(
    model: HybridQuantumInspiredEstimator,
    X: np.ndarray,
) -> np.ndarray:
    return model.predict_proba(X)


def predict_expected_return(
    model: HybridQuantumInspiredEstimator,
    X: np.ndarray,
) -> np.ndarray:
    return model.predict_expected_return(X)

