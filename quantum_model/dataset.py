from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetArrays:
    X: np.ndarray
    y_dir: np.ndarray
    y_return: Optional[np.ndarray]
    timestamps: np.ndarray


def _literal_eval_if_str(value: object):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _parse_embedding(value: object) -> list[float]:
    parsed = _literal_eval_if_str(value)
    if not isinstance(parsed, list):
        raise ValueError("embedding is not a list after parsing")
    return [float(x) for x in parsed]


def _parse_forward_returns(value: object) -> dict[str, float]:
    parsed = _literal_eval_if_str(value)
    if not isinstance(parsed, dict):
        raise ValueError("forward_returns is not a dict after parsing")
    # Values are already floats in practice, but force cast.
    return {str(k): float(v) for k, v in parsed.items()}


def build_training_arrays(
    pattern_csv_path: str,
    symbols: Optional[Sequence[str]] = None,
    timeframes: Optional[Sequence[str]] = None,
    horizon: str = "1h",
    max_rows: Optional[int] = 50_000,
    seed: int = 42,
    embedding_dim: int = 64,
) -> DatasetArrays:
    """
    Build training arrays from `pattern_records_*.csv` produced by the pattern engine.

    Direction label: Good=1, Bad=0. Neutral rows are excluded.
    Expected return regression target: forward_returns[horizon].
    """
    df = pd.read_csv(pattern_csv_path)

    if symbols:
        df = df[df["symbol"].isin(list(symbols))]
    if timeframes:
        df = df[df["timeframe"].isin(list(timeframes))]

    df = df[df["label"].isin(["Good", "Bad"])].copy()
    if len(df) == 0:
        raise ValueError("No Good/Bad rows found after filtering.")

    df["y_dir"] = df["label"].map({"Good": 1, "Bad": 0}).astype(int)

    # Limit to speed up training. We sample stratified to preserve both classes.
    if max_rows is not None and len(df) > max_rows:
        # Deterministic cap: keep the most recent examples per class.
        # This removes random sampling that breaks time-correct training.
        per_class = int(max_rows // 2)
        df = df.sort_values("timestamp")
        df_good_all = df[df["y_dir"] == 1]
        df_bad_all = df[df["y_dir"] == 0]
        df_good = df_good_all.tail(min(per_class, len(df_good_all)))
        df_bad = df_bad_all.tail(min(per_class, len(df_bad_all)))
        df = pd.concat([df_good, df_bad], axis=0).sort_values("timestamp").reset_index(drop=True)

    # Keep timestamps aligned with X/y_dir after filtering/capping.
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)

    # Parse embeddings only for the rows we keep.
    embeddings = []
    y_dir = df["y_dir"].to_numpy(dtype=np.int64)
    y_ret = []

    expected_ret_present = True
    for i, row in df.iterrows():
        emb = _parse_embedding(row["embedding"])
        if len(emb) != embedding_dim:
            raise ValueError(f"Unexpected embedding dim: got {len(emb)} expected {embedding_dim}")
        embeddings.append(emb)

        fr = _parse_forward_returns(row["forward_returns"])
        if horizon not in fr:
            expected_ret_present = False
            y_ret.append(np.nan)
        else:
            y_ret.append(float(fr[horizon]))

    X = np.asarray(embeddings, dtype=np.float32)
    y_ret_arr: Optional[np.ndarray]

    if expected_ret_present:
        y_ret_arr = np.asarray(y_ret, dtype=np.float32)
        if not np.all(np.isfinite(y_ret_arr)):
            # Drop any non-finite targets.
            mask = np.isfinite(y_ret_arr)
            X = X[mask]
            y_dir = y_dir[mask]
            y_ret_arr = y_ret_arr[mask]
    else:
        y_ret_arr = None

    # If y_ret is missing everywhere and gets set to None, X/y_dir still align.
    return DatasetArrays(X=X, y_dir=y_dir, y_return=y_ret_arr, timestamps=timestamps)

