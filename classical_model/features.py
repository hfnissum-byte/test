from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from patterns.embeddings import EMBED_DIM


@dataclass(frozen=True)
class SequenceDataset:
    # Shape: [n_windows, seq_len, embed_dim]
    X_seq: np.ndarray
    y_dir: np.ndarray
    # If present, shape: [n_windows]
    y_return: Optional[np.ndarray]
    # Window end timestamps: shape [n_windows]
    ts_end: np.ndarray


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
    return {str(k): float(v) for k, v in parsed.items()}


def build_sequence_dataset(
    pattern_csv_path: str,
    *,
    symbols: Optional[Sequence[str]] = None,
    timeframes: Optional[Sequence[str]] = None,
    seq_len: int = 12,
    horizon: str = "1h",
    max_labeled_rows: Optional[int] = 25_000,
    seed: int = 42,
    stride: int = 1,
) -> SequenceDataset:
    """
    Convert pattern records into sliding-window sequences of embeddings.

    Targets use the label/forward return of the *last* element in each window.
    """
    df = pd.read_csv(pattern_csv_path)

    if symbols:
        df = df[df["symbol"].isin(list(symbols))]
    if timeframes:
        df = df[df["timeframe"].isin(list(timeframes))]

    df = df[df["label"].isin(["Good", "Bad"])].copy()
    if len(df) == 0:
        raise ValueError("No Good/Bad rows found after filtering.")

    # Deterministic cap: keep most recent labeled rows.
    # This prevents random sampling from breaking time-correct training.
    if max_labeled_rows is not None and len(df) > max_labeled_rows:
        df = df.sort_values("timestamp")
        per_label = int(max_labeled_rows // 2)
        df_good_all = df[df["label"] == "Good"]
        df_bad_all = df[df["label"] == "Bad"]
        df_good = df_good_all.tail(min(per_label, len(df_good_all)))
        df_bad = df_bad_all.tail(min(per_label, len(df_bad_all)))
        df = pd.concat([df_good, df_bad], axis=0).sort_values("timestamp").reset_index(drop=True)

    # Parse + validate embeddings early for deterministic window building.
    embeddings: list[list[float]] = []
    y_dir: list[int] = []
    y_return: list[float] = []
    y_return_keep: list[bool] = []

    label_map = {"Good": 1, "Bad": 0}

    ts_all = df["timestamp"].to_numpy(dtype=np.int64)

    for i, row in df.iterrows():
        emb = _parse_embedding(row["embedding"])
        if len(emb) != EMBED_DIM:
            raise ValueError(f"Unexpected embedding dim: got {len(emb)} expected {EMBED_DIM}")
        embeddings.append(emb)
        y_dir.append(int(label_map[str(row["label"])]))

        fr = _parse_forward_returns(row["forward_returns"])
        if horizon in fr:
            y_return.append(float(fr[horizon]))
            y_return_keep.append(True)
        else:
            y_return.append(float("nan"))
            y_return_keep.append(False)

    X = np.asarray(embeddings, dtype=np.float32)
    y_dir_arr = np.asarray(y_dir, dtype=np.int64)
    y_return_arr: Optional[np.ndarray]
    y_return_arr_full = np.asarray(y_return, dtype=np.float32)
    y_return_keep_arr = np.asarray(y_return_keep, dtype=bool)

    # Align dataframe with parsed arrays, including dropping samples that
    # don't have the requested horizon.
    df2 = df.reset_index(drop=True).copy()
    df2 = df2.assign(_x_index=np.arange(len(df2)))

    if np.any(y_return_keep_arr):
        y_return_arr = y_return_arr_full[y_return_keep_arr]
        X = X[y_return_keep_arr]
        y_dir_arr = y_dir_arr[y_return_keep_arr]
        df2 = df2.loc[y_return_keep_arr].reset_index(drop=True)
    else:
        y_return_arr = None
        # Keep all samples for classification even if regression target is missing everywhere.


    # Build windows within each (symbol, timeframe) group to avoid cross-asset leakage.
    X_windows: list[np.ndarray] = []
    y_dir_windows: list[int] = []
    y_ret_windows: list[float] = []
    ts_windows: list[int] = []

    ts_arr = df2["timestamp"].to_numpy(dtype=np.int64)

    for (sym, tf), grp in df2.groupby(["symbol", "timeframe"]):
        grp = grp.sort_values("timestamp")
        idxs = grp["_x_index"].to_numpy(dtype=int)

        if len(idxs) < seq_len:
            continue

        for start in range(0, len(idxs) - seq_len + 1, stride):
            window_idxs = idxs[start : start + seq_len]
            last_idx = int(window_idxs[-1])
            X_windows.append(X[window_idxs])
            y_dir_windows.append(int(y_dir_arr[last_idx]))
            if y_return_arr is not None:
                y_ret_windows.append(float(y_return_arr[last_idx]))
            ts_windows.append(int(ts_arr[last_idx]))

    if len(X_windows) == 0:
        raise ValueError("Not enough labeled rows to build any sequence windows.")

    X_seq = np.asarray(X_windows, dtype=np.float32)
    y_dir_out = np.asarray(y_dir_windows, dtype=np.int64)
    if y_return_arr is not None:
        y_return_out: Optional[np.ndarray] = np.asarray(y_ret_windows, dtype=np.float32)
    else:
        y_return_out = None

    ts_end_out = np.asarray(ts_windows, dtype=np.int64)
    return SequenceDataset(X_seq=X_seq, y_dir=y_dir_out, y_return=y_return_out, ts_end=ts_end_out)

