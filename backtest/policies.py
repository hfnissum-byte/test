from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

import numpy.typing as npt

from classical_model.inference import load_model as load_classical_model
from patterns.embeddings import EMBED_DIM
from patterns.vector_db import create_vector_db
from quantum_model.inference import load_model as load_quantum_model


def _literal_eval(value: Any) -> Any:
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def parse_embedding_row(embedding_value: Any) -> np.ndarray:
    emb = _literal_eval(embedding_value)
    if not isinstance(emb, list):
        raise ValueError("embedding row is not a list")
    arr = np.asarray([float(x) for x in emb], dtype=float)
    if arr.ndim != 1 or arr.size != EMBED_DIM:
        raise ValueError(f"embedding dim mismatch: got {arr.size}, expected {EMBED_DIM}")
    return arr


@dataclass(frozen=True)
class Signal:
    # +1 long, -1 short, 0 no trade
    direction: int
    confidence: float
    p_long: float


class BasePolicy:
    def predict_signals(self, *, events: pd.DataFrame) -> list[Signal]:
        raise NotImplementedError


def thresholds_to_direction(*, p_long: float, min_confidence: float, p_long_threshold: float, p_short_threshold: float) -> int:
    if p_long < 0.0 or p_long > 1.0:
        return 0
    p_short = 1.0 - p_long
    if max(p_long, p_short) < min_confidence:
        return 0
    if p_long >= p_long_threshold:
        return 1
    if p_short >= p_short_threshold:
        return -1
    return 0


@dataclass(frozen=True)
class ModelPolicyConfig:
    min_confidence: float = 0.55
    p_long_threshold: float = 0.55
    p_short_threshold: float = 0.55


class QuantumModelPolicy(BasePolicy):
    def __init__(
        self,
        *,
        model_path: str,
        cfg: ModelPolicyConfig,
    ) -> None:
        self.model = load_quantum_model(model_path)
        self.cfg = cfg

    def predict_signals(self, *, events: pd.DataFrame) -> list[Signal]:
        # events must contain embedding column.
        embeddings = np.vstack([parse_embedding_row(x) for x in events["embedding"].to_list()]).astype(float)
        proba_long = np.asarray(self.model.predict_proba(embeddings), dtype=float)

        out: list[Signal] = []
        for p in proba_long.tolist():
            direction = thresholds_to_direction(
                p_long=p,
                min_confidence=self.cfg.min_confidence,
                p_long_threshold=self.cfg.p_long_threshold,
                p_short_threshold=self.cfg.p_short_threshold,
            )
            confidence = float(max(p, 1.0 - p))
            out.append(Signal(direction=direction, confidence=confidence, p_long=float(p)))
        return out


class PatternSimilarityPolicy(BasePolicy):
    """
    Uses nearest-neighbor pattern embeddings from vector memory.
    For each event at timestamp t, neighbors with timestamp >= t are discarded
    to reduce lookahead bias.
    """

    def __init__(
        self,
        *,
        pattern_vector_db_base_dir: str,
        cfg: ModelPolicyConfig,
        top_k: int = 50,
    ) -> None:
        # Uses the project’s vector DB factory. If Chroma/FAISS are unavailable,
        # the factory falls back to the local brute-force memory store.
        self.vdb = create_vector_db(base_dir=pattern_vector_db_base_dir, prefer="chroma")
        self.cfg = cfg
        self.top_k = int(top_k)

    def _estimate_p_long(self, *, emb: np.ndarray, event_row: pd.Series, symbol: str, timeframe: str) -> tuple[float, float]:
        # Query in-memory (local cosine similarity fallback).
        results = self.vdb.query(
            embedding=emb.tolist(),
            top_k=self.top_k,
            symbol=symbol,
            timeframe=timeframe,
        )
        if not results:
            return 0.5, 0.0

        # Discard neighbors at/after the event timestamp.
        event_ts = int(event_row["timestamp"])
        good_w = 0.0
        bad_w = 0.0
        total_w = 0.0
        for r in results:
            neigh_ts = int(r.get("timestamp"))
            if neigh_ts >= event_ts:
                continue
            sim = float(r.get("similarity", 0.0))
            w = max(sim, 0.0)
            total_w += w
            label = str(r.get("label", "Neutral"))
            if label == "Good":
                good_w += w
            elif label == "Bad":
                bad_w += w

        if total_w <= 0:
            return 0.5, 0.0

        p_long = good_w / (good_w + bad_w + 1e-12)
        confidence = float(abs(p_long - 0.5) * 2.0)  # 0..1-ish
        return float(p_long), confidence

    def predict_signals(self, *, events: pd.DataFrame) -> list[Signal]:
        # Parse embeddings once.
        embs = np.vstack([parse_embedding_row(x) for x in events["embedding"].to_list()]).astype(float)
        symbol = str(events["symbol"].iloc[0])
        timeframe = str(events["timeframe"].iloc[0])

        out: list[Signal] = []
        for i in range(len(events)):
            p_long, conf = self._estimate_p_long(emb=embs[i], event_row=events.iloc[i], symbol=symbol, timeframe=timeframe)
            direction = thresholds_to_direction(
                p_long=p_long,
                min_confidence=self.cfg.min_confidence,
                p_long_threshold=self.cfg.p_long_threshold,
                p_short_threshold=self.cfg.p_short_threshold,
            )
            confidence_final = float(max(p_long, 1.0 - p_long))
            chosen_conf = confidence_final if direction != 0 else conf
            out.append(Signal(direction=direction, confidence=float(chosen_conf), p_long=float(p_long)))
        return out


class ClassicalModelPolicy(BasePolicy):
    def __init__(
        self,
        *,
        model_path: str,
        cfg: ModelPolicyConfig,
        seq_stride: Optional[int] = None,
    ) -> None:
        self.model = load_classical_model(model_path)
        self.cfg = cfg
        self.seq_stride = seq_stride

    def _build_windows_with_timestamps(
        self,
        events_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        # For classical model we need sliding windows of embeddings.
        seq_len = int(self.model.cfg.seq_len)
        stride = int(self.seq_stride if self.seq_stride is not None else 1)
        if stride <= 0:
            stride = 1

        # Sort by timestamp.
        grp = events_df.sort_values("timestamp").reset_index(drop=True)
        emb_arr = np.vstack([parse_embedding_row(x) for x in grp["embedding"].to_list()]).astype(float)
        ts_arr = grp["timestamp"].to_numpy(dtype=np.int64)

        windows: list[np.ndarray] = []
        win_timestamps: list[int] = []
        if len(ts_arr) < seq_len:
            return np.zeros((0, seq_len, EMBED_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        for start in range(0, len(ts_arr) - seq_len + 1, stride):
            last = start + seq_len - 1
            windows.append(emb_arr[start : last + 1])
            win_timestamps.append(int(ts_arr[last]))

        X_seq = np.asarray(windows, dtype=np.float32)
        timestamps = np.asarray(win_timestamps, dtype=np.int64)
        return X_seq, timestamps

    def predict_signals(self, *, events: pd.DataFrame) -> list[Signal]:
        # We interpret each *window end* as a decision event at that timestamp.
        # Return signals aligned 1:1 with the `events` dataframe rows.
        X_seq, win_ts = self._build_windows_with_timestamps(events)

        out: list[Signal] = []
        if X_seq.shape[0] == 0:
            for _ in range(len(events)):
                out.append(Signal(direction=0, confidence=0.0, p_long=0.5))
            return out

        proba_long = np.asarray(self.model.predict_proba(X_seq), dtype=float)
        # Map timestamp -> p_long/proba for quick lookup.
        # If duplicates exist, keep the most recent window probability.
        ts_to_p: dict[int, float] = {}
        for ts, p in zip(win_ts.tolist(), proba_long.tolist()):
            ts_to_p[int(ts)] = float(p)

        for ts_val in events["timestamp"].to_numpy(dtype=np.int64).tolist():
            p_long = ts_to_p.get(int(ts_val), 0.5)
            direction = thresholds_to_direction(
                p_long=p_long,
                min_confidence=self.cfg.min_confidence,
                p_long_threshold=self.cfg.p_long_threshold,
                p_short_threshold=self.cfg.p_short_threshold,
            )
            confidence = float(max(p_long, 1.0 - p_long)) if direction != 0 else 0.0
            out.append(Signal(direction=direction, confidence=confidence, p_long=float(p_long)))
        return out

