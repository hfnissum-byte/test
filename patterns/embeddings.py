from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


EMBED_DIM = 64
ONEHOT_DIM = 24


def _stable_hash_int(s: str) -> int:
    digest = hashlib.md5(s.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def onehot_pattern_embedding(pattern_type: str, *, k: int = 6) -> np.ndarray:
    """
    Deterministic sparse embedding based on hashing (stable across runs/Python versions).
    """
    vec = np.zeros(ONEHOT_DIM, dtype=float)
    for i in range(k):
        idx = _stable_hash_int(f"{pattern_type}|{i}") % ONEHOT_DIM
        vec[idx] = 1.0
    return vec


def resample_series(y: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a 1D series to length n via linear interpolation.
    """
    if y.size == 0:
        return np.zeros(n, dtype=float)
    if y.size == n:
        return y.astype(float, copy=False)
    x_old = np.linspace(0.0, 1.0, num=y.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=True)
    return np.interp(x_new, x_old, y).astype(float)


def slope_and_curvature(y: np.ndarray) -> tuple[float, float]:
    """
    Compute mean slope and mean absolute curvature from a resampled series.
    Curvature is estimated using second differences (discrete curvature proxy).
    """
    y = y.astype(float, copy=False)
    if y.size < 3:
        return 0.0, 0.0

    dy = np.diff(y)
    d2y = np.diff(y, n=2)
    mean_slope = float(np.mean(dy))
    mean_abs_curv = float(np.mean(np.abs(d2y)))
    return mean_slope, mean_abs_curv


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Basic DTW with O(N^2). Intended for small resampled signatures (e.g., 32 points).
    """
    a = a.astype(float, copy=False)
    b = b.astype(float, copy=False)
    n = a.size
    m = b.size
    if n == 0 or m == 0:
        return float("inf")

    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[n, m])


def compute_curve_signature(df: pd.DataFrame, *, window: int = 48) -> dict[str, float]:
    """
    Compute a geometry/curve-aware signature from a price path window.
    """
    close = df["close"].to_numpy(dtype=float)
    if close.size == 0:
        return {"slope": 0.0, "curvature": 0.0}
    if close.size > window:
        close = close[-window:]

    # Normalize to reduce scale issues.
    first = close[0]
    norm = close / first if first != 0 else close
    signature = resample_series(norm, n=32)
    slope, curvature = slope_and_curvature(signature)
    return {"slope": slope, "curvature": curvature}


def build_embedding(
    *,
    pattern_type: str,
    detector_features: dict[str, Any],
    geometry_metrics: dict[str, float],
    volatility_context: dict[str, float],
) -> list[float]:
    """
    Build a fixed-length embedding vector from geometry + volatility context.
    """
    vec = np.zeros(EMBED_DIM, dtype=float)
    pattern_part = onehot_pattern_embedding(pattern_type)
    vec[:ONEHOT_DIM] = pattern_part

    # Geometry metrics (pick a stable subset).
    geom_keys = [
        "confidence_breakout",
        "symmetry_score",
        "curvature_mean",
        "slope_mean",
        "shape_dtw_to_u",
        "shape_dtw_to_invu",
        "height_tolerance",
        "depth_tolerance",
    ]
    for i, k in enumerate(geom_keys):
        if i >= EMBED_DIM - ONEHOT_DIM:
            break
        vec[ONEHOT_DIM + i] = float(geometry_metrics.get(k, 0.0))

    # Volatility context (normalized).
    atr_percent = float(volatility_context.get("atr_percent", 0.0))
    atr_regime_enc = float({"low": 0.0, "mid": 0.5, "high": 1.0}.get(str(volatility_context.get("regime", "mid")), 0.5))
    base = ONEHOT_DIM + len(geom_keys)
    if base + 2 <= EMBED_DIM:
        vec[base] = atr_percent
        vec[base + 1] = atr_regime_enc

    # Add a small amount of bounded numeric info from detector_features.
    # This remains auditable because it is a direct transform of computed metrics.
    numeric_keys = [k for k in detector_features.keys() if isinstance(detector_features[k], (int, float))]
    numeric_keys = sorted(numeric_keys)[:10]
    fill_start = ONEHOT_DIM + 10
    for j, k in enumerate(numeric_keys):
        idx = fill_start + j
        if idx >= EMBED_DIM:
            break
        vec[idx] = float(detector_features[k])

    # Normalize to keep embeddings bounded.
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return [float(x) for x in vec.tolist()]

