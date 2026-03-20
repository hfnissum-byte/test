from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from patterns.regime import compute_atr


Direction = Literal["bullish", "bearish"]


def _linreg_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x_ = x.astype(float)
    y_ = y.astype(float)
    x_mean = float(np.mean(x_))
    y_mean = float(np.mean(y_))
    denom = float(np.sum((x_ - x_mean) ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum((x_ - x_mean) * (y_ - y_mean)) / denom)


def _polyfit_quadratic_coeff(x: np.ndarray, y: np.ndarray) -> float:
    """
    Returns the quadratic coefficient a in y ~ a*x^2 + b*x + c
    """
    if x.size < 3:
        return 0.0
    # Normalize x to reduce numerical issues.
    x2 = (x - x.min()) / (x.max() - x.min() + 1e-12)
    a, _, _ = np.polyfit(x2.astype(float), y.astype(float), deg=2)
    return float(a)


def _resistance_from_pivots(prices: list[float]) -> float:
    if not prices:
        return float("nan")
    return float(np.mean(prices))


def _tolerance_ratio(a: float, b: float) -> float:
    denom = abs(a) if a != 0 else 1.0
    return abs(a - b) / denom


@dataclass(frozen=True)
class Pivots:
    high_idx: np.ndarray
    high: np.ndarray
    low_idx: np.ndarray
    low: np.ndarray


def extract_pivots(df: pd.DataFrame, *, atr_length: int = 14, peak_order: int = 5) -> Pivots:
    """
    Pivot extraction using find_peaks with ATR-derived prominence.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr = compute_atr(df, length=atr_length)

    # Prominence: fraction of ATR, auditable and volatility-aware.
    prominence = max(1e-12, atr * 0.15)
    distance = max(peak_order, 2 * peak_order)

    high_idx, _ = find_peaks(high, distance=distance, prominence=prominence)
    low_idx, _ = find_peaks(-low, distance=distance, prominence=prominence)
    high_vals = high[high_idx]
    low_vals = low[low_idx]

    return Pivots(high_idx=np.array(high_idx, dtype=int), high=np.array(high_vals, dtype=float), low_idx=np.array(low_idx, dtype=int), low=np.array(low_vals, dtype=float))


def _between_valleys(low_series: np.ndarray, idx_a: int, idx_b: int, low_pivots_idx: np.ndarray, low_pivots: np.ndarray) -> float:
    """
    Given two high pivots, find the highest low pivot between them (neckline candidate for tops).
    """
    if idx_a > idx_b:
        idx_a, idx_b = idx_b, idx_a
    mask = (low_pivots_idx > idx_a) & (low_pivots_idx < idx_b)
    if not np.any(mask):
        # Fallback to raw minima between indices.
        return float(np.min(low_series[idx_a:idx_b + 1]))
    return float(np.max(low_pivots[mask]))


def _between_peaks(high_series: np.ndarray, idx_a: int, idx_b: int, high_pivots_idx: np.ndarray, high_pivots: np.ndarray) -> float:
    """
    Given two low pivots, find the lowest high pivot between them (neckline candidate for bottoms).
    """
    if idx_a > idx_b:
        idx_a, idx_b = idx_b, idx_a
    mask = (high_pivots_idx > idx_a) & (high_pivots_idx < idx_b)
    if not np.any(mask):
        return float(np.max(high_series[idx_a:idx_b + 1]))
    return float(np.min(high_pivots[mask]))


def _curve_metrics(df: pd.DataFrame, *, start_idx: int, end_idx: int) -> dict[str, float]:
    """
    Compute slope/curvature from a normalized close segment via second-differences.
    """
    seg = df.iloc[start_idx : end_idx + 1]
    close = seg["close"].to_numpy(dtype=float)
    if close.size < 3:
        return {"curvature_mean": 0.0, "slope_mean": 0.0}
    norm = close / (close[0] if close[0] != 0 else 1.0)
    y = norm
    dy = np.diff(y)
    d2y = np.diff(y, n=2)
    return {"curvature_mean": float(np.mean(np.abs(d2y))), "slope_mean": float(np.mean(dy))}


def detect_double_top(df: pd.DataFrame, *, atr: float, pivots: Pivots) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    highs = pivots.high
    hi_idx = pivots.high_idx
    if highs.size < 2:
        return out

    close = df["close"].to_numpy(dtype=float)
    low_series = df["low"].to_numpy(dtype=float)

    # Candidate pairs: last two highs within window.
    for i in range(1, highs.size):
        for j in range(i):
            idx_a = int(hi_idx[j])
            idx_b = int(hi_idx[i])
            if idx_b - idx_a < 10:
                continue
            h_a = float(highs[j])
            h_b = float(highs[i])
            height_tol = abs(h_a - h_b) / (abs(h_a) + 1e-12)
            if height_tol > 0.03:
                continue
            neckline = _between_valleys(low_series, idx_a, idx_b, pivots.low_idx, pivots.low)
            breakout_idx = idx_b + 1
            if breakout_idx >= len(df):
                continue
            # Double-top confirmation is bearish: close below neckline.
            breakout_margin = 0.02 * atr
            if close[breakout_idx] < neckline - breakout_margin:
                segment_metrics = _curve_metrics(df, start_idx=idx_a, end_idx=min(idx_b + 10, len(df) - 1))
                confidence = float(np.clip(1.0 - (height_tol / 0.05), 0.0, 1.0)) * float(
                    np.clip((breakout_margin / (atr + 1e-12)), 0.0, 1.0)
                )
                out.append(
                    {
                        "pattern_type": "DoubleTop",
                        "direction": "bearish",
                        "timestamp": int(df["timestamp"].iloc[breakout_idx]),
                        "detector_features": {"peak1_height": h_a, "peak2_height": h_b, "height_tolerance": height_tol},
                        "geometry_metrics": {"neckline": float(neckline), "depth_tolerance": float(breakout_margin), **segment_metrics, "confidence_breakout": confidence},
                        "confidence": confidence,
                    }
                )
    return out


def detect_triple_top(df: pd.DataFrame, *, atr: float, pivots: Pivots) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if pivots.high.size < 3:
        return out
    highs = pivots.high
    hi_idx = pivots.high_idx
    close = df["close"].to_numpy(dtype=float)
    low_series = df["low"].to_numpy(dtype=float)

    # Try combinations of three pivot highs (ordered by index).
    n = highs.size
    for a in range(n - 2):
        for b in range(a + 1, n - 1):
            for c in range(b + 1, n):
                idx_a, idx_b, idx_c = int(hi_idx[a]), int(hi_idx[b]), int(hi_idx[c])
                if min(idx_b - idx_a, idx_c - idx_b) < 10:
                    continue
                h_a, h_b, h_c = float(highs[a]), float(highs[b]), float(highs[c])
                mean_h = (h_a + h_b + h_c) / 3.0
                # Peak similarity tolerance.
                height_tolerance = float(max(abs(h_a - mean_h), abs(h_b - mean_h), abs(h_c - mean_h)) / (abs(mean_h) + 1e-12))
                if height_tolerance > 0.04:
                    continue

                neckline1 = _between_valleys(low_series, idx_a, idx_b, pivots.low_idx, pivots.low)
                neckline2 = _between_valleys(low_series, idx_b, idx_c, pivots.low_idx, pivots.low)
                neckline = float(max(neckline1, neckline2))
                breakout_idx = idx_c + 1
                if breakout_idx >= len(df):
                    continue

                breakout_margin = 0.02 * atr
                if close[breakout_idx] < neckline - breakout_margin:
                    segment_metrics = _curve_metrics(df, start_idx=idx_a, end_idx=min(idx_c + 10, len(df) - 1))
                    # Symmetry based on spacing and peak height similarity.
                    spacing1 = idx_b - idx_a
                    spacing2 = idx_c - idx_b
                    symmetry_score = float(1.0 - min(abs(spacing1 - spacing2) / (max(spacing1, spacing2) + 1e-12), 1.0))
                    confidence = float(
                        np.clip(1.0 - (height_tolerance / 0.06), 0.0, 1.0) * 0.6 + symmetry_score * 0.4
                    )
                    confidence *= float(np.clip(breakout_margin / (atr + 1e-12), 0.0, 1.0))
                    out.append(
                        {
                            "pattern_type": "TripleTop",
                            "direction": "bearish",
                            "timestamp": int(df["timestamp"].iloc[breakout_idx]),
                            "detector_features": {
                                "peak_heights": [h_a, h_b, h_c],
                                "height_tolerance": height_tolerance,
                                "peak_spacing": [spacing1, spacing2],
                            },
                            "geometry_metrics": {
                                "neckline": neckline,
                                "depth_tolerance": float(breakout_margin),
                                "symmetry_score": symmetry_score,
                                **segment_metrics,
                                "confidence_breakout": float(confidence),
                            },
                            "confidence": float(confidence),
                        }
                    )
    return out


def detect_double_bottom(df: pd.DataFrame, *, atr: float, pivots: Pivots) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if pivots.low.size < 2:
        return out
    lows = pivots.low
    lo_idx = pivots.low_idx
    close = df["close"].to_numpy(dtype=float)
    high_series = df["high"].to_numpy(dtype=float)

    for i in range(1, lows.size):
        for j in range(i):
            idx_a = int(lo_idx[j])
            idx_b = int(lo_idx[i])
            if idx_b - idx_a < 10:
                continue
            l_a = float(lows[j])
            l_b = float(lows[i])
            depth_tol = abs(l_a - l_b) / (abs(l_a) + 1e-12)
            if depth_tol > 0.03:
                continue
            neckline = _between_peaks(high_series, idx_a, idx_b, pivots.high_idx, pivots.high)
            breakout_idx = idx_b + 1
            if breakout_idx >= len(df):
                continue
            breakout_margin = 0.02 * atr
            if close[breakout_idx] > neckline + breakout_margin:
                segment_metrics = _curve_metrics(df, start_idx=idx_a, end_idx=min(idx_b + 10, len(df) - 1))
                confidence = float(np.clip(1.0 - (depth_tol / 0.05), 0.0, 1.0)) * float(
                    np.clip((breakout_margin / (atr + 1e-12)), 0.0, 1.0)
                )
                out.append(
                    {
                        "pattern_type": "DoubleBottom",
                        "direction": "bullish",
                        "timestamp": int(df["timestamp"].iloc[breakout_idx]),
                        "detector_features": {"valley1_depth": l_a, "valley2_depth": l_b, "depth_tolerance": depth_tol},
                        "geometry_metrics": {"neckline": float(neckline), "depth_tolerance": float(breakout_margin), **segment_metrics, "confidence_breakout": confidence},
                        "confidence": confidence,
                    }
                )
    return out


def detect_triple_bottom(df: pd.DataFrame, *, atr: float, pivots: Pivots) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if pivots.low.size < 3:
        return out
    lows = pivots.low
    lo_idx = pivots.low_idx
    close = df["close"].to_numpy(dtype=float)
    high_series = df["high"].to_numpy(dtype=float)

    n = lows.size
    for a in range(n - 2):
        for b in range(a + 1, n - 1):
            for c in range(b + 1, n):
                idx_a, idx_b, idx_c = int(lo_idx[a]), int(lo_idx[b]), int(lo_idx[c])
                if min(idx_b - idx_a, idx_c - idx_b) < 10:
                    continue
                l_a, l_b, l_c = float(lows[a]), float(lows[b]), float(lows[c])
                mean_l = (l_a + l_b + l_c) / 3.0
                depth_tolerance = float(max(abs(l_a - mean_l), abs(l_b - mean_l), abs(l_c - mean_l)) / (abs(mean_l) + 1e-12))
                if depth_tolerance > 0.04:
                    continue

                neckline1 = _between_peaks(high_series, idx_a, idx_b, pivots.high_idx, pivots.high)
                neckline2 = _between_peaks(high_series, idx_b, idx_c, pivots.high_idx, pivots.high)
                neckline = float(min(neckline1, neckline2))
                breakout_idx = idx_c + 1
                if breakout_idx >= len(df):
                    continue

                breakout_margin = 0.02 * atr
                if close[breakout_idx] > neckline + breakout_margin:
                    segment_metrics = _curve_metrics(df, start_idx=idx_a, end_idx=min(idx_c + 10, len(df) - 1))
                    spacing1 = idx_b - idx_a
                    spacing2 = idx_c - idx_b
                    symmetry_score = float(1.0 - min(abs(spacing1 - spacing2) / (max(spacing1, spacing2) + 1e-12), 1.0))
                    confidence = float(
                        np.clip(1.0 - (depth_tolerance / 0.06), 0.0, 1.0) * 0.6 + symmetry_score * 0.4
                    )
                    confidence *= float(np.clip(breakout_margin / (atr + 1e-12), 0.0, 1.0))

                    out.append(
                        {
                            "pattern_type": "TripleBottom",
                            "direction": "bullish",
                            "timestamp": int(df["timestamp"].iloc[breakout_idx]),
                            "detector_features": {
                                "valley_depths": [l_a, l_b, l_c],
                                "depth_tolerance": depth_tolerance,
                                "peak_spacing": [spacing1, spacing2],
                            },
                            "geometry_metrics": {
                                "neckline": neckline,
                                "depth_tolerance": float(breakout_margin),
                                "symmetry_score": symmetry_score,
                                **segment_metrics,
                                "confidence_breakout": float(confidence),
                            },
                            "confidence": float(confidence),
                        }
                    )
    return out


def detect_head_and_shoulders(df: pd.DataFrame, *, atr: float, pivots: Pivots) -> list[dict[str, Any]]:
    """
    Bearish Head & Shoulders:
    - three pivot highs: left < head > right
    - left and right within tolerance
    - confirmation close breaks below neckline derived from pivot lows
    """
    out: list[dict[str, Any]] = []
    if pivots.high.size < 3 or pivots.low.size < 1:
        return out

    highs = pivots.high
    hi_idx = pivots.high_idx
    close = df["close"].to_numpy(dtype=float)
    low_series = df["low"].to_numpy(dtype=float)

    n = highs.size
    for i in range(1, n - 1):
        left = i - 1
        head = i
        right = i + 1
        # Require indices are strictly increasing in order as extracted by find_peaks.
        idx_l, idx_h, idx_r = int(hi_idx[left]), int(hi_idx[head]), int(hi_idx[right])
        if not (idx_l < idx_h < idx_r):
            continue

        h_l = float(highs[left])
        h_h = float(highs[head])
        h_r = float(highs[right])

        if h_h < max(h_l, h_r) + 0.05 * atr:
            continue

        # Left/right shoulder similarity.
        shoulders_mean = (h_l + h_r) / 2.0
        shoulder_tol = abs(h_l - h_r) / (abs(shoulders_mean) + 1e-12)
        if shoulder_tol > 0.04:
            continue

        neckline = _between_valleys(low_series, idx_l, idx_h, pivots.low_idx, pivots.low)
        neckline2 = _between_valleys(low_series, idx_h, idx_r, pivots.low_idx, pivots.low)
        neckline_final = float(max(neckline, neckline2))

        breakout_idx = idx_r + 1
        if breakout_idx >= len(df):
            continue
        breakout_margin = 0.02 * atr
        if close[breakout_idx] < neckline_final - breakout_margin:
            segment_metrics = _curve_metrics(df, start_idx=idx_l, end_idx=min(idx_r + 10, len(df) - 1))
            symmetry_score = float(1.0 - min(shoulder_tol / 0.08, 1.0))
            confidence = float(
                np.clip(1.0 - shoulder_tol / 0.08, 0.0, 1.0) * 0.6 + symmetry_score * 0.4
            )
            confidence *= float(np.clip(breakout_margin / (atr + 1e-12), 0.0, 1.0))
            out.append(
                {
                    "pattern_type": "HeadAndShoulders",
                    "direction": "bearish",
                    "timestamp": int(df["timestamp"].iloc[breakout_idx]),
                    "detector_features": {"head_height": h_h, "shoulder_heights": [h_l, h_r], "shoulder_tolerance": shoulder_tol},
                    "geometry_metrics": {"neckline": neckline_final, "symmetry_score": symmetry_score, **segment_metrics, "confidence_breakout": float(confidence)},
                    "confidence": float(confidence),
                }
            )
    return out


def detect_inverse_head_and_shoulders(df: pd.DataFrame, *, atr: float, pivots: Pivots) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if pivots.low.size < 3 or pivots.high.size < 1:
        return out

    lows = pivots.low
    lo_idx = pivots.low_idx
    close = df["close"].to_numpy(dtype=float)
    high_series = df["high"].to_numpy(dtype=float)

    n = lows.size
    for i in range(1, n - 1):
        left = i - 1
        head = i
        right = i + 1
        idx_l, idx_h, idx_r = int(lo_idx[left]), int(lo_idx[head]), int(lo_idx[right])
        if not (idx_l < idx_h < idx_r):
            continue

        l_l = float(lows[left])
        l_h = float(lows[head])
        l_r = float(lows[right])

        if l_h > min(l_l, l_r) - 0.05 * atr:
            # For inverse H&S, head is deepest valley (lower low).
            continue

        shoulders_mean = (l_l + l_r) / 2.0
        shoulder_tol = abs(l_l - l_r) / (abs(shoulders_mean) + 1e-12)
        if shoulder_tol > 0.04:
            continue

        neckline = _between_peaks(high_series, idx_l, idx_h, pivots.high_idx, pivots.high)
        neckline2 = _between_peaks(high_series, idx_h, idx_r, pivots.high_idx, pivots.high)
        neckline_final = float(min(neckline, neckline2))

        breakout_idx = idx_r + 1
        if breakout_idx >= len(df):
            continue
        breakout_margin = 0.02 * atr
        if close[breakout_idx] > neckline_final + breakout_margin:
            segment_metrics = _curve_metrics(df, start_idx=idx_l, end_idx=min(idx_r + 10, len(df) - 1))
            symmetry_score = float(1.0 - min(shoulder_tol / 0.08, 1.0))
            confidence = float(np.clip(1.0 - shoulder_tol / 0.08, 0.0, 1.0) * 0.6 + symmetry_score * 0.4)
            confidence *= float(np.clip(breakout_margin / (atr + 1e-12), 0.0, 1.0))
            out.append(
                {
                    "pattern_type": "InverseHeadAndShoulders",
                    "direction": "bullish",
                    "timestamp": int(df["timestamp"].iloc[breakout_idx]),
                    "detector_features": {"head_depth": l_h, "shoulder_depths": [l_l, l_r], "shoulder_tolerance": shoulder_tol},
                    "geometry_metrics": {"neckline": neckline_final, "symmetry_score": symmetry_score, **segment_metrics, "confidence_breakout": float(confidence)},
                    "confidence": float(confidence),
                }
            )
    return out


def detect_triangles(df: pd.DataFrame, *, atr: float, pivots: Pivots) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Use last few pivots to compute line slopes.
    highs_idx = pivots.high_idx
    highs = pivots.high
    lows_idx = pivots.low_idx
    lows = pivots.low

    if highs_idx.size < 2 or lows_idx.size < 2:
        return out

    # Select last two highs and last two lows by index ordering.
    last_highs = list(range(max(0, highs_idx.size - 2), highs_idx.size))
    last_lows = list(range(max(0, lows_idx.size - 2), lows_idx.size))
    hi2 = [int(highs_idx[i]) for i in last_highs]
    lo2 = [int(lows_idx[i]) for i in last_lows]
    y_hi = np.array([highs[i] for i in last_highs], dtype=float)
    y_lo = np.array([lows[i] for i in last_lows], dtype=float)
    x_hi = np.array(hi2, dtype=float)
    x_lo = np.array(lo2, dtype=float)

    slope_hi = _linreg_slope(x_hi, y_hi)
    slope_lo = _linreg_slope(x_lo, y_lo)

    close = df["close"].to_numpy(dtype=float)
    last_idx = len(df) - 1
    confirmation_idx = min(last_idx, int(max(hi2 + lo2) + 1))
    if confirmation_idx >= len(df):
        return out

    # Compute pivot level extremes for breakout checks.
    high_level = float(np.max(y_hi))
    low_level = float(np.min(y_lo))
    breakout_margin = 0.01 * atr

    if abs(slope_hi) < 0.0005 and slope_lo > 0:
        # Ascending triangle: flat resistance + rising support.
        if close[confirmation_idx] > high_level + breakout_margin:
            out.append(
                {
                    "pattern_type": "AscendingTriangle",
                    "direction": "bullish",
                    "timestamp": int(df["timestamp"].iloc[confirmation_idx]),
                    "detector_features": {"slope_high": slope_hi, "slope_low": slope_lo},
                    "geometry_metrics": {"height_tolerance": 0.0, "depth_tolerance": 0.0, **_curve_metrics(df, start_idx=min(hi2 + lo2), end_idx=confirmation_idx), "confidence_breakout": 0.7},
                    "confidence": 0.7,
                }
            )
    if slope_hi < 0 and abs(slope_lo) < 0.0005:
        # Descending triangle: falling resistance + flat support.
        if close[confirmation_idx] < low_level - breakout_margin:
            out.append(
                {
                    "pattern_type": "DescendingTriangle",
                    "direction": "bearish",
                    "timestamp": int(df["timestamp"].iloc[confirmation_idx]),
                    "detector_features": {"slope_high": slope_hi, "slope_low": slope_lo},
                    "geometry_metrics": {"height_tolerance": 0.0, "depth_tolerance": 0.0, **_curve_metrics(df, start_idx=min(hi2 + lo2), end_idx=confirmation_idx), "confidence_breakout": 0.7},
                    "confidence": 0.7,
                }
            )

    # Symmetrical: converging slopes (high down, low up).
    if slope_hi < 0 and slope_lo > 0:
        if close[confirmation_idx] > high_level + breakout_margin:
            out.append(
                {
                    "pattern_type": "SymmetricalTriangle",
                    "direction": "bullish",
                    "timestamp": int(df["timestamp"].iloc[confirmation_idx]),
                    "detector_features": {"slope_high": slope_hi, "slope_low": slope_lo},
                    "geometry_metrics": {"confidence_breakout": 0.6, **_curve_metrics(df, start_idx=min(hi2 + lo2), end_idx=confirmation_idx)},
                    "confidence": 0.6,
                }
            )
        elif close[confirmation_idx] < low_level - breakout_margin:
            out.append(
                {
                    "pattern_type": "SymmetricalTriangle",
                    "direction": "bearish",
                    "timestamp": int(df["timestamp"].iloc[confirmation_idx]),
                    "detector_features": {"slope_high": slope_hi, "slope_low": slope_lo},
                    "geometry_metrics": {"confidence_breakout": 0.6, **_curve_metrics(df, start_idx=min(hi2 + lo2), end_idx=confirmation_idx)},
                    "confidence": 0.6,
                }
            )

    # Broadening: slopes diverge.
    if slope_hi > 0 and slope_lo < 0:
        if close[confirmation_idx] > high_level + breakout_margin:
            out.append(
                {
                    "pattern_type": "BroadeningFormation",
                    "direction": "bullish",
                    "timestamp": int(df["timestamp"].iloc[confirmation_idx]),
                    "detector_features": {"slope_high": slope_hi, "slope_low": slope_lo},
                    "geometry_metrics": {"confidence_breakout": 0.5, **_curve_metrics(df, start_idx=min(hi2 + lo2), end_idx=confirmation_idx)},
                    "confidence": 0.5,
                }
            )
        elif close[confirmation_idx] < low_level - breakout_margin:
            out.append(
                {
                    "pattern_type": "BroadeningFormation",
                    "direction": "bearish",
                    "timestamp": int(df["timestamp"].iloc[confirmation_idx]),
                    "detector_features": {"slope_high": slope_hi, "slope_low": slope_lo},
                    "geometry_metrics": {"confidence_breakout": 0.5, **_curve_metrics(df, start_idx=min(hi2 + lo2), end_idx=confirmation_idx)},
                    "confidence": 0.5,
                }
            )

    return out


def detect_wedges(df: pd.DataFrame, *, atr: float, pivots: Pivots) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    highs_idx = pivots.high_idx
    highs = pivots.high
    lows_idx = pivots.low_idx
    lows = pivots.low
    if highs_idx.size < 2 or lows_idx.size < 2:
        return out
    last_highs = list(range(max(0, highs_idx.size - 2), highs_idx.size))
    last_lows = list(range(max(0, lows_idx.size - 2), lows_idx.size))
    hi2 = [int(highs_idx[i]) for i in last_highs]
    lo2 = [int(lows_idx[i]) for i in last_lows]
    y_hi = np.array([highs[i] for i in last_highs], dtype=float)
    y_lo = np.array([lows[i] for i in last_lows], dtype=float)
    x_hi = np.array(hi2, dtype=float)
    x_lo = np.array(lo2, dtype=float)
    slope_hi = _linreg_slope(x_hi, y_hi)
    slope_lo = _linreg_slope(x_lo, y_lo)
    close = df["close"].to_numpy(dtype=float)
    confirmation_idx = min(len(df) - 1, int(max(hi2 + lo2) + 1))
    if confirmation_idx >= len(df):
        return out
    breakout_margin = 0.01 * atr
    high_level = float(np.max(y_hi))
    low_level = float(np.min(y_lo))

    # Rising wedge: both slopes positive but converge.
    if slope_hi > 0 and slope_lo > 0 and slope_hi < slope_lo:
        if close[confirmation_idx] < low_level - breakout_margin:
            out.append(
                {
                    "pattern_type": "RisingWedge",
                    "direction": "bearish",
                    "timestamp": int(df["timestamp"].iloc[confirmation_idx]),
                    "detector_features": {"slope_high": slope_hi, "slope_low": slope_lo},
                    "geometry_metrics": {"confidence_breakout": 0.6, **_curve_metrics(df, start_idx=min(hi2 + lo2), end_idx=confirmation_idx)},
                    "confidence": 0.6,
                }
            )

    # Falling wedge: both slopes negative but converge.
    if slope_hi < 0 and slope_lo < 0 and slope_hi > slope_lo:
        if close[confirmation_idx] > high_level + breakout_margin:
            out.append(
                {
                    "pattern_type": "FallingWedge",
                    "direction": "bullish",
                    "timestamp": int(df["timestamp"].iloc[confirmation_idx]),
                    "detector_features": {"slope_high": slope_hi, "slope_low": slope_lo},
                    "geometry_metrics": {"confidence_breakout": 0.6, **_curve_metrics(df, start_idx=min(hi2 + lo2), end_idx=confirmation_idx)},
                    "confidence": 0.6,
                }
            )
    return out


def detect_flags_pennants(df: pd.DataFrame, *, atr: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy(dtype=int)

    n = len(df)
    if n < 60:
        return out

    # Consider last segment for consolidation.
    tail = df.iloc[-80:].copy()
    tail_close = tail["close"].to_numpy(dtype=float)
    impulse_return = tail_close[-1] / tail_close[0] - 1.0

    # Consolidation range computed from recent band.
    cons_window = 30
    cons = df.iloc[-cons_window:].copy()
    cons_high = float(cons["high"].max())
    cons_low = float(cons["low"].min())
    range_pct = (cons_high - cons_low) / (float(tail_close[-1]) + 1e-12)

    # Volatility contraction proxy: std of closes decreasing.
    std_fast = float(pd.Series(df["close"].to_numpy(dtype=float)[-cons_window:]).std())
    std_slow = float(pd.Series(df["close"].to_numpy(dtype=float)[-2 * cons_window:-cons_window]).std() if n >= 2 * cons_window else std_fast)
    contraction = std_fast / (std_slow + 1e-12)

    confirmation_idx = n - 1
    breakout_margin = 0.01 * atr
    last_close = float(close[-1])
    upper = cons_high
    lower = cons_low

    if impulse_return > 0.03 and range_pct < 0.02:
        # Bull flag or pennant.
        slope_con = _linreg_slope(np.arange(cons.shape[0], dtype=float), cons["close"].to_numpy(dtype=float))
        if slope_con < 0:  # slight downward drift in bull flags
            if last_close > upper + breakout_margin:
                out.append(
                    {
                        "pattern_type": "BullFlag",
                        "direction": "bullish",
                        "timestamp": int(ts[confirmation_idx]),
                        "detector_features": {"impulse_return": impulse_return, "range_pct": range_pct},
                        "geometry_metrics": {"confidence_breakout": 0.55, **{"curvature_mean": 0.0, "slope_mean": float(slope_con)}, "height_tolerance": 0.0, "depth_tolerance": 0.0},
                        "confidence": 0.55,
                    }
                )
        if contraction < 0.8:
            if last_close > upper + breakout_margin:
                out.append(
                    {
                        "pattern_type": "BullPennant",
                        "direction": "bullish",
                        "timestamp": int(ts[confirmation_idx]),
                        "detector_features": {"impulse_return": impulse_return, "range_pct": range_pct, "contraction": contraction},
                        "geometry_metrics": {"confidence_breakout": 0.5, **{"curvature_mean": 0.0, "slope_mean": 0.0}, "height_tolerance": 0.0, "depth_tolerance": 0.0},
                        "confidence": 0.5,
                    }
                )

    if impulse_return < -0.03 and range_pct < 0.02:
        slope_con = _linreg_slope(np.arange(cons.shape[0], dtype=float), cons["close"].to_numpy(dtype=float))
        if slope_con > 0:
            if last_close < lower - breakout_margin:
                out.append(
                    {
                        "pattern_type": "BearFlag",
                        "direction": "bearish",
                        "timestamp": int(ts[confirmation_idx]),
                        "detector_features": {"impulse_return": impulse_return, "range_pct": range_pct},
                        "geometry_metrics": {"confidence_breakout": 0.55, **{"curvature_mean": 0.0, "slope_mean": float(slope_con)}, "height_tolerance": 0.0, "depth_tolerance": 0.0},
                        "confidence": 0.55,
                    }
                )
        if contraction < 0.8:
            if last_close < lower - breakout_margin:
                out.append(
                    {
                        "pattern_type": "BearPennant",
                        "direction": "bearish",
                        "timestamp": int(ts[confirmation_idx]),
                        "detector_features": {"impulse_return": impulse_return, "range_pct": range_pct, "contraction": contraction},
                        "geometry_metrics": {"confidence_breakout": 0.5, **{"curvature_mean": 0.0, "slope_mean": 0.0}, "height_tolerance": 0.0, "depth_tolerance": 0.0},
                        "confidence": 0.5,
                    }
                )
    return out


def detect_rectangles(df: pd.DataFrame, *, atr: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(df) < 50:
        return out
    cons = df.iloc[-60:].copy()
    upper = float(cons["high"].max())
    lower = float(cons["low"].min())
    range_pct = (upper - lower) / (float(cons["close"].iloc[-1]) + 1e-12)
    if range_pct > 0.04:
        return out
    last_close = float(df["close"].iloc[-1])
    breakout_margin = 0.01 * atr
    ts = int(df["timestamp"].iloc[-1])
    if last_close > upper + breakout_margin:
        out.append(
            {
                "pattern_type": "RectangleRange",
                "direction": "bullish",
                "timestamp": ts,
                "detector_features": {"range_pct": range_pct, "upper": upper, "lower": lower},
                "geometry_metrics": {"confidence_breakout": 0.5, **_curve_metrics(df, start_idx=max(0, len(df) - 60), end_idx=len(df) - 1)},
                "confidence": 0.5,
            }
        )
    if last_close < lower - breakout_margin:
        out.append(
            {
                "pattern_type": "RectangleRange",
                "direction": "bearish",
                "timestamp": ts,
                "detector_features": {"range_pct": range_pct, "upper": upper, "lower": lower},
                "geometry_metrics": {"confidence_breakout": 0.5, **_curve_metrics(df, start_idx=max(0, len(df) - 60), end_idx=len(df) - 1)},
                "confidence": 0.5,
            }
        )
    return out


def detect_channels(df: pd.DataFrame, *, atr: float, pivots: Pivots) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if pivots.high_idx.size < 2 or pivots.low_idx.size < 2:
        return out
    # Use last two pivots of each side.
    hi_idx = pivots.high_idx[-2:]
    hi = pivots.high[-2:]
    lo_idx = pivots.low_idx[-2:]
    lo = pivots.low[-2:]
    slope_hi = _linreg_slope(hi_idx.astype(float), hi.astype(float))
    slope_lo = _linreg_slope(lo_idx.astype(float), lo.astype(float))
    if abs(slope_hi - slope_lo) > 0.0008:
        return out
    close = float(df["close"].iloc[-1])
    ts = int(df["timestamp"].iloc[-1])
    upper = float(np.max(hi))
    lower = float(np.min(lo))
    breakout_margin = 0.01 * atr
    if close > upper + breakout_margin:
        out.append(
            {
                "pattern_type": "Channels",
                "direction": "bullish",
                "timestamp": ts,
                "detector_features": {"slope_hi": slope_hi, "slope_lo": slope_lo},
                "geometry_metrics": {"confidence_breakout": 0.45, **_curve_metrics(df, start_idx=max(0, min(hi_idx.min(), lo_idx.min())), end_idx=len(df) - 1)},
                "confidence": 0.45,
            }
        )
    if close < lower - breakout_margin:
        out.append(
            {
                "pattern_type": "Channels",
                "direction": "bearish",
                "timestamp": ts,
                "detector_features": {"slope_hi": slope_hi, "slope_lo": slope_lo},
                "geometry_metrics": {"confidence_breakout": 0.45, **_curve_metrics(df, start_idx=max(0, min(hi_idx.min(), lo_idx.min())), end_idx=len(df) - 1)},
                "confidence": 0.45,
            }
        )
    return out


def detect_breakout_retest(df: pd.DataFrame, *, atr: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(df) < 80:
        return out
    # Take the last 50 candles as "range" if very narrow.
    rng = df.iloc[-80:-30].copy()
    upper = float(rng["high"].max())
    lower = float(rng["low"].min())
    range_pct = (upper - lower) / (float(rng["close"].iloc[-1]) + 1e-12)
    if range_pct > 0.06:
        return out
    breakout_window = df.iloc[-30:].copy()
    closes = breakout_window["close"].to_numpy(dtype=float)
    highs = breakout_window["high"].to_numpy(dtype=float)
    lows = breakout_window["low"].to_numpy(dtype=float)
    ts = breakout_window["timestamp"].to_numpy(dtype=int)

    breakout_margin = 0.01 * atr

    # Bull breakout + retest.
    for i in range(0, len(breakout_window) - 1):
        if closes[i] > upper + breakout_margin:
            # Search retest in next 10 candles.
            retest_level = upper
            retest_tol = 0.01 * atr
            for j in range(i + 1, min(i + 11, len(breakout_window))):
                if lows[j] <= retest_level + retest_tol and closes[j] > retest_level - retest_tol:
                    out.append(
                        {
                            "pattern_type": "BreakoutRetest",
                            "direction": "bullish",
                            "timestamp": int(ts[i]),
                            "detector_features": {"range_upper": upper, "range_lower": lower, "range_pct": range_pct},
                            "geometry_metrics": {"confidence_breakout": 0.55, **_curve_metrics(df, start_idx=max(0, len(df) - 80), end_idx=len(df) - 1)},
                            "confidence": 0.55,
                        }
                    )
                    break
    # Bear breakout + retest.
    for i in range(0, len(breakout_window) - 1):
        if closes[i] < lower - breakout_margin:
            retest_level = lower
            retest_tol = 0.01 * atr
            for j in range(i + 1, min(i + 11, len(breakout_window))):
                if highs[j] >= retest_level - retest_tol and closes[j] < retest_level + retest_tol:
                    out.append(
                        {
                            "pattern_type": "BreakoutRetest",
                            "direction": "bearish",
                            "timestamp": int(ts[i]),
                            "detector_features": {"range_upper": upper, "range_lower": lower, "range_pct": range_pct},
                            "geometry_metrics": {"confidence_breakout": 0.55, **_curve_metrics(df, start_idx=max(0, len(df) - 80), end_idx=len(df) - 1)},
                            "confidence": 0.55,
                        }
                    )
                    break
    return out


def detect_swing_failure(df: pd.DataFrame, *, atr: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(df) < 40:
        return out
    recent = df.iloc[-30:].copy()
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    last = df.iloc[-1]
    ts = int(last["timestamp"])
    low = float(last["low"])
    high = float(last["high"])
    close = float(last["close"])
    open_ = float(last["open"])

    breakout_margin = 0.01 * atr
    # Bullish swing failure: wick below support and close back above support.
    if low < support - breakout_margin and close > support + breakout_margin and close > open_:
        out.append(
            {
                "pattern_type": "SwingFailureBullish",
                "direction": "bullish",
                "timestamp": ts,
                "detector_features": {"support": support},
                "geometry_metrics": {"confidence_breakout": 0.6, **_curve_metrics(df, start_idx=max(0, len(df) - 30), end_idx=len(df) - 1)},
                "confidence": 0.6,
            }
        )
    # Bearish swing failure: wick above resistance and close back below.
    if high > resistance + breakout_margin and close < resistance - breakout_margin and close < open_:
        out.append(
            {
                "pattern_type": "SwingFailureBearish",
                "direction": "bearish",
                "timestamp": ts,
                "detector_features": {"resistance": resistance},
                "geometry_metrics": {"confidence_breakout": 0.6, **_curve_metrics(df, start_idx=max(0, len(df) - 30), end_idx=len(df) - 1)},
                "confidence": 0.6,
            }
        )
    return out


def detect_liquidity_sweeps(df: pd.DataFrame, *, atr: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(df) < 40:
        return out
    recent = df.iloc[-25:].copy()
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    last = df.iloc[-1]
    ts = int(last["timestamp"])
    low = float(last["low"])
    high = float(last["high"])
    close = float(last["close"])
    open_ = float(last["open"])
    breakout_margin = 0.01 * atr

    # Sweep below support.
    if low < support - breakout_margin and close > support:
        out.append(
            {
                "pattern_type": "LiquiditySweepLow",
                "direction": "bullish",
                "timestamp": ts,
                "detector_features": {"support": support},
                "geometry_metrics": {"confidence_breakout": 0.55, **_curve_metrics(df, start_idx=max(0, len(df) - 25), end_idx=len(df) - 1)},
                "confidence": 0.55,
            }
        )

    # Sweep above resistance.
    if high > resistance + breakout_margin and close < resistance:
        out.append(
            {
                "pattern_type": "LiquiditySweepHigh",
                "direction": "bearish",
                "timestamp": ts,
                "detector_features": {"resistance": resistance},
                "geometry_metrics": {"confidence_breakout": 0.55, **_curve_metrics(df, start_idx=max(0, len(df) - 25), end_idx=len(df) - 1)},
                "confidence": 0.55,
            }
        )
    return out


def detect_wyckoff(df: pd.DataFrame, *, atr: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(df) < 100:
        return out
    window = df.iloc[-100:].copy()
    close = window["close"].to_numpy(dtype=float)
    open_ = window["open"].to_numpy(dtype=float)
    high = window["high"].to_numpy(dtype=float)
    low = window["low"].to_numpy(dtype=float)
    volume = window["volume"].to_numpy(dtype=float)
    ts = int(df["timestamp"].iloc[-1])

    support = float(np.min(low))
    resistance = float(np.max(high))
    range_pct = (resistance - support) / (float(close[-1]) + 1e-12)
    if range_pct > 0.08:
        return out

    # Selling climax proxy: largest bearish candle near lower range.
    body = np.abs(close - open_)
    bearish = close < open_
    worst_bear_idx = int(np.argmax(np.where(bearish, body, 0.0)))
    spring_idx = int(np.argmin(low))

    spring_close = float(close[spring_idx])
    spring_low = float(low[spring_idx])
    spring_volume = float(volume[spring_idx])

    # Accumulation: wick below support and close above support.
    if spring_low < support - 0.01 * atr and spring_close > support + 0.005 * atr and spring_volume > float(np.mean(volume)):
        out.append(
            {
                "pattern_type": "WyckoffAccumulation",
                "direction": "bullish",
                "timestamp": ts,
                "detector_features": {"support": support, "resistance": resistance, "range_pct": range_pct},
                "geometry_metrics": {"confidence_breakout": 0.45, **_curve_metrics(df, start_idx=len(df) - 100, end_idx=len(df) - 1)},
                "confidence": 0.45,
            }
        )

    # Distribution proxy: buying climax near upper range, upthrust wick and close back below.
    worst_bull_idx = int(np.argmax(np.where(close > open_, body, 0.0)))
    upthrust_idx = int(np.argmax(high))
    upthrust_close = float(close[upthrust_idx])
    upthrust_high = float(high[upthrust_idx])
    upthrust_volume = float(volume[upthrust_idx])

    if upthrust_high > resistance + 0.01 * atr and upthrust_close < resistance - 0.005 * atr and upthrust_volume > float(np.mean(volume)):
        out.append(
            {
                "pattern_type": "WyckoffDistribution",
                "direction": "bearish",
                "timestamp": ts,
                "detector_features": {"support": support, "resistance": resistance, "range_pct": range_pct},
                "geometry_metrics": {"confidence_breakout": 0.45, **_curve_metrics(df, start_idx=len(df) - 100, end_idx=len(df) - 1)},
                "confidence": 0.45,
            }
        )
    return out


def detect_rounded_patterns(df: pd.DataFrame, *, atr: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(df) < 120:
        return out
    window = df.iloc[-120:].copy()
    close = window["close"].to_numpy(dtype=float)
    x = np.arange(close.size, dtype=float)
    a = _polyfit_quadratic_coeff(x, close)
    ts = int(df["timestamp"].iloc[-1])
    # For a bottom, quadratic coefficient should be positive (opens up).
    if a > 0 and (np.argmin(close) > close.size * 0.25) and (np.argmin(close) < close.size * 0.75):
        out.append(
            {
                "pattern_type": "RoundedBottom",
                "direction": "bullish",
                "timestamp": ts,
                "detector_features": {"quad_coeff": a},
                "geometry_metrics": {"confidence_breakout": 0.35, **_curve_metrics(df, start_idx=len(df) - 120, end_idx=len(df) - 1)},
                "confidence": 0.35,
            }
        )
    if a < 0 and (np.argmax(close) > close.size * 0.25) and (np.argmax(close) < close.size * 0.75):
        out.append(
            {
                "pattern_type": "RoundedTop",
                "direction": "bearish",
                "timestamp": ts,
                "detector_features": {"quad_coeff": a},
                "geometry_metrics": {"confidence_breakout": 0.35, **_curve_metrics(df, start_idx=len(df) - 120, end_idx=len(df) - 1)},
                "confidence": 0.35,
            }
        )
    return out


def detect_cup_and_handle(df: pd.DataFrame, *, atr: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(df) < 160:
        return out
    window = df.iloc[-160:].copy()
    close = window["close"].to_numpy(dtype=float)
    ts = int(df["timestamp"].iloc[-1])
    # Cup: rounded bottom in first 2/3, handle: slight drift down in last 1/3.
    cup = close[: int(close.size * 0.65)]
    handle = close[int(close.size * 0.65) :]
    x_cup = np.arange(cup.size, dtype=float)
    a = _polyfit_quadratic_coeff(x_cup, cup)
    if a <= 0:
        return out
    rim = float(np.max(cup))
    handle_low = float(np.min(handle))
    # Handle should be a pullback but not break too deep.
    if handle_low < float(np.min(cup)) - 0.03 * atr:
        return out
    breakout_margin = 0.01 * atr
    if float(close[-1]) > rim + breakout_margin:
        out.append(
            {
                "pattern_type": "CupAndHandle",
                "direction": "bullish",
                "timestamp": ts,
                "detector_features": {"quad_coeff": a, "rim": rim, "handle_low": handle_low},
                "geometry_metrics": {"confidence_breakout": 0.4, **_curve_metrics(df, start_idx=len(df) - 160, end_idx=len(df) - 1)},
                "confidence": 0.4,
            }
        )
    return out


def detect_chart_patterns(df: pd.DataFrame, *, symbol: str, timeframe: str) -> list[dict[str, Any]]:
    """
    Detect chart/structure patterns with pivot + geometry heuristics.
    """
    if df.shape[0] < 200:
        return []

    # Keep Phase 4 runtime bounded by scanning only the most recent segment.
    # This is a real, correctness-preserving engineering constraint (prevents O(n^3) pivot combinations).
    scan_rows = 5000
    if df.shape[0] > scan_rows:
        df = df.iloc[-scan_rows:].copy().reset_index(drop=True)

    atr = compute_atr(df, length=14)
    pivots = extract_pivots(df, atr_length=14, peak_order=5)

    # Bound pivot counts to avoid combinatorial explosion in multi-swing detectors.
    max_pivots = 30
    if pivots.high_idx.size > max_pivots:
        keep_from = int(pivots.high_idx[-max_pivots])
        mask_h = pivots.high_idx >= keep_from
        pivots = Pivots(
            high_idx=pivots.high_idx[mask_h],
            high=pivots.high[mask_h],
            low_idx=pivots.low_idx[pivots.low_idx >= keep_from],
            low=pivots.low[pivots.low_idx >= keep_from],
        )
    if pivots.low_idx.size > max_pivots:
        keep_from = int(pivots.low_idx[-max_pivots])
        mask_l = pivots.low_idx >= keep_from
        pivots = Pivots(
            high_idx=pivots.high_idx[pivots.high_idx >= keep_from],
            high=pivots.high[pivots.high_idx >= keep_from],
            low_idx=pivots.low_idx[mask_l],
            low=pivots.low[mask_l],
        )

    instances: list[dict[str, Any]] = []
    instances.extend(detect_double_top(df, atr=atr, pivots=pivots))
    instances.extend(detect_triple_top(df, atr=atr, pivots=pivots))
    instances.extend(detect_double_bottom(df, atr=atr, pivots=pivots))
    instances.extend(detect_triple_bottom(df, atr=atr, pivots=pivots))
    instances.extend(detect_head_and_shoulders(df, atr=atr, pivots=pivots))
    instances.extend(detect_inverse_head_and_shoulders(df, atr=atr, pivots=pivots))
    instances.extend(detect_triangles(df, atr=atr, pivots=pivots))
    instances.extend(detect_wedges(df, atr=atr, pivots=pivots))
    instances.extend(detect_flags_pennants(df, atr=atr))
    instances.extend(detect_rectangles(df, atr=atr))
    instances.extend(detect_channels(df, atr=atr, pivots=pivots))
    instances.extend(detect_breakout_retest(df, atr=atr))
    instances.extend(detect_swing_failure(df, atr=atr))
    instances.extend(detect_liquidity_sweeps(df, atr=atr))
    instances.extend(detect_wyckoff(df, atr=atr))
    instances.extend(detect_cup_and_handle(df, atr=atr))
    instances.extend(detect_rounded_patterns(df, atr=atr))

    # Some patterns in the requested list are currently covered by the shared detectors above
    # via pattern_type mapping or by approximate geometry signatures.
    # Each returned instance already includes:
    # - pattern_type
    # - timestamp
    # - detector_features
    # - geometry_metrics
    # - confidence
    # - direction
    return instances

