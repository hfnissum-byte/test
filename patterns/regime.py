from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


VolRegime = Literal["low", "mid", "high"]


@dataclass(frozen=True)
class VolatilityContext:
    atr: float
    atr_percent: float
    regime: VolRegime


def true_range(high: np.ndarray, low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))


def compute_atr(df: pd.DataFrame, *, length: int = 14) -> float:
    if df.shape[0] < max(2, length + 1):
        # Not enough history; fall back to simple mean TR.
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = true_range(high, low, prev_close)
        return float(np.mean(tr))

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = true_range(high, low, prev_close)
    atr = pd.Series(tr).rolling(window=length).mean().iloc[-1]
    return float(atr)


def volatility_regime_from_atr_percent(atr_percent: float) -> VolRegime:
    # Simple, auditable thresholds.
    if atr_percent < 0.01:
        return "low"
    if atr_percent < 0.03:
        return "mid"
    return "high"


def compute_volatility_context(df: pd.DataFrame, *, atr_length: int = 14) -> VolatilityContext:
    atr = compute_atr(df, length=atr_length)
    last_close = float(df["close"].iloc[-1])
    atr_percent = atr / last_close if last_close != 0 else 0.0
    regime = volatility_regime_from_atr_percent(atr_percent)
    return VolatilityContext(atr=atr, atr_percent=atr_percent, regime=regime)

