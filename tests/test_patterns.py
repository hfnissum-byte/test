from __future__ import annotations

import numpy as np
import pandas as pd

from patterns.candlestick import detect_candlestick_patterns
from patterns.chart_patterns import detect_chart_patterns


def _df_from_ohlc(close: np.ndarray) -> pd.DataFrame:
    # Simple helper for synthetic tests: make high/low around close.
    ts = np.arange(close.size, dtype=int) * 3_600_000
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = np.ones_like(close, dtype=float) * 100.0
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close.astype(float),
            "volume": volume,
        }
    )


def test_detect_doji() -> None:
    close = np.array([100.0, 101.0, 100.5, 100.5, 103.0, 104.0], dtype=float)
    df = _df_from_ohlc(close)
    # Force candle at index 3 to be a doji.
    i = 3
    df.loc[i, "open"] = 100.5
    df.loc[i, "close"] = 100.5
    df.loc[i, "high"] = 105.0
    df.loc[i, "low"] = 95.0

    inst = detect_candlestick_patterns(df, symbol="BTC/USDT", timeframe="1h")
    assert any(x["pattern_type"] == "Doji" or x["pattern_type"] == "LongLeggedDoji" for x in inst)


def test_detect_triple_top_structure() -> None:
    n = 220
    close = np.ones(n, dtype=float) * 100.0

    # Three peaks (roughly equal heights) and valleys in between.
    peak_idxs = [50, 100, 150]
    valley_idxs = [75, 125]
    for pi in peak_idxs:
        close[pi] = 120.0
    for vi in valley_idxs:
        close[vi] = 95.0

    # Shape around peaks: smooth rises/falls.
    for pi in peak_idxs:
        for k in range(-5, 6):
            j = pi + k
            if 0 <= j < n:
                close[j] = 120.0 - abs(k) * 1.0
    for vi in valley_idxs:
        for k in range(-5, 6):
            j = vi + k
            if 0 <= j < n:
                close[j] = 95.0 + abs(k) * 1.0

    # Breakout confirmation: after third peak, drop below neckline.
    close[151] = 90.0
    close[152] = 89.0

    df = _df_from_ohlc(close)
    # Ensure highs/lows align with our close path.
    df["high"] = df["close"] + 2.0
    df["low"] = df["close"] - 2.0

    inst = detect_chart_patterns(df, symbol="BTC/USDT", timeframe="1h")
    assert any(x["pattern_type"] == "TripleTop" for x in inst)

