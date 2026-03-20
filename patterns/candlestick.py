from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from patterns.regime import compute_atr


Direction = Literal["bullish", "bearish", "neutral"]


def _body(open_: float, close: float) -> float:
    return abs(close - open_)


def _candle_range(high: float, low: float) -> float:
    return max(high - low, 1e-12)


def _upper_wick(high: float, open_: float, close: float) -> float:
    return high - max(open_, close)


def _lower_wick(low: float, open_: float, close: float) -> float:
    return min(open_, close) - low


def _direction(open_: float, close: float) -> Direction:
    if close > open_:
        return "bullish"
    if close < open_:
        return "bearish"
    return "neutral"


def detect_candlestick_patterns(df: pd.DataFrame, *, symbol: str, timeframe: str) -> list[dict[str, Any]]:
    """
    Candlestick pattern detection based on auditable candle geometry.
    Returns instances at the confirmation candle index (typically the last candle in the pattern).
    """
    n = len(df)
    if n < 5:
        return []

    atr = compute_atr(df, length=14)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy(dtype=int)

    def features(i: int) -> dict[str, float]:
        rng = _candle_range(float(high[i]), float(low[i]))
        body = _body(float(open_[i]), float(close[i]))
        up = _upper_wick(float(high[i]), float(open_[i]), float(close[i]))
        lo = _lower_wick(float(low[i]), float(open_[i]), float(close[i]))
        return {
            "open": float(open_[i]),
            "close": float(close[i]),
            "high": float(high[i]),
            "low": float(low[i]),
            "range": float(rng),
            "body": float(body),
            "body_pct": float(body / rng),
            "upper_wick": float(up),
            "upper_wick_pct": float(up / rng),
            "lower_wick": float(lo),
            "lower_wick_pct": float(lo / rng),
            "volume": float(volume[i]),
            "direction": float(1.0 if close[i] > open_[i] else (-1.0 if close[i] < open_[i] else 0.0)),
        }

    def trend_slope(i: int, lookback: int = 10) -> float:
        start = max(0, i - lookback)
        if i - start < 2:
            return 0.0
        x = np.arange(i - start, dtype=float)
        y = close[start:i]
        # linear regression slope
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        denom = float(np.sum((x - x_mean) ** 2)) + 1e-12
        return float(np.sum((x - x_mean) * (y - y_mean)) / denom)

    out: list[dict[str, Any]] = []

    # Single-candle patterns
    for i in range(2, n):
        f = features(i)
        body_pct = f["body_pct"]
        up_pct = f["upper_wick_pct"]
        lo_pct = f["lower_wick_pct"]
        rng = f["range"]

        is_doji = body_pct <= 0.1
        if is_doji:
            if lo_pct >= 0.5 and up_pct <= 0.2:
                out.append(
                    {"pattern_type": "DragonflyDoji", "direction": "bullish" if close[i] >= open_[i] else "bearish", "timestamp": int(ts[i]), "detector_features": {"body_pct": body_pct}, "geometry_metrics": {"confidence_breakout": 0.6}, "confidence": 0.6}
                )
            elif up_pct >= 0.5 and lo_pct <= 0.2:
                out.append(
                    {"pattern_type": "GravestoneDoji", "direction": "bearish" if close[i] <= open_[i] else "bullish", "timestamp": int(ts[i]), "detector_features": {"body_pct": body_pct}, "geometry_metrics": {"confidence_breakout": 0.6}, "confidence": 0.6}
                )
            elif lo_pct >= 0.35 and up_pct >= 0.35:
                out.append(
                    {"pattern_type": "LongLeggedDoji", "direction": "neutral", "timestamp": int(ts[i]), "detector_features": {"body_pct": body_pct}, "geometry_metrics": {"confidence_breakout": 0.55}, "confidence": 0.55}
                )
            else:
                out.append(
                    {"pattern_type": "Doji", "direction": "neutral", "timestamp": int(ts[i]), "detector_features": {"body_pct": body_pct}, "geometry_metrics": {"confidence_breakout": 0.5}, "confidence": 0.5}
                )

        # Hammer / Inverted Hammer / Hanging Man / Shooting Star: use wick/body proportions + prior trend.
        down_trend = trend_slope(i) < 0
        up_trend = trend_slope(i) > 0
        if f["lower_wick_pct"] >= 0.6 and f["body_pct"] <= 0.3 and f["upper_wick_pct"] <= 0.25:
            if up_trend and f["close"] <= f["open"] + 0.2 * rng:
                out.append(
                    {"pattern_type": "HangingMan", "direction": "bearish", "timestamp": int(ts[i]), "detector_features": {"lower_wick_pct": lo_pct}, "geometry_metrics": {"confidence_breakout": 0.55}, "confidence": 0.55}
                )
            else:
                out.append(
                    {"pattern_type": "Hammer", "direction": "bullish", "timestamp": int(ts[i]), "detector_features": {"lower_wick_pct": lo_pct}, "geometry_metrics": {"confidence_breakout": 0.55}, "confidence": 0.55}
                )

        if f["upper_wick_pct"] >= 0.6 and f["body_pct"] <= 0.3 and f["lower_wick_pct"] <= 0.25:
            if down_trend and f["close"] >= f["open"] - 0.2 * rng:
                out.append(
                    {"pattern_type": "ShootingStar", "direction": "bearish", "timestamp": int(ts[i]), "detector_features": {"upper_wick_pct": up_pct}, "geometry_metrics": {"confidence_breakout": 0.55}, "confidence": 0.55}
                )
            else:
                out.append(
                    {"pattern_type": "InvertedHammer", "direction": "bullish" if not down_trend else "bearish", "timestamp": int(ts[i]), "detector_features": {"upper_wick_pct": up_pct}, "geometry_metrics": {"confidence_breakout": 0.5}, "confidence": 0.5}
                )

        # Spinning Top and Marubozu and Belt Hold and Kicker (single-candle aspects)
        if f["body_pct"] <= 0.25 and lo_pct >= 0.3 and up_pct >= 0.3:
            out.append(
                {"pattern_type": "SpinningTop", "direction": "neutral", "timestamp": int(ts[i]), "detector_features": {"body_pct": body_pct}, "geometry_metrics": {"confidence_breakout": 0.45}, "confidence": 0.45}
            )

        if f["upper_wick_pct"] <= 0.03 and f["lower_wick_pct"] <= 0.03:
            # Marubozu-like: strong candle with minimal wicks.
            out.append(
                {"pattern_type": "Marubozu", "direction": _direction(f["open"], f["close"]), "timestamp": int(ts[i]), "detector_features": {"body_pct": body_pct}, "geometry_metrics": {"confidence_breakout": 0.65}, "confidence": 0.65}
            )

        # Belt Hold: open at extreme + close near opposite side.
        if _direction(f["open"], f["close"]) == "bullish" and lo_pct <= 0.05 and f["close"] >= f["high"] - 0.05 * rng:
            out.append(
                {"pattern_type": "BeltHold", "direction": "bullish", "timestamp": int(ts[i]), "detector_features": {"lo_pct": lo_pct}, "geometry_metrics": {"confidence_breakout": 0.55}, "confidence": 0.55}
            )
        if _direction(f["open"], f["close"]) == "bearish" and up_pct <= 0.05 and f["close"] <= f["low"] + 0.05 * rng:
            out.append(
                {"pattern_type": "BeltHold", "direction": "bearish", "timestamp": int(ts[i]), "detector_features": {"up_pct": up_pct}, "geometry_metrics": {"confidence_breakout": 0.55}, "confidence": 0.55}
            )

    # Two-candle patterns
    for i in range(1, n):
        f1 = features(i - 1)
        f2 = features(i)

        rng1 = max(f1["range"], 1e-12)
        rng2 = max(f2["range"], 1e-12)

        bullish1 = f1["close"] > f1["open"]
        bearish1 = f1["close"] < f1["open"]
        bullish2 = f2["close"] > f2["open"]
        bearish2 = f2["close"] < f2["open"]

        # Engulfing
        if bearish1 and bullish2:
            if f2["open"] < f1["close"] and f2["close"] > f1["open"]:
                out.append(
                    {"pattern_type": "BullishEngulfing", "direction": "bullish", "timestamp": int(ts[i]), "detector_features": {"prev_body_pct": f1["body_pct"], "next_body_pct": f2["body_pct"]}, "geometry_metrics": {"confidence_breakout": 0.6}, "confidence": 0.6}
                )
        if bullish1 and bearish2:
            if f2["open"] > f1["close"] and f2["close"] < f1["open"]:
                out.append(
                    {"pattern_type": "BearishEngulfing", "direction": "bearish", "timestamp": int(ts[i]), "detector_features": {"prev_body_pct": f1["body_pct"], "next_body_pct": f2["body_pct"]}, "geometry_metrics": {"confidence_breakout": 0.6}, "confidence": 0.6}
                )

        # Harami
        if bearish1 and bullish2:
            # Harami bullish: first large bearish, second small bullish inside body.
            if f2["body"] < f1["body"] and f2["open"] > f1["close"] and f2["close"] < f1["open"]:
                out.append(
                    {"pattern_type": "Harami", "direction": "bullish", "timestamp": int(ts[i]), "detector_features": {"harami_body_ratio": float(f2["body"] / (f1["body"] + 1e-12))}, "geometry_metrics": {"confidence_breakout": 0.45}, "confidence": 0.45}
                )
                # Harami Cross: second candle is doji within body.
                if f2["body_pct"] <= 0.1:
                    out.append(
                        {"pattern_type": "HaramiCross", "direction": "bullish", "timestamp": int(ts[i]), "detector_features": {"body_pct": f2["body_pct"]}, "geometry_metrics": {"confidence_breakout": 0.5}, "confidence": 0.5}
                    )
        if bullish1 and bearish2:
            if f2["body"] < f1["body"] and f2["open"] < f1["close"] and f2["close"] > f1["open"]:
                out.append(
                    {"pattern_type": "Harami", "direction": "bearish", "timestamp": int(ts[i]), "detector_features": {"harami_body_ratio": float(f2["body"] / (f1["body"] + 1e-12))}, "geometry_metrics": {"confidence_breakout": 0.45}, "confidence": 0.45}
                )
                if f2["body_pct"] <= 0.1:
                    out.append(
                        {"pattern_type": "HaramiCross", "direction": "bearish", "timestamp": int(ts[i]), "detector_features": {"body_pct": f2["body_pct"]}, "geometry_metrics": {"confidence_breakout": 0.5}, "confidence": 0.5}
                    )

        # Piercing Line and Dark Cloud Cover
        if bullish1 and bearish2:
            # Dark Cloud Cover: second bearish closes below mid of first.
            mid = (f1["open"] + f1["close"]) / 2.0
            if f2["open"] >= f1["close"] and f2["close"] <= mid and f2["close"] > f1["low"]:
                out.append(
                    {"pattern_type": "DarkCloudCover", "direction": "bearish", "timestamp": int(ts[i]), "detector_features": {"mid": mid}, "geometry_metrics": {"confidence_breakout": 0.6}, "confidence": 0.6}
                )
        if bearish1 and bullish2:
            # Piercing Line: second bullish closes above mid but below first open.
            mid = (f1["open"] + f1["close"]) / 2.0
            if f2["open"] <= f1["close"] and f2["close"] >= mid and f2["close"] < f1["open"]:
                out.append(
                    {"pattern_type": "PiercingLine", "direction": "bullish", "timestamp": int(ts[i]), "detector_features": {"mid": mid}, "geometry_metrics": {"confidence_breakout": 0.6}, "confidence": 0.6}
                )

        # Inside Bar and Outside Bar
        if f2["high"] <= f1["high"] and f2["low"] >= f1["low"]:
            out.append(
                {"pattern_type": "InsideBar", "direction": "neutral", "timestamp": int(ts[i]), "detector_features": {"range_pct": float(f2["range"] / (f1["range"] + 1e-12))}, "geometry_metrics": {"confidence_breakout": 0.4}, "confidence": 0.4}
            )
        if f2["high"] >= f1["high"] and f2["low"] <= f1["low"]:
            out.append(
                {"pattern_type": "OutsideBar", "direction": _direction(f2["open"], f2["close"]), "timestamp": int(ts[i]), "detector_features": {"range_pct": float(f2["range"] / (f1["range"] + 1e-12))}, "geometry_metrics": {"confidence_breakout": 0.4}, "confidence": 0.4}
            )

        # Tweezer Top/Bottom: same high/low within tolerance + opposite direction.
        height_tol = abs(f2["high"] - f1["high"]) / (float(f1["high"]) + 1e-12)
        depth_tol = abs(f2["low"] - f1["low"]) / (float(f1["low"]) + 1e-12)
        if height_tol <= 0.01 and bearish2 and bullish1:
            out.append(
                {"pattern_type": "TweezerTop", "direction": "bearish", "timestamp": int(ts[i]), "detector_features": {"height_tolerance": float(height_tol)}, "geometry_metrics": {"confidence_breakout": 0.55}, "confidence": 0.55}
            )
        if depth_tol <= 0.01 and bullish2 and bearish1:
            out.append(
                {"pattern_type": "TweezerBottom", "direction": "bullish", "timestamp": int(ts[i]), "detector_features": {"depth_tolerance": float(depth_tol)}, "geometry_metrics": {"confidence_breakout": 0.55}, "confidence": 0.55}
            )

        # Kicker: small candle followed by strong opposite/engulf.
        if _direction(f1["open"], f1["close"]) == "bearish" and _direction(f2["open"], f2["close"]) == "bullish":
            # Bullish kicker: first small bearish, second bullish with strong body.
            if f1["body_pct"] <= 0.2 and f2["body"] >= 1.2 * f1["body"] and f2["close"] > f1["high"]:
                out.append(
                    {"pattern_type": "Kicker", "direction": "bullish", "timestamp": int(ts[i]), "detector_features": {"body1_pct": f1["body_pct"], "body2": f2["body"]}, "geometry_metrics": {"confidence_breakout": 0.6}, "confidence": 0.6}
                )
        if _direction(f1["open"], f1["close"]) == "bullish" and _direction(f2["open"], f2["close"]) == "bearish":
            if f1["body_pct"] <= 0.2 and f2["body"] >= 1.2 * f1["body"] and f2["close"] < f1["low"]:
                out.append(
                    {"pattern_type": "Kicker", "direction": "bearish", "timestamp": int(ts[i]), "detector_features": {"body1_pct": f1["body_pct"], "body2": f2["body"]}, "geometry_metrics": {"confidence_breakout": 0.6}, "confidence": 0.6}
                )

    # Three-candle patterns
    for i in range(2, n):
        f0 = features(i - 2)
        f1 = features(i - 1)
        f2 = features(i)

        bearish0 = f0["close"] < f0["open"]
        bullish0 = f0["close"] > f0["open"]

        # Morning Star variants
        if bearish0:
            mid = (f0["open"] + f0["close"]) / 2.0
            small1 = f1["body_pct"] <= 0.35
            if small1 and bullish0 is False:
                pass
            if small1 and f2["close"] > mid and f2["close"] > f2["open"]:
                out.append(
                    {"pattern_type": "MorningStar", "direction": "bullish", "timestamp": int(ts[i]), "detector_features": {"mid": mid}, "geometry_metrics": {"confidence_breakout": 0.55}, "confidence": 0.55}
                )
            # Doji star
            if small1 and f1["body_pct"] <= 0.1 and f2["close"] > mid and f2["close"] > f2["open"]:
                out.append(
                    {"pattern_type": "MorningDojiStar", "direction": "bullish", "timestamp": int(ts[i]), "detector_features": {"mid": mid, "star_doji_body_pct": f1["body_pct"]}, "geometry_metrics": {"confidence_breakout": 0.6}, "confidence": 0.6}
                )

        # Evening Star variants
        if bullish0:
            mid = (f0["open"] + f0["close"]) / 2.0
            small1 = f1["body_pct"] <= 0.35
            if small1 and f2["close"] < mid and f2["close"] < f2["open"]:
                out.append(
                    {"pattern_type": "EveningStar", "direction": "bearish", "timestamp": int(ts[i]), "detector_features": {"mid": mid}, "geometry_metrics": {"confidence_breakout": 0.55}, "confidence": 0.55}
                )
            if small1 and f1["body_pct"] <= 0.1 and f2["close"] < mid and f2["close"] < f2["open"]:
                out.append(
                    {"pattern_type": "EveningDojiStar", "direction": "bearish", "timestamp": int(ts[i]), "detector_features": {"mid": mid, "star_doji_body_pct": f1["body_pct"]}, "geometry_metrics": {"confidence_breakout": 0.6}, "confidence": 0.6}
                )

    # Multi-candle sequences: Three Soldiers/Crows
    for i in range(3, n):
        # Three White Soldiers: last 3 candles bullish with higher closes and opens within previous body.
        c1, c2, c3 = close[i - 3], close[i - 2], close[i - 1]
        o1, o2, o3 = open_[i - 3], open_[i - 2], open_[i - 1]
        if c1 > o1 and c2 > o2 and c3 > o3 and c2 > c1 and c3 > c2:
            # Each close near highs and body sufficiently large.
            if (abs(c1 - o1) / (_candle_range(float(high[i - 3]), float(low[i - 3])) + 1e-12) > 0.2) and (
                abs(c2 - o2) / (_candle_range(float(high[i - 2]), float(low[i - 2])) + 1e-12) > 0.2
            ) and (
                abs(c3 - o3) / (_candle_range(float(high[i - 1]), float(low[i - 1])) + 1e-12) > 0.2
            ):
                # Opens within previous body.
                if o2 <= c1 and o2 >= o1 and o3 <= c2 and o3 >= o2:
                    out.append(
                        {"pattern_type": "ThreeWhiteSoldiers", "direction": "bullish", "timestamp": int(ts[i - 1]), "detector_features": {}, "geometry_metrics": {"confidence_breakout": 0.6}, "confidence": 0.6}
                    )

        # Three Black Crows
        if c1 < o1 and c2 < o2 and c3 < o3 and c2 < c1 and c3 < c2:
            if (abs(c1 - o1) / (_candle_range(float(high[i - 3]), float(low[i - 3])) + 1e-12) > 0.2) and (
                abs(c2 - o2) / (_candle_range(float(high[i - 2]), float(low[i - 2])) + 1e-12) > 0.2
            ) and (
                abs(c3 - o3) / (_candle_range(float(high[i - 1]), float(low[i - 1])) + 1e-12) > 0.2
            ):
                if o2 >= c1 and o2 <= o1 and o3 >= c2 and o3 <= o2:
                    out.append(
                        {"pattern_type": "ThreeBlackCrows", "direction": "bearish", "timestamp": int(ts[i - 1]), "detector_features": {}, "geometry_metrics": {"confidence_breakout": 0.6}, "confidence": 0.6}
                    )

    return out

