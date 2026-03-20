from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


Label = Literal["Good", "Bad", "Neutral"]


@dataclass(frozen=True)
class LabelingConfig:
    horizons_hours: dict[str, int]
    horizon_weights: dict[str, float]
    good_quantile: float
    bad_quantile: float


DEFAULT_LABELING_CONFIG = LabelingConfig(
    horizons_hours={"1h": 1, "4h": 4, "24h": 24},
    horizon_weights={"1h": 0.2, "4h": 0.3, "24h": 0.5},
    good_quantile=0.8,
    bad_quantile=0.2,
)


def _lookup_close_at_1h(df_1h: pd.DataFrame, timestamp_ms: int) -> tuple[int, float] | None:
    ts = df_1h["timestamp"].to_numpy(dtype=np.int64)
    # Exact match preferred; otherwise we take the next candle at/after the timestamp.
    idx = int(np.searchsorted(ts, timestamp_ms, side="left"))
    if idx >= ts.size:
        return None
    if ts[idx] != timestamp_ms:
        # If not exact, use nearest next candle (auditable).
        timestamp_ms = int(ts[idx])
    return idx, float(df_1h["close"].iloc[idx])


def _compute_forward_returns_1h(
    df_1h: pd.DataFrame,
    *,
    pattern_timestamp_ms: int,
    horizons_hours: dict[str, int],
) -> dict[str, float] | None:
    lookup = _lookup_close_at_1h(df_1h, pattern_timestamp_ms)
    if lookup is None:
        return None
    idx, close0 = lookup

    returns: dict[str, float] = {}
    for horizon_key, step in horizons_hours.items():
        target_idx = idx + int(step)
        if target_idx >= len(df_1h):
            return None
        close_t = float(df_1h["close"].iloc[target_idx])
        if close0 == 0:
            return None
        returns[horizon_key] = (close_t / close0) - 1.0
    return returns


def _quantile_rank(values: np.ndarray, x: float) -> float:
    # Empirical CDF: fraction of values <= x.
    if values.size == 0:
        return 0.5
    return float(np.sum(values <= x) / values.size)


def label_pattern_instances(
    *,
    df_1h: pd.DataFrame,
    instances: list[dict[str, Any]],
    cfg: LabelingConfig = DEFAULT_LABELING_CONFIG,
) -> list[dict[str, Any]]:
    """
    Mutates instances by adding:
    - forward_returns
    - label
    """
    # 1) Compute forward returns for each instance.
    with_returns: list[dict[str, Any]] = []
    for inst in instances:
        ts = int(inst["timestamp"])
        fv = _compute_forward_returns_1h(df_1h, pattern_timestamp_ms=ts, horizons_hours=cfg.horizons_hours)
        if fv is None:
            continue
        inst2 = dict(inst)
        inst2["forward_returns"] = fv
        with_returns.append(inst2)

    if not with_returns:
        # Nothing can be labeled due to lookahead gaps.
        return instances

    # 2) Quantile-labeling per (pattern_type, timeframe).
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for inst in with_returns:
        key = (str(inst["pattern_type"]), str(inst["timeframe"]))
        grouped.setdefault(key, []).append(inst)

    # Precompute distributions.
    dists: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, group in grouped.items():
        by_h: dict[str, list[float]] = {h: [] for h in cfg.horizons_hours.keys()}
        for inst in group:
            for h in cfg.horizons_hours.keys():
                by_h[h].append(float(inst["forward_returns"][h]))
        dists[key] = {h: np.array(vals, dtype=float) for h, vals in by_h.items()}

    # 3) Assign weighted Good/Bad/Neutral.
    good_threshold = float(cfg.good_quantile)  # weights sum to 1 by default
    bad_threshold = float(cfg.bad_quantile)

    for inst in with_returns:
        key = (str(inst["pattern_type"]), str(inst["timeframe"]))
        dist_by_h = dists[key]

        weighted_score = 0.0
        for horizon_key, weight in cfg.horizon_weights.items():
            q = _quantile_rank(dist_by_h[horizon_key], float(inst["forward_returns"][horizon_key]))
            weighted_score += float(weight) * q

        if weighted_score >= good_threshold:
            label: Label = "Good"
        elif weighted_score <= bad_threshold:
            label = "Bad"
        else:
            label = "Neutral"

        inst["label"] = label
        inst["label_score"] = weighted_score

    # 4) Merge back; instances without returns get Neutral with empty returns.
    out: list[dict[str, Any]] = []
    by_ts_key: dict[tuple[str, str, int], dict[str, Any]] = {
        (str(i["pattern_type"]), str(i["timeframe"]), int(i["timestamp"])): i for i in with_returns
    }
    for inst in instances:
        key = (str(inst["pattern_type"]), str(inst["timeframe"]), int(inst["timestamp"]))
        labeled = by_ts_key.get(key)
        if labeled is None:
            out.append({**inst, "forward_returns": {}, "label": "Neutral", "label_score": 0.5})
        else:
            out.append(labeled)
    return out

