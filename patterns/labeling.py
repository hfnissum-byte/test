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

    # 2) Time-horizon-correct quantile-labeling per (pattern_type, timeframe).
    #
    # The old implementation computed empirical CDFs from the full dataset,
    # which means an event's Good/Bad thresholds were partially influenced
    # by future events. This version computes, for each instance and each
    # horizon key (1h/4h/24h), the quantile rank using only prior events
    # whose forward-return horizon has fully elapsed:
    #   prior_ts <= current_ts - horizon_step_ms
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for inst in with_returns:
        key = (str(inst["pattern_type"]), str(inst["timeframe"]))
        grouped.setdefault(key, []).append(inst)

    # 3) Assign weighted Good/Bad/Neutral.
    good_threshold = float(cfg.good_quantile)  # weights sum to 1 by default
    bad_threshold = float(cfg.bad_quantile)

    # BIT (Fenwick tree) helper for efficient prefix counts.
    def _bit_update(bit: list[int], idx: int, delta: int) -> None:
        # idx is 1-based
        n = len(bit) - 1
        while idx <= n:
            bit[idx] += delta
            idx += idx & -idx

    def _bit_query(bit: list[int], idx: int) -> int:
        # sum of [1..idx], idx is 1-based
        s = 0
        while idx > 0:
            s += bit[idx]
            idx -= idx & -idx
        return s

    for key, group in grouped.items():
        group_sorted = sorted(group, key=lambda x: int(x["timestamp"]))
        ts = np.asarray([int(inst["timestamp"]) for inst in group_sorted], dtype=np.int64)
        m = ts.size
        if m == 0:
            continue

        weighted_score = np.zeros(m, dtype=float)

        # For each horizon, compute horizon-specific quantile ranks q_i.
        for horizon_key, weight in cfg.horizon_weights.items():
            values = np.asarray([float(inst["forward_returns"][horizon_key]) for inst in group_sorted], dtype=float)
            step_hours = float(cfg.horizons_hours[horizon_key])
            step_ms = int(step_hours * 3_600_000)

            # Coordinate compress float values for BIT indexing.
            uniq = np.unique(values)
            # Rank is 1-based.
            ranks = np.searchsorted(uniq, values, side="left").astype(int) + 1

            bit = [0 for _ in range(len(uniq) + 2)]
            eligible_count = 0
            # `eligible_count` tracks how many leading instances are eligible for
            # current index (i.e., their timestamp <= current_ts - step_ms).
            k = 0

            q = np.full(m, 0.5, dtype=float)
            for i in range(m):
                cutoff = int(ts[i] - step_ms)
                # Add instances that are now eligible and strictly before i.
                while k < i and int(ts[k]) <= cutoff:
                    _bit_update(bit, int(ranks[k]), 1)
                    k += 1
                eligible_count = k
                if eligible_count > 0:
                    c_le = _bit_query(bit, int(ranks[i]))
                    q[i] = float(c_le) / float(eligible_count)

            weighted_score += float(weight) * q

        for i, inst in enumerate(group_sorted):
            score = float(weighted_score[i])
            if score >= good_threshold:
                label: Label = "Good"
            elif score <= bad_threshold:
                label = "Bad"
            else:
                label = "Neutral"

            inst["label"] = label
            inst["label_score"] = score

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

