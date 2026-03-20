from __future__ import annotations

import numpy as np
import pandas as pd

from patterns.labeling import DEFAULT_LABELING_CONFIG, LabelingConfig, label_pattern_instances


def test_labeling_time_horizon_correctness() -> None:
    """
    Ensure labels (specifically label_score) are computed using only prior
    events whose forward-return horizons have fully elapsed.

    Construction:
    - constant close => forward returns are always 0
    - two events for the same (pattern_type, timeframe) at ts=0 and ts=5h
    - weights are 0.2 (1h), 0.3 (4h), 0.5 (24h)

    At ts=5h:
    - 1h forward return for prior event at ts=0 is known (0 <= 5h-1h)
    - 4h forward return for prior event at ts=0 is known (0 <= 5h-4h)
    - 24h forward return for prior event at ts=0 is NOT known (0 <= 5h-24h is false)

    Therefore:
      q_1h = 1.0 (only prior eligible value, <= current value)
      q_4h = 1.0
      q_24h = 0.5 (no eligible history)
    label_score = 0.2*1 + 0.3*1 + 0.5*0.5 = 0.75
    """
    n_hours = 40
    ts = np.arange(n_hours, dtype=np.int64) * 3_600_000
    close = np.ones_like(ts, dtype=float) * 100.0
    df_1h = pd.DataFrame({"timestamp": ts, "close": close})

    t0 = int(ts[0])
    t5 = int(ts[5])

    instances = [
        {"pattern_type": "TestPattern", "timeframe": "1h", "timestamp": t0, "confidence": 0.5},
        {"pattern_type": "TestPattern", "timeframe": "1h", "timestamp": t5, "confidence": 0.5},
    ]

    cfg = LabelingConfig(
        horizons_hours=DEFAULT_LABELING_CONFIG.horizons_hours,
        horizon_weights=DEFAULT_LABELING_CONFIG.horizon_weights,
        good_quantile=DEFAULT_LABELING_CONFIG.good_quantile,
        bad_quantile=DEFAULT_LABELING_CONFIG.bad_quantile,
    )

    labeled = label_pattern_instances(df_1h=df_1h, instances=instances, cfg=cfg)
    labeled_by_ts = {int(x["timestamp"]): x for x in labeled}

    assert int(labeled_by_ts[t0]["timestamp"]) == t0
    assert int(labeled_by_ts[t5]["timestamp"]) == t5

    # The exact score for t=5h depends only on the horizon eligibility logic.
    score_t5 = float(labeled_by_ts[t5]["label_score"])
    assert abs(score_t5 - 0.75) < 1e-9

