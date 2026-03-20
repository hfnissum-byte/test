"""
Backtesting engine (Phase 9).

Supports pluggable signal generators (policies) and shared execution + risk logic.
PPO is intentionally optional and NOT the default primary policy.
"""

from .engine import backtest_and_compare  # noqa: F401
from .compare import rank_backtest_policies  # noqa: F401

