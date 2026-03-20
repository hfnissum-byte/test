"""
Continuous learning orchestration (Phase 10+ style).

Runs a loop that:
  1) updates stored OHLCV candles (real exchange via CCXT),
  2) rebuilds/updates pattern DB + embeddings,
  3) retrains quantum + classical models deterministically,
  4) runs baseline backtest comparisons and writes reports.

PPO remains optional and excluded from baseline by design.
"""

from .runner import run_continual_learning  # noqa: F401

