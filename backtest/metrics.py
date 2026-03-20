from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BacktestTrade:
    pnl: float
    return_pct: float
    direction: int  # +1 long, -1 short


def _safe_std(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=1))


def compute_equity_curve_drawdown(equity_curve: Sequence[float]) -> float:
    if len(equity_curve) == 0:
        return 0.0
    equity = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    return float(np.max(dd)) if dd.size else 0.0


def compute_drawdown_curve(equity_curve: Sequence[float]) -> np.ndarray:
    """
    Returns drawdown series (peak-to-current) in the same order as `equity_curve`.
    """
    if len(equity_curve) == 0:
        return np.asarray([], dtype=float)
    equity = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    return dd.astype(float, copy=False)


def compute_metrics(
    *,
    initial_equity: float,
    equity_curve: Sequence[float],
    trades: Sequence[BacktestTrade],
) -> dict[str, float]:
    equity_curve_arr = np.asarray(equity_curve, dtype=float)
    final_equity = float(equity_curve_arr[-1]) if equity_curve_arr.size else float(initial_equity)
    net_return = (final_equity / float(initial_equity)) - 1.0

    max_drawdown = compute_equity_curve_drawdown(equity_curve_arr)

    if not trades:
        return {
            "net_return": float(net_return),
            "max_drawdown": float(max_drawdown),
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "trade_count": 0.0,
        }

    pnl_arr = np.asarray([t.pnl for t in trades], dtype=float)
    ret_arr = np.asarray([t.return_pct for t in trades], dtype=float)

    wins = pnl_arr[pnl_arr > 0.0]
    losses = pnl_arr[pnl_arr < 0.0]

    win_rate = float(wins.size / pnl_arr.size)

    avg_win = float(np.mean(wins)) if wins.size else 0.0
    avg_loss = float(np.mean(losses)) if losses.size else 0.0  # negative

    expectancy = float(np.mean(pnl_arr))

    profit_factor = float(np.sum(wins) / abs(np.sum(losses))) if losses.size and np.sum(losses) != 0 else 0.0

    # Sharpe/Sortino on trade returns; no annualization here (baseline determinism).
    mean_ret = float(np.mean(ret_arr))
    std_ret = _safe_std(ret_arr)
    sharpe = float(mean_ret / std_ret) if std_ret > 0 else 0.0

    downside = ret_arr[ret_arr < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size >= 2 else 0.0
    sortino = float(mean_ret / downside_std) if downside_std > 0 else 0.0

    calmar = float(net_return / max_drawdown) if max_drawdown > 0 else 0.0

    return {
        "net_return": float(net_return),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy),
        "profit_factor": float(profit_factor),
        "trade_count": float(pnl_arr.size),
    }

