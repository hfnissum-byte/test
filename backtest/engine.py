from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from backtest.metrics import BacktestTrade, compute_drawdown_curve, compute_metrics
from backtest.policies import ClassicalModelPolicy, ModelPolicyConfig, PatternSimilarityPolicy, QuantumModelPolicy
from data.storage import ParquetCandleStorage
from patterns.regime import true_range, volatility_regime_from_atr_percent


logger = logging.getLogger(__name__)


def _parse_embedding(value: Any) -> np.ndarray:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    if not isinstance(parsed, list):
        raise ValueError("embedding is not a list")
    return np.asarray([float(x) for x in parsed], dtype=float)


def _timeframe_to_minutes(tf: str) -> int:
    tf = str(tf).strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 24 * 60
    raise ValueError(f"Unsupported timeframe duration: {tf}")


def _compute_atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, *, length: int) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high=high, low=low, prev_close=prev_close)
    atr = np.full_like(tr, fill_value=np.nan, dtype=float)
    if len(tr) < length + 1:
        return np.full_like(tr, float(np.mean(tr)), dtype=float)

    for i in range(length, len(tr)):
        atr[i] = float(np.mean(tr[i - length + 1 : i + 1]))
    first_valid = int(np.where(np.isfinite(atr))[0][0])
    atr[:first_valid] = atr[first_valid]
    return atr


def _find_exec_index(ts_array: np.ndarray, ts_ms: int) -> int:
    # Find the first candle with open time >= ts_ms.
    idx = int(np.searchsorted(ts_array, ts_ms, side="left"))
    if idx >= ts_array.size:
        return ts_array.size - 1
    return idx


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    decision_timeframe: str
    execution_timeframe: str
    exit_horizon: str = "1h"

    exchange_id: str = "binance"
    eval_start_fraction: float = 0.7
    max_events: Optional[int] = None

    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 0.01
    max_risk_per_trade_pct: float = 0.02

    fee_pct: float = 0.001
    slippage_pct: float = 0.0005

    atr_period: int = 14
    atr_mult_sl: float = 1.5
    atr_mult_tp: float = 2.0

    min_confidence: float = 0.55
    p_long_threshold: float = 0.55
    p_short_threshold: float = 0.55

    pattern_sim_top_k: int = 50

    # Reporting output base directory.
    report_base_dir: str = "backtest/results"
    # Logging control: Phase 9 can be quite verbose (per-trade). Default off.
    log_trades: bool = False


def _backtest_policy(
    *,
    policy_name: str,
    policy: Any,
    patterns_df_all: pd.DataFrame,
    exec_df: pd.DataFrame,
    cfg: BacktestConfig,
    eval_start_ts: int,
) -> dict[str, Any]:
    """
    Single-position backtest (no overlap). Uses event timestamps from patterns_df_all.
    """
    exec_ts = exec_df["timestamp"].to_numpy(dtype=np.int64)
    o = exec_df["open"].to_numpy(dtype=float)
    h = exec_df["high"].to_numpy(dtype=float)
    l = exec_df["low"].to_numpy(dtype=float)
    c = exec_df["close"].to_numpy(dtype=float)

    atr = _compute_atr_series(h, l, c, length=cfg.atr_period)

    horizon_minutes = _timeframe_to_minutes(cfg.exit_horizon)
    exec_tf_minutes = _timeframe_to_minutes(cfg.execution_timeframe)
    horizon_steps = max(1, int(round(horizon_minutes / exec_tf_minutes)))

    patterns_sorted = patterns_df_all.sort_values("timestamp").reset_index(drop=True)
    if patterns_sorted.empty:
        return {"policy": policy_name, "metrics": compute_metrics(initial_equity=cfg.initial_equity, equity_curve=[cfg.initial_equity], trades=[]) }

    # Evaluation/trading events:
    trade_events = patterns_sorted[patterns_sorted["timestamp"] >= eval_start_ts].copy()
    if cfg.max_events is not None:
        trade_events = trade_events.head(int(cfg.max_events)).copy()
    if trade_events.empty:
        return {
            "policy": policy_name,
            "metrics": compute_metrics(initial_equity=cfg.initial_equity, equity_curve=[cfg.initial_equity], trades=[]),
        }

    # For signal generation:
    # - classical needs seq_len-1 prior events as context for window-end predictions
    # - others only need the evaluation/trade events
    if policy_name == "classical" and hasattr(policy, "model"):
        try:
            seq_len = int(policy.model.cfg.seq_len)  # type: ignore[attr-defined]
        except Exception:
            seq_len = 1
        # Find idx of first trade event in the full patterns_sorted frame.
        first_trade_ts = int(trade_events["timestamp"].iloc[0])
        idx_first_trade = int(patterns_sorted.index[patterns_sorted["timestamp"] == first_trade_ts][0])
        context_start_idx = max(0, idx_first_trade - max(seq_len - 1, 0))
        signal_events = patterns_sorted.iloc[context_start_idx:].copy()
    else:
        signal_events = trade_events

    signals = policy.predict_signals(events=signal_events)
    if len(signals) != len(signal_events):
        raise RuntimeError(f"Policy returned {len(signals)} signals for {len(signal_events)} events")

    equity = float(cfg.initial_equity)
    # Time-series points for report: list of dicts -> saved to CSV.
    equity_points: list[dict[str, Any]] = [{"timestamp": int(eval_start_ts), "equity": float(equity)}]
    trades: list[BacktestTrade] = []
    trade_logs: list[dict[str, Any]] = []
    executed_trade_cap = int(cfg.max_events) if cfg.max_events is not None else None

    last_exit_ts = -1
    # Iterate signal events chronologically.
    for i in range(len(signal_events)):
        ev = signal_events.iloc[i]
        ts_event = int(ev["timestamp"])
        # Only take decisions in evaluation window.
        if ts_event < eval_start_ts:
            continue
        if ts_event < last_exit_ts:
            # No overlap: prior trade hasn’t exited yet.
            continue

        sig = signals[i]
        if sig.direction == 0:
            continue
        if sig.confidence < cfg.min_confidence:
            continue

        dir_ = int(sig.direction)

        entry_idx = _find_exec_index(exec_ts, ts_event)
        if entry_idx < 0 or entry_idx >= exec_ts.size - 2:
            continue

        entry_raw = float(o[entry_idx])
        entry_eff = entry_raw * (1.0 + cfg.slippage_pct) if dir_ == 1 else entry_raw * (1.0 - cfg.slippage_pct)

        atr_entry = float(atr[entry_idx])
        if not np.isfinite(atr_entry) or atr_entry <= 0:
            continue

        # Volatility regime-aware multipliers (auditable, deterministic).
        close_entry = float(c[entry_idx])
        atr_pct = atr_entry / close_entry if close_entry != 0 else 0.0
        regime = volatility_regime_from_atr_percent(atr_pct)

        if regime == "low":
            sl_mult = cfg.atr_mult_sl * 0.85
            tp_mult = cfg.atr_mult_tp * 0.9
        elif regime == "high":
            sl_mult = cfg.atr_mult_sl * 1.15
            tp_mult = cfg.atr_mult_tp * 1.2
        else:
            sl_mult = cfg.atr_mult_sl
            tp_mult = cfg.atr_mult_tp

        sl_raw = entry_raw - sl_mult * atr_entry if dir_ == 1 else entry_raw + sl_mult * atr_entry
        tp_raw = entry_raw + tp_mult * atr_entry if dir_ == 1 else entry_raw - tp_mult * atr_entry

        # Risk-based sizing.
        risk_amount = equity * float(cfg.risk_per_trade_pct)
        max_risk_amount = equity * float(cfg.max_risk_per_trade_pct)
        risk_amount = min(risk_amount, max_risk_amount)

        sl_dist_eff = (entry_eff - sl_raw) if dir_ == 1 else (sl_raw - entry_eff)
        if sl_dist_eff <= 0:
            continue

        qty_by_risk = risk_amount / sl_dist_eff
        qty_by_cash = equity / entry_eff if entry_eff != 0 else 0.0
        qty = float(min(qty_by_risk, qty_by_cash))
        if qty <= 0 or not np.isfinite(qty):
            continue

        fees_entry = qty * abs(entry_eff) * float(cfg.fee_pct)

        # Simulate within horizon: stop-loss has priority if both hit.
        exit_idx = min(exec_ts.size - 1, entry_idx + horizon_steps)
        exit_reason = "horizon"
        exit_raw: float = float(c[exit_idx])
        hit_stop = False
        hit_tp = False

        for j in range(entry_idx, exit_idx + 1):
            if dir_ == 1:
                if l[j] <= sl_raw:
                    hit_stop = True
                    exit_raw = sl_raw
                    exit_reason = "stop_loss"
                    break
                if h[j] >= tp_raw:
                    hit_tp = True
                    exit_raw = tp_raw
                    exit_reason = "take_profit"
                    break
            else:
                if h[j] >= sl_raw:
                    hit_stop = True
                    exit_raw = sl_raw
                    exit_reason = "stop_loss"
                    break
                if l[j] <= tp_raw:
                    hit_tp = True
                    exit_raw = tp_raw
                    exit_reason = "take_profit"
                    break

        if hit_stop and hit_tp:
            exit_reason = "stop_loss"

        # Apply slippage on exit.
        exit_eff = exit_raw * (1.0 - cfg.slippage_pct) if dir_ == 1 else exit_raw * (1.0 + cfg.slippage_pct)
        fees_exit = qty * abs(exit_eff) * float(cfg.fee_pct)

        pnl = qty * (exit_eff - entry_eff) if dir_ == 1 else qty * (entry_eff - exit_eff)
        net_pnl = pnl - (fees_entry + fees_exit)

        equity_before = equity
        equity += float(net_pnl)

        exit_ts = int(exec_ts[exit_idx])
        last_exit_ts = exit_ts

        equity_points.append({"timestamp": int(exit_ts), "equity": float(equity)})
        ret_pct = float(net_pnl / equity_before) if equity_before != 0 else 0.0
        trades.append(BacktestTrade(pnl=float(net_pnl), return_pct=ret_pct, direction=dir_))
        trade_logs.append(
            {
                "entry_ts": int(ts_event),
                "exit_ts": int(exec_ts[exit_idx]),
                "direction": int(dir_),
                "confidence": float(sig.confidence),
                "entry_price": float(entry_raw),
                "entry_eff_price": float(entry_eff),
                "exit_price": float(exit_raw),
                "exit_eff_price": float(exit_eff),
                "qty": float(qty),
                "fees_entry": float(fees_entry),
                "fees_exit": float(fees_exit),
                "net_pnl": float(net_pnl),
                "return_pct": float(ret_pct),
                "exit_reason": str(exit_reason),
                "atr_entry": float(atr_entry),
                "atr_pct": float(atr_pct),
                "regime": str(regime),
                "sl_mult": float(sl_mult),
                "tp_mult": float(tp_mult),
                "sl_raw": float(sl_raw),
                "tp_raw": float(tp_raw),
                "horizon_steps": int(horizon_steps),
            }
        )

        if executed_trade_cap is not None and len(trades) >= executed_trade_cap:
            break

        if bool(cfg.log_trades):
            logger.info(
                "policy=%s entry_ts=%s dir=%s entry=%.4f exit=%.4f reason=%s net_pnl=%.4f equity=%.2f",
                policy_name,
                ts_event,
                dir_,
                entry_eff,
                exit_eff,
                exit_reason,
                net_pnl,
                equity,
            )

    equity_curve_arr = [float(p["equity"]) for p in equity_points]
    dd_series = compute_drawdown_curve(equity_curve_arr)
    metrics = compute_metrics(initial_equity=cfg.initial_equity, equity_curve=equity_curve_arr, trades=trades)

    # Persist reports.
    report_dir = Path(cfg.report_base_dir) / policy_name
    report_dir.mkdir(parents=True, exist_ok=True)

    # Equity + drawdown curves.
    eq_df = pd.DataFrame(equity_points)
    eq_df.to_csv(report_dir / "equity_curve.csv", index=False)

    dd_df = pd.DataFrame({"timestamp": eq_df["timestamp"].astype(np.int64), "drawdown": dd_series})
    dd_df.to_csv(report_dir / "drawdown_curve.csv", index=False)

    trades_df = pd.DataFrame(trade_logs)
    trades_df.to_csv(report_dir / "trades.csv", index=False)

    metrics_out = {
        "policy": policy_name,
        "symbol": cfg.symbol,
        "decision_timeframe": cfg.decision_timeframe,
        "execution_timeframe": cfg.execution_timeframe,
        "exit_horizon": cfg.exit_horizon,
        "eval_start_fraction": cfg.eval_start_fraction,
        "eval_start_ts": int(eval_start_ts),
        "initial_equity": cfg.initial_equity,
        "risk_per_trade_pct": cfg.risk_per_trade_pct,
        "max_risk_per_trade_pct": cfg.max_risk_per_trade_pct,
        "fee_pct": cfg.fee_pct,
        "slippage_pct": cfg.slippage_pct,
        "atr_period": cfg.atr_period,
        "atr_mult_sl": cfg.atr_mult_sl,
        "atr_mult_tp": cfg.atr_mult_tp,
        "min_confidence": cfg.min_confidence,
        "p_long_threshold": cfg.p_long_threshold,
        "p_short_threshold": cfg.p_short_threshold,
        **metrics,
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    return {
        "policy": policy_name,
        "metrics": metrics,
        "trade_count": len(trades),
        "equity_final": equity,
        "report_dir": str(report_dir),
    }


def backtest_and_compare(
    *,
    storage: ParquetCandleStorage,
    pattern_csv_path: str,
    classical_model_path: str,
    quantum_model_path: str,
    vector_db_base_dir: str,
    cfg: BacktestConfig,
    policies: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    patterns_df = pd.read_csv(pattern_csv_path)
    patterns_df = patterns_df[patterns_df["symbol"] == cfg.symbol]
    patterns_df = patterns_df[patterns_df["timeframe"] == cfg.decision_timeframe]
    patterns_df = patterns_df.sort_values("timestamp").reset_index(drop=True)
    if patterns_df.empty:
        raise RuntimeError(f"No pattern records found for symbol={cfg.symbol} timeframe={cfg.decision_timeframe}")

    ts_all = patterns_df["timestamp"].to_numpy(dtype=np.int64)
    eval_start_ts = int(np.quantile(ts_all, float(cfg.eval_start_fraction)))

    exec_df = storage.load_candles(exchange=cfg.exchange_id, symbol=cfg.symbol, timeframe=cfg.execution_timeframe)
    if exec_df.empty:
        raise RuntimeError(
            f"No stored candles for exchange={cfg.exchange_id} symbol={cfg.symbol} timeframe={cfg.execution_timeframe}"
        )
    exec_df = exec_df.sort_values("timestamp").reset_index(drop=True)

    policy_cfg = ModelPolicyConfig(
        min_confidence=cfg.min_confidence,
        p_long_threshold=cfg.p_long_threshold,
        p_short_threshold=cfg.p_short_threshold,
    )

    if policies is None:
        # Baseline comparison policies (Phase 9 start: no PPO).
        policies = {
            "quantum": QuantumModelPolicy(model_path=quantum_model_path, cfg=policy_cfg),
            "classical": ClassicalModelPolicy(model_path=classical_model_path, cfg=policy_cfg, seq_stride=None),
            "pattern-similarity": PatternSimilarityPolicy(
                pattern_vector_db_base_dir=vector_db_base_dir, cfg=policy_cfg, top_k=cfg.pattern_sim_top_k
            ),
        }

    results: dict[str, Any] = {}
    for policy_name, policy in policies.items():
        res = _backtest_policy(
            policy_name=str(policy_name),
            policy=policy,
            patterns_df_all=patterns_df,
            exec_df=exec_df,
            cfg=cfg,
            eval_start_ts=eval_start_ts,
        )
        results[str(policy_name)] = res["metrics"]

    comparison = {
        "decision_timeframe": cfg.decision_timeframe,
        "execution_timeframe": cfg.execution_timeframe,
        "exit_horizon": cfg.exit_horizon,
        "eval_start_fraction": cfg.eval_start_fraction,
        "eval_start_ts": eval_start_ts,
        "results": {
            **results,
        },
    }
    return comparison

