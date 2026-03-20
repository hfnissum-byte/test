from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from data.storage import ParquetCandleStorage

from patterns.regime import compute_atr, volatility_regime_from_atr_percent, true_range


logger = logging.getLogger(__name__)


try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover (optional dependency)
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

# Make this env a real Gymnasium env (SB3 checks `isinstance(..., gym.Env)`).
_GymBase = gym.Env if gym is not None else object


@dataclass(frozen=True)
class TradingEnvConfig:
    symbol: str
    timeframe: str
    episode_length: int = 250
    lookback: int = 30

    # One position at a time; long-only (action close just exits).
    initial_equity: float = 10_000.0

    fee_pct: float = 0.001  # 0.10%
    slippage_pct: float = 0.0005  # 0.05% effective price impact

    risk_per_trade_pct: float = 0.01  # capped via position sizing at entry
    max_drawdown_pct: float = 0.10  # episode terminates if equity drops below (1-max_dd)

    atr_period: int = 14
    atr_mult_sl: float = 1.5
    atr_mult_tp: float = 2.0

    # Environment behavior
    random_start: bool = True
    seed: int = 42


def _safe_float_array(series: np.ndarray) -> np.ndarray:
    out = np.asarray(series, dtype=float)
    if out.ndim != 1:
        raise ValueError("Expected 1D array")
    return out


class LongOnlyTradingEnv(_GymBase):
    """
    Long-only single-position trading environment backed by stored OHLCV candles.

    Action space (Discrete(3)):
      0 = hold (do nothing)
      1 = enter long if flat; otherwise hold
      2 = close long if in position; otherwise do nothing
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        storage: ParquetCandleStorage,
        exchange_candidates: Sequence[str],
        cfg: TradingEnvConfig,
    ) -> None:
        if gym is None or spaces is None:  # pragma: no cover
            raise RuntimeError(
                "gymnasium is required to create the trading environment. "
                "Install `gymnasium` to enable Phase 8 PPO training."
            )

        self.storage = storage
        self.exchange_candidates = [str(x) for x in exchange_candidates]
        self.cfg = cfg

        self._rng = np.random.default_rng(cfg.seed)

        self._data_df = self._load_candles()
        if len(self._data_df) < self.cfg.lookback + self.cfg.episode_length + 2:
            raise RuntimeError(
                "Not enough stored candles to run an episode. "
                f"Have={len(self._data_df)}, need>={self.cfg.lookback + self.cfg.episode_length + 2}."
            )

        # Pre-extract arrays for speed.
        self._open = _safe_float_array(self._data_df["open"].to_numpy())
        self._high = _safe_float_array(self._data_df["high"].to_numpy())
        self._low = _safe_float_array(self._data_df["low"].to_numpy())
        self._close = _safe_float_array(self._data_df["close"].to_numpy())
        self._volume = _safe_float_array(self._data_df["volume"].to_numpy())
        self._n = len(self._close)

        # Precompute ATR (rolling mean of TR).
        self._atr = self._compute_atr_series()

        # Observation vector:
        # [close_rel_sma, ret1, ret3, ret5, atr_pct, volume_rel, pos_flag, sl_dist_pct, tp_dist_pct, regime_enc]
        self._obs_dim = 10

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # State
        self._reset_state()
        self._current_step = 0
        self._episode_start = 0
        self._episode_end = 0

    def _load_candles(self):
        # Try each exchange candidate in order.
        last_err: Optional[Exception] = None
        for ex in self.exchange_candidates:
            try:
                df = self.storage.load_candles(
                    exchange=ex,
                    symbol=self.cfg.symbol,
                    timeframe=self.cfg.timeframe,
                )
                if not df.empty and {"open", "high", "low", "close", "volume"}.issubset(set(df.columns)):
                    df = df.sort_values("timestamp").reset_index(drop=True)
                    if len(df) > 0:
                        logger.info("Using exchange=%s for %s %s", ex, self.cfg.symbol, self.cfg.timeframe)
                        return df
            except Exception as e:
                last_err = e
        raise RuntimeError(
            f"Unable to load stored candles for {self.cfg.symbol} {self.cfg.timeframe}. "
            f"Tried exchanges={self.exchange_candidates}. last_err={last_err}"
        )

    def _compute_atr_series(self) -> np.ndarray:
        high = self._high
        low = self._low
        close = self._close
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = true_range(high=high, low=low, prev_close=prev_close)
        atr = np.full_like(tr, fill_value=np.nan, dtype=float)
        length = self.cfg.atr_period
        if len(tr) < length + 1:
            # Fallback: constant mean TR to avoid NaNs; correctness-first for small samples.
            mean_tr = float(np.mean(tr))
            return np.full_like(tr, mean_tr, dtype=float)

        # Rolling mean; align so atr[i] uses last `length` TR up to i.
        for i in range(length, len(tr)):
            atr[i] = float(np.mean(tr[i - length + 1 : i + 1]))
        # Fill early indices with first valid ATR.
        first_valid = int(np.where(np.isfinite(atr))[0][0])
        atr[:first_valid] = atr[first_valid]
        return atr

    def _reset_state(self) -> None:
        self._cash = float(self.cfg.initial_equity)
        self._position_qty = 0.0
        self._entry_price = 0.0
        self._stop_loss = 0.0
        self._take_profit = 0.0
        self._active = False
        self._equity = float(self.cfg.initial_equity)

        self._last_equity = self._equity

        self._last_reward = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._reset_state()

        max_start = self._n - (self.cfg.episode_length + 2)
        min_start = self.cfg.lookback + 2
        if max_start <= min_start:
            raise RuntimeError("Episode parameters exceed available candle history.")

        if self.cfg.random_start:
            self._episode_start = int(self._rng.integers(min_start, max_start))
        else:
            self._episode_start = min_start

        self._episode_end = self._episode_start + self.cfg.episode_length
        self._current_step = self._episode_start

        obs = self._get_obs(self._current_step)
        info: dict[str, Any] = {"equity": self._equity, "exchange_loaded": self._infer_exchange_loaded()}
        return obs, info

    def _infer_exchange_loaded(self) -> str:
        # Best-effort: data comes from a single exchange df load. We'll rely on file existence.
        # Not critical for training.
        return self.exchange_candidates[0]

    def _get_obs(self, idx: int) -> np.ndarray:
        lookback = self.cfg.lookback
        start = max(0, idx - lookback + 1)
        closes = self._close[start : idx + 1]

        sma = float(np.mean(closes)) if closes.size else float(self._close[idx])
        close_rel_sma = float(self._close[idx] / sma - 1.0) if sma != 0 else 0.0

        # Returns: guard for early indices.
        ret1 = float(self._close[idx] / self._close[idx - 1] - 1.0) if idx - 1 >= 0 else 0.0
        ret3 = float(self._close[idx] / self._close[idx - 3] - 1.0) if idx - 3 >= 0 else 0.0
        ret5 = float(self._close[idx] / self._close[idx - 5] - 1.0) if idx - 5 >= 0 else 0.0

        atr = float(self._atr[idx])
        atr_pct = float(atr / self._close[idx]) if self._close[idx] != 0 else 0.0
        regime = volatility_regime_from_atr_percent(atr_pct)
        regime_enc = {"low": 0.0, "mid": 0.5, "high": 1.0}[regime]

        vol_window = self._volume[start : idx + 1]
        vol_mean = float(np.mean(vol_window)) if vol_window.size else float(self._volume[idx])
        volume_rel = float(self._volume[idx] / vol_mean - 1.0) if vol_mean != 0 else 0.0

        pos_flag = 1.0 if self._active else 0.0
        sl_dist_pct = (self._entry_price - self._stop_loss) / self._entry_price if self._active and self._entry_price != 0 else 0.0
        tp_dist_pct = (self._take_profit - self._entry_price) / self._entry_price if self._active and self._entry_price != 0 else 0.0

        obs = np.array(
            [
                close_rel_sma,
                ret1,
                ret3,
                ret5,
                atr_pct,
                volume_rel,
                pos_flag,
                sl_dist_pct,
                tp_dist_pct,
                regime_enc,
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: int):
        if not self._active and action not in (0, 1, 2):
            raise ValueError("Invalid action")

        idx = self._current_step
        prev_equity = self._equity

        # Read this bar.
        o = float(self._open[idx])
        h = float(self._high[idx])
        l = float(self._low[idx])
        c = float(self._close[idx])
        # v = float(self._volume[idx])  # not currently used in reward

        # 1) Stop-loss / take-profit have priority (conservative ordering).
        if self._active:
            stop_hit = l <= self._stop_loss
            take_hit = h >= self._take_profit

            exit_reason: Optional[str] = None
            if stop_hit:
                exit_reason = "stop_loss"
                raw_exit = self._stop_loss
            elif take_hit:
                exit_reason = "take_profit"
                raw_exit = self._take_profit
            else:
                raw_exit = None

            if raw_exit is not None:
                eff_exit = raw_exit * (1.0 - self.cfg.slippage_pct)
                notional = self._position_qty * eff_exit
                fee_exit = notional * self.cfg.fee_pct
                cash_delta = notional - fee_exit
                self._cash += cash_delta
                self._position_qty = 0.0
                self._entry_price = 0.0
                self._stop_loss = 0.0
                self._take_profit = 0.0
                self._active = False

        # 2) Agent action after risk checks.
        if not self._active:
            if int(action) == 1:
                # Enter long at close with slippage.
                entry = c * (1.0 + self.cfg.slippage_pct)
                atr = float(self._atr[idx])
                if not np.isfinite(atr) or atr <= 0:
                    # Can't compute meaningful SL/TP; skip entry.
                    pass
                else:
                    sl = entry - self.cfg.atr_mult_sl * atr
                    tp = entry + self.cfg.atr_mult_tp * atr
                    sl_dist = entry - sl
                    if sl_dist <= 0:
                        pass
                    else:
                        # Risk-based sizing (spot-like cap by available cash).
                        risk_amount = float(self._cash) * self.cfg.risk_per_trade_pct
                        qty_by_risk = risk_amount / sl_dist
                        qty_by_cash = float(self._cash) / entry if entry != 0 else 0.0
                        qty = min(float(qty_by_risk), float(qty_by_cash))

                        if qty > 0:
                            cost = qty * entry
                            fee_entry = cost * self.cfg.fee_pct
                            total_cost = cost + fee_entry
                            if total_cost <= self._cash and cost > 0:
                                self._cash -= total_cost
                                self._position_qty = float(qty)
                                self._entry_price = float(entry)
                                self._stop_loss = float(sl)
                                self._take_profit = float(tp)
                                self._active = True
        else:
            # In position: action=2 closes at close with slippage.
            if int(action) == 2:
                eff_exit = c * (1.0 - self.cfg.slippage_pct)
                notional = self._position_qty * eff_exit
                fee_exit = notional * self.cfg.fee_pct
                self._cash += notional - fee_exit
                self._position_qty = 0.0
                self._entry_price = 0.0
                self._stop_loss = 0.0
                self._take_profit = 0.0
                self._active = False

        # 3) Mark-to-market at close (if still in position).
        if self._active:
            pos_value = self._position_qty * c
            self._equity = float(self._cash + pos_value)
        else:
            self._equity = float(self._cash)

        reward = (self._equity - prev_equity) / float(self.cfg.initial_equity)
        self._last_reward = float(reward)

        terminated = False
        truncated = False

        # Max drawdown constraint.
        if self._equity <= self.cfg.initial_equity * (1.0 - self.cfg.max_drawdown_pct):
            terminated = True

        # End of episode by time.
        next_idx = idx + 1
        if next_idx >= self._episode_end:
            truncated = True

        if terminated or truncated:
            obs = self._get_obs(min(next_idx, self._n - 1))
        else:
            obs = self._get_obs(next_idx)

        info: dict[str, Any] = {
            "equity": self._equity,
            "cash": self._cash,
            "in_position": self._active,
            "reward": self._last_reward,
            "step_idx": idx,
        }
        return obs, float(reward), terminated, truncated, info

