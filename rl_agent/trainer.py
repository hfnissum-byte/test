from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from data.storage import ParquetCandleStorage

from .env import LongOnlyTradingEnv, TradingEnvConfig
from .ppo_agent import train_ppo

logger = logging.getLogger(__name__)


def train_ppo_from_storage(
    *,
    storage: ParquetCandleStorage,
    exchange_candidates: Sequence[str],
    symbol: str,
    timeframe: str,
    out_dir: str | Path,
    episode_length: int = 250,
    lookback: int = 30,
    total_timesteps: int = 50_000,
    seed: int = 42,
) -> Path:
    env_cfg = TradingEnvConfig(
        symbol=symbol,
        timeframe=timeframe,
        episode_length=episode_length,
        lookback=lookback,
        seed=seed,
    )

    env = LongOnlyTradingEnv(
        storage=storage,
        exchange_candidates=exchange_candidates,
        cfg=env_cfg,
    )

    model_path = train_ppo(env=env, out_dir=out_dir, total_timesteps=total_timesteps, seed=seed)
    logger.info("PPO model saved to %s", model_path)
    return model_path

