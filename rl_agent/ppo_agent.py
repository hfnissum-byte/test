from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _try_import_ppo():
    try:
        import gymnasium as gym  # noqa: F401
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
    except ModuleNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Import error: {type(e).__name__}: {e}"
    return {"PPO": PPO, "DummyVecEnv": DummyVecEnv}, ""


def train_ppo(
    *,
    env,
    out_dir: str | Path,
    total_timesteps: int = 50_000,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    n_steps: int = 2048,
    batch_size: int = 64,
    seed: int = 42,
    verbose: int = 1,
) -> Path:
    """
    Train PPO via stable-baselines3.

    This is dependency-optional: if gymnasium/stable-baselines3 aren't
    installed, this raises a clear RuntimeError.
    """
    imports, err = _try_import_ppo()
    if imports is None:  # pragma: no cover
        raise RuntimeError(
            "PPO training requires optional dependencies: `gymnasium` and `stable-baselines3` "
            f"(import error: {err}). Install them to enable Phase 8 PPO training."
        )

    PPO = imports["PPO"]
    DummyVecEnv = imports["DummyVecEnv"]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / "ppo_model"

    vec_env = DummyVecEnv([lambda: env])
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=verbose,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)

    model.save(str(model_path))
    meta_path = out_path / "ppo_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path

