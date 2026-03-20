from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(data)}")
    return data


def ensure_symbol_list(symbols: Iterable[str]) -> list[str]:
    out: list[str] = []
    for s in symbols:
        s2 = str(s).strip()
        if not s2:
            continue
        # Basic validation only (real validation happens at CCXT symbol parsing time).
        if "/" not in s2:
            raise ValueError(f"Invalid symbol '{s2}'. Expected format like 'BTC/USDT'.")
        out.append(s2)
    if not out:
        raise ValueError("No symbols provided.")
    # Stable ordering is helpful for deterministic ingestion loops.
    return sorted(set(out))


def get_env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_env_optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None

