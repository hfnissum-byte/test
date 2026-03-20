from __future__ import annotations

import time
from dataclasses import dataclass


def now_ms() -> int:
    return int(time.time() * 1000)


def ms_to_iso(ms: int) -> str:
    # Use UTC "Z" format without extra dependencies.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ms / 1000))


def safe_int(value: int | float) -> int:
    return int(value)


@dataclass(frozen=True)
class TimeframeInfo:
    timeframe: str
    milliseconds: int

