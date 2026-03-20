from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.preprocessing import validate_ohlcv_dataframe
from data.schemas import CandleOHLCV
from data.storage import ParquetCandleStorage


HAS_PYARROW = True
try:
    import pyarrow  # noqa: F401
except ModuleNotFoundError:
    HAS_PYARROW = False


def _storage(tmp_path: Path) -> ParquetCandleStorage:
    cfg = {"storage": {"candles_subdir": "candles", "parquet_compression": "snappy"}}
    return ParquetCandleStorage(data_dir=tmp_path, cfg=cfg)


def test_candle_schema_from_ccxt_tuple() -> None:
    c = CandleOHLCV.from_ccxt(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        ohlcv=[1710000000000, "60000", "61000", "59000", "60500", "123.4"],
    )
    assert c.exchange == "binance"
    assert c.symbol == "BTC/USDT"
    assert c.timeframe == "1h"
    assert c.timestamp == 1710000000000
    assert isinstance(c.open, float)
    assert isinstance(c.volume, float)


def test_storage_upsert_dedup_by_timestamp(tmp_path: Path) -> None:
    storage = _storage(tmp_path)

    c1 = CandleOHLCV(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp=1710000000000,
        open=60000,
        high=61000,
        low=59000,
        close=60500,
        volume=10,
    )
    c2 = CandleOHLCV(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp=1710000000000,
        open=60000,
        high=61000,
        low=59000,
        close=60600,
        volume=11,
    )

    storage.upsert_candles([c1])
    storage.upsert_candles([c2])

    df = storage.load_candles(exchange="binance", symbol="BTC/USDT", timeframe="1h")
    assert len(df) == 1
    assert float(df.loc[0, "close"]) == 60600.0
    assert float(df.loc[0, "volume"]) == 11.0


def test_preprocessing_validate_ohlcv_dataframe() -> None:
    df = pd.DataFrame(
        [
            {
                "exchange": "binance",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "timestamp": 2,
                "open": 101.0,
                "high": 110.0,
                "low": 90.0,
                "close": 100.0,
                "volume": 1.0,
            },
            {
                "exchange": "binance",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "timestamp": 1,
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1.0,
            },
        ]
    )
    out = validate_ohlcv_dataframe(df)
    assert list(out["timestamp"]) == [1, 2]

