from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.schemas import PatternRecord
from patterns.embeddings import EMBED_DIM
from patterns.vector_db import create_vector_db


def _make_record(*, symbol: str, timeframe: str, timestamp: int, pattern_type: str, label: str, confidence: float, emb_idx: int) -> PatternRecord:
    emb = np.zeros((EMBED_DIM,), dtype=float)
    emb[emb_idx % EMBED_DIM] = 1.0
    return PatternRecord(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp,
        pattern_type=pattern_type,
        label=label,
        detector_features={},
        geometry_metrics={},
        volatility_context={},
        volume_context={},
        forward_returns={},
        embedding=[float(x) for x in emb.tolist()],
        confidence=float(confidence),
    )


def test_vector_db_local_upsert_and_query(tmp_path: Path) -> None:
    # chromadb/faiss aren't required in this runtime; factory should fall back cleanly.
    vdb = create_vector_db(base_dir=tmp_path / "vdb", prefer="chroma")

    r1 = _make_record(symbol="BTC/USDT", timeframe="1d", timestamp=1000, pattern_type="Hammer", label="Good", confidence=0.9, emb_idx=1)
    r2 = _make_record(symbol="BTC/USDT", timeframe="1d", timestamp=2000, pattern_type="Doji", label="Bad", confidence=0.8, emb_idx=2)
    vdb.upsert([r1, r2])

    # Query with r1 embedding should return Hammer as top result.
    q_emb = r1.embedding
    res = vdb.query(embedding=q_emb, top_k=2, symbol="BTC/USDT", timeframe="1d")
    assert len(res) >= 1
    assert res[0]["pattern_type"] == "Hammer"

    # Filtering by label should narrow results.
    res_bad = vdb.query(embedding=r2.embedding, top_k=2, symbol="BTC/USDT", timeframe="1d", label="Bad")
    assert all(x["label"] == "Bad" for x in res_bad)

