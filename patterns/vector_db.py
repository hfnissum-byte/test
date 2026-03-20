from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from data.schemas import PatternRecord
from patterns.embeddings import EMBED_DIM

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class LocalEmbeddingMemory:
    """
    Minimal local vector store using brute-force cosine similarity.
    This keeps Phase 5 runnable without Chroma/FAISS dependencies.
    """

    def __init__(self, *, memory_dir: Path, embedding_dim: int = EMBED_DIM):
        self._memory_dir = Path(memory_dir)
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        self._meta_path = self._memory_dir / "metadata.csv"
        self._emb_path = self._memory_dir / "embeddings.npy"
        self._embedding_dim = int(embedding_dim)

        self._metadata: pd.DataFrame
        self._embeddings: np.ndarray

        if self._meta_path.exists() and self._emb_path.exists():
            self._metadata = pd.read_csv(self._meta_path)
            self._embeddings = np.load(self._emb_path)
        else:
            self._metadata = pd.DataFrame(columns=["symbol", "timeframe", "timestamp", "pattern_type", "label", "confidence"])
            self._embeddings = np.zeros((0, self._embedding_dim), dtype=float)

    def _make_key_mask(
        self,
        *,
        symbol: str,
        timeframe: str,
        timestamp: int,
        pattern_type: str,
    ) -> pd.Series:
        mask = (
            (self._metadata["symbol"] == symbol)
            & (self._metadata["timeframe"] == timeframe)
            & (self._metadata["timestamp"] == int(timestamp))
            & (self._metadata["pattern_type"] == pattern_type)
        )
        return mask

    def upsert(self, records: Sequence[PatternRecord]) -> None:
        if not records:
            return

        # Convert to rows.
        new_meta_rows = []
        new_embeddings = []

        for r in records:
            emb = np.array(r.embedding, dtype=float)
            if emb.ndim != 1:
                raise ValueError("Embedding must be 1D.")
            if emb.size != self._embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self._embedding_dim}, got {emb.size}")
            if self._embeddings.shape[0] == 0 and self._embeddings.shape[1] != emb.size:
                self._embeddings = np.zeros((0, self._embedding_dim), dtype=float)

            mask = self._make_key_mask(symbol=r.symbol, timeframe=r.timeframe, timestamp=r.timestamp, pattern_type=r.pattern_type)
            if mask.any():
                idx = int(np.where(mask.to_numpy())[0][0])
                self._embeddings[idx, :] = emb
                self._metadata.loc[mask.index[idx], "label"] = r.label
                self._metadata.loc[mask.index[idx], "confidence"] = float(r.confidence)
            else:
                new_meta_rows.append(
                    {
                        "symbol": r.symbol,
                        "timeframe": r.timeframe,
                        "timestamp": int(r.timestamp),
                        "pattern_type": r.pattern_type,
                        "label": r.label,
                        "confidence": float(r.confidence),
                    }
                )
                new_embeddings.append(emb)

        if new_meta_rows:
            self._metadata = pd.concat([self._metadata, pd.DataFrame(new_meta_rows)], ignore_index=True)
            self._embeddings = np.vstack([self._embeddings, np.vstack(new_embeddings)])

        self._metadata.to_csv(self._meta_path, index=False)
        np.save(self._emb_path, self._embeddings)

    def query(
        self,
        *,
        embedding: Sequence[float],
        top_k: int = 10,
        symbol: str | None = None,
        timeframe: str | None = None,
        pattern_type: str | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._embeddings.shape[0] == 0:
            return []

        emb_q = np.array(embedding, dtype=float)

        mask = pd.Series(True, index=self._metadata.index)
        if symbol is not None:
            mask &= self._metadata["symbol"] == symbol
        if timeframe is not None:
            mask &= self._metadata["timeframe"] == timeframe
        if pattern_type is not None:
            mask &= self._metadata["pattern_type"] == pattern_type
        if label is not None:
            mask &= self._metadata["label"] == label

        idxs = self._metadata.index[mask].to_numpy()
        if idxs.size == 0:
            return []

        sims: list[tuple[float, int]] = []
        for idx in idxs:
            sims.append((_cosine_similarity(emb_q, self._embeddings[int(idx)]), int(idx)))

        sims.sort(key=lambda x: x[0], reverse=True)
        top = sims[: int(top_k)]
        results: list[dict[str, Any]] = []
        for sim, idx in top:
            row = self._metadata.iloc[int(idx)].to_dict()
            results.append({"similarity": float(sim), **row})
        return results


def _record_id(r: PatternRecord) -> str:
    # Stable ID for upserts across backends.
    return f"{r.symbol}|{r.timeframe}|{int(r.timestamp)}|{r.pattern_type}"


class ChromaEmbeddingMemory:
    """
    Chroma persistent vector store.

    Notes:
    - Uses metadata fields for filtering.
    - If Chroma isn't installed, this class isn't importable at runtime; callers should use factory.
    """

    def __init__(self, *, persist_dir: Path, collection_name: str):
        import chromadb  # type: ignore[import-not-found]

        self._persist_dir = Path(persist_dir)
        self._collection_name = collection_name

        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, records: Sequence[PatternRecord]) -> None:
        if not records:
            return

        ids = [_record_id(r) for r in records]
        embeddings = [list(map(float, r.embedding)) for r in records]
        metadatas = [
            {
                "symbol": r.symbol,
                "timeframe": r.timeframe,
                "timestamp": int(r.timestamp),
                "pattern_type": r.pattern_type,
                "label": r.label,
                "confidence": float(r.confidence),
            }
            for r in records
        ]

        # Prefer upsert if available; otherwise emulate with delete+add.
        upsert_fn = getattr(self._collection, "upsert", None)
        if callable(upsert_fn):
            # chromadb signature supports ids/embeddings/metadatas/documents.
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            return

        # Fallback: delete by ids then add.
        delete_fn = getattr(self._collection, "delete", None)
        if callable(delete_fn):
            self._collection.delete(ids=ids)
        add_fn = getattr(self._collection, "add", None)
        if not callable(add_fn):
            raise RuntimeError("Chroma collection has neither upsert nor add methods.")
        self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(
        self,
        *,
        embedding: Sequence[float],
        top_k: int = 10,
        symbol: str | None = None,
        timeframe: str | None = None,
        pattern_type: str | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        # Chroma expects `where` to use operators, e.g. {"symbol": {"$eq": "BTC/USDT"}}.
        # For multiple fields, combine with {"$and": [clause1, clause2, ...]}.
        clauses: list[dict[str, Any]] = []
        if symbol is not None:
            clauses.append({"symbol": {"$eq": symbol}})
        if timeframe is not None:
            clauses.append({"timeframe": {"$eq": timeframe}})
        if pattern_type is not None:
            clauses.append({"pattern_type": {"$eq": pattern_type}})
        if label is not None:
            clauses.append({"label": {"$eq": label}})

        where: dict[str, Any] | None
        if not clauses:
            where = None
        elif len(clauses) == 1:
            where = clauses[0]
        else:
            where = {"$and": clauses}

        res = self._collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=int(top_k),
            where=where,
            include=["metadatas", "distances", "embeddings"],
        )

        results: list[dict[str, Any]] = []
        # Chroma returns lists per query.
        for meta, dist, _id in zip(res.get("metadatas", [[]])[0], res.get("distances", [[]])[0], res.get("ids", [[]])[0]):
            results.append(
                {
                    "similarity": float(1.0 - dist) if dist is not None else 0.0,
                    "id": _id,
                    **(meta or {}),
                }
            )
        return results


class FaissEmbeddingMemory:
    """
    FAISS persistent index with local metadata CSV.

    Note: requires `faiss-cpu` installed at runtime.
    """

    def __init__(self, *, index_dir: Path, embedding_dim: int = EMBED_DIM, index_name: str = "index.faiss"):
        import faiss  # type: ignore[import-not-found]

        self._faiss = faiss
        self._index_dir = Path(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._index_dir / index_name
        self._embedding_dim = int(embedding_dim)
        self._meta_path = self._index_dir / "metadata.csv"

        if self._index_path.exists():
            self._index = self._faiss.read_index(str(self._index_path))
        else:
            # Inner product index on normalized vectors => cosine similarity.
            self._index = self._faiss.IndexFlatIP(self._embedding_dim)

        if self._meta_path.exists():
            self._metadata = pd.read_csv(self._meta_path)
        else:
            self._metadata = pd.DataFrame(columns=["id", "symbol", "timeframe", "timestamp", "pattern_type", "label", "confidence"])

    def upsert(self, records: Sequence[PatternRecord]) -> None:
        if not records:
            return

        # For correctness: easiest upsert is rebuild if ids overlap.
        # For Phase 5 dataset builds (offline), rebuild cost is acceptable.
        ids_new = [_record_id(r) for r in records]
        new_df = pd.DataFrame(
            [
                {
                    "id": ids_new[i],
                    "symbol": r.symbol,
                    "timeframe": r.timeframe,
                    "timestamp": int(r.timestamp),
                    "pattern_type": r.pattern_type,
                    "label": r.label,
                    "confidence": float(r.confidence),
                }
                for i, r in enumerate(records)
            ]
        )

        # Offline Phase 5 builds typically re-run with the full dataset.
        # For correctness without storing historical embeddings, we overwrite the index
        # and metadata from the provided record batch.
        merged = new_df

        embeddings = []
        for r in records:
            emb = np.array(r.embedding, dtype=float)
            embeddings.append(emb)

        # Rebuild index from the provided record batch.
        vecs = np.vstack([np.array(r.embedding, dtype=float) for r in records])
        # Normalize for cosine similarity.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms

        self._index = self._faiss.IndexFlatIP(self._embedding_dim)
        self._index.add(vecs.astype(np.float32))
        self._metadata = merged

        merged.to_csv(self._meta_path, index=False)
        self._faiss.write_index(self._index, str(self._index_path))

    def query(
        self,
        *,
        embedding: Sequence[float],
        top_k: int = 10,
        symbol: str | None = None,
        timeframe: str | None = None,
        pattern_type: str | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._metadata.empty or self._index.ntotal == 0:
            return []

        emb = np.array(embedding, dtype=float)
        if emb.size != self._embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self._embedding_dim}, got {emb.size}")
        norm = float(np.linalg.norm(emb) + 1e-12)
        q = (emb / norm).astype(np.float32).reshape(1, -1)

        sims, idxs = self._index.search(q, int(top_k))
        sims = sims[0].tolist()
        idxs = idxs[0].tolist()

        # Filter results by metadata, then take first K after filtering.
        results: list[dict[str, Any]] = []
        for sim, idx in zip(sims, idxs):
            if idx < 0 or idx >= len(self._metadata):
                continue
            row = self._metadata.iloc[int(idx)]
            if symbol is not None and row["symbol"] != symbol:
                continue
            if timeframe is not None and row["timeframe"] != timeframe:
                continue
            if pattern_type is not None and row["pattern_type"] != pattern_type:
                continue
            if label is not None and row["label"] != label:
                continue
            results.append(
                {
                    "similarity": float(sim),
                    **row.to_dict(),
                }
            )
        return results[: int(top_k)]


class VectorDB:
    """
    Unified interface over Chroma/FAISS/local backends.
    """

    def __init__(self, backend: Any):
        self._backend = backend

    def upsert(self, records: Sequence[PatternRecord]) -> None:
        return self._backend.upsert(records)

    def query(
        self,
        *,
        embedding: Sequence[float],
        top_k: int = 10,
        symbol: str | None = None,
        timeframe: str | None = None,
        pattern_type: str | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._backend.query(
            embedding=embedding,
            top_k=top_k,
            symbol=symbol,
            timeframe=timeframe,
            pattern_type=pattern_type,
            label=label,
        )


def create_vector_db(
    *,
    base_dir: Path,
    prefer: str = "chroma",
    collection_name: str = "pattern_embeddings",
    embedding_dim: int = EMBED_DIM,
) -> VectorDB:
    """
    Create vector DB backend with graceful fallbacks.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    prefer = str(prefer).lower().strip()
    last_err: Optional[Exception] = None

    if prefer in {"chroma", "chroma_first", "auto"}:
        try:
            backend = ChromaEmbeddingMemory(persist_dir=base_dir / "chroma", collection_name=collection_name)
            # If Chroma is present but the collection is empty (e.g. created earlier
            # when Chroma deps were missing and we fell back to local), fall back.
            try:
                n = int(getattr(backend, "_collection").count())
            except Exception:
                n = 0
            if n <= 0:
                raise RuntimeError("Chroma collection is empty; falling back to local.")
            return VectorDB(backend)
        except Exception as e:
            last_err = e

    if prefer in {"faiss", "faiss_first", "auto"}:
        try:
            backend = FaissEmbeddingMemory(index_dir=base_dir / "faiss", embedding_dim=embedding_dim)
            return VectorDB(backend)
        except Exception as e:
            last_err = e

    # Always available.
    logger.warning("Vector DB backend not available; falling back to local store. last_err=%s", last_err)
    backend = LocalEmbeddingMemory(memory_dir=base_dir / "local", embedding_dim=embedding_dim)
    return VectorDB(backend)

