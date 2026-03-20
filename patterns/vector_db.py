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


def _record_id(r: PatternRecord) -> str:
    # Stable ID for upserts across backends.
    return f"{r.symbol}|{r.timeframe}|{int(r.timestamp)}|{r.pattern_type}"


class LocalEmbeddingMemory:
    """
    Minimal local vector store using brute-force cosine similarity.

    For this project's workflow, pattern DB builds are effectively full offline
    rebuilds, so `upsert()` overwrites the local store from the provided batch
    instead of doing slow row-by-row mask scans.
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
            if self._embeddings.ndim != 2 or (
                self._embeddings.shape[1] != self._embedding_dim and self._embeddings.shape[0] > 0
            ):
                raise ValueError(
                    f"Stored local embedding matrix has wrong shape: {self._embeddings.shape}, "
                    f"expected (*, {self._embedding_dim})"
                )
        else:
            self._metadata = pd.DataFrame(
                columns=["symbol", "timeframe", "timestamp", "pattern_type", "label", "confidence"]
            )
            self._embeddings = np.zeros((0, self._embedding_dim), dtype=float)

    def upsert(self, records: Sequence[PatternRecord]) -> None:
        if not records:
            return

        meta_rows: list[dict[str, Any]] = []
        embeddings: list[np.ndarray] = []

        for r in records:
            emb = np.asarray(r.embedding, dtype=float)
            if emb.ndim != 1:
                raise ValueError("Embedding must be 1D.")
            if emb.size != self._embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self._embedding_dim}, got {emb.size}"
                )

            meta_rows.append(
                {
                    "symbol": r.symbol,
                    "timeframe": r.timeframe,
                    "timestamp": int(r.timestamp),
                    "pattern_type": r.pattern_type,
                    "label": r.label,
                    "confidence": float(r.confidence),
                }
            )
            embeddings.append(emb)

        self._metadata = pd.DataFrame(meta_rows)
        self._embeddings = (
            np.vstack(embeddings)
            if embeddings
            else np.zeros((0, self._embedding_dim), dtype=float)
        )

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

        emb_q = np.asarray(embedding, dtype=float)
        if emb_q.ndim != 1 or emb_q.size != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, got {emb_q.size}"
            )

        mask = pd.Series(True, index=self._metadata.index)
        if symbol is not None:
            mask &= self._metadata["symbol"] == symbol
        if timeframe is not None:
            mask &= self._metadata["timeframe"] == timeframe
        if pattern_type is not None:
            mask &= self._metadata["pattern_type"] == pattern_type
        if label is not None:
            mask &= self._metadata["label"] == label

        idxs = self._metadata.index[mask].to_numpy(dtype=int)
        if idxs.size == 0:
            return []

        sub = self._embeddings[idxs]
        q_norm = float(np.linalg.norm(emb_q))
        if q_norm == 0.0:
            return []

        sub_norms = np.linalg.norm(sub, axis=1)
        denom = (sub_norms * q_norm) + 1e-12
        sims = (sub @ emb_q) / denom

        k = min(int(top_k), sims.shape[0])
        order = np.argsort(-sims)[:k]

        results: list[dict[str, Any]] = []
        for pos in order:
            idx = int(idxs[int(pos)])
            row = self._metadata.iloc[idx].to_dict()
            results.append({"similarity": float(sims[int(pos)]), **row})
        return results


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

        # Chroma enforces an internal max batch size for upsert/add operations.
        # We chunk to stay under the hard limit and keep the critical path
        # runnable for large pattern rebuilds.
        max_batch_size = 5000

        def _batched(seq: Sequence[PatternRecord], size: int):
            for i in range(0, len(seq), size):
                yield seq[i : i + size]

        upsert_fn = getattr(self._collection, "upsert", None)
        if callable(upsert_fn):
            for batch in _batched(records, max_batch_size):
                ids = [_record_id(r) for r in batch]
                embeddings = [list(map(float, r.embedding)) for r in batch]
                metadatas = [
                    {
                        "symbol": r.symbol,
                        "timeframe": r.timeframe,
                        "timestamp": int(r.timestamp),
                        "pattern_type": r.pattern_type,
                        "label": r.label,
                        "confidence": float(r.confidence),
                    }
                    for r in batch
                ]
                self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            return

        # Fallback for older Chroma APIs: delete then add, also in chunks.
        delete_fn = getattr(self._collection, "delete", None)
        add_fn = getattr(self._collection, "add", None)
        if not callable(add_fn):
            raise RuntimeError("Chroma collection has neither upsert nor add methods.")

        if callable(delete_fn):
            for batch in _batched(records, max_batch_size):
                ids = [_record_id(r) for r in batch]
                self._collection.delete(ids=ids)

        for batch in _batched(records, max_batch_size):
            ids = [_record_id(r) for r in batch]
            embeddings = [list(map(float, r.embedding)) for r in batch]
            metadatas = [
                {
                    "symbol": r.symbol,
                    "timeframe": r.timeframe,
                    "timestamp": int(r.timestamp),
                    "pattern_type": r.pattern_type,
                    "label": r.label,
                    "confidence": float(r.confidence),
                }
                for r in batch
            ]
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
            include=["metadatas", "distances"],
        )

        results: list[dict[str, Any]] = []
        for meta, dist, _id in zip(
            res.get("metadatas", [[]])[0],
            res.get("distances", [[]])[0],
            res.get("ids", [[]])[0],
        ):
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
            self._index = self._faiss.IndexFlatIP(self._embedding_dim)

        if self._meta_path.exists():
            self._metadata = pd.read_csv(self._meta_path)
        else:
            self._metadata = pd.DataFrame(
                columns=["id", "symbol", "timeframe", "timestamp", "pattern_type", "label", "confidence"]
            )

    def upsert(self, records: Sequence[PatternRecord]) -> None:
        if not records:
            return

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

        vecs = np.vstack([np.asarray(r.embedding, dtype=float) for r in records])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms

        self._index = self._faiss.IndexFlatIP(self._embedding_dim)
        self._index.add(vecs.astype(np.float32))
        self._metadata = new_df

        new_df.to_csv(self._meta_path, index=False)
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

        emb = np.asarray(embedding, dtype=float)
        if emb.size != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, got {emb.size}"
            )

        norm = float(np.linalg.norm(emb) + 1e-12)
        q = (emb / norm).astype(np.float32).reshape(1, -1)

        sims, idxs = self._index.search(q, int(top_k))
        sims = sims[0].tolist()
        idxs = idxs[0].tolist()

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

    Important: an empty Chroma collection is valid on first run, so we do not
    treat "count() == 0" as an error.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    prefer = str(prefer).lower().strip()
    last_err: Optional[Exception] = None

    if prefer in {"chroma", "chroma_first", "auto"}:
        try:
            backend = ChromaEmbeddingMemory(
                persist_dir=base_dir / "chroma",
                collection_name=collection_name,
            )
            return VectorDB(backend)
        except Exception as e:
            last_err = e

    if prefer in {"faiss", "faiss_first", "auto"}:
        try:
            backend = FaissEmbeddingMemory(
                index_dir=base_dir / "faiss",
                embedding_dim=embedding_dim,
            )
            return VectorDB(backend)
        except Exception as e:
            last_err = e

    logger.warning(
        "Vector DB backend not available; falling back to local store. last_err=%s",
        last_err,
    )
    backend = LocalEmbeddingMemory(memory_dir=base_dir / "local", embedding_dim=embedding_dim)
    return VectorDB(backend)