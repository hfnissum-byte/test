from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from data.schemas import PatternRecord
from data.storage import ParquetCandleStorage
from patterns.candlestick import detect_candlestick_patterns
from patterns.chart_patterns import detect_chart_patterns
from patterns.embeddings import build_embedding, compute_curve_signature, dtw_distance, resample_series
from patterns.labeling import DEFAULT_LABELING_CONFIG, label_pattern_instances
from patterns.vector_db import create_vector_db
from patterns.regime import compute_volatility_context


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatternBuildConfig:
    symbol: str
    primary_exchange: str = "binance"
    detection_timeframes: list[str] = None  # type: ignore[assignment]
    curve_window: int = 48
    canonical_template_points: int = 32


def _timestamp_to_index(df: pd.DataFrame, ts_ms: int) -> int | None:
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    idx = int(np.searchsorted(ts, ts_ms, side="left"))
    if idx >= ts.size:
        return None
    if ts[idx] != ts_ms:
        # Use the nearest next candle at/after ts.
        if idx + 1 < ts.size:
            return idx
        return None
    return idx


def _canonical_u_signature(n: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, num=n, endpoint=True)
    # Convex parabola (U-shape). Normalize to start at 1.
    y = (x - 0.5) ** 2
    y = y / (y[0] + 1e-12)
    return y.astype(float)


def _canonical_inv_u_signature(n: int) -> np.ndarray:
    return 1.0 / (_canonical_u_signature(n) + 1e-12)


def compute_volume_context(df_slice: pd.DataFrame) -> dict[str, float]:
    vols = df_slice["volume"].to_numpy(dtype=float)
    if vols.size == 0:
        return {"volume_mean": 0.0, "volume_z": 0.0}
    mean = float(np.mean(vols[:-1])) if vols.size > 1 else float(np.mean(vols))
    std = float(np.std(vols[:-1])) if vols.size > 2 else float(np.std(vols))
    last = float(vols[-1])
    z = (last - mean) / (std + 1e-12)
    return {"volume_mean": mean, "volume_z": float(z)}


def _enhance_instance_with_geometry(
    *,
    df: pd.DataFrame,
    inst: dict[str, Any],
    timeframe: str,
    curve_window: int,
) -> dict[str, Any]:
    ts_ms = int(inst["timestamp"])
    idx = _timestamp_to_index(df, ts_ms)
    if idx is None:
        return inst

    start = max(0, idx - curve_window // 2)
    end = min(len(df) - 1, idx + curve_window // 2)
    seg = df.iloc[start : end + 1].copy()
    curve_sig = compute_curve_signature(seg, window=curve_window)

    # DTW distances to canonical U / inverse-U shapes (curve-aware).
    close = seg["close"].to_numpy(dtype=float)
    first = close[0]
    norm = close / (first if first != 0 else 1.0)
    sig = resample_series(norm, n=32)
    u = _canonical_u_signature(32)
    invu = _canonical_inv_u_signature(32)
    dist_u = dtw_distance(sig, u)
    dist_inv = dtw_distance(sig, invu)

    geometry_metrics = dict(inst.get("geometry_metrics", {}))
    geometry_metrics["curvature_mean"] = float(curve_sig["curvature"])
    geometry_metrics["slope_mean"] = float(curve_sig["slope"])
    geometry_metrics["shape_dtw_to_u"] = float(dist_u)
    geometry_metrics["shape_dtw_to_invu"] = float(dist_inv)
    inst["geometry_metrics"] = geometry_metrics
    return inst


def build_pattern_db(
    *,
    storage: ParquetCandleStorage,
    symbol: str,
    detection_timeframes: Sequence[str],
    out_dir: Path,
    curve_window: int = 48,
    primary_exchange_id: str = "binance",
) -> list[PatternRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 1h for labeling horizons.
    df_1h = storage.load_candles(exchange=primary_exchange_id, symbol=symbol, timeframe="1h")
    if df_1h.empty:
        raise RuntimeError(f"Missing 1h data for labeling for {symbol}.")

    all_instances: list[dict[str, Any]] = []
    by_timeframe: dict[str, list[dict[str, Any]]] = {}
    df_by_timeframe: dict[str, pd.DataFrame] = {}

    for tf in detection_timeframes:
        df_tf = storage.load_candles(exchange=primary_exchange_id, symbol=symbol, timeframe=tf)
        if df_tf.empty:
            logger.warning("No candles for %s timeframe %s", symbol, tf)
            continue
        df_by_timeframe[str(tf)] = df_tf

        # Candlesticks + chart structure patterns.
        candles = detect_candlestick_patterns(df_tf, symbol=symbol, timeframe=tf)
        chart = detect_chart_patterns(df_tf, symbol=symbol, timeframe=tf)

        # Normalize instance format.
        instances_tf: list[dict[str, Any]] = []
        for inst in candles + chart:
            instances_tf.append(
                {
                    "symbol": symbol,
                    "timeframe": tf,
                    "timestamp": int(inst["timestamp"]),
                    "pattern_type": str(inst["pattern_type"]),
                    "detector_features": dict(inst.get("detector_features", {})),
                    "geometry_metrics": dict(inst.get("geometry_metrics", {})),
                    "confidence": float(inst.get("confidence", 0.5)),
                }
            )

        # Deduplicate exact (pattern_type, timestamp) keys to prevent explosion from overlapping detectors.
        best_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for inst in instances_tf:
            key = (str(inst["pattern_type"]), int(inst["timestamp"]))
            prev = best_by_key.get(key)
            if prev is None or float(inst.get("confidence", 0.5)) > float(prev.get("confidence", 0.5)):
                best_by_key[key] = inst

        instances_tf_dedup = list(best_by_key.values())

        by_timeframe[tf] = instances_tf_dedup
        all_instances.extend(instances_tf_dedup)
        logger.info("Detected %d instances for %s %s", len(instances_tf_dedup), symbol, tf)

    if not all_instances:
        logger.warning("No pattern instances detected. Exiting.")
        return []

    # Cross-timeframe alignment scoring (only for requested swing frames).
    if len(detection_timeframes) >= 2:
        primary = str(detection_timeframes[0])
        secondary = str(detection_timeframes[1])
        sec_by_type: dict[str, list[int]] = {}
        for inst in by_timeframe.get(secondary, []):
            sec_by_type.setdefault(str(inst["pattern_type"]), []).append(int(inst["timestamp"]))
        # Use a loose window: 2 * secondary timeframe in ms.
        # Approximate milliseconds mapping for these specific timeframes.
        tf_ms_map = {"1d": 86_400_000, "4h": 14_400_000, "1h": 3_600_000, "15m": 900_000, "5m": 300_000}
        window_ms = 2 * tf_ms_map.get(secondary, 14_400_000)
        for inst in by_timeframe.get(primary, []):
            ts_ms = int(inst["timestamp"])
            ts_list = sec_by_type.get(str(inst["pattern_type"]), [])
            matches = sum(1 for x in ts_list if abs(x - ts_ms) <= window_ms)
            if matches > 0:
                inst["confidence"] = float(np.clip(inst["confidence"] * (1.0 + 0.1 * matches), 0.0, 1.0))

    # Enhance with volatility + volume + curve-aware geometry.
    for inst in all_instances:
        tf = str(inst["timeframe"])
        df_tf = df_by_timeframe.get(tf)
        if df_tf is None or df_tf.empty:
            continue
        # Volatility context from recent window.
        idx = _timestamp_to_index(df_tf, int(inst["timestamp"]))
        if idx is None:
            continue
        start = max(0, idx - 50)
        seg = df_tf.iloc[start : idx + 1].copy()
        vol_ctx = compute_volatility_context(seg, atr_length=14)
        inst["volatility_context"] = vol_ctx.__dict__
        inst["volume_context"] = compute_volume_context(seg)

        # Curve-aware signature.
        inst = _enhance_instance_with_geometry(df=df_tf, inst=inst, timeframe=tf, curve_window=curve_window)

    # Compute forward labels.
    labeled_instances = label_pattern_instances(
        df_1h=df_1h,
        instances=all_instances,
        cfg=DEFAULT_LABELING_CONFIG,
    )

    # Build embeddings and materialize PatternRecord objects.
    records: list[PatternRecord] = []
    for inst in labeled_instances:
        if not inst.get("forward_returns"):
            # Could not compute due to lookahead gaps; keep Neutral.
            label = "Neutral"
            forward_returns = {}
        else:
            label = str(inst.get("label", "Neutral"))
            forward_returns = dict(inst.get("forward_returns", {}))

        vol_ctx = inst.get("volatility_context", {"atr_percent": 0.0, "regime": "mid", "atr": 0.0})
        vol_ctx = {k: float(v) if k != "regime" else str(v) for k, v in vol_ctx.items()}
        vol_ctx.setdefault("atr_percent", 0.0)
        vol_ctx.setdefault("regime", "mid")

        geom = dict(inst.get("geometry_metrics", {}))
        detector_features = dict(inst.get("detector_features", {}))
        volume_context = dict(inst.get("volume_context", {}))

        embedding = build_embedding(
            pattern_type=str(inst["pattern_type"]),
            detector_features=detector_features,
            geometry_metrics=geom,
            volatility_context=vol_ctx,
        )

        records.append(
            PatternRecord(
                symbol=symbol,
                timeframe=str(inst["timeframe"]),
                timestamp=int(inst["timestamp"]),
                pattern_type=str(inst["pattern_type"]),
                label=label,
                detector_features=detector_features,
                geometry_metrics={k: float(v) for k, v in geom.items() if isinstance(v, (int, float, np.floating))},
                volatility_context={k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in vol_ctx.items()},
                volume_context={k: float(v) for k, v in volume_context.items() if isinstance(v, (int, float, np.floating))},
                forward_returns={k: float(v) for k, v in forward_returns.items()},
                embedding=embedding,
                confidence=float(inst.get("confidence", 0.5)),
            )
        )

    # Persist to disk.
    df_out = pd.DataFrame([r.model_dump() for r in records])
    csv_path = out_dir / f"pattern_records_{symbol.replace('/', '_')}.csv"
    df_out.to_csv(csv_path, index=False)
    logger.info("Wrote %d pattern records to %s", len(records), csv_path)

    # Phase 5: store embeddings and labeled metadata in a vector DB backend.
    # The factory will attempt Chroma/FAISS if available, and fall back to a local cosine store.
    vector_db = create_vector_db(base_dir=out_dir / "vector_db", prefer="chroma")
    vector_db.upsert(records)
    logger.info("Vector DB upserted %d pattern records into %s", len(records), str(out_dir / "vector_db"))
    return records

