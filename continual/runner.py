from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from classical_model.trainer import train_classical_model
from quantum_model.trainer import train_quantum_model
from patterns.intelligence_engine import build_pattern_db
from backtest.engine import BacktestConfig, backtest_and_compare
from data.downloader import CCXTDownloader
from data.storage import ParquetCandleStorage
from utils.validation import ensure_symbol_list
from utils.time_utils import now_ms

logger = logging.getLogger(__name__)

def _maybe_choose_primary_exchange_id(cfg: dict[str, Any]) -> str:
    exchanges_cfg = cfg.get("exchanges", {})
    return str(exchanges_cfg.get("primary", {}).get("id", "binance"))


def _latest_closed_candle_ts(
    *,
    storage: ParquetCandleStorage,
    exchange_ids: Sequence[str],
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> Optional[int]:
    """
    Determine the latest timestamp that likely corresponds to a *closed* candle.

    Storage tracks open timestamps; treat a candle as closed when:
      ts <= now_ms - timeframe_ms
    """
    latest: Optional[int] = None
    now = int(now_ms())

    # Local copy of timeframe parsing (keeps this runner independent).
    def tf_to_ms(tf: str) -> int:
        tf = str(tf).strip().lower()
        if tf.endswith("m"):
            return int(tf[:-1]) * 60 * 1000
        if tf.endswith("h"):
            return int(tf[:-1]) * 60 * 60 * 1000
        if tf.endswith("d"):
            return int(tf[:-1]) * 24 * 60 * 60 * 1000
        raise ValueError(f"Unsupported timeframe: {tf}")

    for symbol in symbols:
        for timeframe in timeframes:
            tf_ms = tf_to_ms(timeframe)
            cutoff = now - tf_ms
            for exchange_id in exchange_ids:
                ts = storage.get_latest_timestamp(exchange=exchange_id, symbol=symbol, timeframe=timeframe)
                if ts is None:
                    continue
                # We store candle "open time". A candle is closed when open_ts + tf_ms <= now.
                # If the latest stored candle is still forming (ts > cutoff), then the most
                # recent closed candle is very likely the previous one: (ts - tf_ms).
                candidate = ts if ts <= cutoff else ts - tf_ms
                if candidate <= cutoff:
                    latest = candidate if latest is None else max(latest, candidate)
    return latest


@dataclass(frozen=True)
class ContinualConfig:
    symbols: Sequence[str]
    timeframes: Sequence[str]

    poll_interval_seconds: float = 300.0
    max_cycles: Optional[int] = None  # for testing; None means infinite

    # Pattern/embedding refresh
    pattern_out_dir: str = "data/patterns"
    curve_window: int = 48
    pattern_detection_timeframes: Optional[Sequence[str]] = None  # defaults to ["1d","4h"] if present

    # Training outputs (deterministic updates)
    quantum_model_out_dir: str = "data/models/quantum_current"
    classical_model_out_dir: str = "data/models/classical_current"

    # Promotion targets (only replaced if challenger beats champion).
    quantum_champion_out_dir: str = "data/models/quantum_champion"
    classical_champion_out_dir: str = "data/models/classical_champion"

    # Backtest outputs (reports overwrite via new run timestamp in dir per cycle if desired)
    run_backtest_each_cycle: bool = True
    backtest_execution_timeframe: str = "5m"
    backtest_exit_horizon: str = "15m"

    # Backtest window + risk knobs
    backtest_eval_start_fraction: float = 0.95
    backtest_max_events: int = 30
    backtest_initial_equity: float = 10_000.0
    backtest_risk_per_trade_pct: float = 0.01
    backtest_max_risk_per_trade_pct: float = 0.02
    backtest_fee_pct: float = 0.001
    backtest_slippage_pct: float = 0.0005
    backtest_atr_period: int = 14
    backtest_atr_mult_sl: float = 1.5
    backtest_atr_mult_tp: float = 2.0
    backtest_min_confidence: float = 0.55
    backtest_p_long_threshold: float = 0.55
    backtest_p_short_threshold: float = 0.55
    backtest_pattern_sim_top_k: int = 50

    # Promotion gate (auditable and deterministic).
    promotion_enabled: bool = True
    promotion_min_trade_count: float = 5.0
    promotion_require_sharpe_ge: bool = True
    promotion_require_net_return_ge: bool = True
    promotion_require_max_drawdown_le: bool = True

    # Determinism
    train_seed: int = 42
    horizon: str = "1h"
    max_train_rows: int = 50_000


async def run_continual_learning(
    *,
    storage: ParquetCandleStorage,
    cfg: dict[str, Any],
    symbols: Sequence[str],
    timeframes: Sequence[str],
    poll_interval_seconds: float,
    max_cycles: Optional[int],
    pattern_out_dir: str,
    quantum_model_out_dir: str,
    classical_model_out_dir: str,
    curve_window: int,
    train_seed: int,
    backtest_each_cycle: bool,
    pattern_detection_timeframes_override: Optional[Sequence[str]] = None,
) -> None:
    """
    Main orchestration loop (REST incremental updates).

    This is the safest always-on mode in environments where threading + websocket
    scheduling could complicate correctness.
    """
    symbols = ensure_symbol_list(symbols)
    timeframes = [str(tf) for tf in timeframes]

    exchanges_cfg = cfg.get("exchanges", {})
    primary_exchange_id = _maybe_choose_primary_exchange_id(cfg)

    # Determine detection timeframes for pattern engine + "learning trigger".
    # Default keeps Phase-1 style cost control: if both 1d and 4h exist, use only those.
    if pattern_detection_timeframes_override:
        pattern_detection_timeframes = [str(tf) for tf in pattern_detection_timeframes_override]
    else:
        decision_tfs: list[str] = ["1d", "4h"]
        tf_set = set(timeframes)
        if "1d" in tf_set and "4h" in tf_set:
            pattern_detection_timeframes = decision_tfs
        else:
            pattern_detection_timeframes = list(timeframes)

    continual_cfg = ContinualConfig(
        symbols=symbols,
        timeframes=timeframes,
        poll_interval_seconds=float(poll_interval_seconds),
        max_cycles=max_cycles,
        pattern_out_dir=pattern_out_dir,
        curve_window=int(curve_window),
        pattern_detection_timeframes=pattern_detection_timeframes,
        quantum_model_out_dir=quantum_model_out_dir,
        classical_model_out_dir=classical_model_out_dir,
        quantum_champion_out_dir=str(Path(quantum_model_out_dir).parent / "quantum_champion"),
        classical_champion_out_dir=str(Path(classical_model_out_dir).parent / "classical_champion"),
        run_backtest_each_cycle=bool(backtest_each_cycle),
        train_seed=int(train_seed),
    )

    # Ensure we always download the execution timeframe needed for Phase 9 backtesting.
    download_timeframes = sorted(set(timeframes) | {continual_cfg.backtest_execution_timeframe})

    # Initial timestamp baseline.
    last_global_ts: Optional[int] = None

    # Exchange candidates used for candle updates (matches downloader fallback logic).
    exchange_ids = [primary_exchange_id, str(exchanges_cfg.get("fallback", {}).get("id", "bybit"))]

    for cycle in range(int(max_cycles) if max_cycles is not None else 10**18):
        start_t = time.time()
        try:
            # 1) Update stored candles incrementally (REST).
            downloader = CCXTDownloader(cfg=cfg, storage=storage)
            await downloader.download_incremental(symbols=symbols, timeframes=download_timeframes)

            # 2) Detect if we received new candles since last cycle.
            latest_ts = _latest_closed_candle_ts(
                storage=storage,
                exchange_ids=exchange_ids,
                symbols=symbols,
                timeframes=list(continual_cfg.pattern_detection_timeframes or timeframes),
            )
            if latest_ts is None:
                logger.info("continual cycle=%d no closed candles found yet; sleeping %.1fs", cycle, continual_cfg.poll_interval_seconds)
                await asyncio.sleep(continual_cfg.poll_interval_seconds)
                continue

            if last_global_ts is not None and latest_ts <= last_global_ts:
                logger.info("continual cycle=%d no new candles; sleeping %.1fs", cycle, continual_cfg.poll_interval_seconds)
                await asyncio.sleep(max(0.1, continual_cfg.poll_interval_seconds - (time.time() - start_t)))
                continue

            logger.info(
                "continual cycle=%d new data detected (prev_ts=%s new_ts=%s); rebuilding patterns + training + backtest",
                cycle,
                last_global_ts,
                latest_ts,
            )

            # 3) Build/update pattern DB + vector store.
            pattern_out_path = Path(continual_cfg.pattern_out_dir)
            logger.info(
                "continual cycle=%d building pattern DB (symbols=%s, trigger_tfs=%s)",
                cycle,
                list(symbols),
                list(continual_cfg.pattern_detection_timeframes or timeframes),
            )
            for sym in symbols:
                logger.info("continual cycle=%d building patterns for %s", cycle, sym)
                build_pattern_db(
                    storage=storage,
                    symbol=sym,
                    detection_timeframes=list(continual_cfg.pattern_detection_timeframes or timeframes),
                    out_dir=pattern_out_path,
                    curve_window=int(continual_cfg.curve_window),
                    primary_exchange_id=primary_exchange_id,
                )

            # 4) Train quantum + classical from the updated pattern records.
            for sym in symbols:
                pattern_csv_path_sym = str(pattern_out_path / f"pattern_records_{sym.replace('/', '_')}.csv")

                # Quantum
                logger.info("continual cycle=%d training quantum model for %s", cycle, sym)
                t_q = time.time()
                train_quantum_model(
                    pattern_csv_path=pattern_csv_path_sym,
                    out_dir=continual_cfg.quantum_model_out_dir,
                    symbols=[sym],
                    timeframes=list(continual_cfg.pattern_detection_timeframes or timeframes),
                    horizon=continual_cfg.horizon,
                    max_rows=continual_cfg.max_train_rows,
                    seed=int(continual_cfg.train_seed),
                )
                logger.info(
                    "continual cycle=%d quantum training done for %s (%.1fs)",
                    cycle,
                    sym,
                    time.time() - t_q,
                )

                # Classical
                logger.info("continual cycle=%d training classical model for %s", cycle, sym)
                t_c = time.time()
                train_classical_model(
                    pattern_csv_path=pattern_csv_path_sym,
                    out_dir=continual_cfg.classical_model_out_dir,
                    symbols=[sym],
                    timeframes=list(continual_cfg.pattern_detection_timeframes or timeframes),
                    horizon=continual_cfg.horizon,
                    seq_len=12,
                    stride=1,
                    max_labeled_rows=25_000,
                    max_windows=None,
                    seed=int(continual_cfg.train_seed),
                )
                logger.info(
                    "continual cycle=%d classical training done for %s (%.1fs)",
                    cycle,
                    sym,
                    time.time() - t_c,
                )

            # 5) Run baseline backtest to measure improvement.
            if continual_cfg.run_backtest_each_cycle:
                for sym in symbols:
                    logger.info("continual cycle=%d running baseline backtest for %s", cycle, sym)
                    candidate_classical_model_path = str(Path(continual_cfg.classical_model_out_dir) / "classical_model.joblib")
                    candidate_quantum_model_path = str(Path(continual_cfg.quantum_model_out_dir) / "quantum_model.joblib")
                    champion_classical_model_path = str(Path(continual_cfg.classical_champion_out_dir) / "classical_model.joblib")
                    champion_quantum_model_path = str(Path(continual_cfg.quantum_champion_out_dir) / "quantum_model.joblib")
                    vector_db_base_dir = str(pattern_out_path / "vector_db")

                    backtest_cfg = BacktestConfig(
                        symbol=sym,
                        decision_timeframe=list(continual_cfg.pattern_detection_timeframes or timeframes)[0],
                        execution_timeframe=continual_cfg.backtest_execution_timeframe,
                        exit_horizon=continual_cfg.backtest_exit_horizon,
                        exchange_id=primary_exchange_id,
                        eval_start_fraction=continual_cfg.backtest_eval_start_fraction,
                        max_events=continual_cfg.backtest_max_events,
                        initial_equity=continual_cfg.backtest_initial_equity,
                        risk_per_trade_pct=continual_cfg.backtest_risk_per_trade_pct,
                        max_risk_per_trade_pct=continual_cfg.backtest_max_risk_per_trade_pct,
                        fee_pct=continual_cfg.backtest_fee_pct,
                        slippage_pct=continual_cfg.backtest_slippage_pct,
                        atr_period=continual_cfg.backtest_atr_period,
                        atr_mult_sl=continual_cfg.backtest_atr_mult_sl,
                        atr_mult_tp=continual_cfg.backtest_atr_mult_tp,
                        min_confidence=continual_cfg.backtest_min_confidence,
                        p_long_threshold=continual_cfg.backtest_p_long_threshold,
                        p_short_threshold=continual_cfg.backtest_p_short_threshold,
                        pattern_sim_top_k=continual_cfg.backtest_pattern_sim_top_k,
                    )

                    def _should_promote(champ: dict[str, Any], cand: dict[str, Any]) -> bool:
                        # Backtest returns trade_count as float; treat as numeric.
                        if float(cand.get("trade_count", 0.0)) < float(continual_cfg.promotion_min_trade_count):
                            return False
                        if continual_cfg.promotion_require_sharpe_ge:
                            if float(cand.get("sharpe", 0.0)) < float(champ.get("sharpe", 0.0)):
                                return False
                        if continual_cfg.promotion_require_net_return_ge:
                            if float(cand.get("net_return", 0.0)) < float(champ.get("net_return", 0.0)):
                                return False
                        if continual_cfg.promotion_require_max_drawdown_le:
                            if float(cand.get("max_drawdown", 1.0)) > float(champ.get("max_drawdown", 0.0)):
                                return False
                        return True

                    # Determine whether we have a champion yet.
                    champion_quantum_exists = Path(champion_quantum_model_path).exists()
                    champion_classical_exists = Path(champion_classical_model_path).exists()

                    cycle_dir = Path("backtest/results/continual") / f"cycle_{cycle:06d}_{sym.replace('/', '_')}"
                    cycle_dir.mkdir(parents=True, exist_ok=True)

                    # 5.1 Evaluate champion on the locked forward window.
                    if not champion_quantum_exists or not champion_classical_exists or not continual_cfg.promotion_enabled:
                        # No champion yet (or promotion disabled): promote candidates immediately.
                        Path(continual_cfg.quantum_champion_out_dir).mkdir(parents=True, exist_ok=True)
                        Path(continual_cfg.classical_champion_out_dir).mkdir(parents=True, exist_ok=True)
                        # Copy model artifacts deterministically if present.
                        for src, dst in [
                            (candidate_quantum_model_path, champion_quantum_model_path),
                            (candidate_classical_model_path, champion_classical_model_path),
                        ]:
                            src_p = Path(src)
                            if not src_p.exists():
                                raise RuntimeError(f"Candidate model missing for promotion: {src}")
                            Path(dst).write_bytes(src_p.read_bytes())

                        logger.info(
                            "continual cycle=%d promoted initial challenger to champion for %s",
                            cycle,
                            sym,
                        )
                        # Run baseline once using the promoted champions (writes audit reports).
                        backtest_cfg.report_base_dir = str(cycle_dir / "champion_initial")
                        backtest_and_compare(
                            storage=storage,
                            pattern_csv_path=str(pattern_out_path / f"pattern_records_{sym.replace('/', '_')}.csv"),
                            classical_model_path=champion_classical_model_path,
                            quantum_model_path=champion_quantum_model_path,
                            vector_db_base_dir=vector_db_base_dir,
                            cfg=backtest_cfg,
                        )
                        logger.info(
                            "continual cycle=%d baseline backtest done for %s (initial promotion)",
                            cycle,
                            sym,
                        )
                        continue

                    # 5.2 Backtest champion and challenger and compare.
                    backtest_cfg.report_base_dir = str(cycle_dir / "champion_eval")
                    res_champ = backtest_and_compare(
                        storage=storage,
                        pattern_csv_path=str(pattern_out_path / f"pattern_records_{sym.replace('/', '_')}.csv"),
                        classical_model_path=champion_classical_model_path,
                        quantum_model_path=champion_quantum_model_path,
                        vector_db_base_dir=vector_db_base_dir,
                        cfg=backtest_cfg,
                    )

                    backtest_cfg.report_base_dir = str(cycle_dir / "challenger_eval")
                    res_cand = backtest_and_compare(
                        storage=storage,
                        pattern_csv_path=str(pattern_out_path / f"pattern_records_{sym.replace('/', '_')}.csv"),
                        classical_model_path=candidate_classical_model_path,
                        quantum_model_path=candidate_quantum_model_path,
                        vector_db_base_dir=vector_db_base_dir,
                        cfg=backtest_cfg,
                    )

                    # Promotion decision per model.
                    champ_q = res_champ["results"]["quantum"]
                    cand_q = res_cand["results"]["quantum"]
                    champ_c = res_champ["results"]["classical"]
                    cand_c = res_cand["results"]["classical"]

                    promote_q = _should_promote(champ_q, cand_q)
                    promote_c = _should_promote(champ_c, cand_c)

                    promotion_out = {
                        "cycle": cycle,
                        "symbol": sym,
                        "promote_quantum": promote_q,
                        "promote_classical": promote_c,
                        "champion_quantum_metrics": champ_q,
                        "challenger_quantum_metrics": cand_q,
                        "champion_classical_metrics": champ_c,
                        "challenger_classical_metrics": cand_c,
                    }
                    (cycle_dir / "promotion.json").write_text(
                        __import__("json").dumps(promotion_out, indent=2), encoding="utf-8"
                    )

                    # Copy promoted artifacts (if any).
                    if promote_q:
                        Path(continual_cfg.quantum_champion_out_dir).mkdir(parents=True, exist_ok=True)
                        Path(champion_quantum_model_path).write_bytes(Path(candidate_quantum_model_path).read_bytes())
                    if promote_c:
                        Path(continual_cfg.classical_champion_out_dir).mkdir(parents=True, exist_ok=True)
                        Path(champion_classical_model_path).write_bytes(Path(candidate_classical_model_path).read_bytes())

                    logger.info(
                        "continual cycle=%d promotion decision for %s: quantum=%s classical=%s",
                        cycle,
                        sym,
                        promote_q,
                        promote_c,
                    )

            last_global_ts = latest_ts

            elapsed = time.time() - start_t
            await asyncio.sleep(max(0.1, continual_cfg.poll_interval_seconds - elapsed))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("continual cycle=%d failed: %s", cycle, e)
            await asyncio.sleep(max(10.0, continual_cfg.poll_interval_seconds))

