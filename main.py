from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Sequence

from data.downloader import CCXTDownloader
from data.live_feed import BinanceLiveFeed
from data.storage import ParquetCandleStorage
from patterns.intelligence_engine import build_pattern_db
from quantum_model.trainer import train_quantum_model
from classical_model.trainer import train_classical_model
from utils.logging_utils import configure_logging
from utils.validation import ensure_symbol_list
from utils.validation import load_yaml_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="quantumprofit-bot")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help='Override symbols (e.g. "BTC/USDT" "XRP/USDT"). Defaults to config.',
    )
    common.add_argument(
        "--timeframes",
        nargs="*",
        default=None,
        help="Override timeframes (e.g. 1d 4h 1h 15m 5m). Defaults to config.",
    )

    p_download = subparsers.add_parser("download-data", parents=[common], help="Download earliest-to-today OHLCV")
    p_download.set_defaults(_run_mode="historical_full")

    p_update = subparsers.add_parser("update-data", parents=[common], help="Incremental OHLCV update; optional live mode")
    p_update.add_argument("--live", action="store_true", help="Also start live ingestion (closed candles only).")
    p_update.set_defaults(_run_mode="historical_incremental")

    p_build_patterns = subparsers.add_parser(
        "build-pattern-db",
        parents=[common],
        help="Detect/label/embed market patterns and write pattern records.",
    )
    p_build_patterns.add_argument("--out", default="data/patterns", help="Output directory for pattern DB artifacts.")
    p_build_patterns.add_argument(
        "--curve-window",
        type=int,
        default=48,
        help="Window size (candles) for curve-aware geometry embeddings.",
    )
    p_build_patterns.set_defaults(_run_mode="build_pattern_db")

    p_train_quantum = subparsers.add_parser(
        "train-quantum",
        parents=[common],
        help="Train quantum-inspired hybrid model from pattern embeddings.",
    )
    p_train_quantum.add_argument(
        "--pattern-csv",
        default="data/patterns/pattern_records_BTC_USDT.csv",
        help="Path to pattern_records CSV produced by `build-pattern-db`.",
    )
    p_train_quantum.add_argument("--out", default="data/models/quantum", help="Output directory for the trained model.")
    p_train_quantum.add_argument("--horizon", default="1h", help="Forward-return horizon to learn (e.g. 1h, 4h, 24h).")
    p_train_quantum.add_argument("--max-rows", type=int, default=50_000, help="Max labeled rows to parse (Good/Bad only).")
    p_train_quantum.add_argument("--seed", type=int, default=42, help="Random seed for sampling/splits.")
    p_train_quantum.set_defaults(_run_mode="train_quantum")

    p_train_classical = subparsers.add_parser(
        "train-classical",
        parents=[common],
        help="Train classical Transformer+LSTM benchmark from pattern embeddings.",
    )
    p_train_classical.add_argument(
        "--pattern-csv",
        default="data/patterns/pattern_records_BTC_USDT.csv",
        help="Path to pattern_records CSV produced by `build-pattern-db`.",
    )
    p_train_classical.add_argument("--out", default="data/models/classical", help="Output directory for the trained model.")
    p_train_classical.add_argument("--horizon", default="1h", help="Forward-return horizon to learn (e.g. 1h, 4h, 24h).")
    p_train_classical.add_argument("--seq-len", type=int, default=12, help="Sliding window length over embedding sequences.")
    p_train_classical.add_argument("--stride", type=int, default=1, help="Stride for sliding windows.")
    p_train_classical.add_argument(
        "--max-labeled-rows",
        type=int,
        default=25_000,
        help="Max Good/Bad pattern rows to parse before windowing.",
    )
    p_train_classical.add_argument("--max-windows", type=int, default=None, help="Optional cap on derived windows.")
    p_train_classical.add_argument("--seed", type=int, default=42, help="Random seed for sampling/splits.")
    p_train_classical.set_defaults(_run_mode="train_classical")

    p_train_ppo = subparsers.add_parser(
        "train-ppo",
        parents=[common],
        help="Train PPO agent on stored OHLCV candles (long-only). Requires gymnasium+stable-baselines3.",
    )
    p_train_ppo.add_argument(
        "--env-timeframe",
        default="4h",
        help="Candle timeframe for the trading environment (e.g. 4h, 1h, 15m, 5m, 1d).",
    )
    p_train_ppo.add_argument("--episode-length", type=int, default=250, help="Steps per episode.")
    p_train_ppo.add_argument("--lookback", type=int, default=30, help="Observation lookback window (bars).")
    p_train_ppo.add_argument("--total-timesteps", type=int, default=50_000, help="PPO training timesteps.")
    p_train_ppo.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_train_ppo.add_argument("--out", default="data/models/ppo", help="Output directory for PPO artifacts.")
    p_train_ppo.set_defaults(_run_mode="train_ppo")

    p_backtest = subparsers.add_parser(
        "backtest",
        parents=[common],
        help="Backtest and compare classical vs quantum vs pattern-similarity policies (PPO optional later).",
    )
    p_backtest.add_argument("--pattern-csv", default=None, help="Pattern records CSV path. Defaults from symbol.")
    p_backtest.add_argument("--vector-db-base", default="data/patterns/vector_db", help="Vector DB base dir (for pattern similarity).")
    p_backtest.add_argument("--classical-model-path", default="data/models/classical_run/classical_model.joblib")
    p_backtest.add_argument("--quantum-model-path", default="data/models/quantum_run/quantum_model.joblib")
    p_backtest.add_argument("--decision-timeframe", default="4h", help="Decision/pattern timeframe (e.g. 4h, 1d).")
    p_backtest.add_argument("--execution-timeframe", default="5m", help="Candle timeframe for trade simulation.")
    p_backtest.add_argument("--exit-horizon", default="1h", help="Trade horizon duration (e.g. 1h, 15m, 4h, 1d).")
    p_backtest.add_argument("--eval-start-fraction", type=float, default=0.7, help="Start eval at timestamp quantile (0..1).")
    p_backtest.add_argument("--max-events", type=int, default=500, help="Max number of traded events in the eval window.")
    p_backtest.add_argument("--min-confidence", type=float, default=0.55)
    p_backtest.add_argument("--p-long-threshold", type=float, default=0.55)
    p_backtest.add_argument("--p-short-threshold", type=float, default=0.55)
    p_backtest.add_argument("--initial-equity", type=float, default=10_000.0)
    p_backtest.add_argument("--risk-per-trade-pct", type=float, default=0.01)
    p_backtest.add_argument("--max-risk-per-trade-pct", type=float, default=0.02)
    p_backtest.add_argument("--fee-pct", type=float, default=0.001)
    p_backtest.add_argument("--slippage-pct", type=float, default=0.0005)
    p_backtest.add_argument("--atr-period", type=int, default=14)
    p_backtest.add_argument("--atr-mult-sl", type=float, default=1.5)
    p_backtest.add_argument("--atr-mult-tp", type=float, default=2.0)
    p_backtest.add_argument("--pattern-sim-top-k", type=int, default=50)
    p_backtest.set_defaults(_run_mode="backtest")

    p_backtest_rank = subparsers.add_parser(
        "backtest-rank",
        parents=[common],
        help="Rank previously run backtests by Sharpe and net return (no PPO baseline).",
    )
    p_backtest_rank.add_argument("--base-dir", default="backtest/results", help="Backtest results base directory.")
    p_backtest_rank.set_defaults(_run_mode="backtest_rank")

    p_continual = subparsers.add_parser(
        "continual-learning",
        parents=[common],
        help="Always-on loop: incremental data update -> rebuild patterns -> retrain -> baseline backtest.",
    )
    p_continual.add_argument("--poll-interval-seconds", type=float, default=300.0, help="Loop sleep between poll cycles.")
    p_continual.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Stop after N cycles (0 means infinite). Useful for smoke tests.",
    )
    p_continual.add_argument("--pattern-out", default="data/patterns", help="Output dir for pattern DB artifacts.")
    p_continual.add_argument("--curve-window", type=int, default=48, help="Curve window size for embeddings.")
    p_continual.add_argument("--quantum-model-out", default="data/models/quantum_current", help="Quantum model output dir.")
    p_continual.add_argument("--classical-model-out", default="data/models/classical_current", help="Classical model output dir.")
    p_continual.add_argument("--train-seed", type=int, default=42, help="Deterministic training seed.")
    p_continual.add_argument("--no-backtest", action="store_true", help="Disable baseline backtest each time patterns update.")
    p_continual.add_argument(
        "--pattern-detection-timeframes",
        nargs="*",
        default=None,
        help="Override which timeframes trigger pattern rebuild/training (e.g. 1h 15m 5m). Defaults to 1d/4h when both exist.",
    )
    p_continual.set_defaults(_run_mode="continual_learning")

    return parser.parse_args()


def _get_run_params(cfg: dict, args: argparse.Namespace) -> tuple[list[str], list[str]]:
    symbols = cfg.get("symbols", [])
    timeframes = cfg.get("timeframes", [])

    if args.symbols is not None:
        symbols = args.symbols
    if args.timeframes is not None:
        timeframes = args.timeframes

    symbols = ensure_symbol_list(symbols)
    timeframes = [str(tf) for tf in timeframes]
    return symbols, timeframes


async def _run_download(cfg: dict, storage: ParquetCandleStorage, symbols: Sequence[str], timeframes: Sequence[str]) -> None:
    downloader = CCXTDownloader(cfg=cfg, storage=storage)
    await downloader.download_earliest_to_today(symbols=symbols, timeframes=timeframes)


async def _run_update(
    cfg: dict,
    storage: ParquetCandleStorage,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    live: bool,
) -> None:
    downloader = CCXTDownloader(cfg=cfg, storage=storage)
    await downloader.download_incremental(symbols=symbols, timeframes=timeframes)

    if not live:
        return

    live_feed = BinanceLiveFeed(cfg=cfg, storage=storage, symbols=symbols, timeframes=timeframes)
    await live_feed.run_forever()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    cfg = load_yaml_config(config_path)

    configure_logging(level=str(cfg.get("app", {}).get("log_level", "INFO")).upper())
    logging.getLogger(__name__).info("Loaded config: %s", config_path)

    symbols, timeframes = _get_run_params(cfg, args)
    data_dir = Path(cfg.get("data_dir", "./data"))
    storage = ParquetCandleStorage(data_dir=data_dir, cfg=cfg)

    if args._run_mode == "historical_full":
        asyncio.run(_run_download(cfg, storage, symbols, timeframes))
    elif args._run_mode == "historical_incremental":
        asyncio.run(_run_update(cfg, storage, symbols, timeframes, live=args.live))
    elif args._run_mode == "build_pattern_db":
        # For swing mode in Phase 4, detection uses only the provided timeframes (defaulting to config timeframes).
        out_dir = Path(args.out)
        # Swing mode primary decision frames are 1d and 4h.
        detection_timeframes = ["1d", "4h"] if "1d" in timeframes and "4h" in timeframes else timeframes
        exchanges_cfg = cfg.get("exchanges", {})
        primary_exchange_id = str(exchanges_cfg.get("primary", {}).get("id", "binance"))
        for sym in symbols:
            build_pattern_db(
                storage=storage,
                symbol=sym,
                detection_timeframes=detection_timeframes,
                out_dir=out_dir,
                curve_window=int(args.curve_window),
                primary_exchange_id=primary_exchange_id,
            )
    elif args._run_mode == "train_quantum":
        metrics = train_quantum_model(
            pattern_csv_path=args.pattern_csv,
            out_dir=args.out,
            symbols=symbols,
            timeframes=timeframes,
            horizon=str(args.horizon),
            max_rows=int(args.max_rows),
            seed=int(args.seed),
        )
        logging.getLogger(__name__).info("Quantum model training completed: %s", metrics)
    elif args._run_mode == "train_classical":
        metrics = train_classical_model(
            pattern_csv_path=args.pattern_csv,
            out_dir=args.out,
            symbols=symbols,
            timeframes=timeframes,
            horizon=str(args.horizon),
            seq_len=int(args.seq_len),
            stride=int(args.stride),
            max_labeled_rows=int(args.max_labeled_rows),
            max_windows=args.max_windows if args.max_windows is not None else None,
            seed=int(args.seed),
        )
        logging.getLogger(__name__).info("Classical model training completed: %s", metrics)
    elif args._run_mode == "train_ppo":
        # Lazy import so CLI/help doesn't hard-fail when optional RL deps are missing.
        import importlib.util

        missing: list[str] = []
        if importlib.util.find_spec("torch") is None:
            missing.append("torch")
        if importlib.util.find_spec("gymnasium") is None:
            missing.append("gymnasium")
        if importlib.util.find_spec("stable_baselines3") is None:
            missing.append("stable-baselines3")

        if missing:
            raise RuntimeError(
                "Phase 8 PPO requires optional dependencies that are not installed in this Python runtime: "
                f"{', '.join(missing)}. "
                "Install them, then retry `train-ppo`."
            )

        try:
            from rl_agent.trainer import train_ppo_from_storage
        except Exception as e:
            raise RuntimeError(f"Failed to import RL trainer. Ensure Phase 8 deps are installed. Error: {e}") from e

        exchanges_cfg = cfg.get("exchanges", {})
        primary_id = str(exchanges_cfg.get("primary", {}).get("id", "binance"))
        fallback_id = str(exchanges_cfg.get("fallback", {}).get("id", "bybit"))
        exchange_candidates = [primary_id, fallback_id]

        for sym in symbols:
            train_ppo_from_storage(
                storage=storage,
                exchange_candidates=exchange_candidates,
                symbol=sym,
                timeframe=str(args.env_timeframe),
                out_dir=args.out,
                episode_length=int(args.episode_length),
                lookback=int(args.lookback),
                total_timesteps=int(args.total_timesteps),
                seed=int(args.seed),
            )
    elif args._run_mode == "backtest":
        from backtest.engine import BacktestConfig, backtest_and_compare

        # Baseline comparison: classical + quantum + pattern-similarity. PPO is excluded by design.
        exchanges_cfg = cfg.get("exchanges", {})
        exchange_id = str(exchanges_cfg.get("primary", {}).get("id", "binance"))

        pattern_csv_default = args.pattern_csv
        results_by_symbol: dict[str, Any] = {}
        for sym in symbols:
            if pattern_csv_default is not None:
                pattern_csv_path = str(pattern_csv_default)
            else:
                pattern_csv_path = str(Path("data/patterns") / f"pattern_records_{sym.replace('/', '_')}.csv")

            bt_cfg = BacktestConfig(
                symbol=sym,
                decision_timeframe=str(args.decision_timeframe),
                execution_timeframe=str(args.execution_timeframe),
                exit_horizon=str(args.exit_horizon),
                exchange_id=exchange_id,
                eval_start_fraction=float(args.eval_start_fraction),
                max_events=int(args.max_events) if args.max_events > 0 else None,
                initial_equity=float(args.initial_equity),
                risk_per_trade_pct=float(args.risk_per_trade_pct),
                max_risk_per_trade_pct=float(args.max_risk_per_trade_pct),
                fee_pct=float(args.fee_pct),
                slippage_pct=float(args.slippage_pct),
                atr_period=int(args.atr_period),
                atr_mult_sl=float(args.atr_mult_sl),
                atr_mult_tp=float(args.atr_mult_tp),
                min_confidence=float(args.min_confidence),
                p_long_threshold=float(args.p_long_threshold),
                p_short_threshold=float(args.p_short_threshold),
                pattern_sim_top_k=int(args.pattern_sim_top_k),
            )
            results_by_symbol[sym] = backtest_and_compare(
                storage=storage,
                pattern_csv_path=pattern_csv_path,
                classical_model_path=str(args.classical_model_path),
                quantum_model_path=str(args.quantum_model_path),
                vector_db_base_dir=str(args.vector_db_base),
                cfg=bt_cfg,
            )

        logging.getLogger(__name__).info("Backtest comparison complete: %s", results_by_symbol)
    elif args._run_mode == "backtest_rank":
        from backtest.compare import rank_backtest_policies

        rank_backtest_policies(report_base_dir=args.base_dir)
    elif args._run_mode == "continual_learning":
        from continual.runner import run_continual_learning

        max_cycles = None if int(args.max_cycles) <= 0 else int(args.max_cycles)
        asyncio.run(
            run_continual_learning(
                storage=storage,
                cfg=cfg,
                symbols=symbols,
                timeframes=timeframes,
                poll_interval_seconds=float(args.poll_interval_seconds),
                max_cycles=max_cycles,
                pattern_out_dir=str(args.pattern_out),
                quantum_model_out_dir=str(args.quantum_model_out),
                classical_model_out_dir=str(args.classical_model_out),
                curve_window=int(args.curve_window),
                train_seed=int(args.train_seed),
                backtest_each_cycle=not bool(args.no_backtest),
                pattern_detection_timeframes_override=args.pattern_detection_timeframes,
            )
        )
    else:
        raise ValueError(f"Unknown run mode: {args._run_mode}")


if __name__ == "__main__":
    main()

