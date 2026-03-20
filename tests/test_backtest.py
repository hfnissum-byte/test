from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from backtest.engine import BacktestConfig, backtest_and_compare
from data.storage import ParquetCandleStorage
from utils.validation import load_yaml_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_backtest_baseline_comparison(tmp_path: Path) -> None:
    root = _repo_root()
    cfg = load_yaml_config(root / "config.yaml")
    storage = ParquetCandleStorage(data_dir=root / "data", cfg=cfg)

    pattern_csv = root / "data" / "patterns" / "pattern_records_BTC_USDT.csv"
    quantum_model_path = root / "data" / "models" / "quantum_run" / "quantum_model.joblib"
    classical_model_path = root / "data" / "models" / "classical_run" / "classical_model.joblib"
    vector_db_base_dir = root / "data" / "patterns" / "vector_db"

    if not (pattern_csv.exists() and quantum_model_path.exists() and classical_model_path.exists() and vector_db_base_dir.exists()):
        pytest.skip("Models/pattern DB artifacts not available in this environment.")

    # Continual-learning overwrites `pattern_records_BTC_USDT.csv` based on whatever
    # detection timeframes were last run. Pick an available decision timeframe
    # so the backtest doesn't fail due to missing `4h` rows.
    df = pd.read_csv(pattern_csv)
    available_tfs = set(df["timeframe"].unique().tolist())
    decision_tf = "4h" if "4h" in available_tfs else ("1h" if "1h" in available_tfs else ("15m" if "15m" in available_tfs else sorted(available_tfs)[0]))

    bt_cfg = BacktestConfig(
        symbol="BTC/USDT",
        decision_timeframe=decision_tf,
        execution_timeframe="5m",
        exit_horizon="15m",
        exchange_id=str(cfg.get("exchanges", {}).get("primary", {}).get("id", "binance")),
        eval_start_fraction=0.95,
        max_events=30,
        initial_equity=10_000.0,
        risk_per_trade_pct=0.01,
        max_risk_per_trade_pct=0.02,
        fee_pct=0.001,
        slippage_pct=0.0005,
        atr_period=14,
        atr_mult_sl=1.5,
        atr_mult_tp=2.0,
        min_confidence=0.55,
        p_long_threshold=0.55,
        p_short_threshold=0.55,
        pattern_sim_top_k=20,
        report_base_dir=str(tmp_path / "backtest_results"),
        log_trades=False,
    )

    res = backtest_and_compare(
        storage=storage,
        pattern_csv_path=str(pattern_csv),
        classical_model_path=str(classical_model_path),
        quantum_model_path=str(quantum_model_path),
        vector_db_base_dir=str(vector_db_base_dir),
        cfg=bt_cfg,
    )

    results = res["results"]
    required = {
        "net_return",
        "max_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "win_rate",
        "avg_win",
        "avg_loss",
        "expectancy",
        "profit_factor",
        "trade_count",
    }

    for policy_key in ("quantum", "classical", "pattern-similarity"):
        assert policy_key in results
        assert required.issubset(set(results[policy_key].keys()))

        policy_dir = tmp_path / "backtest_results" / policy_key
        assert (policy_dir / "equity_curve.csv").exists()
        assert (policy_dir / "drawdown_curve.csv").exists()
        assert (policy_dir / "trades.csv").exists()
        assert (policy_dir / "metrics.json").exists()

