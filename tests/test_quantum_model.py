from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantum_model.dataset import build_training_arrays
from quantum_model.inference import load_model
from quantum_model.trainer import train_quantum_model


def _repo_root() -> Path:
    # tests/ -> quantumprofit-bot/
    return Path(__file__).resolve().parents[1]


def test_train_and_infer_quantum_fallback(tmp_path: Path) -> None:
    pattern_csv = _repo_root() / "data" / "patterns" / "pattern_records_BTC_USDT.csv"
    assert pattern_csv.exists()

    metrics = train_quantum_model(
        pattern_csv_path=str(pattern_csv),
        out_dir=str(tmp_path),
        symbols=["BTC/USDT"],
        timeframes=None,
        horizon="1h",
        max_rows=2000,
        seed=123,
    )
    assert "val_accuracy" in metrics

    model_path = tmp_path / "quantum_model.joblib"
    assert model_path.exists()

    model = load_model(model_path)
    assert model is not None

    arrays = build_training_arrays(
        pattern_csv_path=str(pattern_csv),
        symbols=["BTC/USDT"],
        timeframes=None,
        horizon="1h",
        max_rows=200,
        seed=321,
        embedding_dim=64,
    )

    proba = model.predict_proba(arrays.X)
    assert proba.shape == (len(arrays.X),)
    assert np.all(np.isfinite(proba))
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)

    # Return predictions should also be runnable when available.
    if arrays.y_return is not None:
        ret_pred = model.predict_expected_return(arrays.X)
        assert ret_pred.shape == (len(arrays.X),)
        assert np.all(np.isfinite(ret_pred))

