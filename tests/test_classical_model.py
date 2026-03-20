from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from classical_model.features import build_sequence_dataset
from classical_model.inference import load_model
from classical_model.trainer import train_classical_model


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_train_and_infer_classical_fallback(tmp_path: Path) -> None:
    pattern_csv = _repo_root() / "data" / "patterns" / "pattern_records_BTC_USDT.csv"
    assert pattern_csv.exists()

    df = pd.read_csv(pattern_csv)
    available_tfs = sorted(df["timeframe"].unique().tolist())
    # Prefer the original default, but fall back if continual-learning overwrote artifacts.
    chosen_tf = "4h" if "4h" in available_tfs else ("1h" if "1h" in available_tfs else available_tfs[0])

    # Restrict to one timeframe to keep runtime bounded.
    metrics = train_classical_model(
        pattern_csv_path=str(pattern_csv),
        out_dir=str(tmp_path),
        symbols=["BTC/USDT"],
        timeframes=[chosen_tf],
        horizon="1h",
        seq_len=6,
        stride=1,
        max_labeled_rows=2000,
        max_windows=2000,
        seed=123,
    )
    assert "val_accuracy" in metrics

    model_path = tmp_path / "classical_model.joblib"
    assert model_path.exists()

    model = load_model(model_path)
    assert model is not None

    dataset = build_sequence_dataset(
        pattern_csv_path=str(pattern_csv),
        symbols=["BTC/USDT"],
        timeframes=[chosen_tf],
        seq_len=6,
        horizon="1h",
        max_labeled_rows=2000,
        seed=321,
        stride=1,
    )

    proba = model.predict_proba(dataset.X_seq)
    assert proba.shape == (len(dataset.X_seq),)
    assert np.all(np.isfinite(proba))
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)

    if dataset.y_return is not None:
        ret_pred = model.predict_expected_return(dataset.X_seq)
        assert ret_pred.shape == (len(dataset.X_seq),)
        assert np.all(np.isfinite(ret_pred))

