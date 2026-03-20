"""
Classical benchmark models.

Phase 7 provides a Transformer+LSTM hybrid sequence model interface.
If torch is unavailable at runtime, a fully functional sklearn fallback is
used so the pipeline remains runnable.
"""

from .inference import load_model, predict_proba, predict_expected_return  # noqa: F401

