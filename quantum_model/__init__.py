"""
Quantum-inspired model stack.

This phase provides a hybrid quantum-classical estimator interface.
If PennyLane/PyTorch are unavailable at runtime, the model falls back to a
fully functional classical benchmark while keeping the quantum code paths
available for environments that do support them.
"""

from .inference import load_model, predict_proba, predict_expected_return  # noqa: F401

