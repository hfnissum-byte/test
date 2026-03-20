from __future__ import annotations

"""
Quantum neural network components (PennyLane + PyTorch).

This module is intentionally dependency-optional: the rest of the project can
train and run inference using the classical fallback estimator when PyTorch or
PennyLane are not installed in the runtime.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import importlib.util


BackendName = Literal["quantum", "classical_fallback"]


def _is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def quantum_backend_available() -> bool:
    # Torch is required for TorchLayer integration.
    return _is_available("torch") and _is_available("pennylane")


@dataclass(frozen=True)
class QuantumCircuitConfig:
    embedding_dim: int
    n_qubits: int = 8
    n_layers: int = 2
    rotation: str = "Y"


def create_quantum_frontend_torch(cfg: QuantumCircuitConfig):
    """
    Create a torch.nn.Module that:
      embedding_dim -> n_qubits angle projection -> quantum circuit expectation values.

    Returns a torch module only if both `torch` and `pennylane` are importable.
    """
    if not quantum_backend_available():
        raise RuntimeError(
            "Quantum backend requested but dependencies are missing. "
            "Install `torch` and `pennylane` to enable the quantum path."
        )

    # Local imports so importing this module never hard-fails.
    import torch  # type: ignore
    import pennylane as qml  # type: ignore

    dev = qml.device("default.qubit", wires=cfg.n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=list(range(cfg.n_qubits)), rotation=cfg.rotation)
        qml.StronglyEntanglingLayers(weights, wires=list(range(cfg.n_qubits)))
        # Return one expectation value per qubit.
        return [qml.expval(qml.PauliZ(i)) for i in range(cfg.n_qubits)]

    # For StronglyEntanglingLayers, weights shape is (n_layers, n_wires, 3).
    weight_shapes = {"weights": (cfg.n_layers, cfg.n_qubits, 3)}
    q_layer = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)

    class QuantumFrontend(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.angle_proj = torch.nn.Linear(cfg.embedding_dim, cfg.n_qubits)
            self.q_layer = q_layer

        def forward(self, x):
            angles = self.angle_proj(x)
            return self.q_layer(angles)

    return QuantumFrontend()


def backend_name() -> BackendName:
    return "quantum" if quantum_backend_available() else "classical_fallback"


