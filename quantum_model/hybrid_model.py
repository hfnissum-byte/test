from __future__ import annotations

import joblib
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .qnn import QuantumCircuitConfig, backend_name, create_quantum_frontend_torch


@dataclass(frozen=True)
class HybridModelConfig:
    embedding_dim: int = 64
    # Quantum-ish parameters (only used if torch + pennylane are available).
    n_qubits: int = 8
    n_layers: int = 2
    rotation: str = "Y"

    # Classical fallback training parameters.
    direction_max_iter: int = 2000
    direction_solver: str = "lbfgs"
    return_reg_alpha: float = 1.0

    # Shared training params (used by the torch path).
    epochs: int = 25
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42


class HybridQuantumInspiredEstimator:
    """
    Direction classifier (Good=1, Bad=0) and optional expected return regressor.

    If torch + pennylane are present, trains a hybrid quantum-classical torch model.
    Otherwise it falls back to sklearn models (fully functional).
    """

    def __init__(self, cfg: HybridModelConfig) -> None:
        self.cfg = cfg
        self.backend = backend_name()

        # We always normalize inputs to stabilize training/inference.
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()

        self._direction_clf: Any = None
        self._return_reg: Any = None

        # Torch objects are created lazily only when available.
        self._torch_model: Any = None
        self._torch_dir_head: Any = None
        self._torch_ret_head: Any = None

        if self.backend == "quantum":
            self._init_torch_model()
        else:
            self._init_sklearn_models()

    def _init_sklearn_models(self) -> None:
        from sklearn.linear_model import LogisticRegression, Ridge

        self._direction_clf = LogisticRegression(
            max_iter=self.cfg.direction_max_iter,
            solver=self.cfg.direction_solver,
        )
        self._return_reg = Ridge(alpha=self.cfg.return_reg_alpha)

    def _init_torch_model(self) -> None:
        # Local import to avoid hard dependency on torch at module import time.
        import torch  # type: ignore

        q_cfg = QuantumCircuitConfig(
            embedding_dim=self.cfg.embedding_dim,
            n_qubits=self.cfg.n_qubits,
            n_layers=self.cfg.n_layers,
            rotation=self.cfg.rotation,
        )
        quantum_frontend = create_quantum_frontend_torch(q_cfg)

        class TorchHybridModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quantum_frontend = quantum_frontend
                self.mlp = torch.nn.Sequential(
                    # QuantumFrontend returns one expectation value per qubit.
                    torch.nn.Linear(cfg.n_qubits, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 8),
                    torch.nn.ReLU(),
                )
                self.dir_head = torch.nn.Linear(8, 1)  # logit
                self.ret_head = torch.nn.Linear(8, 1)  # regression

            def forward(self, x):
                q_feat = self.quantum_frontend(x)
                h = self.mlp(q_feat)
                dir_logit = self.dir_head(h).squeeze(-1)
                ret = self.ret_head(h).squeeze(-1)
                return dir_logit, ret

        self._torch_model = TorchHybridModel()

    def fit(
        self,
        X: np.ndarray,
        y_dir: np.ndarray,
        y_return: Optional[np.ndarray] = None,
        val_fraction: float = 0.2,
        *,
        X_val: Optional[np.ndarray] = None,
        y_dir_val: Optional[np.ndarray] = None,
        y_return_val: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        if X.ndim != 2:
            raise ValueError("X must be 2D [n_samples, embedding_dim]")
        if len(X) != len(y_dir):
            raise ValueError("X and y_dir length mismatch")

        Xs = self.scaler.fit_transform(X).astype(np.float32, copy=False)

        if self.backend == "classical_fallback":
            self._direction_clf.fit(Xs, y_dir)
            metrics: dict[str, float] = {}

            if y_return is not None:
                if len(y_return) != len(X):
                    raise ValueError("y_return length mismatch")
                self._return_reg.fit(Xs, y_return)

                metrics["return_train_r2"] = float(self._return_reg.score(Xs, y_return))
            return metrics

        # Torch path
        if X_val is None or y_dir_val is None:
            # Fallback for callers that don't supply explicit validation data.
            # This is not time-correct and exists only for compatibility.
            metrics = self._fit_torch_random_split(Xs, y_dir, y_return=y_return, val_fraction=val_fraction)
            return metrics

        metrics = self._fit_torch_explicit_val(
            X_train=Xs,
            y_dir_train=y_dir,
            y_return_train=y_return,
            X_val=X_val,
            y_dir_val=y_dir_val,
            y_return_val=y_return_val,
        )
        return metrics

    def _fit_torch_explicit_val(
        self,
        *,
        X_train: np.ndarray,
        y_dir_train: np.ndarray,
        y_return_train: Optional[np.ndarray],
        X_val: np.ndarray,
        y_dir_val: np.ndarray,
        y_return_val: Optional[np.ndarray],
    ) -> dict[str, float]:
        import torch  # type: ignore
        from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model: torch.nn.Module = self._torch_model.to(device)

        opt = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        bce = torch.nn.BCEWithLogitsLoss()
        mse = torch.nn.MSELoss()

        # X_train is already scaled (Xs).
        X_train_t = torch.from_numpy(X_train).to(device)
        y_dir_train_t = torch.from_numpy(y_dir_train.astype(np.float32)).to(device)

        # Val arrays must be scaled using the already-fitted scaler.
        X_val_s = self.scaler.transform(X_val).astype(np.float32, copy=False)
        X_val_t = torch.from_numpy(X_val_s).to(device)
        y_dir_val_t = torch.from_numpy(y_dir_val.astype(np.float32)).to(device)

        y_ret_train_t = None
        y_ret_val_t = None
        if y_return_train is not None:
            y_ret_train_t = torch.from_numpy(y_return_train.astype(np.float32)).to(device)
        if y_return_val is not None:
            y_ret_val_t = torch.from_numpy(y_return_val.astype(np.float32)).to(device)

        n = X_train_t.shape[0]
        for _epoch in range(self.cfg.epochs):
            # Simple mini-batch loop (no dependency on DataLoader).
            perm = torch.randperm(n, device=device)
            for start in range(0, n, self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]
                xb = X_train_t[idx]
                yb_dir = y_dir_train_t[idx]

                opt.zero_grad(set_to_none=True)
                dir_logit, ret_pred = model(xb)
                loss_dir = bce(dir_logit, yb_dir)
                loss = loss_dir
                if y_ret_train_t is not None:
                    yb_ret = y_ret_train_t[idx]
                    loss_ret = mse(ret_pred, yb_ret)
                    loss = loss_dir + loss_ret
                loss.backward()
                opt.step()

        # Validation metrics
        model.eval()
        with torch.no_grad():
            dir_logit_val, ret_pred_val = model(X_val_t)
            prob_val = torch.sigmoid(dir_logit_val).detach().cpu().numpy()

        y_pred_val = (prob_val >= 0.5).astype(int)
        acc = float(accuracy_score(y_dir_val, y_pred_val))
        auc = float(roc_auc_score(y_dir_val, prob_val))
        metrics: dict[str, float] = {"val_accuracy": acc}
        # ROC-AUC requires both classes.
        if len(np.unique(y_dir_val)) == 2:
            metrics["val_roc_auc"] = auc

        if y_ret_val_t is not None:
            y_ret_true = y_ret_val_t.detach().cpu().numpy()
            y_ret_pred = ret_pred_val.detach().cpu().numpy()
            y_ret_mae = mean_absolute_error(y_ret_true, y_ret_pred)
            metrics["val_return_mae"] = float(y_ret_mae)

        return metrics

    def _fit_torch_random_split(
        self,
        Xs: np.ndarray,
        y_dir: np.ndarray,
        y_return: Optional[np.ndarray],
        val_fraction: float,
    ) -> dict[str, float]:
        """
        Compatibility fallback that performs a random validation split.

        For time-correct training/validation, callers should pass explicit
        X_val/y_dir_val/y_return_val and use `_fit_torch_explicit_val`.
        """
        import torch  # type: ignore
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

        X_train, X_val, y_dir_train, y_dir_val = train_test_split(
            Xs, y_dir, test_size=val_fraction, random_state=self.cfg.seed, stratify=y_dir
        )
        if y_return is not None:
            _, _, y_ret_train, y_ret_val = train_test_split(
                Xs, y_return, test_size=val_fraction, random_state=self.cfg.seed, stratify=y_dir
            )
        else:
            y_ret_train = None
            y_ret_val = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model: torch.nn.Module = self._torch_model.to(device)

        opt = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        bce = torch.nn.BCEWithLogitsLoss()
        mse = torch.nn.MSELoss()

        X_train_t = torch.from_numpy(X_train).to(device)
        y_dir_train_t = torch.from_numpy(y_dir_train.astype(np.float32)).to(device)

        # X_val comes from already-scaled Xs, but keep consistent typing.
        X_val_t = torch.from_numpy(X_val).to(device)
        y_dir_val_t = torch.from_numpy(y_dir_val.astype(np.float32)).to(device)

        y_ret_train_t = None
        y_ret_val_t = None
        if y_ret_train is not None:
            y_ret_train_t = torch.from_numpy(y_ret_train.astype(np.float32)).to(device)
        if y_ret_val is not None:
            y_ret_val_t = torch.from_numpy(y_ret_val.astype(np.float32)).to(device)

        n = X_train_t.shape[0]
        for _epoch in range(self.cfg.epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]
                xb = X_train_t[idx]
                yb_dir = y_dir_train_t[idx]

                opt.zero_grad(set_to_none=True)
                dir_logit, ret_pred = model(xb)
                loss_dir = bce(dir_logit, yb_dir)
                loss = loss_dir
                if y_ret_train_t is not None:
                    yb_ret = y_ret_train_t[idx]
                    loss_ret = mse(ret_pred, yb_ret)
                    loss = loss_dir + loss_ret
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            dir_logit_val, ret_pred_val = model(X_val_t)
            prob_val = torch.sigmoid(dir_logit_val).detach().cpu().numpy()

        y_pred_val = (prob_val >= 0.5).astype(int)
        acc = float(accuracy_score(y_dir_val, y_pred_val))
        metrics: dict[str, float] = {"val_accuracy": acc}

        if len(np.unique(y_dir_val)) == 2:
            metrics["val_roc_auc"] = float(roc_auc_score(y_dir_val, prob_val))

        if y_ret_val_t is not None:
            y_ret_true = y_ret_val_t.detach().cpu().numpy()
            y_ret_pred = ret_pred_val.detach().cpu().numpy()
            metrics["val_return_mae"] = float(mean_absolute_error(y_ret_true, y_ret_pred))

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns P(direction=1) with shape [n_samples].
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D [n_samples, embedding_dim]")
        Xs = self.scaler.transform(X).astype(np.float32, copy=False)

        if self.backend == "classical_fallback":
            if self._direction_clf is None:
                raise RuntimeError("Model is not trained")
            probs = self._direction_clf.predict_proba(Xs)[:, 1]
            return probs.astype(np.float32)

        import torch  # type: ignore

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model: torch.nn.Module = self._torch_model.to(device)
        model.eval()
        X_t = torch.from_numpy(Xs).to(device)
        with torch.no_grad():
            dir_logit, _ = model(X_t)
            probs = torch.sigmoid(dir_logit).detach().cpu().numpy()
        return probs.astype(np.float32)

    def predict_expected_return(self, X: np.ndarray) -> np.ndarray:
        """
        Returns expected return regression with shape [n_samples].
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D [n_samples, embedding_dim]")
        Xs = self.scaler.transform(X).astype(np.float32, copy=False)

        if self.backend == "classical_fallback":
            if self._return_reg is None:
                raise RuntimeError("Model has no return regressor (train with y_return).")
            preds = self._return_reg.predict(Xs)
            return preds.astype(np.float32)

        import torch  # type: ignore

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model: torch.nn.Module = self._torch_model.to(device)
        model.eval()
        X_t = torch.from_numpy(Xs).to(device)
        with torch.no_grad():
            _, ret = model(X_t)
            preds = ret.detach().cpu().numpy()
        return preds.astype(np.float32)

    def save(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "cfg": self.cfg.__dict__,
            "backend": self.backend,
            "scaler": self.scaler,
        }

        if self.backend == "classical_fallback":
            payload["direction_clf"] = self._direction_clf
            payload["return_reg"] = self._return_reg
        else:
            # Store torch model weights; requires torch + pennylane on load.
            payload["torch_state_dict"] = self._torch_model.state_dict()

        joblib.dump(payload, out_path)

    @classmethod
    def load(cls, model_path: Path) -> "HybridQuantumInspiredEstimator":
        payload = joblib.load(model_path)
        cfg = HybridModelConfig(**payload["cfg"])
        est = cls(cfg=cfg)
        est.backend = payload["backend"]  # type: ignore[assignment]
        est.scaler = payload["scaler"]

        if est.backend == "classical_fallback":
            est._direction_clf = payload["direction_clf"]
            est._return_reg = payload["return_reg"]
        else:
            if not backend_name() == "quantum":
                raise RuntimeError(
                    "Loaded a quantum model but torch/pennylane are not available "
                    "in this runtime."
                )
            # Recreate torch model and load weights.
            est._init_torch_model()
            est._torch_model.load_state_dict(payload["torch_state_dict"])
        return est

