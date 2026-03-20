from __future__ import annotations

import joblib
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ClassicalSequenceModelConfig:
    embedding_dim: int = 64
    seq_len: int = 12

    # Torch path parameters (only used if torch is available).
    model_dim: int = 128
    n_heads: int = 4
    transformer_layers: int = 2
    lstm_hidden: int = 64
    lstm_layers: int = 1
    dropout: float = 0.1

    # Fallback (sklearn) parameters.
    direction_max_iter: int = 2000
    return_reg_alpha: float = 1.0

    seed: int = 42


def _torch_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False


class ClassicalSequenceEstimator:
    """
    Transformer+LSTM hybrid estimator interface.

    If `torch` isn't available in the runtime, this falls back to a fully
    functional sklearn model that uses flattened embedding-window features.
    """

    def __init__(self, cfg: ClassicalSequenceModelConfig) -> None:
        self.cfg = cfg
        self.backend = "torch" if _torch_available() else "sklearn_fallback"

        self._scaler: Any = None
        self._dir_clf: Any = None
        self._ret_reg: Any = None

        # Torch objects (lazy).
        self._torch_model: Any = None

        if self.backend == "sklearn_fallback":
            self._init_sklearn()
        else:
            self._init_torch()

    def _init_sklearn(self) -> None:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        self._dir_clf = LogisticRegression(
            max_iter=self.cfg.direction_max_iter,
            class_weight="balanced",
            random_state=self.cfg.seed,
        )
        self._ret_reg = Ridge(alpha=self.cfg.return_reg_alpha)

    def _init_torch(self) -> None:
        # Optional full implementation.
        import torch  # type: ignore

        cfg = self.cfg

        class PositionalEncoding(torch.nn.Module):
            def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
                super().__init__()
                self.dropout = torch.nn.Dropout(p=dropout)

                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer("pe", pe)

            def forward(self, x):
                # x: [batch, seq, d_model]
                x = x + self.pe[:, : x.size(1), :]
                return self.dropout(x)

        class TorchTransformerLSTM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_proj = torch.nn.Linear(cfg.embedding_dim, cfg.model_dim)
                self.pos = PositionalEncoding(cfg.model_dim, dropout=cfg.dropout, max_len=cfg.seq_len + 5)

                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=cfg.model_dim,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.model_dim * 2,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
                self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)

                self.lstm = torch.nn.LSTM(
                    input_size=cfg.model_dim,
                    hidden_size=cfg.lstm_hidden,
                    num_layers=cfg.lstm_layers,
                    batch_first=True,
                    bidirectional=False,
                )
                self.dir_head = torch.nn.Linear(cfg.lstm_hidden, 1)
                self.ret_head = torch.nn.Linear(cfg.lstm_hidden, 1)

            def forward(self, x_seq):
                # x_seq: [batch, seq, embedding_dim]
                x = self.input_proj(x_seq)
                x = self.pos(x)
                x = self.transformer(x)
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                dir_logit = self.dir_head(last).squeeze(-1)
                ret = self.ret_head(last).squeeze(-1)
                return dir_logit, ret

        self._torch_model = TorchTransformerLSTM()

    def _flatten(self, X_seq: np.ndarray) -> np.ndarray:
        if X_seq.ndim != 3:
            raise ValueError("X_seq must be [n_samples, seq_len, embedding_dim]")
        n, seq_len, emb_dim = X_seq.shape
        if seq_len != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got {seq_len}")
        if emb_dim != self.cfg.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.cfg.embedding_dim}, got {emb_dim}")
        return X_seq.reshape(n, seq_len * emb_dim)

    def fit(
        self,
        X_seq: np.ndarray,
        y_dir: np.ndarray,
        y_return: Optional[np.ndarray] = None,
        val_fraction: float = 0.2,
        *,
        X_val_seq: Optional[np.ndarray] = None,
        y_dir_val: Optional[np.ndarray] = None,
        y_return_val: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        if self.backend == "sklearn_fallback":
            return self._fit_sklearn(
                X_seq,
                y_dir,
                y_return=y_return,
                X_val_seq=X_val_seq,
                y_dir_val=y_dir_val,
                y_return_val=y_return_val,
            )

        # Torch training loop (not used in this runtime unless torch exists).
        return self._fit_torch(
            X_seq,
            y_dir,
            y_return=y_return,
            val_fraction=val_fraction,
            X_val_seq=X_val_seq,
            y_dir_val=y_dir_val,
            y_return_val=y_return_val,
        )

    def _fit_sklearn(
        self,
        X_seq: np.ndarray,
        y_dir: np.ndarray,
        y_return: Optional[np.ndarray],
        *,
        X_val_seq: Optional[np.ndarray] = None,
        y_dir_val: Optional[np.ndarray] = None,
        y_return_val: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

        X_feat = self._flatten(X_seq).astype(np.float32, copy=False)

        if X_val_seq is not None and y_dir_val is not None:
            X_val_feat = self._flatten(X_val_seq).astype(np.float32, copy=False)
            X_train_s = self._scaler.fit_transform(X_feat)
            X_val_s = self._scaler.transform(X_val_feat)

            self._dir_clf.fit(X_train_s, y_dir)
            proba_val = self._dir_clf.predict_proba(X_val_s)[:, 1]
            y_val = y_dir_val

            y_pred_val = (proba_val >= 0.5).astype(int)
            metrics: dict[str, float] = {"val_accuracy": float(accuracy_score(y_val, y_pred_val))}
            if len(np.unique(y_val)) == 2:
                metrics["val_roc_auc"] = float(roc_auc_score(y_val, proba_val))

            if y_return is not None and y_return_val is not None:
                if len(y_return) != len(y_dir):
                    raise ValueError("y_return length mismatch")
                if len(y_return_val) != len(y_val):
                    raise ValueError("y_return_val length mismatch")
                self._ret_reg.fit(X_train_s, y_return)
                y_ret_pred = self._ret_reg.predict(X_val_s)
                metrics["val_return_mae"] = float(mean_absolute_error(y_return_val, y_ret_pred))
            return metrics

        # Compatibility fallback: random split + random val.
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X_feat, y_dir, test_size=0.2, random_state=self.cfg.seed, stratify=y_dir
        )

        X_train_s = self._scaler.fit_transform(X_train)
        X_val_s = self._scaler.transform(X_val)

        self._dir_clf.fit(X_train_s, y_train)
        proba_val = self._dir_clf.predict_proba(X_val_s)[:, 1]
        y_pred_val = (proba_val >= 0.5).astype(int)

        metrics: dict[str, float] = {"val_accuracy": float(accuracy_score(y_val, y_pred_val))}
        if len(np.unique(y_val)) == 2:
            metrics["val_roc_auc"] = float(roc_auc_score(y_val, proba_val))

        if y_return is not None:
            if len(y_return) != len(y_dir):
                raise ValueError("y_return length mismatch")
            y_train_ret, y_val_ret = train_test_split(
                y_return,
                test_size=0.2,
                random_state=self.cfg.seed,
                stratify=y_dir,
            )
            self._ret_reg.fit(X_train_s, y_train_ret)
            y_ret_pred = self._ret_reg.predict(X_val_s)
            metrics["val_return_mae"] = float(mean_absolute_error(y_val_ret, y_ret_pred))

        return metrics

    def _fit_torch(
        self,
        X_seq: np.ndarray,
        y_dir: np.ndarray,
        y_return: Optional[np.ndarray],
        val_fraction: float,
        *,
        X_val_seq: Optional[np.ndarray] = None,
        y_dir_val: Optional[np.ndarray] = None,
        y_return_val: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        import torch  # type: ignore
        from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

        if X_val_seq is not None and y_dir_val is not None:
            X_train, X_val = X_seq, X_val_seq
            y_dir_train, y_dir_val = y_dir, y_dir_val
            y_ret_train = y_return
            y_ret_val = y_return_val
        else:
            # Compatibility fallback: random split.
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_dir_train, y_dir_val = train_test_split(
                X_seq, y_dir, test_size=val_fraction, random_state=self.cfg.seed, stratify=y_dir
            )

            if y_return is not None:
                _, _, y_ret_train, y_ret_val = train_test_split(
                    X_seq, y_return, test_size=val_fraction, random_state=self.cfg.seed, stratify=y_dir
                )
            else:
                y_ret_train = None
                y_ret_val = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._torch_model.to(device)
        model.train()

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        bce = torch.nn.BCEWithLogitsLoss()
        mse = torch.nn.MSELoss()

        X_train_t = torch.from_numpy(X_train).to(device)
        y_dir_train_t = torch.from_numpy(y_dir_train.astype(np.float32)).to(device)
        X_val_t = torch.from_numpy(X_val).to(device)
        y_dir_val_t = torch.from_numpy(y_dir_val.astype(np.float32)).to(device)

        if y_return is not None and y_ret_train is not None and y_ret_val is not None:
            y_ret_train_t = torch.from_numpy(y_ret_train.astype(np.float32)).to(device)
            y_ret_val_t = torch.from_numpy(y_ret_val.astype(np.float32)).to(device)
        else:
            y_ret_train_t = None
            y_ret_val_t = None

        # Keep training modest; this is the benchmark baseline.
        epochs = 10
        batch_size = 64
        n = X_train_t.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = X_train_t[idx]
                yb_dir = y_dir_train_t[idx]

                opt.zero_grad(set_to_none=True)
                dir_logit, ret_pred = model(xb)
                loss = bce(dir_logit, yb_dir)
                if y_ret_train_t is not None:
                    yb_ret = y_ret_train_t[idx]
                    loss = loss + mse(ret_pred, yb_ret)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            dir_logit_val, ret_pred_val = model(X_val_t)
            prob_val = torch.sigmoid(dir_logit_val).detach().cpu().numpy()
            y_pred_val = (prob_val >= 0.5).astype(int)
            metrics = {"val_accuracy": float(accuracy_score(y_dir_val, y_pred_val))}
            if len(np.unique(y_dir_val)) == 2:
                metrics["val_roc_auc"] = float(roc_auc_score(y_dir_val, prob_val))
            if y_ret_val_t is not None and y_return is not None:
                y_ret_mae = mean_absolute_error(
                    y_ret_val_t.detach().cpu().numpy(), ret_pred_val.detach().cpu().numpy()
                )
                metrics["val_return_mae"] = float(y_ret_mae)

        return metrics

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        if self.backend == "sklearn_fallback":
            X_feat = self._flatten(X_seq).astype(np.float32, copy=False)
            Xs = self._scaler.transform(X_feat)
            return self._dir_clf.predict_proba(Xs)[:, 1].astype(np.float32)

        import torch  # type: ignore

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._torch_model.to(device)
        model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_seq).to(device)
            dir_logit, _ = model(X_t)
            probs = torch.sigmoid(dir_logit).detach().cpu().numpy()
        return probs.astype(np.float32)

    def predict_expected_return(self, X_seq: np.ndarray) -> np.ndarray:
        if self.backend == "sklearn_fallback":
            if self._ret_reg is None:
                raise RuntimeError("No return regressor trained.")
            X_feat = self._flatten(X_seq).astype(np.float32, copy=False)
            Xs = self._scaler.transform(X_feat)
            return self._ret_reg.predict(Xs).astype(np.float32)

        import torch  # type: ignore

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._torch_model.to(device)
        model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_seq).to(device)
            _, ret_pred = model(X_t)
            preds = ret_pred.detach().cpu().numpy()
        return preds.astype(np.float32)

    def save(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"cfg": self.cfg.__dict__, "backend": self.backend}

        if self.backend == "sklearn_fallback":
            payload["scaler"] = self._scaler
            payload["dir_clf"] = self._dir_clf
            payload["ret_reg"] = self._ret_reg
        else:
            payload["torch_state_dict"] = self._torch_model.state_dict()

        joblib.dump(payload, out_path)

    @classmethod
    def load(cls, model_path: Path) -> "ClassicalSequenceEstimator":
        payload = joblib.load(model_path)
        cfg = ClassicalSequenceModelConfig(**payload["cfg"])
        est = cls(cfg=cfg)
        est.backend = payload["backend"]

        if est.backend == "sklearn_fallback":
            est._scaler = payload["scaler"]
            est._dir_clf = payload["dir_clf"]
            est._ret_reg = payload["ret_reg"]
        else:
            if not _torch_available():
                raise RuntimeError("Torch model requested but torch isn't available.")
            est._init_torch()
            est._torch_model.load_state_dict(payload["torch_state_dict"])

        return est

