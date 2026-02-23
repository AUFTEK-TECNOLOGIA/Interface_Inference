"""Construção e treino/reuso de forecasters e regressors."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

import logging
logger = logging.getLogger(__name__)


from .config import (
    TARGET_LEN,
    BATCH_FC,
    EPOCHS_FC,
    BATCH_REG,
    EPOCHS_REG,
    LR,
    DEVICE,
    MODEL_DIR,
    SEED,
    TEST_SIZE,
)


class WindowDS(torch.utils.data.Dataset):
    """Dataset de janelas deslizantes."""

    def __init__(self, series: list[np.ndarray], *, window: int, horizon: int):
        X, Y = [], []
        min_len = window + horizon
        for s in series:
            L = len(s)
            if L < min_len:
                pad = np.zeros((min_len - L, s.shape[1]), dtype=s.dtype)
                s_pad = np.vstack([s, pad])
            else:
                s_pad = s
            L_pad = len(s_pad)
            for t in range(window, L_pad - horizon + 1):
                X.append(s_pad[t - window : t])
                Y.append(s_pad[t : t + horizon])

        if not X:
            # evita DataLoader quebrar sem mensagem clara
            raise ValueError(
                "Sem janelas válidas. Aumente o tamanho das séries ou ajuste window/horizon."
            )

        self.X = torch.as_tensor(np.stack(X), dtype=torch.float32)
        self.Y = torch.as_tensor(np.stack(Y), dtype=torch.float32)

    def __len__(self):
        # len(dataset) -> usado pelo DataLoader
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]



# ─── Forecaster ───────────────────────────────────────────────────────────

def make_forecaster(
    n_in: int,
    hidden: int,
    layers: int,
    *,
    horizon: int,
    dropout: float = 0.0,
    bidirectional: bool = False,
) -> nn.Module:
    """Cria modelo LSTM previsor parametrizado por horizon/dropout/bidirectional."""

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            do = 0.0 if layers == 1 else float(dropout)
            self.lstm = nn.LSTM(
                input_size=n_in,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=do,
                bidirectional=bool(bidirectional),
            )
            num_dir = 2 if bidirectional else 1
            self.fc = nn.Linear(hidden * num_dir, horizon * n_in)

        def forward(self, x):  # noqa: D401
            h, _ = self.lstm(x)
            y = self.fc(h[:, -1])
            return y.view(-1, horizon, n_in)

    return Forecaster()


# ─── Regressor ────────────────────────────────────────────────────────────

def make_regressor(
    arch: str,
    n_in: int,
    n_outputs: int = 1,
    gbm_n_estimators: int = 800,
    gbm_lr: float = 0.03,
    gbm_max_depth: int = 3,
    gbm_min_child_weight: float = 1.0,
    gbm_subsample: float = 1.0,
    gbm_colsample_bytree: float = 1.0,
    gbm_gamma: float = 0.0,
    gbm_reg_lambda: float = 1.0,
    gbm_reg_alpha: float = 0.0,
    cnn_kernel: int = 5,
    cnn_depth: int = 2,
    cnn_channels: int = 16,
    cnn_stride: int = 1,
    cnn_pool_size: int = 2,
    lstm_layers: int = 2,
    lstm_units: int = 64,
    lstm_dropout: float = 0.2,
    rf_n_estimators: int = 100,
    rf_max_depth: int | None = None,
    rf_max_features: str = "sqrt",
    rf_min_samples_split: int = 2,
    rf_min_samples_leaf: int = 1,
    rf_bootstrap: bool = True,
    svr_kernel: str = "rbf",
    svr_c: float = 1.0,
    svr_epsilon: float = 0.1,
    svr_degree: int = 3,
    xgb_n_estimators: int = 200,
    xgb_lr: float = 0.05,
    xgb_max_depth: int = 3,
    xgb_scale_pos_weight: float = 1.0,
    xgb_max_delta_step: int = 0,
    mlp_layers: int = 2,
    mlp_hidden: int = 64,
    mlp_dropout: float = 0.0,
    batch_norm: bool = False,
    mlp_in_features: int | None = None,
    **extra,
):
    a = arch.lower()
    if a == "mlp":
        layers = []
        in_f = mlp_in_features or TARGET_LEN * n_in
        for _ in range(mlp_layers):
            layers.append(nn.Linear(in_f, mlp_hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(mlp_hidden))
            layers.append(nn.ReLU())
            if mlp_dropout > 0:
                layers.append(nn.Dropout(mlp_dropout))
            in_f = mlp_hidden
        layers.append(nn.Linear(in_f, n_outputs))
        return nn.Sequential(*layers)
    if a == "cnn":
        layers = []
        in_ch = n_in
        out_ch = cnn_channels
        for _ in range(cnn_depth):
            layers.append(nn.Conv1d(in_ch, out_ch, cnn_kernel, stride=cnn_stride))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(cnn_pool_size))
            in_ch = out_ch
            out_ch *= 2
        layers.append(nn.AdaptiveAvgPool1d(1))
        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(*layers)
                self.fc = nn.Linear(in_ch, n_outputs)

            def forward(self, x):  # noqa: D401
                x = x.view(x.size(0), n_in, -1)
                x = self.conv(x).squeeze(-1)
                return self.fc(x)

        return CNN()
    if a == "lstm":
        class LSTMreg(nn.Module):
            def __init__(self):
                super().__init__()
                do = 0.0 if lstm_layers == 1 else lstm_dropout
                self.rnn = nn.LSTM(
                    n_in,
                    lstm_units,
                    lstm_layers,
                    batch_first=True,
                    dropout=do,
                )
                self.fc = nn.Linear(lstm_units, n_outputs)

            def forward(self, x):  # noqa: D401
                y, _ = self.rnn(x.view(x.size(0), -1, n_in))
                return self.fc(y[:, -1])

        return LSTMreg()
    if a in ("gbm", "xgboost", "xgb"):
        try:
            from xgboost import XGBRegressor
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError("xgboost nao instalado") from exc

        if a == "gbm":
            n_estimators = gbm_n_estimators
            lr = gbm_lr
            depth = gbm_max_depth
            scale_pos = xgb_scale_pos_weight
            delta = xgb_max_delta_step
            reg_lambda = gbm_reg_lambda
            reg_alpha = gbm_reg_alpha
            gamma = gbm_gamma
            min_child = gbm_min_child_weight
            subsample = gbm_subsample
            colsample = gbm_colsample_bytree
        else:
            n_estimators = xgb_n_estimators
            lr = xgb_lr
            depth = xgb_max_depth
            scale_pos = xgb_scale_pos_weight
            delta = xgb_max_delta_step
            reg_lambda = gbm_reg_lambda
            reg_alpha = gbm_reg_alpha
            gamma = gbm_gamma
            min_child = gbm_min_child_weight
            subsample = gbm_subsample
            colsample = gbm_colsample_bytree

        params = dict(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=depth,
            min_child_weight=min_child,
            subsample=subsample,
            colsample_bytree=colsample,
            gamma=gamma,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            scale_pos_weight=scale_pos,
            max_delta_step=delta,
            objective="reg:squarederror",
            random_state=SEED,
            verbosity=0,
        )
        if DEVICE.type == "cuda":
            params.update(device="cuda", predictor="gpu_predictor")
        base = XGBRegressor(**params)
        return base if n_outputs == 1 else MultiOutputRegressor(base)
    if a == "random_forest_gpu":
        # GPU-based Random Forest via XGBoost
        try:
            from xgboost import XGBRFRegressor
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError("xgboost nao instalado") from exc

        params = dict(
            n_estimators=xgb_n_estimators,
            learning_rate=xgb_lr,
            max_depth=xgb_max_depth,
            objective="reg:squarederror",
            random_state=SEED,
        )
        if DEVICE.type == "cuda":
            params.update(device="cuda", predictor="gpu_predictor")
        base = XGBRFRegressor(**params)
        return base if n_outputs == 1 else MultiOutputRegressor(base)
    if a in ("random_forest", "rf"):
        base = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            max_features=rf_max_features,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            bootstrap=rf_bootstrap,
            random_state=SEED,
            n_jobs=-1,
        )
        return base if n_outputs == 1 else MultiOutputRegressor(base)
    if a == "svr":
        base = SVR(kernel=svr_kernel, C=svr_c, epsilon=svr_epsilon, degree=svr_degree)
        return base if n_outputs == 1 else MultiOutputRegressor(base)
    raise ValueError(f"modelo desconhecido: {arch}")


# ─── Utilitários de treino/carregamento ───────────────────────────────────

# src/models.py  (trecho ilustrativo do fit_or_load_torch)

def fit_or_load_torch(model, loader, epochs, ckpt_path, lr: float = 1e-3):
    # 1) tenta carregar se já existir
    if ckpt_path is not None and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
            model.history = state.get("history", [])
        else:
            model.load_state_dict(state)   # state_dict puro
            model.history = []
        model.eval()
        logger.info("checkpoint carregado de %s", ckpt_path)
        return model

    # 2) senão, treina do zero
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for e in range(1, epochs + 1):
        model.train()
        se, n = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(DEVICE) if hasattr(xb, "to") else xb
            yb = yb.to(DEVICE) if hasattr(yb, "to") else yb
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

            with torch.no_grad():
                diff = pred - yb
                se += float((diff * diff).sum().item())
                n  += int(yb.numel())
        train_mse = se / max(n, 1)
        history.append({"epoch": e, "train_mse": train_mse})
        logger.info("[reg %4d/%d] MSE=%.4f", e, epochs, train_mse)

    model.history = history
    model.eval()

    # 3) salva no caminho correto e com o formato mais robusto
    if ckpt_path is not None:
        torch.save({"model_state": model.state_dict(), "history": history, "lr": lr}, ckpt_path)

    return model




def fit_or_load_sklearn(model, path: Path, x, y, **fit_kwargs):
    """Fit scikit-learn model or load from disk.

    Any extra keyword arguments are forwarded to ``model.fit``. This allows
    passing parameters like ``eval_set`` or ``early_stopping_rounds`` when
    supported by the estimator.
    """
    import joblib
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            print("modelo incompatível → re-treinando…")
    try:
        model.fit(x, y, **fit_kwargs)
    except TypeError as exc:
        if "early_stopping_rounds" in str(exc) and "early_stopping_rounds" in fit_kwargs:
            try:
                from xgboost.callback import EarlyStopping
            except Exception:
                raise
            rounds = fit_kwargs.pop("early_stopping_rounds")
            callbacks = fit_kwargs.pop("callbacks", [])
            callbacks.append(EarlyStopping(rounds=rounds))
            model.fit(x, y, callbacks=callbacks, **fit_kwargs)
        else:
            raise
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)
    print(f"✅ {path.name} salvo")
    return model


def train_torch(model, loader: DataLoader, epochs: int):
    """Treina modelo PyTorch sem salvar."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            if yb.ndim == 2 and yb.shape[1] == 1:
                yb = yb.squeeze(1)
            assert pred.shape == yb.shape, (pred.shape, yb.shape)
            l = loss(pred, yb)
            opt.zero_grad(); l.backward(); opt.step()  # noqa: E702
    return model


from sklearn.base import BaseEstimator, RegressorMixin


class TorchRegressor(BaseEstimator, RegressorMixin):
    """Wrapper scikit-learn para modelos PyTorch."""

    def __init__(self, build_fn, *, batch_size=BATCH_REG, epochs=EPOCHS_REG, device=DEVICE):
        self.build_fn = build_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.model = build_fn().to(device)
        self.is_fitted_ = False

    def fit(self, X, y):  # noqa: D401
        yt = torch.tensor(y, dtype=torch.float32)
        if yt.ndim == 2 and yt.shape[1] == 1:
            yt = yt.squeeze(1)
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            yt,
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        train_torch(self.model, loader, self.epochs)
        self.model.eval()
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        self.is_fitted_ = True
        return self

    def predict(self, X):  # noqa: D401
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X, device=self.device, dtype=torch.float32)
            pred = self.model(xb).cpu().numpy()
        if pred.ndim == 2 and pred.shape[1] == 1:
            return pred.ravel()
        return pred

    # ------------------------------------------------------------------
    # Pickle support ----------------------------------------------------
    def __getstate__(self):  # noqa: D401
        return {
            "state_dict": self.model.state_dict(),
            "build_fn": self.build_fn,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "device": str(self.device),
            "n_features_in_": getattr(self, "n_features_in_", None),
            "is_fitted_": getattr(self, "is_fitted_", False),
        }

    def __setstate__(self, state):  # noqa: D401
        self.build_fn = state["build_fn"]
        self.batch_size = state.get("batch_size", BATCH_REG)
        self.epochs = state.get("epochs", EPOCHS_REG)
        saved_device = torch.device(state.get("device", "cpu"))
        if saved_device.type == "cuda" and not torch.cuda.is_available():
            saved_device = torch.device("cpu")
        self.device = saved_device
        self.model = self.build_fn().to(self.device)
        self.model.load_state_dict(state["state_dict"])
        self.n_features_in_ = state.get("n_features_in_")
        self.is_fitted_ = state.get("is_fitted_", True)


def grid_search_regressor(arch: str, n_in: int, tr_x, tr_y):
    """Busca simples de hiperparâmetros e retorna melhor modelo e params."""
    param_grid = []
    if arch == "cnn":
        for depth in (1, 2, 3):
            for kernel in (3, 5):
                param_grid.append({"cnn_depth": depth, "cnn_kernel": kernel})
    elif arch == "gbm":
        for ne in (400, 800):
            for lr in (0.03, 0.05):
                param_grid.append({"gbm_n_estimators": ne, "gbm_lr": lr})
    else:
        param_grid.append({})

    # Ensure single-target arrays are 1D for scikit-learn compatibility
    if isinstance(tr_y, np.ndarray) and tr_y.ndim == 2 and tr_y.shape[1] == 1:
        tr_y = tr_y.ravel()

    tr_x_sub, val_x, tr_y_sub, val_y = train_test_split(
        tr_x, tr_y, test_size=0.2, random_state=SEED
    )
    if isinstance(tr_y_sub, np.ndarray) and tr_y_sub.ndim == 2 and tr_y_sub.shape[1] == 1:
        tr_y_sub = tr_y_sub.ravel()
        val_y = val_y.ravel()

    best_params = None
    best_rmse = float("inf")
    best_model = None

    for params in param_grid:
        if arch in ("gbm", "xgboost"):
            reg = make_regressor(arch, n_in, **params)
            reg.fit(tr_x_sub, tr_y_sub)
            preds = reg.predict(val_x)
        else:
            ds = TensorDataset(
                torch.tensor(tr_x_sub, dtype=torch.float32),
                torch.tensor(tr_y_sub, dtype=torch.float32),
            )
            reg = make_regressor(arch, n_in, **params).to(DEVICE)
            loader = DataLoader(ds, batch_size=BATCH_REG, shuffle=True)
            reg = train_torch(reg, loader, EPOCHS_REG)
            with torch.no_grad():
                preds = reg(torch.tensor(val_x, device=DEVICE).float()).cpu().numpy().ravel()
        rmse = float(np.sqrt(np.mean((preds - val_y) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    # treina modelo final com melhores params
    if arch in ("gbm", "xgboost"):
        best_model = make_regressor(arch, n_in, **best_params)
        best_model.fit(tr_x, tr_y)
    else:
        ds_full = TensorDataset(
            torch.tensor(tr_x, dtype=torch.float32),
            torch.tensor(tr_y, dtype=torch.float32),
        )
        best_model = make_regressor(arch, n_in, **best_params).to(DEVICE)
        loader = DataLoader(ds_full, batch_size=BATCH_REG, shuffle=True)
        best_model = train_torch(best_model, loader, EPOCHS_REG)

    return best_model, best_params


# ─── Ajuda para split de treino/teste ─────────────────────────────────────

def split_train_test(X, y, uuids, test_size=TEST_SIZE, random_state=SEED):
    """
    Divide X, y e uuids em treino/teste usando o test_size e random_state fornecidos.
    """
    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
    return (
        X[tr_idx],
        X[te_idx],
        y[tr_idx],
        y[te_idx],
        [uuids[i] for i in tr_idx],
        [uuids[i] for i in te_idx],
    )
