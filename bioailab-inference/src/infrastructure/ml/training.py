"""
Treinamento de modelos ML (export para ONNX + scaler joblib).

Objetivo: treinar modelos supervisionados usando X extraído do pipeline e y vindo de lab_results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


DEFAULT_TARGETS_MAP: dict[str, dict[str, str]] = {
    "coliformes_totais": {
        "NMP/100mL": "coliformesTotaisNmp",
        "UFC/mL": "coliformesTotaisUfc",
        # Fallbacks comuns
        "NMP": "coliformesTotaisNmp",
        "UFC": "coliformesTotaisUfc",
    },
    "e_coli": {
        "NMP/100mL": "ecoliNmp",
        "UFC/mL": "ecoliUfc",
        # Fallbacks comuns
        "NMP": "ecoliNmp",
        "UFC": "ecoliUfc",
    },
}


def unit_slug(unit: str) -> str:
    u = (unit or "").strip()
    if not u:
        return "unit"
    return (
        u.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("%", "pct")
        .replace("__", "_")
        .lower()
    )


def select_latest_lab_result(lab_results: Any) -> Optional[dict]:
    if not isinstance(lab_results, list) or not lab_results:
        return None

    def key_fn(item: Any) -> str:
        if not isinstance(item, dict):
            return ""
        # Preferir analysisDate; fallback para _id/id
        v = item.get("analysisDate")
        if v is None:
            v = item.get("analysis_date")
        return str(v or item.get("_id") or item.get("id") or "")

    # Ordenar decrescente pelo campo "analysisDate" (string ISO ou datetime serializado)
    try:
        return sorted([x for x in lab_results if isinstance(x, dict)], key=key_fn, reverse=True)[0]
    except Exception:
        for x in lab_results:
            if isinstance(x, dict):
                return x
        return None


def select_lab_result_for_field(lab_results: Any, field: str) -> Optional[dict]:
    if not field:
        return select_latest_lab_result(lab_results)
    if not isinstance(lab_results, list) or not lab_results:
        return None
    candidates = [
        item
        for item in lab_results
        if isinstance(item, dict) and item.get(field) is not None
    ]
    if not candidates:
        return select_latest_lab_result(lab_results)

    def key_fn(item: Any) -> str:
        if not isinstance(item, dict):
            return ""
        v = item.get("analysisDate")
        if v is None:
            v = item.get("analysis_date")
        return str(v or item.get("_id") or item.get("id") or "")

    try:
        return sorted(candidates, key=key_fn, reverse=True)[0]
    except Exception:
        for x in candidates:
            if isinstance(x, dict):
                return x
        return None


def extract_target_value(lab: dict, field: str) -> Optional[float]:
    if not isinstance(lab, dict):
        return None
    if not field:
        return None
    v = lab.get(field)
    if v is None:
        return None
    try:
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv):
            return None
        return fv
    except Exception:
        return None


def transform_y(values: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "").strip().lower()
    if mode in ("log10p", "log10p1", "log10p_1"):
        return np.log10(np.maximum(0.0, values) + 1.0).astype(np.float32)
    return values.astype(np.float32)


def _metric_key(metric: str) -> str:
    m = (metric or "rmse").strip().lower()
    if m in ("mae", "rmse", "r2"):
        return m
    return "rmse"


def _score_for_selection(metrics: dict[str, Any], metric: str) -> float:
    m = _metric_key(metric)
    # Preferir validação quando existir
    if m == "r2":
        key = "val_r2" if "val_r2" in metrics else "train_r2"
        return float(metrics.get(key) or 0.0)
    if m == "mae":
        key = "val_mae" if "val_mae" in metrics else "train_mae"
        return -float(metrics.get(key) or 0.0)
    # rmse
    key = "val_rmse" if "val_rmse" in metrics else "train_rmse"
    return -float(metrics.get(key) or 0.0)


def _build_model(algorithm: str, params: dict[str, Any], *, random_state: int = 42):
    algo = (algorithm or "ridge").strip().lower()
    prefer_gpu = bool(params.get("prefer_gpu", True))

    if algo in ("rf", "random_forest", "randomforest"):

        def to_int(v: Any, default: int) -> int:
            try:
                return int(v)
            except Exception:
                return default

        def to_float(v: Any, default: float) -> float:
            try:
                return float(v)
            except Exception:
                return default

        def to_bool(v: Any, default: bool) -> bool:
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            if s in ("1", "true", "yes", "y", "sim", "on"):
                return True
            if s in ("0", "false", "no", "n", "nao", "não", "off"):
                return False
            return default

        if prefer_gpu:
            try:
                from lightgbm import LGBMRegressor  # type: ignore

                return LGBMRegressor(
                    boosting_type="rf",
                    n_estimators=to_int(params.get("n_estimators", 300) or 300, 300),
                    max_depth=to_int(params.get("max_depth", -1) or -1, -1),
                    min_child_samples=to_int(params.get("min_samples_leaf", 20) or 20, 20),
                    subsample=to_float(params.get("subsample", 0.8) or 0.8, 0.8),
                    subsample_freq=1,
                    colsample_bytree=to_float(params.get("max_features", 0.8) or 0.8, 0.8),
                    random_state=random_state,
                    device_type="gpu",
                )
            except Exception:
                pass

        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=to_int(params.get("n_estimators", 300) or 300, 300),
            max_depth=(None if params.get("max_depth") in [None, "", 0] else to_int(params.get("max_depth"), 12)),
            min_samples_split=to_int(params.get("min_samples_split", 2) or 2, 2),
            min_samples_leaf=to_int(params.get("min_samples_leaf", 1) or 1, 1),
            max_features=(None if params.get("max_features") in [None, ""] else to_float(params.get("max_features"), 1.0)),
            bootstrap=to_bool(params.get("bootstrap", True), True),
            random_state=random_state,
            n_jobs=to_int(params.get("n_jobs", -1) or -1, -1),
        )

    if algo in ("gbm", "gbr", "gradient_boosting", "lgbm", "lightgbm"):
        def to_int(v: Any, default: int) -> int:
            try:
                return int(v)
            except Exception:
                return default

        def to_float(v: Any, default: float) -> float:
            try:
                return float(v)
            except Exception:
                return default

        if prefer_gpu or algo in ("lgbm", "lightgbm"):
            try:
                from lightgbm import LGBMRegressor  # type: ignore

                return LGBMRegressor(
                    n_estimators=to_int(params.get("n_estimators", 400) or 400, 400),
                    learning_rate=to_float(params.get("learning_rate", 0.03) or 0.03, 0.03),
                    max_depth=to_int(params.get("max_depth", -1) or -1, -1),
                    subsample=to_float(params.get("subsample", 1.0) or 1.0, 1.0),
                    colsample_bytree=to_float(params.get("colsample_bytree", 1.0) or 1.0, 1.0),
                    random_state=random_state,
                    device_type="gpu",
                )
            except Exception:
                pass

        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=to_int(params.get("n_estimators", 400) or 400, 400),
            learning_rate=to_float(params.get("learning_rate", 0.03) or 0.03, 0.03),
            max_depth=to_int(params.get("max_depth", 3) or 3, 3),
            subsample=to_float(params.get("subsample", 1.0) or 1.0, 1.0),
            min_samples_split=to_int(params.get("min_samples_split", 2) or 2, 2),
            min_samples_leaf=to_int(params.get("min_samples_leaf", 1) or 1, 1),
            max_features=(None if params.get("max_features") in [None, ""] else to_float(params.get("max_features"), 1.0)),
            random_state=random_state,
        )

    if algo in ("xgb", "xgboost"):
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception as exc:
            raise RuntimeError("Algoritmo 'xgb' requer a dependência 'xgboost' instalada no servidor.") from exc

        def to_int(v: Any, default: int) -> int:
            try:
                return int(v)
            except Exception:
                return default

        def to_float(v: Any, default: float) -> float:
            try:
                return float(v)
            except Exception:
                return default

        gpu_params: dict[str, Any] = {}
        use_gpu = params.get("use_gpu", prefer_gpu)
        if use_gpu:
            if "tree_method" not in params:
                gpu_params["tree_method"] = "hist"
            if "device" not in params:
                gpu_params["device"] = "cuda"

        return XGBRegressor(
            n_estimators=to_int(params.get("n_estimators", 300) or 300, 300),
            learning_rate=to_float(params.get("learning_rate", 0.05) or 0.05, 0.05),
            max_depth=to_int(params.get("max_depth", 4) or 4, 4),
            subsample=to_float(params.get("subsample", 0.9) or 0.9, 0.9),
            colsample_bytree=to_float(params.get("colsample_bytree", 0.8) or 0.8, 0.8),
            gamma=to_float(params.get("gamma", 0.0) or 0.0, 0.0),
            reg_alpha=to_float(params.get("reg_alpha", 0.0) or 0.0, 0.0),
            reg_lambda=to_float(params.get("reg_lambda", 1.0) or 1.0, 1.0),
            random_state=random_state,
            n_jobs=to_int(params.get("n_jobs", -1) or -1, -1),
            objective="reg:squarederror",
            **gpu_params,
        )

    if algo in ("cat", "catboost"):
        try:
            from catboost import CatBoostRegressor  # type: ignore
        except Exception as exc:
            raise RuntimeError("Algoritmo 'catboost' requer a dependencia 'catboost' instalada no servidor.") from exc

        def to_int(v: Any, default: int) -> int:
            try:
                return int(v)
            except Exception:
                return default

        def to_float(v: Any, default: float) -> float:
            try:
                return float(v)
            except Exception:
                return default

        task_type = "GPU" if prefer_gpu else "CPU"
        return CatBoostRegressor(
            iterations=to_int(params.get("n_estimators", 400) or 400, 400),
            learning_rate=to_float(params.get("learning_rate", 0.1) or 0.1, 0.1),
            depth=to_int(params.get("max_depth", 6) or 6, 6),
            l2_leaf_reg=to_float(params.get("l2_leaf_reg", 3.0) or 3.0, 3.0),
            loss_function=str(params.get("loss_function", "RMSE") or "RMSE"),
            random_seed=random_state,
            task_type=task_type,
            verbose=False,
        )

    if algo in ("svr",):
        from sklearn.svm import SVR

        gamma = params.get("gamma", None)
        if isinstance(gamma, str) and gamma.strip().lower() in ("scale", "auto"):
            gamma_value: Any = gamma.strip().lower()
        elif gamma in [None, ""]:
            gamma_value = "scale"
        else:
            try:
                gamma_value = float(gamma)  # type: ignore[arg-type]
            except Exception:
                gamma_value = "scale"

        return SVR(
            kernel=str(params.get("kernel", "rbf") or "rbf"),
            C=float(params.get("C", 1.0) or 1.0),
            epsilon=float(params.get("epsilon", 0.1) or 0.1),
            gamma=gamma_value,
        )

    if algo in ("mlp", "mlp_regressor"):
        from sklearn.neural_network import MLPRegressor

        hidden = params.get("hidden_layer_sizes", (128, 64))
        if isinstance(hidden, str):
            parts = [p.strip() for p in hidden.split(",") if p.strip()]
            hidden = tuple(int(p) for p in parts) if parts else (128, 64)

        def to_bool(v: Any, default: bool) -> bool:
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            if s in ("1", "true", "yes", "y", "sim", "on"):
                return True
            if s in ("0", "false", "no", "n", "nao", "não", "off"):
                return False
            return default

        return MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=str(params.get("activation", "relu") or "relu"),
            solver=str(params.get("solver", "adam") or "adam"),
            alpha=float(params.get("alpha", 0.0001) or 0.0001),
            max_iter=int(params.get("max_iter", 800) or 800),
            learning_rate_init=float(params.get("learning_rate_init", 0.001) or 0.001),
            early_stopping=to_bool(params.get("early_stopping", True), True),
            random_state=random_state,
        )

    from sklearn.linear_model import Ridge

    return Ridge(alpha=float(params.get("alpha", 1.0) or 1.0))


def _is_lightgbm_model(model: Any) -> bool:
    cls = getattr(model, "__class__", None)
    if cls is None:
        return False
    name = getattr(cls, "__name__", "")
    module = getattr(cls, "__module__", "")
    if name == "LGBMRegressor":
        return True
    return module.startswith("lightgbm")


def _is_xgb_model(model: Any) -> bool:
    cls = getattr(model, "__class__", None)
    if cls is None:
        return False
    name = getattr(cls, "__name__", "")
    module = getattr(cls, "__module__", "")
    if name == "XGBRegressor":
        return True
    return module.startswith("xgboost")


def _predict_with_device(model: Any, X: np.ndarray) -> np.ndarray:
    if not _is_xgb_model(model):
        return model.predict(X)

    params = {}
    if hasattr(model, "get_params"):
        try:
            params = model.get_params()
        except Exception:
            params = {}
    device = str(params.get("device") or "").strip().lower()
    use_gpu = device.startswith("cuda")

    if use_gpu:
        try:
            import cupy as cp  # type: ignore

            Xc = cp.asarray(X)
            return model.predict(Xc)
        except Exception:
            try:
                if hasattr(model, "set_params"):
                    model.set_params(device="cpu")
            except Exception:
                pass

    preds = model.predict(X)
    if use_gpu and hasattr(model, "set_params") and device:
        try:
            model.set_params(device=device)
        except Exception:
            pass
    return preds


def _is_catboost_model(model: Any) -> bool:
    cls = getattr(model, "__class__", None)
    if cls is None:
        return False
    name = getattr(cls, "__name__", "")
    module = getattr(cls, "__module__", "")
    if name == "CatBoostRegressor":
        return True
    return module.startswith("catboost")


def _export_catboost_onnx(model: Any, path: Path) -> None:
    # CatBoost exporta ONNX via save_model
    model.save_model(str(path), format="onnx")


def _should_retry_without_gpu(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        key in msg
        for key in (
            "cuda",
            "gpu",
            "cublas",
            "cudnn",
            "device",
            "xgboost",
        )
    )


def _fit_with_fallback(
    algorithm: str,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
):
    cur_params = dict(params or {})
    prefer_gpu = bool(cur_params.get("prefer_gpu", True))
    mdl = _build_model(algorithm, cur_params, random_state=random_state)
    try:
        mdl.fit(X, y)
        return mdl
    except Exception as exc:
        if not prefer_gpu or not _should_retry_without_gpu(exc):
            raise
        cpu_params = dict(cur_params)
        cpu_params["prefer_gpu"] = False
        cpu_params["use_gpu"] = False
        mdl = _build_model(algorithm, cpu_params, random_state=random_state)
        mdl.fit(X, y)
        return mdl


def _validate_param_grid(grid: Any) -> dict[str, list[Any]]:
    if grid is None:
        return {}
    if not isinstance(grid, dict):
        raise ValueError("param_grid deve ser um objeto/dict")
    out: dict[str, list[Any]] = {}
    for k, v in grid.items():
        key = str(k).strip()
        if not key:
            continue
        if isinstance(v, list):
            out[key] = v
        else:
            out[key] = [v]
    return out


@dataclass
class TorchModelBundle:
    model: Any
    device: str
    algorithm: str
    input_layout: str
    seq_len: Optional[int] = None
    seq_channels: Optional[int] = None


def _is_torch_bundle(obj: Any) -> bool:
    return isinstance(obj, TorchModelBundle)


def _torch_device(prefer_gpu: bool) -> str:
    if not TORCH_AVAILABLE:
        return "cpu"
    if prefer_gpu and torch.cuda.is_available():  # type: ignore[union-attr]
        return "cuda"
    return "cpu"


def _infer_sequence_shape(X: np.ndarray, block_config: dict[str, Any]) -> tuple[int, int, str]:
    input_layout = str(block_config.get("input_layout", "time_channels") or "time_channels").strip().lower()
    channels = block_config.get("channels") or block_config.get("input_channels")
    if isinstance(channels, list):
        ch_count = len(channels)
    elif channels in ("", None):
        ch_count = 1
    else:
        try:
            ch_count = int(channels)
        except Exception:
            ch_count = 1

    max_length = block_config.get("max_length")
    if max_length in ("", None):
        max_length = block_config.get("window")
    seq_len = None
    if max_length not in ("", None):
        try:
            seq_len = int(max_length)
        except Exception:
            seq_len = None

    if seq_len and ch_count and (seq_len * ch_count == X.shape[1]):
        return seq_len, ch_count, input_layout

    if ch_count and X.shape[1] % ch_count == 0:
        return int(X.shape[1] // ch_count), int(ch_count), input_layout

    return int(X.shape[1]), 1, input_layout


def _make_sequence_views(
    X: np.ndarray,
    *,
    seq_len: int,
    seq_channels: int,
    input_layout: str,
) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if input_layout == "channels_time":
        x_ct = X.reshape(n, seq_channels, seq_len)
        x_tc = np.transpose(x_ct, (0, 2, 1))
    else:
        x_tc = X.reshape(n, seq_len, seq_channels)
        x_ct = np.transpose(x_tc, (0, 2, 1))
    return x_tc.astype(np.float32), x_ct.astype(np.float32)


def _parse_hidden_sizes(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(v) for v in value)
        except Exception:
            return default
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if parts:
            try:
                return tuple(int(p) for p in parts)
            except Exception:
                return default
    return default


def _normalize_activation(value: Any) -> str:
    if value is None:
        return "relu"
    if isinstance(value, (int, float)):
        idx = int(value)
        if idx == 1:
            return "tanh"
        if idx == 2:
            return "sigmoid"
        return "relu"
    s = str(value).strip().lower()
    if s in ("1", "tanh"):
        return "tanh"
    if s in ("2", "sigmoid", "logistic"):
        return "sigmoid"
    if s in ("0", "relu"):
        return "relu"
    return "relu"


def _activation_layer(name: str):
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    return nn.ReLU()


def _normalize_optimizer(value: Any) -> str:
    if value is None:
        return "adam"
    if isinstance(value, (int, float)):
        idx = int(value)
        if idx == 1:
            return "sgd"
        if idx == 2:
            return "rmsprop"
        return "adam"
    s = str(value).strip().lower()
    if s in ("1", "sgd"):
        return "sgd"
    if s in ("2", "rmsprop", "rms"):
        return "rmsprop"
    if s in ("0", "adam"):
        return "adam"
    return "adam"


def _build_torch_model(
    algorithm: str,
    *,
    input_dim: int,
    seq_len: Optional[int],
    seq_channels: Optional[int],
    params: dict[str, Any],
) -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Algoritmo requer PyTorch instalado no servidor.")

    algo = (algorithm or "mlp").strip().lower()
    hidden_sizes = _parse_hidden_sizes(params.get("hidden_layer_sizes"), ())
    if not hidden_sizes:
        legacy_layers = params.get("layers")
        legacy_hidden = params.get("hidden")
        try:
            layers_count = int(legacy_layers) if legacy_layers not in ("", None) else 0
        except Exception:
            layers_count = 0
        try:
            hidden_count = int(legacy_hidden) if legacy_hidden not in ("", None) else 0
        except Exception:
            hidden_count = 0
        if layers_count > 0 and hidden_count > 0:
            hidden_sizes = tuple(hidden_count for _ in range(layers_count))
    if not hidden_sizes:
        hidden_sizes = (128, 64)
    hidden_size = int(params.get("hidden_size") or params.get("hidden") or 64)
    num_layers = int(params.get("num_layers") or 1)
    dropout = float(params.get("dropout") or 0.0)
    activation_name = _normalize_activation(params.get("activation"))

    if algo == "mlp":
        layers: list[Any] = []
        in_dim = int(input_dim)
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(_activation_layer(activation_name))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    if algo == "cnn":
        ch = int(seq_channels or 1)
        t_len = int(seq_len or 1)
        conv1 = nn.Conv1d(ch, 32, kernel_size=3, padding=1)
        conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        flatten_dim = 64 * t_len
        dense_size = int(params.get("hidden") or params.get("hidden_size") or 64)
        return nn.Sequential(
            conv1,
            _activation_layer(activation_name),
            conv2,
            _activation_layer(activation_name),
            nn.Flatten(),
            nn.Linear(flatten_dim, dense_size),
            _activation_layer(activation_name),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dense_size, 1),
        )

    if algo == "lstm":
        ch = int(seq_channels or 1)
        lstm = nn.LSTM(
            input_size=ch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        head = nn.Linear(hidden_size, 1)

        class _LSTMRegressor(nn.Module):
            def __init__(self, lstm_layer: Any, head_layer: Any):
                super().__init__()
                self.lstm = lstm_layer
                self.head = head_layer

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.head(last)

        return _LSTMRegressor(lstm, head)

    raise RuntimeError(f"Algoritmo torch desconhecido: {algo}")


def _train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    algorithm: str,
    params: dict[str, Any],
    block_config: dict[str, Any],
    random_state: int,
) -> TorchModelBundle:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Algoritmo requer PyTorch instalado no servidor.")

    torch.manual_seed(int(random_state))  # type: ignore[union-attr]
    prefer_gpu = bool(params.get("prefer_gpu", True))
    device = _torch_device(prefer_gpu)

    algo = (algorithm or "mlp").strip().lower()
    seq_len = None
    seq_channels = None
    input_layout = "flat"
    if algo in ("cnn", "lstm"):
        seq_len, seq_channels, input_layout = _infer_sequence_shape(X_train, block_config)
        x_tc, x_ct = _make_sequence_views(
            X_train,
            seq_len=seq_len,
            seq_channels=seq_channels,
            input_layout=input_layout,
        )
        x_np = x_ct if algo == "cnn" else x_tc
    else:
        x_np = X_train

    input_dim = int(x_np.shape[1]) if x_np.ndim == 2 else int(x_np.shape[1] * x_np.shape[2])
    model = _build_torch_model(
        algo,
        input_dim=input_dim,
        seq_len=seq_len,
        seq_channels=seq_channels,
        params=params,
    ).to(device)

    epochs = int(params.get("epochs") or 200)
    epochs = 1 if epochs < 1 else 1000 if epochs > 1000 else epochs
    batch_size = int(params.get("batch_size") or 64)
    batch_size = 1 if batch_size < 1 else 4096 if batch_size > 4096 else batch_size
    lr = float(params.get("learning_rate") or params.get("learning_rate_init") or 0.001)

    criterion = nn.MSELoss()
    optimizer_name = _normalize_optimizer(params.get("optimizer") or params.get("solver"))
    if optimizer_name == "sgd":
        momentum = float(params.get("momentum") or 0.9)
        optimizer = torch.optim.SGD(  # type: ignore[union-attr]
            model.parameters(), lr=lr, momentum=momentum
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)  # type: ignore[union-attr]
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # type: ignore[union-attr]

    dataset = torch.utils.data.TensorDataset(  # type: ignore[union-attr]
        torch.from_numpy(x_np).float(),
        torch.from_numpy(y_train).float().view(-1, 1),
    )
    loader = torch.utils.data.DataLoader(  # type: ignore[union-attr]
        dataset, batch_size=batch_size, shuffle=True
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    return TorchModelBundle(
        model=model,
        device=device,
        algorithm=algo,
        input_layout=input_layout,
        seq_len=seq_len,
        seq_channels=seq_channels,
    )


def _predict_torch(bundle: TorchModelBundle, X: np.ndarray) -> np.ndarray:
    model = bundle.model
    device = bundle.device
    algo = bundle.algorithm
    model.eval()
    if algo in ("cnn", "lstm"):
        seq_len = int(bundle.seq_len or X.shape[1])
        seq_channels = int(bundle.seq_channels or 1)
        x_tc, x_ct = _make_sequence_views(
            X,
            seq_len=seq_len,
            seq_channels=seq_channels,
            input_layout=bundle.input_layout,
        )
        x_np = x_ct if algo == "cnn" else x_tc
    else:
        x_np = X
    with torch.no_grad():  # type: ignore[union-attr]
        xb = torch.from_numpy(x_np).float().to(device)
        preds = model(xb).detach().cpu().numpy().reshape(-1)
    return preds.astype(np.float32)


def _export_torch_onnx(bundle: TorchModelBundle, X: np.ndarray, out_path: Path) -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Export requer PyTorch instalado no servidor.")
    model = bundle.model
    model.eval()
    algo = bundle.algorithm
    if algo in ("cnn", "lstm"):
        seq_len = int(bundle.seq_len or X.shape[1])
        seq_channels = int(bundle.seq_channels or 1)
        if algo == "cnn":
            dummy = torch.zeros(1, seq_channels, seq_len, dtype=torch.float32)
        else:
            dummy = torch.zeros(1, seq_len, seq_channels, dtype=torch.float32)
    else:
        dummy = torch.zeros(1, int(X.shape[1]), dtype=torch.float32)
    torch.onnx.export(  # type: ignore[union-attr]
        model.cpu(),
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )


@dataclass
class TrainResult:
    model_path: Path
    scaler_path: Path
    metrics: dict[str, Any]
    n_samples: int
    y_transform: str
    metadata_path: Optional[Path] = None  # Caminho do arquivo de metadados JSON


def inverse_transform_y(values: np.ndarray, mode: str) -> np.ndarray:
    """
    Reverte a transformação aplicada ao y durante o treinamento.
    
    Crítico: Se o modelo foi treinado com log10p(y), a predição está em escala log.
    Devemos aplicar 10^pred - 1 para voltar à escala original.
    """
    mode = (mode or "").strip().lower()
    if mode in ("log10p", "log10p1", "log10p_1"):
        # Reverso de log10(y + 1) é 10^pred - 1
        return (np.power(10.0, values) - 1.0).astype(np.float32)
    return values.astype(np.float32)


def train_regressor_export_onnx(
    X: np.ndarray,
    y: np.ndarray,
    *,
    algorithm: str = "ridge",
    params: Optional[dict[str, Any]] = None,
    params_by_algorithm: Optional[dict[str, dict[str, Any]]] = None,
    y_transform_mode: str = "log10p",
    test_size: float = 0.2,
    random_state: int = 42,
    perm_importance: bool = False,
    perm_repeats: int = 10,
    selection_metric: str = "rmse",
    grid_search: bool = False,
    algorithms: Optional[list[str]] = None,
    param_grid: Optional[dict[str, Any]] = None,
    param_grid_by_algorithm: Optional[dict[str, dict[str, Any]]] = None,
    max_trials: int = 60,
    out_dir: Path,
    prefix: str,
    # -------------------------------------------------------------------------
    # NOVO: Configurações específicas do bloco para salvar no metadata
    # -------------------------------------------------------------------------
    block_config: Optional[dict[str, Any]] = None,
) -> TrainResult:
    """
    Treina um regressor em sklearn, salva scaler (.joblib) e exporta o modelo para ONNX (.onnx).

    Observação: o scaler é aplicado fora do ONNX (OnnxAdapter aplica scaler antes da inferência).
    
    Args:
        block_config: Configurações específicas do bloco que serão salvas no metadata.
                      Para forecaster: window, horizon, input_channels, target_channel, output_channel
                      Para inference: input_feature, channel, output_unit
    """
    params = params or {}
    params_by_algorithm = params_by_algorithm or {}
    param_grid_by_algorithm = param_grid_by_algorithm or {}
    block_config = block_config or {}
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    y_t = transform_y(y, y_transform_mode)

    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    ts = float(test_size or 0.0)
    ts = 0.0 if ts < 0 else 0.8 if ts > 0.8 else ts

    if ts and X.shape[0] >= 4:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y_t,
            test_size=ts,
            random_state=int(random_state),
        )
    else:
        X_train, y_train = X, y_t
        X_val, y_val = None, None

    scaler = RobustScaler()
    Xs_train = scaler.fit_transform(X_train).astype(np.float32)
    Xs_val = scaler.transform(X_val).astype(np.float32) if X_val is not None else None

    metric_key = _metric_key(selection_metric)
    max_trials_i = int(max_trials or 60)
    max_trials_i = 1 if max_trials_i < 1 else 500 if max_trials_i > 500 else max_trials_i

    tried: list[dict[str, Any]] = []
    best_model = None
    best_algo = ""
    best_params: dict[str, Any] = {}
    best_metrics: dict[str, Any] | None = None
    best_score = float("-inf")

    def eval_model(mdl) -> dict[str, Any]:
        if _is_torch_bundle(mdl):
            pred_tr = _predict_torch(mdl, Xs_train).astype(np.float32).reshape(-1)
        else:
            pred_tr = _predict_with_device(mdl, Xs_train).astype(np.float32).reshape(-1)
        m: dict[str, Any] = {
            "train_mae": float(mean_absolute_error(y_train, pred_tr)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_tr))),
            "train_r2": float(r2_score(y_train, pred_tr)) if len(y_train) >= 2 else 0.0,
        }
        if Xs_val is not None and y_val is not None and len(y_val) >= 1:
            if _is_torch_bundle(mdl):
                pred_val = _predict_torch(mdl, Xs_val).astype(np.float32).reshape(-1)
            else:
                pred_val = _predict_with_device(mdl, Xs_val).astype(np.float32).reshape(-1)
            m.update(
                {
                    "val_mae": float(mean_absolute_error(y_val, pred_val)),
                    "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred_val))),
                    "val_r2": float(r2_score(y_val, pred_val)) if len(y_val) >= 2 else 0.0,
                }
            )
        return m

    # grid search (bem simples, estilo legado) — varre algoritmos + ParameterGrid(param_grid)
    candidate_algos = [str(a).strip().lower() for a in (algorithms or [algorithm]) if str(a).strip()]
    if not candidate_algos:
        candidate_algos = ["ridge"]

    for algo in candidate_algos:
        base_params = dict(params or {})
        per_algo = params_by_algorithm.get(algo)
        if isinstance(per_algo, dict):
            base_params.update(per_algo)

        grid_raw = None
        if isinstance(param_grid_by_algorithm.get(algo), dict):
            grid_raw = param_grid_by_algorithm.get(algo)
        elif isinstance(param_grid, dict):
            grid_raw = param_grid

        grid = _validate_param_grid(grid_raw) if grid_raw else {}

        combos = [None]
        if grid:
            from sklearn.model_selection import ParameterGrid

            combos = list(ParameterGrid(grid))

        for combo in combos:
            if len(tried) >= max_trials_i:
                break
            cur_params = dict(base_params)
            if isinstance(combo, dict):
                cur_params.update(combo)

            if algo in ("mlp", "cnn", "lstm"):
                mdl = _train_torch_model(
                    Xs_train,
                    y_train,
                    algorithm=algo,
                    params=cur_params,
                    block_config=block_config,
                    random_state=int(random_state),
                )
            else:
                mdl = _fit_with_fallback(
                    algo, cur_params, Xs_train, y_train, random_state=int(random_state)
                )
            m = eval_model(mdl)
            m.update(
                {
                    "test_size": float(ts),
                    "random_state": int(random_state),
                    "y_transform": y_transform_mode or "none",
                }
            )
            score = _score_for_selection(m, metric_key)
            tried.append(
                {
                    "algorithm": algo,
                    "params": cur_params,
                    "score": float(score),
                    "metrics": m,
                }
            )
            if score > best_score:
                best_score = score
                best_model = mdl
                best_algo = algo
                best_params = cur_params
                best_metrics = m
        if len(tried) >= max_trials_i:
            break

    # fallback
    if best_model is None:
        best_algo = (algorithm or "ridge").strip().lower()
        best_params = dict(params or {})
        if best_algo in ("mlp", "cnn", "lstm"):
            best_model = _train_torch_model(
                Xs_train,
                y_train,
                algorithm=best_algo,
                params=best_params,
                block_config=block_config,
                random_state=int(random_state),
            )
        else:
            best_model = _fit_with_fallback(
                best_algo, best_params, Xs_train, y_train, random_state=int(random_state)
            )
        best_metrics = eval_model(best_model)
        best_score = _score_for_selection(best_metrics, metric_key)

    best_is_torch = _is_torch_bundle(best_model)
    metrics = dict(best_metrics or {})
    metrics.update(
        {
            "selection_metric": metric_key,
            "best_algorithm": best_algo,
            "best_params": best_params,
            "best_score": float(best_score),
            "trials_count": int(len(tried)),
        }
    )
    if tried:
        metrics["trials_preview"] = tried[: min(10, len(tried))]

    if Xs_val is not None and y_val is not None and len(y_val) >= 1 and perm_importance and not best_is_torch:
        try:
            from sklearn.inspection import permutation_importance

            repeats = int(perm_repeats or 10)
            repeats = 1 if repeats < 1 else 50 if repeats > 50 else repeats
            r = permutation_importance(
                best_model,
                Xs_val,
                y_val,
                n_repeats=repeats,
                random_state=int(random_state),
                scoring="neg_mean_squared_error",
            )
            metrics["perm_importance_mean"] = [float(x) for x in (r.importances_mean.tolist() or [])]
            metrics["perm_repeats"] = repeats
        except Exception:
            metrics["perm_importance_mean"] = None

    import joblib
    import json
    from datetime import datetime

    scaler_path = out_dir / f"{prefix}_scaler.joblib"
    model_path = out_dir / f"{prefix}_model.onnx"
    metadata_path = out_dir / f"{prefix}_metadata.json"

    joblib.dump(scaler, scaler_path)

    # Exportar somente o modelo (scaler fica externo)
    initial_types = None
    onnx_model = None
    onnx_bytes: Optional[bytes] = None

    # XGBoost precisa de conversor específico (onnxmltools)
    if best_algo in ("xgb", "xgboost"):
        try:
            from onnxmltools import convert_xgboost  # type: ignore
            from onnxmltools.convert.common.data_types import FloatTensorType  # type: ignore

            initial_types = [("input", FloatTensorType([None, int(Xs_train.shape[1])]))]
            onnx_model = convert_xgboost(best_model, initial_types=initial_types, target_opset=13)
        except Exception as exc:
            raise RuntimeError(
                "Falha ao exportar modelo XGBoost para ONNX. Verifique dependências: onnxmltools + xgboost."
            ) from exc
    elif _is_lightgbm_model(best_model):
        try:
            from onnxmltools import convert_lightgbm  # type: ignore
            from onnxmltools.convert.common.data_types import FloatTensorType  # type: ignore

            initial_types = [("input", FloatTensorType([None, int(Xs_train.shape[1])]))]
            onnx_model = convert_lightgbm(best_model, initial_types=initial_types, target_opset=13)
        except Exception as exc:
            raise RuntimeError(
                "Falha ao exportar modelo LightGBM para ONNX. Verifique dependencias: onnxmltools + lightgbm."
            ) from exc
    elif _is_catboost_model(best_model):
        try:
            _export_catboost_onnx(best_model, model_path)
            onnx_bytes = model_path.read_bytes()
        except Exception as exc:
            raise RuntimeError(
                "Falha ao exportar modelo CatBoost para ONNX. Verifique dependencias: catboost."
            ) from exc
    elif best_is_torch:
        try:
            _export_torch_onnx(best_model, Xs_train, model_path)
            onnx_bytes = model_path.read_bytes()
        except Exception as exc:
            raise RuntimeError("Falha ao exportar modelo PyTorch para ONNX.") from exc
    else:
        from skl2onnx import convert_sklearn  # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore

        initial_types = [("input", FloatTensorType([None, int(Xs_train.shape[1])]))]
        onnx_model = convert_sklearn(best_model, initial_types=initial_types, target_opset=13)

    if onnx_model is not None:
        model_path.write_bytes(onnx_model.SerializeToString())
    elif onnx_bytes is not None and not model_path.exists():
        model_path.write_bytes(onnx_bytes)

    # -------------------------------------------------------------------------
    # METADATA DO MODELO - Crítico para inferência correta
    # -------------------------------------------------------------------------
    # Este arquivo contém todas as informações necessárias para o bloco de
    # inferência processar o modelo corretamente. O treinamento "manda" as configs.
    # -------------------------------------------------------------------------
    
    # Calcular estatísticas do X para validação na inferência
    x_stats = {
        "mean": [float(m) for m in np.nanmean(X, axis=0).tolist()],
        "std": [float(s) for s in np.nanstd(X, axis=0).tolist()],
        "min": [float(m) for m in np.nanmin(X, axis=0).tolist()],
        "max": [float(m) for m in np.nanmax(X, axis=0).tolist()],
        "q25": [float(q) for q in np.nanpercentile(X, 25, axis=0).tolist()],
        "q75": [float(q) for q in np.nanpercentile(X, 75, axis=0).tolist()],
    }
    
    # Calcular estatísticas do y (escala original, antes de transform)
    y_stats = {
        "mean": float(np.nanmean(y)),
        "std": float(np.nanstd(y)),
        "min": float(np.nanmin(y)),
        "max": float(np.nanmax(y)),
        "q25": float(np.nanpercentile(y, 25)),
        "q75": float(np.nanpercentile(y, 75)),
    }
    
    metadata = {
        "version": "1.1.0",  # Versão atualizada com block_config
        "created_at": datetime.utcnow().isoformat() + "Z",
        
        # Arquivos associados (paths relativos ao diretório do metadata)
        "model_file": model_path.name,
        "scaler_file": scaler_path.name,
        
        # =====================================================================
        # CONFIGURAÇÕES DO BLOCO - Carregadas automaticamente na inferência
        # =====================================================================
        # Estas são as configurações que o treinamento determina e que o bloco
        # deve usar. O usuário não precisa configurar manualmente.
        # =====================================================================
        "block_config": block_config,
        
        # Configuração do treinamento (algoritmo, hiperparâmetros)
        "training": {
            "algorithm": best_algo,
            "params": best_params,
            "y_transform": y_transform_mode or "none",
            "test_size": float(ts),
            "random_state": int(random_state),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "framework": "torch" if best_is_torch else "sklearn",
        },
        
        # Estatísticas para validação na inferência
        "input_stats": x_stats,
        "output_stats": y_stats,
        
        # Ranges válidos para clipping (baseado nos dados de treino)
        "valid_ranges": {
            "input_min": x_stats["min"],
            "input_max": x_stats["max"],
            "output_min": y_stats["min"],
            "output_max": max(y_stats["max"], y_stats["max"] * 1.5),  # Margem para extrapolação
        },
        
        # Métricas de performance
        "metrics": {
            k: v for k, v in metrics.items()
            if k not in ("trials_preview", "perm_importance_mean")  # Excluir campos grandes
        },
    }
    
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return TrainResult(
        model_path=model_path,
        scaler_path=scaler_path,
        metrics=metrics,
        n_samples=int(X.shape[0]),
        y_transform=y_transform_mode or "none",
        metadata_path=metadata_path,
    )


# =============================================================================
# TREINAMENTO COM SALVAMENTO DE TODOS OS CANDIDATOS
# =============================================================================
# Permite analisar e escolher qual modelo usar, como no pipeline legado.
# Salva cada candidato em arquivo separado para seleção posterior.
# =============================================================================

@dataclass
class CandidateModel:
    """Representa um modelo candidato do grid search."""
    rank: int
    algorithm: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    score: float
    model_path: Path
    scaler_path: Path
    metadata_path: Path
    selected: bool = False


@dataclass
class GridSearchResult:
    """Resultado do grid search com todos os candidatos."""
    candidates: list[CandidateModel]
    best_index: int
    session_path: Path
    n_samples: int
    y_transform: str


def train_with_candidates(
    X: np.ndarray,
    y: np.ndarray,
    *,
    algorithm: str = "ridge",
    params: Optional[dict[str, Any]] = None,
    params_by_algorithm: Optional[dict[str, dict[str, Any]]] = None,
    y_transform_mode: str = "log10p",
    test_size: float = 0.2,
    random_state: int = 42,
    selection_metric: str = "rmse",
    algorithms: Optional[list[str]] = None,
    param_grid: Optional[dict[str, list[Any]]] = None,
    param_grid_by_algorithm: Optional[dict[str, dict[str, list[Any]]]] = None,
    max_trials: int = 60,
    out_dir: Path | str = Path("resources/models"),
    prefix: str = "model",
    block_config: Optional[dict[str, Any]] = None,
    save_all_candidates: bool = True,
) -> GridSearchResult:
    """
    Treina modelos com grid search e SALVA TODOS OS CANDIDATOS.
    
    Diferente de train_regressor_export_onnx, esta função:
    1. Salva cada candidato em arquivo separado
    2. Retorna lista de candidatos para análise/seleção
    3. Permite escolher qual modelo usar posteriormente
    
    Args:
        save_all_candidates: Se True, exporta cada candidato para ONNX.
                            Se False, só exporta o melhor.
    
    Returns:
        GridSearchResult com lista de candidatos e paths dos arquivos.
    """
    import joblib
    import json
    from datetime import datetime
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import train_test_split, ParameterGrid
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar diretório de sessão para os candidatos
    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    candidates_dir = out_dir / f"{prefix}_candidates_{session_id}"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    
    params = params or {}
    params_by_algorithm = params_by_algorithm or {}
    param_grid_by_algorithm = param_grid_by_algorithm or {}
    block_config = block_config or {}
    
    # Transformar y
    y_t = transform_y(y, y_transform_mode)
    
    # Split
    ts = float(test_size) if test_size and 0 < test_size < 1 else 0.0
    if ts > 0 and len(y) >= 5:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_t, test_size=ts, random_state=int(random_state)
        )
    else:
        X_train, y_train = X, y_t
        X_val, y_val = None, None
    
    scaler = RobustScaler()
    Xs_train = scaler.fit_transform(X_train).astype(np.float32)
    Xs_val = scaler.transform(X_val).astype(np.float32) if X_val is not None else None
    
    metric_key = _metric_key(selection_metric)
    max_trials_i = min(max(1, int(max_trials or 60)), 500)
    
    def eval_model(mdl) -> tuple[dict[str, Any], dict[str, Any]]:
        """Retorna (métricas, dados_predicao)"""
        if _is_torch_bundle(mdl):
            pred_tr = _predict_torch(mdl, Xs_train).astype(np.float32).reshape(-1)
        else:
            pred_tr = _predict_with_device(mdl, Xs_train).astype(np.float32).reshape(-1)
        m: dict[str, Any] = {
            "train_mae": float(mean_absolute_error(y_train, pred_tr)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_tr))),
            "train_r2": float(r2_score(y_train, pred_tr)) if len(y_train) >= 2 else 0.0,
        }
        # Dados para gráficos
        pred_data: dict[str, Any] = {
            "train_actual": y_train.tolist(),
            "train_predicted": pred_tr.tolist(),
            "train_residuals": (y_train - pred_tr).tolist(),
        }
        if Xs_val is not None and y_val is not None and len(y_val) >= 1:
            if _is_torch_bundle(mdl):
                pred_val = _predict_torch(mdl, Xs_val).astype(np.float32).reshape(-1)
            else:
                pred_val = _predict_with_device(mdl, Xs_val).astype(np.float32).reshape(-1)
            m.update({
                "val_mae": float(mean_absolute_error(y_val, pred_val)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred_val))),
                "val_r2": float(r2_score(y_val, pred_val)) if len(y_val) >= 2 else 0.0,
            })
            pred_data.update({
                "val_actual": y_val.tolist(),
                "val_predicted": pred_val.tolist(),
                "val_residuals": (y_val - pred_val).tolist(),
            })
        return m, pred_data
    
    # Lista de candidatos
    candidates: list[CandidateModel] = []
    best_score = float("-inf")
    best_index = 0
    
    # Algoritmos candidatos
    candidate_algos = [str(a).strip().lower() for a in (algorithms or [algorithm]) if str(a).strip()]
    if not candidate_algos:
        candidate_algos = ["ridge"]
    
    trial_idx = 0
    for algo in candidate_algos:
        base_params = dict(params or {})
        per_algo = params_by_algorithm.get(algo)
        if isinstance(per_algo, dict):
            base_params.update(per_algo)
        
        grid_raw = param_grid_by_algorithm.get(algo) or param_grid
        grid = _validate_param_grid(grid_raw) if grid_raw else {}
        
        combos = [None] if not grid else list(ParameterGrid(grid))
        
        for combo in combos:
            if trial_idx >= max_trials_i:
                break
            
            cur_params = dict(base_params)
            if isinstance(combo, dict):
                cur_params.update(combo)
            
            if algo in ("mlp", "cnn", "lstm"):
                mdl = _train_torch_model(
                    Xs_train,
                    y_train,
                    algorithm=algo,
                    params=cur_params,
                    block_config=block_config,
                    random_state=int(random_state),
                )
            else:
                mdl = _fit_with_fallback(
                    algo, cur_params, Xs_train, y_train, random_state=int(random_state)
                )
            m, pred_data = eval_model(mdl)
            m.update({
                "test_size": float(ts),
                "random_state": int(random_state),
                "y_transform": y_transform_mode or "none",
            })
            score = _score_for_selection(m, metric_key)
            
            # Salvar este candidato
            cand_prefix = f"candidate_{trial_idx:03d}_{algo}"
            cand_model_path = candidates_dir / f"{cand_prefix}_model.onnx"
            cand_scaler_path = candidates_dir / f"{cand_prefix}_scaler.joblib"
            cand_predictions_path = candidates_dir / f"{cand_prefix}_predictions.json"
            cand_metadata_path = candidates_dir / f"{cand_prefix}_metadata.json"
            
            if save_all_candidates:
                # Salvar scaler
                joblib.dump(scaler, cand_scaler_path)
                
                # Salvar dados de predição para gráficos
                cand_predictions_path.write_text(json.dumps(pred_data, indent=2, ensure_ascii=False), encoding="utf-8")
                
                # Exportar modelo ONNX
                initial_types = None
                onnx_model = None
                onnx_bytes: Optional[bytes] = None
                
                if algo in ("xgb", "xgboost"):
                    try:
                        from onnxmltools import convert_xgboost
                        from onnxmltools.convert.common.data_types import FloatTensorType
                        initial_types = [("input", FloatTensorType([None, int(Xs_train.shape[1])]))]
                        onnx_model = convert_xgboost(mdl, initial_types=initial_types, target_opset=13)
                    except Exception:
                        pass
                elif _is_lightgbm_model(mdl):
                    try:
                        from onnxmltools import convert_lightgbm
                        from onnxmltools.convert.common.data_types import FloatTensorType
                        initial_types = [("input", FloatTensorType([None, int(Xs_train.shape[1])]))]
                        onnx_model = convert_lightgbm(mdl, initial_types=initial_types, target_opset=13)
                    except Exception:
                        pass
                elif _is_catboost_model(mdl):
                    try:
                        _export_catboost_onnx(mdl, cand_model_path)
                        onnx_bytes = cand_model_path.read_bytes()
                    except Exception:
                        pass
                elif _is_torch_bundle(mdl):
                    try:
                        _export_torch_onnx(mdl, Xs_train, cand_model_path)
                        onnx_bytes = cand_model_path.read_bytes()
                    except Exception:
                        pass
                else:
                    try:
                        from skl2onnx import convert_sklearn
                        from skl2onnx.common.data_types import FloatTensorType
                        initial_types = [("input", FloatTensorType([None, int(Xs_train.shape[1])]))]
                        onnx_model = convert_sklearn(mdl, initial_types=initial_types, target_opset=13)
                    except Exception:
                        pass
                
                if onnx_model:
                    cand_model_path.write_bytes(onnx_model.SerializeToString())
                elif onnx_bytes is not None and not cand_model_path.exists():
                    cand_model_path.write_bytes(onnx_bytes)
                
                # Salvar metadata do candidato
                cand_metadata = {
                    "version": "1.1.0",
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "candidate_index": trial_idx,
                    "model_file": cand_model_path.name,
                    "scaler_file": cand_scaler_path.name,
                    "block_config": block_config,
                    "training": {
                        "algorithm": algo,
                        "params": cur_params,
                        "y_transform": y_transform_mode or "none",
                        "test_size": float(ts),
                        "random_state": int(random_state),
                        "n_samples": int(X.shape[0]),
                        "n_features": int(X.shape[1]),
                        "framework": "torch" if _is_torch_bundle(mdl) else "sklearn",
                    },
                    "metrics": m,
                    "score": float(score),
                    "rank": -1,  # Será atualizado depois
                }
                cand_metadata_path.write_text(json.dumps(cand_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            
            candidate = CandidateModel(
                rank=-1,  # Será atualizado
                algorithm=algo,
                params=cur_params,
                metrics=m,
                score=float(score),
                model_path=cand_model_path,
                scaler_path=cand_scaler_path,
                metadata_path=cand_metadata_path,
            )
            candidates.append(candidate)
            
            if score > best_score:
                best_score = score
                best_index = trial_idx
            
            trial_idx += 1
        
        if trial_idx >= max_trials_i:
            break
    
    # Ordenar candidatos por score (maior = melhor) e atribuir ranks
    sorted_indices = sorted(range(len(candidates)), key=lambda i: candidates[i].score, reverse=True)
    for rank, idx in enumerate(sorted_indices):
        candidates[idx].rank = rank + 1
        # Atualizar rank no metadata
        if save_all_candidates and candidates[idx].metadata_path.exists():
            meta = json.loads(candidates[idx].metadata_path.read_text(encoding="utf-8"))
            meta["rank"] = rank + 1
            candidates[idx].metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Salvar sessão completa
    session_path = candidates_dir / "_session.json"
    session_data = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "y_transform": y_transform_mode or "none",
        "selection_metric": metric_key,
        "best_index": best_index,
        "candidates": [
            {
                "rank": c.rank,
                "algorithm": c.algorithm,
                "params": c.params,
                "metrics": c.metrics,
                "score": c.score,
                "model_file": c.model_path.name if c.model_path.exists() else None,
                "scaler_file": c.scaler_path.name if c.scaler_path.exists() else None,
                "metadata_file": c.metadata_path.name if c.metadata_path.exists() else None,
            }
            for c in candidates
        ],
    }
    session_path.write_text(json.dumps(session_data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    return GridSearchResult(
        candidates=candidates,
        best_index=best_index,
        session_path=session_path,
        n_samples=int(X.shape[0]),
        y_transform=y_transform_mode or "none",
    )


def select_candidate(
    session_path: Path | str,
    candidate_index: int,
    out_dir: Path | str,
    prefix: str = "selected_model",
) -> TrainResult:
    """
    Seleciona um candidato e copia para o diretório final.
    
    Args:
        session_path: Caminho do arquivo _session.json
        candidate_index: Índice do candidato a selecionar (0-based)
        out_dir: Diretório de destino
        prefix: Prefixo dos arquivos finais
    
    Returns:
        TrainResult com paths dos arquivos selecionados
    """
    import json
    import shutil
    
    session_path = Path(session_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not session_path.exists():
        raise FileNotFoundError(f"Sessão não encontrada: {session_path}")
    
    session = json.loads(session_path.read_text(encoding="utf-8"))
    candidates = session.get("candidates", [])
    
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise ValueError(f"Índice inválido: {candidate_index} (máx: {len(candidates) - 1})")
    
    selected = candidates[candidate_index]
    candidates_dir = session_path.parent
    
    # Copiar arquivos do candidato selecionado
    src_model = candidates_dir / selected["model_file"]
    src_scaler = candidates_dir / selected["scaler_file"]
    src_metadata = candidates_dir / selected["metadata_file"]
    
    dst_model = out_dir / f"{prefix}_model.onnx"
    dst_scaler = out_dir / f"{prefix}_scaler.joblib"
    dst_metadata = out_dir / f"{prefix}_metadata.json"
    
    if src_model.exists():
        shutil.copy2(src_model, dst_model)
    if src_scaler.exists():
        shutil.copy2(src_scaler, dst_scaler)
    if src_metadata.exists():
        # Atualizar metadata com info de seleção
        meta = json.loads(src_metadata.read_text(encoding="utf-8"))
        meta["selected"] = True
        meta["selected_from_session"] = session.get("session_id")
        meta["model_file"] = dst_model.name
        meta["scaler_file"] = dst_scaler.name
        dst_metadata.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    
    return TrainResult(
        model_path=dst_model,
        scaler_path=dst_scaler,
        metrics=selected.get("metrics", {}),
        n_samples=session.get("n_samples", 0),
        y_transform=session.get("y_transform", "none"),
        metadata_path=dst_metadata,
    )
