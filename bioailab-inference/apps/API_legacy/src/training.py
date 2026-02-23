# src/training.py

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import joblib


from .config import (
    SEED,
    TEST_SIZE,
    BATCH_FC,
    EPOCHS_FC,
    BATCH_REG,
    EPOCHS_REG,
    DEVICE,
    MODEL_DIR,
    FCT_HIDDEN,
    FCT_LAYERS,
    WINDOW,
    HORIZON,
    TARGET_LEN,
    LR,
)
from .loader import load_dataset
from .models import (
    make_forecaster,
    make_regressor,
    fit_or_load_torch,
    WindowDS,
    TorchRegressor,
)
from .utils import complete_series
from .augmentation import augment_dataset


# ===== remova a tabela DEFAULTS e aliases =====

# Mapas de validação
ALIASES = {
    "random_forest": "rf",
    "xgboost": "xgb",
}
REQUIRED: dict[str, set[str]] = {
    "gbm": {
        "gbm_n_estimators","gbm_lr","gbm_max_depth","gbm_min_child_weight",
        "gbm_subsample","gbm_colsample_bytree","gbm_gamma",
        "gbm_reg_lambda","gbm_reg_alpha",
    },
    "xgb": {
        "xgb_n_estimators","xgb_lr","xgb_max_depth","xgb_scale_pos_weight","xgb_max_delta_step",
        "gbm_reg_lambda","gbm_reg_alpha","gbm_gamma","gbm_min_child_weight",
        "gbm_subsample","gbm_colsample_bytree",
    },
    "rf": {
        "rf_n_estimators","rf_max_depth","rf_max_features",
        "rf_min_samples_split","rf_min_samples_leaf","rf_bootstrap",
    },
    "svr": {"svr_kernel","svr_c","svr_epsilon","svr_degree"},
    "mlp": {"mlp_layers","mlp_hidden","mlp_dropout","batch_norm"},
    "cnn": {"cnn_kernel","cnn_depth","cnn_channels","cnn_stride","cnn_pool_size"},
    "lstm":{"lstm_layers","lstm_units","lstm_dropout"},
}
# parâmetros tolerados extra (não “de modelo”)
ALLOWED_EXTRA = {"mlp_in_features"}  # será auto-preenchido se faltar

def _canon(name: str) -> str:
    n = name.lower()
    return ALIASES.get(n, n)

def _validate_single(name: str, params: dict) -> dict:
    arch = _canon(name)
    if arch not in REQUIRED:
        raise ValueError(f"modelo desconhecido: {name}")
    need = REQUIRED[arch]
    given = set(params.keys())
    missing = sorted(list(need - given))
    if missing:
        raise ValueError(f"{name}: faltam parâmetros obrigatórios: {missing}")
    unknown = sorted(list(given - need - ALLOWED_EXTRA))
    if unknown:
        raise ValueError(f"{name}: parâmetros desconhecidos: {unknown}")
    return params

def _validate_grid(name: str, grid: dict, params: dict | None = None) -> dict:
    arch = _canon(name)
    if arch not in REQUIRED:
        raise ValueError(f"modelo desconhecido: {name}")
    if not grid:
        raise ValueError(f"{name}: grid vazio.")

    keys = set(grid.keys())
    params_keys = set(params.keys()) if params else set()
    need = REQUIRED[arch]

    # exige que a união entre params e grid cubra todas as chaves obrigatórias
    missing = sorted(list(need - keys - params_keys))
    if missing:
        raise ValueError(
            f"{name}: faltam parâmetros obrigatórios em params/grid: {missing}"
        )

    unknown_grid = sorted(list(keys - need - ALLOWED_EXTRA))
    if unknown_grid:
        raise ValueError(f"{name}: chaves desconhecidas no grid: {unknown_grid}")

    if params:
        unknown_params = sorted(list(params_keys - need - ALLOWED_EXTRA))
        if unknown_params:
            raise ValueError(f"{name}: parâmetros desconhecidos: {unknown_params}")

    return grid

def compute_perm_importance(model, X_val, y_val, n_repeats: int = 10):
    """Compute permutation importance for sklearn or torch models."""
    try:
        if isinstance(model, torch.nn.Module):
            class _Wrap:
                def __init__(self, net):
                    self.net = net.eval()
                def fit(self, X, y):
                    pass
                def predict(self, X):
                    with torch.no_grad():
                        t = torch.tensor(X, device=DEVICE, dtype=torch.float32)
                        return self.net(t).cpu().numpy()

            wrapper = _Wrap(model)
            scorer = "neg_mean_squared_error"
            r = permutation_importance(
                wrapper, X_val, y_val, n_repeats=n_repeats, scoring=scorer, random_state=SEED
            )
        else:
            r = permutation_importance(
                model, X_val, y_val, n_repeats=n_repeats, random_state=SEED
            )
        return r.importances_mean.tolist()
    except Exception as e:
        logging.debug("Perm importance failed: %s", e)
        return None



def train_forecaster(
    mixes: dict[str, list[str]],
    units: list[str],
    params,
    dataset_path: Path,
):
    """Treina o forecaster e retorna dados processados; gera um .npz por trial."""
    MODEL_DIR.mkdir(exist_ok=True)



    # ── Carrega e filtra datasets por sensor ─────────────────────────────────
    data_json = Path(dataset_path)
    datasets = {}
    for sensor, channels in mixes.items():
        c, y, uids, ts = load_dataset(
            sensor,
            channels,
            units,
            data_json,
            slice_start=getattr(params, "sliceStart", 0),
            slice_end=getattr(params, "sliceEnd", 0),
        )
        datasets[sensor] = {
            uid: (curve, yl, t)
            for curve, yl, uid, t in zip(c, y, uids, ts)
        }

    # ── Listas alinhadas de curvas/targets/uuids ─────────────────────────────
    if not datasets:
        raise ValueError("Nenhum ensaio após filtros.")
    common = sorted(set.intersection(*(set(d.keys()) for d in datasets.values())))
    curves, ylog, uuids, timestamps = [], [], [], []
    for uid in common:
        parts = [datasets[s][uid] for s in mixes]
        curves.append(np.hstack([p[0] for p in parts]))
        ylog.append(parts[0][1])
        uuids.append(uid)
        ts_ref = parts[0][2]
        for p in parts[1:]:
            if not np.array_equal(p[2], ts_ref):
                raise ValueError("timestamps inconsistentes entre sensores")
        timestamps.append(ts_ref)

    orig_curves = list(curves)
    rng = np.random.default_rng(SEED)
    for seq in orig_curves:
        if len(seq) > 1:
            n = int(rng.integers(1, len(seq)))
            curves.append(seq[:n])

    n_channels = sum(len(chs) for chs in mixes.values())
    canon    = "|".join(f"{s}-{','.join(chs)}" for s, chs in mixes.items())
    unit_str = ",".join(units)
    tag      = hashlib.md5(f"{canon}-{unit_str}".encode()).hexdigest()[:6]
    # diretório para salvar .npz por trial
    out_dir = Path(MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Split TREINO/VAL em nível de sequência (VAL não é usado na seleção aqui) ─
    idx_all = np.arange(len(curves))
    idx_tr_seq, _idx_val_seq = train_test_split(
        idx_all,
        test_size=getattr(params, "testSize", TEST_SIZE),
        random_state=SEED,
    )

    # parâmetros padrão (podem ser sobrescritos pelo grid)
    default_window  = params.window
    default_horizon = params.horizon
    default_epochs  = params.epochs
    default_batch   = params.batchSize
    default_lr      = params.learningRate


    # Dataset de treino = apenas sequências de treino (janeladas)
    ds_tr = WindowDS([curves[i] for i in idx_tr_seq], window=default_window, horizon=default_horizon)
    loader = DataLoader(
        ds_tr,
        batch_size=default_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE.type=="cuda"),
    )

    # ── Helpers para obter o train_mse ───────────────────────────────────────
    def _loader_mse(model, loader_) -> float:
        """Fallback: MSE médio no loader (janelas de treino), caso não haja history."""
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader_:
                xb = xb.to(DEVICE) if hasattr(xb, "to") else xb
                yb = yb.to(DEVICE) if hasattr(yb, "to") else yb
                pred = model(xb)
                diff = pred - yb
                se += float((diff * diff).sum().item())
                n  += int(np.prod(yb.shape))
        return se / max(n, 1)

    def _get_train_mse_from_history(m) -> float | None:
        if hasattr(m, "history") and m.history:
            h = m.history[-1]
            if "train_mse" in h:
                return float(h["train_mse"])
            if "mse" in h:
                return float(h["mse"])
            if "loss" in h:
                return float(h["loss"])
        return None

    def _long_extrap_mse(model, curves_eval, *, window, horizon) -> float:
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for seq in curves_eval:
                if len(seq) <= window:
                    continue
                pred = complete_series(
                    seq[:window],
                    model,
                    window=window,
                    horizon=horizon,
                    target_len=len(seq),
                )
                true = seq
                diff = pred[window:] - true[window:]
                se += float((diff * diff).sum())
                n  += diff.size
        return se / max(n, 1)

    # ── Grid search (param_search) ───────────────────────────────────────────
    grid = getattr(params, "paramGrid", None)
    param_search = getattr(params, "paramSearch", False)

    trials: list[dict] = []
    best_params = None

    if param_search and grid:
        best_score = float("inf")
        best_model = None
        best_combo = None

        combos = list(ParameterGrid(grid))
        for i, combo in enumerate(combos):
            logging.info("Forecaster grid %d/%d: %s", i + 1, len(combos), combo)

            hidden = int(combo.get("hiddenUnits", params.hiddenUnits))
            layers = int(combo.get("layers", params.layers))
            dropout = float(combo.get("dropout", params.dropout))
            bidir = bool(combo.get("bidirectional", params.bidirectional))

            window  = int(combo.get("window",  params.window))
            horizon = int(combo.get("horizon", params.horizon))
            epochs  = int(combo.get("epochs", params.epochs))
            batch   = int(combo.get("batchSize",  params.batchSize))
            lr      = float(combo.get("learningRate",      params.learningRate))


            # reconstroi dataset/loader para este combo
            ds_tr = WindowDS([curves[i] for i in idx_tr_seq], window=window, horizon=horizon)
            loader = DataLoader(
                ds_tr,
                batch_size=batch,
                shuffle=True,
                num_workers=4,
                pin_memory=(DEVICE.type == "cuda"),
            )

            model = make_forecaster(
                n_channels,
                hidden,
                layers,
                horizon=horizon,
                dropout=dropout,
                bidirectional=bidir,
            ).to(DEVICE)

            # Checkpoint temporário por combo
            tmp_ckpt = MODEL_DIR / f"fct_{tag}_grid_{i}.pth"
            mdl = fit_or_load_torch(model, loader, epochs, tmp_ckpt, lr=lr)

            train_mse = _long_extrap_mse(
                mdl,
                orig_curves,
                window=window,
                horizon=horizon,
            )

            torch.save(
                {
                    "model_state": mdl.state_dict(),
                    "hidden": int(hidden),
                    "layers": int(layers),
                    "dropout": float(dropout),
                    "bidirectional": bool(bidir),
                    "window": int(window),
                    "horizon": int(horizon),
                    "targetLen": int(getattr(params, "targetLen", TARGET_LEN)),
                    "learningRate": float(lr),
                    "epochs": int(epochs),
                    "batchSize": int(batch),
                },
                tmp_ckpt,
            )

            # Registra trial
            trial_rec = {
                "modelFile": tmp_ckpt.name,
                "params": {
                    "hiddenUnits": int(hidden),
                    "layers": int(layers),
                    "dropout": float(dropout),
                    "bidirectional": bool(bidir),
                    "window": int(window),
                    "horizon": int(horizon),
                    "learningRate": float(lr),
                    "epochs": int(epochs),
                    "batchSize": int(batch),
                },
                "trainMse": float(train_mse),
            }
            trials.append(trial_rec)

            # Seleção do melhor pelo menor train_mse (apenas para retorno/curvas agregadas)
            score = float("inf") if train_mse is None else float(train_mse)
            if score < best_score or best_model is None:
                best_score = score
                best_model = mdl
                best_combo = trial_rec["params"].copy()

        if best_model is None:
            raise RuntimeError("Grid search do forecaster não encontrou nenhum modelo válido.")

        fct = best_model
        best_params = best_combo

    else:
        # Treino único (sem grid) – ainda geramos um "trial" consistente (_grid_0)
        hidden   = int(params.hiddenUnits)
        layers   = int(params.layers)
        dropout  = float(params.dropout)
        bidir    = bool(params.bidirectional)
        window   = int(params.window)
        horizon  = int(params.horizon)
        epochs   = int(params.epochs)
        batch    = int(params.batchSize)
        lr       = float(params.learningRate)


        # reconstruir loader com os defaults (garantia)
        ds_tr = WindowDS([curves[i] for i in idx_tr_seq], window=window, horizon=horizon)
        loader = DataLoader(
            ds_tr,
            batch_size=batch,
            shuffle=True,
            num_workers=4,
            pin_memory=(DEVICE.type == "cuda"),
        )
        fct = fit_or_load_torch(
            make_forecaster(
                n_channels,
                hidden,
                layers,
                horizon=horizon,
                dropout=dropout,
                bidirectional=bidir,
            ).to(DEVICE),
            loader,
            epochs,
            None,  # não usar caminho canônico aqui
            lr=lr,
        )
        best_params = {
            "hiddenUnits": int(hidden),
            "layers": int(layers),
            "dropout": float(dropout),
            "bidirectional": bool(bidir),
            "window": int(window),
            "horizon": int(horizon),
            "learningRate": float(lr),
            "epochs": int(epochs),
            "batchSize": int(batch),
        }

        train_mse_single = _long_extrap_mse(
            fct,
            orig_curves,
            window=window,
            horizon=horizon,
        )

        tmp_ckpt = MODEL_DIR / f"fct_{tag}_grid_0.pth"
        torch.save(
            {
                "model_state": fct.state_dict(),
                "hidden": int(hidden),
                "layers": int(layers),
                "dropout": float(dropout),
                "bidirectional": bool(bidir),
                "window": int(window),
                "horizon": int(horizon),
                "targetLen": int(getattr(params, "targetLen", TARGET_LEN)),
                "learningRate": float(lr),
                "epochs": int(epochs),
                "batchSize": int(batch),
            },
            tmp_ckpt,
        )

        trials.append(
            {
                "modelFile": tmp_ckpt.name,
                "params": {
                    "hiddenUnits": int(hidden),
                    "layers": int(layers),
                    "dropout": float(dropout),
                    "bidirectional": bool(bidir),
                    "window": int(window),
                    "horizon": int(horizon),
                    "learningRate": float(lr),
                    "epochs": int(epochs),
                    "batchSize": int(batch),
                },
                "trainMse": float(train_mse_single),
            }
        )

    # ── Gera curvas completas (usando o melhor modelo, para consumo imediato) ─
    with torch.no_grad():
        full_curves = [
            complete_series(
                seq,
                fct,
                window=int(best_params["window"]),
                horizon=int(best_params["horizon"]),
                target_len=int(getattr(params, "targetLen", TARGET_LEN)),
                timestamps=ts,
            )
            for seq, ts in zip(orig_curves, timestamps)
        ]


    y_all = np.asarray(ylog, float)
    if y_all.ndim == 2 and y_all.shape[1] == 1:
        y_all = y_all.ravel()

    # ── Gera .npz por trial com as curvas do PRÓPRIO forecaster ──────────────
    for i, t in enumerate(trials):
        ckpt_path = MODEL_DIR / t["modelFile"]
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

        hidden = int(state.get("hidden", t["params"]["hiddenUnits"]))
        layers = int(state.get("layers", t["params"]["layers"]))
        dropout = float(state.get("dropout", t["params"].get("dropout", 0.0)))
        bidir = bool(state.get("bidirectional", t["params"].get("bidirectional", False)))
        window = int(state.get("window", t["params"].get("window", default_window)))
        horizon = int(state.get("horizon", t["params"].get("horizon", default_horizon)))
        tgt_len = int(state.get("targetLen", getattr(params, "targetLen", TARGET_LEN)))

        mdl_trial = make_forecaster(
            n_channels,
            hidden,
            layers,
            horizon=horizon,
            dropout=dropout,
            bidirectional=bidir,
        ).to(DEVICE)
        mdl_trial.load_state_dict(state["model_state"])  # <- pega só os pesos

        full_trial = [
            complete_series(
                seq,
                mdl_trial,
                window=window,
                horizon=horizon,
                target_len=tgt_len,
                timestamps=ts,
            )
            for seq, ts in zip(orig_curves, timestamps)
        ]
        arr_full = np.stack(full_trial, axis=0).astype(np.float32)
        arr_y    = np.asarray(y_all)
        arr_ids  = np.asarray(uuids, dtype=object)

        feat_name = f"features_{tag}_grid_{i}.npz"
        np.savez_compressed(
            out_dir / feat_name,
            full_curves=arr_full,
            y=arr_y,
            uuids=arr_ids,
            channels=np.array(mixes, dtype=object),
            tag=tag,
            timestamps=np.array(timestamps, dtype=object),
        )
        # anexa no registro do trial
        t["featureFile"] = feat_name

    return {
        "forecaster": fct,
        "full_curves": full_curves,  # derivadas do 'melhor' apenas para uso imediato
        "y":          y_all,
        "uuids":      uuids,
        "curves":     curves,
        "channels":  mixes,
        "timestamps": timestamps,
        "tag":        tag,
        "bestParams": best_params,
        "trials": trials,   # {modelFile, params, trainMse, featureFile?}
    }



def train_regressor(
    model_specs: list[dict],
    training_data: dict,
    units: list[str],
    test_size: float = TEST_SIZE,
    augment: str | None = None,
    n_augs: int = 1,
    perm_imp: bool = False,
):
    if not model_specs:
        raise ValueError("lista de modelos vazia")

    full_curves = training_data["full_curves"]
    y_all       = training_data["y"]
    uuids_all   = training_data["uuids"]
    channels    = training_data.get("channels")
    n_channels  = (sum(len(chs) for chs in channels.values())
                   if channels is not None else training_data["n_channels"])
    tag         = training_data["tag"]

    idx_tr, idx_te = train_test_split(
        np.arange(len(full_curves)),
        test_size=test_size,
        random_state=SEED,
    )
    tr_curves = [full_curves[i] for i in idx_tr]
    te_curves = [full_curves[i] for i in idx_te]
    tr_y      = y_all[idx_tr]
    te_y      = y_all[idx_te]

    if augment:
        tr_curves, tr_y, _ = augment_dataset(
            tr_curves, tr_y, [uuids_all[i] for i in idx_tr],
            method=augment, n_augs=n_augs
        )

    X_train = np.stack([c.reshape(-1) for c in tr_curves]).astype(np.float32)
    X_test  = np.stack([c.reshape(-1) for c in te_curves]).astype(np.float32)

    results: list[dict] = []
    path_idx = 0

    for spec in model_specs:
        raw_name = spec.get("name", "")
        name = _canon(raw_name)
        params = spec.get("params") or {}
        grid   = spec.get("grid")

        # validação rígida
        if grid:
            grid = _validate_grid(name, grid, params)
            combos = list(ParameterGrid(grid))
        else:
            if not params:
                raise ValueError(f"{raw_name}: envie 'params' ou 'grid'.")
            params = _validate_single(name, params)
            combos = [{}]

        for combo in combos:
            cur_params = {**params, **combo}

            # Derivação automática só do input MLP (não é hiperparâmetro)
            if name == "mlp" and "mlp_in_features" not in cur_params:
                cur_params["mlp_in_features"] = X_train.shape[1]

            # Constroi regressor
            if name in ("mlp", "cnn", "lstm"):
                from functools import partial
                reg = TorchRegressor(
                    partial(make_regressor, name, n_channels, n_outputs=len(units), **cur_params)
                )
            else:
                reg = make_regressor(name, n_channels, n_outputs=len(units), **cur_params)

            pipe = make_pipeline(RobustScaler(quantile_range=(5,95)), reg)
            pipe.fit(X_train, tr_y)

            preds_te = pipe.predict(X_test)
            if preds_te.ndim == 1:
                preds_te = preds_te.reshape(-1, 1)

            # métricas
            metrics_unit: dict[str, dict[str, float]] = {}
            for i, u in enumerate(units):
                y_true_u = te_y if te_y.ndim==1 else te_y[:,i]
                y_pred_u = preds_te if preds_te.ndim==1 else preds_te[:,i]
                mse_u  = mean_squared_error(y_true_u, y_pred_u)
                rmse_u = float(np.sqrt(mse_u))
                mae_u  = mean_absolute_error(y_true_u, y_pred_u)
                r2_u   = r2_score(y_true_u, y_pred_u)
                metrics_unit[u] = {"mse": mse_u, "rmse": rmse_u, "mae": mae_u, "r2": r2_u}

            mse = float(np.mean([m["mse"] for m in metrics_unit.values()]))
            rmse = float(np.mean([m["rmse"] for m in metrics_unit.values()]))
            mae = float(np.mean([m["mae"] for m in metrics_unit.values()]))
            r2 = float(np.mean([m["r2"] for m in metrics_unit.values()]))

            # importâncias
            imp = None
            reg_step = pipe.named_steps[list(pipe.named_steps.keys())[-1]]
            if hasattr(reg_step, "feature_importances_"):
                imp = reg_step.feature_importances_.tolist()
            elif hasattr(reg_step, "coef_"):
                imp = np.abs(reg_step.coef_).ravel().tolist()
            elif perm_imp:
                imp = compute_perm_importance(pipe, X_test, te_y)

            model_path = MODEL_DIR / f"reg_{tag}_{name}_{path_idx}.pkl"
            joblib.dump(pipe, model_path)
            results.append({
                "name": name,
                "model_file": model_path,
                "params": cur_params,
                "metrics": {
                    "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "per_unit": metrics_unit
                },
                "feature_importances_": imp,
            })
            path_idx += 1

    return results
