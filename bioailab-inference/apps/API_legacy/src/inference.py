from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Tuple, Dict, List, Mapping, Any

import numpy as np
import torch
import joblib

from src.utils   import complete_series
from src.loader  import load_dataset_from_obj         # ← novo
from src.models  import make_forecaster
from src.config  import DEVICE

MODEL_DIR      = Path("models")
FORECASTER_DIR = MODEL_DIR

def _compute_tag(sensors: Dict[str, list[str]], units: list[str]) -> str:
    canon = "|".join(f"{s}-{','.join(chs)}" for s, chs in sensors.items())
    unit_str = ",".join(units)
    return hashlib.md5(f"{canon}-{unit_str}".encode()).hexdigest()[:6]


def _build_seq_from_experiment(
    ensaio: Mapping[str, Any],
    sensors: Dict[str, List[str]],
    units:   List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Empilha (hstack) todos os sensores/canais de UM experimento."""
    seqs = []
    ts_ref = None
    T_ref = None

    for sensor, channels in sensors.items():
        curves, _, _, ts_list = load_dataset_from_obj(
            sensor=sensor, channels=channels, units=units, exp=ensaio,
            slice_start=0, slice_end=0
        )
        if not curves:
            raise ValueError(f"Nenhum dado válido para '{sensor}:{','.join(channels)}'.")

        curve = curves[0]                 # (T, n_ch_sensor)

        if T_ref is None:
            T_ref = curve.shape[0]
        elif curve.shape[0] != T_ref:
            raise ValueError(
                f"Comprimento inconsistente: T={curve.shape[0]} em '{sensor}', esperado T={T_ref}."
            )

        ts_cur = ts_list[0]
        if ts_ref is None:
            ts_ref = ts_cur
        elif not np.array_equal(ts_cur, ts_ref):
            raise ValueError("timestamps inconsistentes entre sensores")

        seqs.append(curve)

    seq = np.hstack(seqs).astype(np.float32)     # (T, n_total_ch)
    return seq, ts_ref


def inference(
    ensaio:           Mapping[str, Any],
    sensors:          Dict[str, List[str]],
    units:            List[str],
    model_name:       str,
    model_file:       str | None = None,
    forecaster_file:  str | None = None,
    slice_start:      int   = 0,
    device:           str   = DEVICE,
) -> Tuple[np.ndarray, float, np.ndarray, int]:
    """
    Retorna
      y_full           – série prevista ``target_len``×n_ch
      y_regression     – saída do regressor (float)
      y_true_partial   – parte observada (slice_start:growth_idx)
      growth_idx       – minuto detectado (aqui = len(seq))
    """
    # 1) tag consistente com o treino
    tag = _compute_tag(sensors, units)

    # 2) Carrega checkpoint do forecaster (apenas para pesos/arquitetura)
    fct_path = (
        (FORECASTER_DIR / forecaster_file) if forecaster_file
        else (FORECASTER_DIR / f"fct_{tag}.pth")
    )
    ckpt = torch.load(fct_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise RuntimeError(f"Checkpoint {fct_path} incompatível.")

    # Metadados do forecaster (com fallbacks razoáveis)
    hidden     = int(ckpt.get("hidden",       64))
    layers     = int(ckpt.get("layers",       2))
    dropout    = float(ckpt.get("dropout",    0.0))
    bidir      = bool(ckpt.get("bidirectional", False))
    window     = int(ckpt.get("window",       60))
    horizon    = int(ckpt.get("horizon",      1))
    target_len = int(ckpt.get("targetLen",    1440))

    # 3) Sequência do próprio ensaio
    seq, ts = _build_seq_from_experiment(
        ensaio, sensors, units
    )
    n_channels = seq.shape[1]

    # 4) Reconstroi forecaster com a mesma arquitetura e carrega pesos
    fct = make_forecaster(
        n_channels,
        hidden,
        layers,
        horizon=horizon,
        dropout=dropout,
        bidirectional=bidir,
    ).to(device)
    fct.load_state_dict(ckpt["model_state"])
    fct.eval()

    # 5) Índice de crescimento (aqui não há detecção automática)
    growth_idx = len(seq)

    # 6) Parte observada bruta
    y_true_partial = seq[slice_start:growth_idx]

    # 7) Completa série diretamente em valores brutos
    y_full = complete_series(
        seq,
        fct,
        window=window,
        horizon=horizon,
        target_len=target_len,
        timestamps=None,
    )

    # 8) Regressor (.pkl). Aceita nome direto ou busca por tag+nome
    if str(model_name).lower() in {"noop", "none", "skip", ""}:
        return y_full, float("nan"), y_true_partial, growth_idx

    if model_file:
        reg_path = MODEL_DIR / model_file
        if not reg_path.exists():
            raise FileNotFoundError(f"Regressor não encontrado: {reg_path}")
    else:
        # 1) novo padrão com sufixo
        cands = sorted(MODEL_DIR.glob(f"reg_{tag}_{model_name}_*.pkl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            reg_path = cands[0]
        else:
            # 2) compat: arquivo sem sufixo
            legacy = MODEL_DIR / f"reg_{tag}_{model_name}.pkl"
            if legacy.exists():
                reg_path = legacy
            else:
                raise FileNotFoundError(
                    f"Regressor não encontrado: {MODEL_DIR / f'reg_{tag}_{model_name}_*.pkl'} "
                    f"(ou {legacy})"
                )

    pipe = joblib.load(reg_path)
    X = y_full.reshape(1, -1).astype(np.float32)
    y_reg = float(pipe.predict(X)[0])

    return y_full, y_reg, y_true_partial, growth_idx

