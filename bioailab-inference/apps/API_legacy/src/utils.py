"""Funções auxiliares utilizadas no pipeline."""
from __future__ import annotations

import numpy as np
import torch

from .config import WINDOW, TARGET_LEN, DEVICE

from .sensor import MACROS, RAW_BANDS


def parse_sensors(sensor_args: list[str]) -> dict[str, list[str]]:
    """Parse repeated ``-s`` arguments.

    Parameters
    ----------
    sensor_args: list[str]
        List of strings in the form ``SENSOR:ch1,ch2,...``.

    Returns
    -------
    dict[str, list[str]]
        Ordered mapping from sensor to list of unique channels.
    """

    valid_sensors = {"VIS1", "VIS2", "UV", "AUTO"}
    valid_channels = {"R", "G", "B", "C", "M", "Y", "K", "H", "S", "V", *RAW_BANDS}

    mixes: dict[str, list[str]] = {}
    for arg in sensor_args:
        if ":" not in arg:
            raise ValueError("-s deve ser sensor:canal1,canal2,…")
        sensor, ch_txt = arg.split(":", 1)
        sensor = sensor.strip().upper()
        if sensor not in valid_sensors:
            raise ValueError(f"sensor desconhecido: {sensor}")
        if not ch_txt.strip():
            raise ValueError(f"canais ausentes para {sensor}")

        items = []
        for tok in ch_txt.split(","):
            tok = tok.strip().upper()
            if not tok:
                continue
            items.extend(MACROS.get(tok, [tok]))

        seen: list[str] = mixes.get(sensor, [])
        for c in items:
            if c not in valid_channels:
                raise ValueError(f"canal desconhecido: {c}")
            if c not in seen:
                seen.append(c)
        mixes[sensor] = seen

    return mixes




@torch.no_grad()
def complete_series(
    seq: np.ndarray,
    model,
    *,
    window: int = WINDOW,
    horizon: int = 1,
    target_len: int = TARGET_LEN,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """Completa uma série ``seq`` até ``target_len`` pontos usando o forecaster."""
    buf = seq.astype(np.float32).copy()          # (T,C)
    # ───────────────────────────────────────────────────────────────────
    if timestamps is not None and len(timestamps):
        ts_arr = np.asarray(timestamps, dtype=np.int64)
        min_len = min(len(ts_arr), len(buf))
        ts_arr  = ts_arr[:min_len]
        buf     = buf[:min_len]

        # Assume série quase-regular: usa mediana do passo real
        step = int(np.median(np.diff(ts_arr))) if ts_arr.size > 1 else 1
        # grade de referência até onde JÁ existe dado
        full_ts = np.arange(ts_arr[0], ts_arr[-1] + step, step, dtype=np.int64)

        # aloca e forward-fill buracos -----------------------------------
        C        = buf.shape[1]
        full_buf = np.empty((len(full_ts), C), np.float32)
        full_buf[:] = np.nan
        idx      = np.searchsorted(full_ts, ts_arr)
        full_buf[idx] = buf                         # coloca observados

        # preenche NaNs usando o último valor visto em cada coluna
        mask = np.isnan(full_buf[:, 0])
        for c in range(C):
            col = full_buf[:, c]
            last = col[~mask][0] if (~mask).any() else 0.
            for i in range(len(col)):
                if not np.isnan(col[i]):
                    last = col[i]
                else:
                    col[i] = last

        buf = full_buf                              # agora sem buracos
    # ───────────────────────────────────────────────────────────────────



    while buf.shape[0] < target_len:
        win = buf[-window:]
        if win.shape[0] < window:
            pad = np.zeros((window - win.shape[0], buf.shape[1]), np.float32)
            win = np.vstack([pad, win])
        inp = torch.tensor(win[None], device=DEVICE)  # (1,window,C)
        nxt = model(inp).cpu().numpy()[0]             # (horizon,C)
        buf = np.vstack([buf, nxt])
    return buf[:target_len]


def detect_growth_start_bipolar(
    seq,
    ch_idx=0,
    win=10,
    thr_ratio=0.05,
    min_sustain=10,
    skip_initial=30,
) -> int:
    """Detecta início de crescimento pela curva bruta."""
    if seq.shape[0] < skip_initial + win + min_sustain:
        return 0
    raw = seq[:, ch_idx]
    base = np.median(raw[skip_initial : skip_initial + win])
    lim = abs(base) * thr_ratio + 1e-9
    for t in range(skip_initial + win, len(raw) - min_sustain):
        delta = raw[t] - base
        if abs(delta) > lim:
            sign = np.sign(delta)
            if all(sign * (raw[t : t + min_sustain] - base) > lim):
                return t
    return 0
