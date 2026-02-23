"""Carregamento e pré-processamento dos JSONs de experimentos."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np
from typing import Any, Mapping, List, Tuple

from .sensor import (
    generate_rgb_colors,
    RAW_BANDS,
    rgb_to_cmyk,
    rgb_to_hsb,
)
from .sensor_as7341 import normalize_block


def load_experiments(path_json: Path) -> list[dict]:
    """Load experiment list from a wrapper JSON file.

    Each wrapper element must contain a ``body`` field with the experiment data
    serialized as JSON. Any entries missing the field or containing invalid JSON
    are skipped with a warning.
    """
    with open(path_json, "r", encoding="utf-8") as f:
        wrappers = json.load(f)

    experiments = []
    for idx, w in enumerate(wrappers):
        body = w.get("body", "")
        if not body:
            logging.warning("Experiment wrapper %d missing body", idx)
            continue
        try:
            experiments.append(json.loads(body))
        except json.JSONDecodeError:
            logging.warning("Invalid JSON in experiment wrapper %d", idx)
            continue

    return experiments


def load_dataset(
    sensor: str,
    channels: list[str],
    units: list[str],
    json_path: Path,
    color_model: str = "RGB",
    slice_start: int = 0,
    slice_end: int = 0,

) -> tuple:
    """Load dataset applying optional color model conversion.

    The ``color_model`` argument is preserved for backward compatibility but
    mixed channels no longer require specifying a single model. The function
    now converts RGB values to CMYK/HSV on demand based on ``channels``.
    """
    curves, ylog, uuids, timestamps = [], [], [], []
    for exp in load_experiments(json_path):
        cal_vals = []
        for u in units:
            cal = next(
                (c for c in exp.get("calibration", {}).values() if c.get("unit") == u and c.get("count", 0) > 0),
                None,
            )
            if cal is None:
                cal_vals = []
                break
            cal_vals.append(np.log10(cal["count"] + 1))
        if not cal_vals:
            continue
        ylog.append(cal_vals)
        uuids.append(exp.get("experiment_UUID", ""))

        ts = exp.get("timestamps", [])
        cfg_v2 = exp.get("light_sensor_config", {}).get("visible_2", {})

        blocks = {
            "VIS2": exp.get("spectral_vis_2", {}),
            "VIS1": exp.get("spectral_vis_1", {}),
            "UV": exp.get("spectral_uv", {}),
        }
        blk = blocks.get(sensor, {}) if sensor != "AUTO" else next(
            (blocks[k] for k in ("VIS2", "VIS1", "UV") if isinstance(blocks[k], dict)),
            {},
        )
        if not isinstance(blk, dict):
            continue

        blk_norm = normalize_block(blk, cfg_v2)

        rgb = None
        rgb_needed = any(
            c in ("R", "G", "B", "C", "M", "Y", "K", "H", "S", "V") for c in channels
        )
        if rgb_needed:
            rgb = np.asarray(generate_rgb_colors(blk_norm, ts, False, "VIS"), float)
            if rgb.size == 0:
                continue

        color_data: dict[str, np.ndarray] = {}
        if rgb is not None:
            if any(c in ("R", "G", "B") for c in channels):
                color_data.update({"R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2]})
            if any(c in ("C", "M", "Y", "K") for c in channels):
                cmyk = np.array(
                    [list(rgb_to_cmyk({"R": r, "G": g, "B": b}).values()) for r, g, b in rgb],
                    float,
                )
                color_data.update({"C": cmyk[:, 0], "M": cmyk[:, 1], "Y": cmyk[:, 2], "K": cmyk[:, 3]})
            if any(c in ("H", "S", "V") for c in channels):
                hsv = np.array(
                    [list(rgb_to_hsb({"R": r, "G": g, "B": b}).values()) for r, g, b in rgb],
                    float,
                )
                color_data.update({"H": hsv[:, 0], "S": hsv[:, 1], "V": hsv[:, 2]})

        cols = []
        try:
            for c in channels:
                if c in color_data:
                    cols.append(color_data[c])
                elif c in RAW_BANDS:
                    cols.append(blk_norm.get(c, np.array([], float)))
                else:
                    raise KeyError
        except KeyError:
            continue

        mat = np.column_stack(cols).astype(np.float32)
        ts_arr = np.asarray(ts, dtype=np.int64)
        if slice_start or slice_end:
            end_idx = len(mat) - slice_end if slice_end > 0 else len(mat)
            mat = mat[slice_start:end_idx]
            ts_arr = ts_arr[slice_start:end_idx]

        curves.append(mat.astype(np.float32))
        timestamps.append(ts_arr)

    if not ylog:
        return [], np.array([]), [], []

    ylog = np.asarray(ylog, float)
    # mediana por coluna (suporta multi-saída)
    med = np.median(ylog, axis=0)
    # desvio absoluto mediano por coluna, substituindo zeros por 1e-6
    mad = np.median(np.abs(ylog - med), axis=0)
    mad = np.where(mad == 0, 1e-6, mad)
    # filtra ensaios dentro de 3*MAD em todas as colunas
    keep = np.all(np.abs(ylog - med) <= 3 * mad, axis=1)

    return (
        [c for c, k in zip(curves, keep) if k],
        ylog[keep],
        [u for u, k in zip(uuids, keep) if k],
        [t for t, k in zip(timestamps, keep) if k],
    )



def load_dataset_from_obj(
    sensor: str,
    channels: List[str],
    units: List[str],
    exp: Mapping[str, Any],
    color_model: str = "RGB",
    slice_start: int = 0,
    slice_end: int = 0,
) -> Tuple[List[np.ndarray], np.ndarray, List[str], List[np.ndarray]]:
    """
    Versão 'single-experiment' do ``load_dataset`` sem normalização min-max.

    Retorna:
      curves: [np.ndarray(T, n_ch)]
      ylog:   np.ndarray(1, n_targets) OU np.array([]) se não houver calibração
      uuids:  [str]
      timestamps: [np.ndarray(T)]
    """
    # --- calibração (igual ao load_dataset) --------------------------------
    cal_vals = []
    for u in units:
        cal = next(
            (c for c in exp.get("calibration", {}).values()
             if c.get("unit") == u and c.get("count", 0) > 0),
            None,
        )
        if cal is None:
            cal_vals = []
            break
        cal_vals.append(np.log10(cal["count"] + 1))

    if cal_vals:
        ylog = np.asarray([cal_vals], dtype=float)  # (1, n_targets)
    else:
        ylog = np.array([])

    uuid = exp.get("experiment_UUID", "")

    # --- blocos e normalização do bloco -----------------------------------
    ts = exp.get("timestamps", [])
    cfg_v2 = exp.get("light_sensor_config", {}).get("visible_2", {})

    blocks = {
        "VIS2": exp.get("spectral_vis_2", {}),
        "VIS1": exp.get("spectral_vis_1", {}),
        "UV":   exp.get("spectral_uv", {}),
    }
    blk = blocks.get(sensor, {}) if sensor != "AUTO" else next(
        (blocks[k] for k in ("VIS2", "VIS1", "UV") if isinstance(blocks[k], dict)),
        {},
    )
    if not isinstance(blk, dict):
        return [], np.array([]), [], []

    blk_norm = normalize_block(blk, cfg_v2)

    # --- cores derivadas sob demanda (igual ao load_dataset) ---------------
    rgb = None
    rgb_needed = any(c in ("R", "G", "B", "C", "M", "Y", "K", "H", "S", "V") for c in channels)
    if rgb_needed:
        rgb = np.asarray(generate_rgb_colors(blk_norm, ts, False, "VIS"), float)
        if rgb.size == 0:
            return [], np.array([]), [], []

    color_data: dict[str, np.ndarray] = {}
    if rgb is not None:
        if any(c in ("R", "G", "B") for c in channels):
            color_data.update({"R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2]})
        if any(c in ("C", "M", "Y", "K") for c in channels):
            cmyk = np.array([list(rgb_to_cmyk({"R": r, "G": g, "B": b}).values()) for r, g, b in rgb], float)
            color_data.update({"C": cmyk[:, 0], "M": cmyk[:, 1], "Y": cmyk[:, 2], "K": cmyk[:, 3]})
        if any(c in ("H", "S", "V") for c in channels):
            hsv = np.array([list(rgb_to_hsb({"R": r, "G": g, "B": b}).values()) for r, g, b in rgb], float)
            color_data.update({"H": hsv[:, 0], "S": hsv[:, 1], "V": hsv[:, 2]})

    # --- seleção de colunas ------------------------------------------------
    cols = []
    try:
        for c in channels:
            if c in color_data:
                cols.append(color_data[c])
            elif c in RAW_BANDS:
                arr = blk_norm.get(c, np.array([], float))
                if arr.size == 0:
                    raise KeyError
                cols.append(arr)
            else:
                raise KeyError
    except KeyError:
        return [], np.array([]), [], []

    mat = np.column_stack(cols).astype(np.float32)
    ts_arr = np.asarray(ts, dtype=np.int64)

    if slice_start or slice_end:
        end_idx = len(mat) - slice_end if slice_end > 0 else len(mat)
        mat = mat[slice_start:end_idx]
        ts_arr = ts_arr[slice_start:end_idx]

    curves = [mat.astype(np.float32)]
    uuids = [uuid]
    timestamps = [ts_arr]

    return curves, ylog, uuids, timestamps
