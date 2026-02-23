"""Plotagem hora a hora do pipeline."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")        # <<< aqui

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .config import TARGET_LEN, MIN_PER_HR, HOURS_TOT
from .utils import complete_series, detect_growth_start_bipolar


def plot_hourly_sim(
    seq,
    model,
    channels,
    uuid="",
    ch_idx=0,
    detect=False,
    thr=0.05,
    save_dir: Path | None = None,
    sensor_colors: dict[str, str] | None = None,
):
    """Plota ou salva as projeções hora a hora."""
    idx = np.arange(TARGET_LEN)
    seq_len = min(len(seq), TARGET_LEN)
    gt = np.full(TARGET_LEN, np.nan, np.float32)
    gt[:seq_len] = seq[:seq_len, ch_idx]

    plt.figure(figsize=(10, 5))
    color = None
    if sensor_colors is not None:
        sensor = channels[ch_idx].split(":")[0]
        color = sensor_colors.get(sensor)
    plt.plot(idx, gt, color or "k", lw=2, label="real")

    t0 = 0
    if detect:
        t0 = detect_growth_start_bipolar(seq, ch_idx=ch_idx, thr_ratio=thr)
    h0 = int(np.ceil(t0 / MIN_PER_HR)) or 1

    for h in range(h0, HOURS_TOT):
        t = h * MIN_PER_HR
        if t > seq_len:
            break
        comp = complete_series(seq[:t], model)[:, ch_idx]
        plt.plot(idx, comp, ls="--", alpha=0.45, label=f"prev {h}h", color=color)
        plt.axvline(t - 1, c="k", ls=":", alpha=0.2)

    plt.title(f"{uuid}  (start≈{t0}min)")
    plt.xlabel("Minutos")
    plt.ylabel(channels[ch_idx])
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ch_label = channels[ch_idx].replace(":", "-")
        fname = save_dir / f"{uuid}_{ch_label}.png"
        plt.savefig(fname)
        plt.close()


def plot_residuals_hist(residuals, model_name: str, save_dir: Path | None = None):
    """Plota histograma de resíduos."""
    plt.figure()
    plt.hist(residuals, bins=30, alpha=0.7, color="tab:blue")
    plt.title(f"Resíduos - {model_name}")
    plt.xlabel("resíduo")
    plt.ylabel("frequência")
    plt.grid(True)
    plt.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fname = Path(save_dir) / f"hist_{model_name}.png"
        plt.savefig(fname)
        plt.close()


def plot_pred_vs_true(y_true, y_pred, model_name: str, save_dir: Path | None = None):
    """Plota curva predito vs real."""
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "k--", lw=1)
    plt.title(f"Predição vs Real - {model_name}")
    plt.xlabel("real")
    plt.ylabel("predito")
    plt.grid(True)
    plt.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fname = Path(save_dir) / f"pred_vs_real_{model_name}.png"
        plt.savefig(fname)
        plt.close()


def plot_rmse_bar(df: pd.DataFrame, save_dir: Path | None = None, mix_str: str = ""):
    """Plota comparação de RMSE entre modelos."""
    plt.figure(figsize=(6, 4))
    plt.bar(df["model"], df["rmse"], color="tab:orange")
    plt.ylabel("RMSE")
    title = "Comparativo de modelos"
    if mix_str:
        title += f" - {mix_str}"
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fname = Path(save_dir) / "rmse_comparison.png"
        plt.savefig(fname)
        plt.close()


def plot_metrics_comparison(results: list[dict], out_dir: Path, mix_str: str = ""):
    """Generate comparison plots for each model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    plot_rmse_bar(df, out_dir, mix_str)
    for r in results:
        if "preds" not in r:
            continue
        model = r.get("model", "model")
        y_true = r["y_true"]
        y_pred = r["preds"]
        plot_pred_vs_true(y_true, y_pred, model, out_dir)
        plot_residuals_hist(y_true - y_pred, model, out_dir)
