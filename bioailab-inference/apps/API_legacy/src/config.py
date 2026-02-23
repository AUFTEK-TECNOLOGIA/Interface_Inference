# src/config.py

import torch
from pathlib import Path

# Detecta automaticamente se CUDA está disponível
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Mapeia categoria espectral → chave em light_sensor_config
SPECTRAL_MAP = {
    "spectral_uv":     "uv",
    "spectral_vis_1":  "visible_1",
    "spectral_vis_2":  "visible_2"
}

# Chaves que não são canais numéricos
META_KEYS = {
    'experiment_UUID',
    'serial_number',
    'calibration',
    'features',
    'general_info',
    'light_sensor_config',
    'timestamps'
}

# ───────────────── seeds ───────────────────────────────────────────────
SEED       = 42

# ── tempo / janelas ─────────────────────────────────────────────────────
MIN_PER_HR = 60
HOURS_TOT  = 24
TARGET_LEN = MIN_PER_HR * HOURS_TOT      # 1440
WINDOW     = 180                         # minutos de histórico
HORIZON    = 60                          # minutos de previsão

# ── forecaster (LSTM) ───────────────────────────────────────────────────
FCT_HIDDEN = 64
FCT_LAYERS = 2
BATCH_FC   = 256
EPOCHS_FC  = 5 #padraõ 30

# ── regressão – defaults ────────────────────────────────────────────────
BATCH_REG   = 64
EPOCHS_REG  = 500

# ── otimização / split ─────────────────────────────────────────────────
LR         = 1e-3
TEST_SIZE  = 0.20

# ── device / dirs ───────────────────────────────────────────────────────
# DEVICE foi definido de acordo com a disponibilidade de GPU
MODEL_DIR  = Path("models")
