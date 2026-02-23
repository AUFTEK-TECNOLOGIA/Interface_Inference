"""Interface utilitário para o sensor AS7341"""
from __future__ import annotations

import numpy as np

# ===== AS7341 basic counts & gain =====
GAIN_ENUM_MAP = {
    0: 0.5,
    1: 1.0,
    2: 2.0,
    3: 4.0,
    4: 8.0,
    5: 16.0,
    6: 32.0,
    7: 64.0,
    8: 128.0,
    9: 256.0,
    10: 512.0,
}

RAW_BANDS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "CLR", "NIR"]
MACROS = {"RGB": ["R", "G", "B"], "RAW": RAW_BANDS}


def get_gain_factor(gain_enum: int) -> float:
    """Retorna o fator de ganho real do AS7341."""
    return GAIN_ENUM_MAP.get(int(gain_enum), 1.0)


def as7341_basic_counts(raw_values, config: dict) -> np.ndarray:
    """Converte leituras em "basic counts" normalizados."""
    atime = config.get("atime", 0)
    astep = config.get("astep", 0)
    gain_enum = config.get("gain", 1)
    gain = get_gain_factor(gain_enum)
    tint = (int(atime) + 1) * (int(astep) + 1) * 2.78
    arr = np.array(raw_values, dtype=float)
    if tint <= 0 or gain <= 0:
        return arr
    return arr / (gain * tint)


# ===== conversão de espectro para RGB =====
XYZ_MATRICES = {
    "VIS": np.array(
        [
            [0.39814, 1.29540, 0.36956, 0.10902, 0.71942, 1.78180, 1.10110, -0.03991, -0.27597, -0.02347],
            [0.01396, 0.16748, 0.23538, 1.42750, 1.88670, 1.14200, 0.46497, -0.02702, -0.24468, -0.01993],
            [1.95010, 6.45490, 2.78010, 0.18501, 0.15325, 0.09539, 0.10563, 0.08866, -0.61140, -0.00938],
        ]
    ),
    "Fluorescência": np.array(
        [
            [0.39814, 1.29540, 0.36956, 0.10902, 0.71942, 1.78180, 1.10110, -0.03991, 0.0, 0.0],
            [0.01396, 0.16748, 0.23538, 1.42750, 1.88670, 1.14200, 0.46497, -0.02702, 0.0, 0.0],
            [1.95010, 6.45490, 2.78010, 0.18501, 0.15325, 0.09539, 0.10563, 0.08866, 0.0, 0.0],
        ]
    ),
}


def correct_gamma(channel: float) -> float:
    """Aplica correção de gama sRGB."""
    if channel <= 0.0031308:
        return 12.92 * channel
    return 1.055 * (channel ** (1 / 2.4)) - 0.055


def spectrum_to_xyz(spectrum: list[float], matrix: np.ndarray) -> np.ndarray:
    """Converte espectro (10 bandas) em coordenadas XYZ normalizadas."""
    spec = np.array(spectrum, dtype=float)
    xyz = matrix.dot(spec)
    total = xyz.sum()
    if total > 0:
        xyz = xyz / total
    return xyz


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    """Converte XYZ para sRGB linear e aplica gama."""
    x, y, z = xyz
    r_lin = 3.2406 * x - 1.5372 * y - 0.4986 * z
    g_lin = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b_lin = 0.0557 * x - 0.2040 * y + 1.0570 * z

    r = r_lin
    g = g_lin
    b = b_lin
    return np.array([r, g, b])


def generate_rgb_colors(
    data: dict[str, list[float]],
    timestamps: list[float],
    is_inverse: bool = False,
    type: str = "VIS",
) -> list[tuple[int, int, int]]:
    """Gera lista de cores RGB para cada timestamp."""
    matrix = XYZ_MATRICES.get(type, XYZ_MATRICES["VIS"])
    length = len(timestamps)
    colors = []
    bands = RAW_BANDS
    for i in range(length):
        spectrum = [data.get(b, [0] * length)[i] for b in bands]
        xyz = spectrum_to_xyz(spectrum, matrix)
        rgb = xyz_to_rgb(xyz)
        r, g, b = rgb
        colors.append((r, g, b))
    if is_inverse:
        colors.reverse()
    return colors


def xyz_to_rgb_gamma(xyz_values: np.ndarray) -> dict:
    """Converte XYZ normalizado para RGB aplicando correção de gama."""
    x, y, z = xyz_values

    r_linear = 3.2406 * x - 1.5372 * y - 0.4986 * z
    g_linear = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b_linear = 0.0557 * x - 0.2040 * y + 1.0570 * z

    r = correct_gamma(r_linear)
    g = correct_gamma(g_linear)
    b = correct_gamma(b_linear)

    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)

    return {"R": r, "G": g, "B": b}


def rgb_to_cmyk(rgb_values: dict) -> dict:
    """Converte valores RGB para o espaço CMYK."""
    r = rgb_values["R"]
    g = rgb_values["G"]
    b = rgb_values["B"]

    k = 1 - max(r, g, b)
    if k < 1:
        c = (1 - r - k) / (1 - k)
        m = (1 - g - k) / (1 - k)
        y = (1 - b - k) / (1 - k)
    else:
        c = m = y = 0

    return {"C": c, "M": m, "Y": y, "K": k}


def rgb_to_hsb(rgb_values: dict) -> dict:
    """Converte valores RGB para o espaço HSB/HSV."""
    r = rgb_values["R"]
    g = rgb_values["G"]
    b = rgb_values["B"]

    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val

    h = 0.0
    if delta != 0:
        if max_val == r:
            h = ((g - b) / delta) % 6
        elif max_val == g:
            h = ((b - r) / delta) + 2
        else:
            h = ((r - g) / delta) + 4
        h *= 60

    s = 0.0 if max_val == 0 else delta / max_val
    v = max_val

    return {"H": h, "S": s, "V": v}


def xyz_to_lab(xyz_abs_values: np.ndarray) -> dict:
    """Converte coordenadas XYZ absolutas para o espaço LAB."""
    x, y, z = xyz_abs_values

    x_ref, y_ref, z_ref = 95.047, 100.0, 108.883

    x = x / x_ref
    y = y / y_ref
    z = z / z_ref

    x = lab_f(x)
    y = lab_f(y)
    z = lab_f(z)

    l = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    return {"L*": l, "a*": a, "b*": b}


def lab_f(t: float) -> float:
    """Função auxiliar para o cálculo de valores LAB."""
    delta = 6 / 29
    if t > delta ** 3:
        return t ** (1 / 3)
    return (t / (3 * delta ** 2)) + (4 / 29)
