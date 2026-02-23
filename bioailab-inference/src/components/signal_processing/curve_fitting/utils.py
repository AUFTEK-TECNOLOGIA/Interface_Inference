"""
Utilitários para ajuste de curvas.
"""

import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter1d

# Re-exportar do módulo normalizers para compatibilidade
from ..normalizers import normalize_data, denormalize_data


def dedupe_and_sort(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ordena e remove duplicatas (agregando por média).
    
    Args:
        x: Array de timestamps
        y: Array de valores
    
    Returns:
        Tupla (x_unique, y_aggregated)
    """
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    
    # Agrega duplicatas
    unique_x = []
    unique_y = []
    i = 0
    
    while i < len(x):
        j = i + 1
        acc = y[i]
        count = 1
        
        while j < len(x) and np.isclose(x[j], x[i]):
            acc += y[j]
            count += 1
            j += 1
        
        unique_x.append(x[i])
        unique_y.append(acc / count)
        i = j
    
    return np.array(unique_x), np.array(unique_y)


def determine_window(
    x: np.ndarray,
    y: np.ndarray,
    threshold_start: float = 0.05,
    threshold_end: float = 0.05,
    smooth_sigma: float = 5.0,
    min_window_size: int = 20
) -> Tuple[float, float]:
    """
    Detecta a janela de interpolação baseada no pico da derivada.
    
    Args:
        x: Array de timestamps
        y: Array de valores
        threshold_start: Threshold para início da janela (% do pico)
        threshold_end: Threshold para fim da janela (% do pico)
        smooth_sigma: Sigma para suavização gaussiana
        min_window_size: Tamanho mínimo da janela
    
    Returns:
        Tupla (window_start, window_end) ou (None, None) se falhar
    """
    if len(y) < min_window_size:
        return None, None
    
    x = np.array(x, dtype=float).flatten()
    y = np.array(y, dtype=float).flatten()
    
    # Deduplica e ordena
    x, y = dedupe_and_sort(x, y)
    
    if len(x) < min_window_size:
        return None, None
    
    # Suaviza e calcula derivada
    y_smooth = gaussian_filter1d(y, sigma=smooth_sigma)
    dy = np.gradient(y_smooth, x)
    dy_smooth = gaussian_filter1d(dy, sigma=smooth_sigma)
    dy_abs = np.abs(dy_smooth)
    
    # Encontra pico da derivada
    max_idx = np.argmax(dy_abs)
    max_value = dy_abs[max_idx]
    
    if max_value <= 0:
        return None, None
    
    # Encontra limites da janela
    start_idx = 0
    for i in range(max_idx, 0, -1):
        if dy_abs[i] <= threshold_start * max_value:
            start_idx = i
            break
    
    end_idx = len(x) - 1
    for i in range(max_idx, len(x)):
        if dy_abs[i] <= threshold_end * max_value:
            end_idx = i
            break
    
    # Garante tamanho mínimo
    if end_idx - start_idx < min_window_size:
        center = (start_idx + end_idx) // 2
        half = min_window_size // 2
        start_idx = max(0, center - half)
        end_idx = min(len(x) - 1, center + half)
    
    return x[start_idx], x[end_idx]
