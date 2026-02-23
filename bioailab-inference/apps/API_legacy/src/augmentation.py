import numpy as np
from scipy.interpolate import CubicSpline


def jitter(x: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """
    Adiciona ruído Gaussiano proporcional ao desvio padrão de cada canal.
    """
    noise = np.random.normal(loc=0.0, scale=sigma * np.std(x, axis=0), size=x.shape)
    return x + noise


def magnitude_warp(x: np.ndarray, knots: int = 4, sigma: float = 0.2) -> np.ndarray:
    """
    Aplica uma curva suave multiplicativa aos valores da série.
    """
    N, D = x.shape
    orig_steps = np.arange(N)
    # gera valores aleatórios nos knots
    warp_steps = np.linspace(0, N - 1, num=knots + 2)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knots + 2, D))
    # interpola spline para cada canal
    warper = np.vstack([
        CubicSpline(warp_steps, random_warps[:, d])(orig_steps)
        for d in range(D)
    ]).T
    return x * warper


def time_warp(x: np.ndarray, knots: int = 4, sigma: float = 0.2) -> np.ndarray:
    """
    Deforma o eixo temporal seguindo uma curva suave.
    """
    N, D = x.shape
    orig_steps = np.arange(N)
    warp_steps = np.linspace(0, N - 1, num=knots + 2)
    random_curves = np.random.normal(loc=1.0, scale=sigma, size=(knots + 2,))
    time_warp = CubicSpline(warp_steps, random_curves)(orig_steps)
    # normaliza e acumula
    cumwarp = np.cumsum(time_warp)
    cumwarp = (cumwarp - cumwarp.min()) / (cumwarp.max() - cumwarp.min()) * (N - 1)
    # reamostra para o grid original
    warped = np.vstack([
        np.interp(orig_steps, cumwarp, x[:, d]) for d in range(D)
    ]).T
    return warped


def window_slice(x: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
    """
    Corta uma janela aleatória de comprimento reduzido e reamostra ao tamanho original.
    """
    N, D = x.shape
    win_len = int(np.ceil(N * reduce_ratio))
    start = np.random.randint(0, N - win_len + 1)
    window = x[start:start + win_len]
    # interpola de volta a N
    new_idx = np.linspace(0, win_len - 1, num=N)
    return np.vstack([
        np.interp(new_idx, np.arange(win_len), window[:, d]) for d in range(D)
    ]).T


def permute_segments(x: np.ndarray, n_segments: int = 4) -> np.ndarray:
    """
    Divide a série em segmentos e embaralha a ordem.
    """
    N, D = x.shape
    # pontos de corte
    cuts = np.linspace(0, N, n_segments + 1, dtype=int)
    segments = [x[cuts[i]:cuts[i+1]] for i in range(n_segments)]
    np.random.shuffle(segments)
    return np.vstack(segments)


def mixup(x: np.ndarray, y: np.ndarray, alpha: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Combina dois exemplos (x,y) e (x',y') segundo peso lam ~ Beta(alpha,alpha).
    """
    lam = np.random.beta(alpha, alpha)
    # escolhe segundo exemplo aleatório
    idx = np.random.randint(0, len(x)) if isinstance(x, np.ndarray) and x.ndim == 3 else None
    # se x for 2D (um exemplo), idx virá externo em apply_augmentation
    return lam * x + (1 - lam) * x, lam * y + (1 - lam) * y


def apply_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    uuids: list[str],
    method: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Gera dados aumentados de X,y, retornando também novos UUIDs.

    method em {{'jitter','magnitude_warp','time_warp',
                'window_slice','permute_segments','mixup'}}
    """
    X_aug, y_aug, ids_aug = [], [], []
    rng = np.random.RandomState()
    for i, (xi, yi, ui) in enumerate(zip(X, y, uuids)):
        if method == 'jitter':
            X_aug.append(jitter(xi))
            y_aug.append(yi)
            ids_aug.append(f"{ui}_jitter")
        elif method == 'magnitude_warp':
            X_aug.append(magnitude_warp(xi))
            y_aug.append(yi)
            ids_aug.append(f"{ui}_magwarp")
        elif method == 'time_warp':
            X_aug.append(time_warp(xi))
            y_aug.append(yi)
            ids_aug.append(f"{ui}_timewarp")
        elif method == 'window_slice':
            X_aug.append(window_slice(xi))
            y_aug.append(yi)
            ids_aug.append(f"{ui}_winslice")
        elif method == 'permute_segments':
            X_aug.append(permute_segments(xi))
            y_aug.append(yi)
            ids_aug.append(f"{ui}_permseg")
        elif method == 'mixup':
            # escolhe outro índice aleatório
            j = rng.randint(len(X))
            x2, y2 = X[j], y[j]
            lam = rng.beta(0.2, 0.2)
            xm = lam * xi + (1 - lam) * x2
            ym = lam * yi + (1 - lam) * y2
            X_aug.append(xm)
            y_aug.append(ym)
            ids_aug.append(f"{ui}_mixup_{j}")
    if not X_aug:
        return X, y, uuids
    # concatena originais + augmentados
    X_all = np.vstack([X, np.stack(X_aug)])
    y_all = np.concatenate([y, np.stack(y_aug)])
    ids_all = uuids + ids_aug
    return X_all, y_all, ids_all


def augment_dataset(
    curves: list[np.ndarray],
    ylog: np.ndarray,
    uuids: list[str],
    method: str,
    n_augs: int
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Gera n_augs cópias augmentadas para cada série."""
    all_curves = list(curves)
    all_y    = list(ylog)
    all_ids  = list(uuids)
    for _ in range(n_augs):
        X_aug, y_aug, ids_aug = apply_augmentation(
            np.stack(curves),
            np.asarray(ylog),
            uuids,
            method
        )
        # apply_augmentation já retorna originais+augmentados,
        # mas queremos só as parte novas:
        new_start = len(curves)
        all_curves.extend(X_aug[new_start:])
        all_y     .extend(y_aug[new_start:])
        all_ids   .extend(ids_aug[new_start:])
    return all_curves, np.array(all_y), all_ids
