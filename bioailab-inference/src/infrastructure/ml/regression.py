"""
Módulo de Regressões Matemáticas

Suporta ajuste e predição de regressões simples como alternativa a modelos ML.
Os coeficientes são salvos no metadata.json e podem ser usados nos blocos de inferência.

Tipos suportados:
- linear: y = a*x + b
- quadratic: y = a*x² + b*x + c
- exponential: y = a * exp(b*x) + c
- logarithmic: y = a * ln(x) + b
- power: y = a * x^b + c
- polynomial: y = aₙxⁿ + ... + a₁x + a₀

Remoção de outliers:
- RANSAC: Random Sample Consensus para robustez contra outliers
- IQR: Interquartile Range para detecção estatística de outliers
"""

from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np
from scipy import optimize
from scipy import stats
import warnings

# Tipos de regressão suportados
RegressionType = Literal["linear", "quadratic", "exponential", "logarithmic", "power", "polynomial"]

SUPPORTED_REGRESSIONS = ["linear", "quadratic", "exponential", "logarithmic", "power", "polynomial"]

# Métodos de remoção de outliers
OutlierMethod = Literal["none", "ransac", "iqr", "zscore"]
SUPPORTED_OUTLIER_METHODS = ["none", "ransac", "iqr", "zscore"]


@dataclass
class RegressionResult:
    """Resultado do ajuste de uma regressão."""
    
    regression_type: str
    coefficients: dict[str, float]
    equation: str
    r2_score: float
    rmse: float
    mae: float
    n_samples: int
    success: bool = True
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    
    # Para visualização
    x_range: tuple[float, float] | None = None
    
    # Informações de outliers
    outlier_method: str = "none"
    n_outliers_removed: int = 0
    outlier_indices: list[int] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Converte para dicionário (para salvar no metadata.json)."""
        return {
            "model_type": self.regression_type,
            "regression": {
                "equation": self.equation,
                "coefficients": self.coefficients,
                "r2_score": round(self.r2_score, 6),
                "rmse": round(self.rmse, 6),
                "mae": round(self.mae, 6),
                "n_samples": self.n_samples,
                "x_range": list(self.x_range) if self.x_range else None,
                "outlier_method": self.outlier_method,
                "n_outliers_removed": self.n_outliers_removed,
            },
            "success": self.success,
            "error": self.error,
            "warnings": self.warnings,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RegressionResult":
        """Reconstrói a partir de um dicionário."""
        reg = data.get("regression", {})
        return cls(
            regression_type=data.get("model_type", "unknown"),
            coefficients=reg.get("coefficients", {}),
            equation=reg.get("equation", ""),
            r2_score=reg.get("r2_score", 0.0),
            rmse=reg.get("rmse", 0.0),
            mae=reg.get("mae", 0.0),
            n_samples=reg.get("n_samples", 0),
            success=data.get("success", True),
            error=data.get("error"),
            warnings=data.get("warnings", []),
            x_range=tuple(reg["x_range"]) if reg.get("x_range") else None,
        )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Calcula R², RMSE e MAE."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    return r2, rmse, mae


# =============================================================================
# Funções de Remoção de Outliers
# =============================================================================

def _detect_outliers_iqr_1d(
    values: np.ndarray, 
    multiplier: float = 1.5
) -> np.ndarray:
    """
    Detecta outliers em 1D usando o método IQR (Interquartile Range).
    
    Args:
        values: Array de valores
        multiplier: Multiplicador do IQR (1.5 = padrão, 3.0 = extremos)
    
    Returns:
        Máscara booleana (True = inlier, False = outlier)
    """
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    if iqr < 1e-10:
        # Se IQR muito pequeno, usar MAD
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad < 1e-10:
            return np.ones(len(values), dtype=bool)
        lower_bound = median - multiplier * 2.5 * mad
        upper_bound = median + multiplier * 2.5 * mad
    else:
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
    
    return (values >= lower_bound) & (values <= upper_bound)


def _detect_outliers_combined(
    x: np.ndarray,
    y: np.ndarray,
    multiplier: float = 2.0
) -> np.ndarray:
    """
    Detecta outliers considerando X e resíduos da regressão.
    
    Abordagem conservadora:
    1. Detecta outliers MUITO extremos em X (usando multiplicador maior)
    2. Ajusta regressão robusta nos pontos restantes
    3. Detecta outliers nos resíduos
    
    Um ponto é outlier se:
    - For extremamente fora do range em X (IQR * 2.5), OU
    - Tiver resíduo muito alto após ajuste robusto
    
    Args:
        x: Array de features
        y: Array de targets
        multiplier: Multiplicador do IQR (2.0 = mais conservador)
    
    Returns:
        Máscara booleana (True = inlier, False = outlier)
    """
    n = len(x)
    
    # 1. Detectar outliers MUITO extremos em X (usar multiplicador maior)
    # Esses são pontos claramente fora da distribuição principal
    x_extreme_outliers = ~_detect_outliers_iqr_1d(x, multiplier * 1.5)  # 3.0 IQR
    
    # 2. Ajustar regressão robusta em pontos não-extremos
    initial_mask = ~x_extreme_outliers
    
    if np.sum(initial_mask) >= 3:
        x_init = x[initial_mask]
        y_init = y[initial_mask]
        
        # Usar regressão robusta (Theil-Sen)
        try:
            slope, intercept = _theil_sen_fit(x_init, y_init)
        except:
            slope, intercept = np.polyfit(x_init, y_init, 1)
        
        # Calcular resíduos em todos os pontos
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        # Detectar outliers nos resíduos (mais conservador)
        residual_outliers = ~_detect_outliers_iqr_1d(residuals, multiplier)
        
        # Ponto é outlier se:
        # - For extremo em X, OU
        # - Tiver resíduo muito alto E não estiver no cluster principal de X
        x_moderate_outliers = ~_detect_outliers_iqr_1d(x, multiplier)  # 2.0 IQR
        
        # Outlier = extremo em X OU (resíduo alto E X moderadamente fora)
        outlier_mask = x_extreme_outliers | (residual_outliers & x_moderate_outliers)
        inliers_mask = ~outlier_mask
    else:
        # Poucos pontos - apenas remover extremos em X
        inliers_mask = ~x_extreme_outliers
    
    return inliers_mask


def _theil_sen_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Ajuste robusto usando o estimador Theil-Sen (mediana das inclinações).
    Muito mais robusto contra outliers que mínimos quadrados.
    """
    n = len(x)
    if n < 2:
        return 0.0, np.median(y)
    
    # Calcular todas as inclinações entre pares de pontos
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(x[j] - x[i]) > 1e-10:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    
    if not slopes:
        return 0.0, np.median(y)
    
    # Mediana das inclinações
    slope = np.median(slopes)
    
    # Intercepto: mediana de (y - slope * x)
    intercept = np.median(y - slope * x)
    
    return slope, intercept


def _detect_outliers_zscore(
    residuals: np.ndarray, 
    threshold: float = 2.5
) -> np.ndarray:
    """
    Detecta outliers usando Z-score.
    
    Args:
        residuals: Array de resíduos
        threshold: Limite do Z-score (2.5-3.0 são valores comuns)
    
    Returns:
        Máscara booleana (True = inlier, False = outlier)
    """
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    
    if std_res < 1e-10:
        return np.ones(len(residuals), dtype=bool)
    
    z_scores = np.abs((residuals - mean_res) / std_res)
    return z_scores <= threshold


def _ransac_regression(
    x: np.ndarray,
    y: np.ndarray,
    fit_func,
    predict_func,
    n_iterations: int = 100,
    sample_ratio: float = 0.5,
    inlier_threshold: float | None = None,
    min_inliers_ratio: float = 0.6,
) -> tuple[np.ndarray, list[int]]:
    """
    Implementação genérica do RANSAC para regressão.
    
    O RANSAC (Random Sample Consensus) é robusto contra outliers:
    1. Seleciona aleatoriamente um subconjunto dos dados
    2. Ajusta o modelo no subconjunto
    3. Conta quantos pontos são "inliers" (erro pequeno)
    4. Repete e mantém o modelo com mais inliers
    
    Args:
        x: Array de features
        y: Array de targets
        fit_func: Função que ajusta o modelo (x, y) -> coeffs
        predict_func: Função que prediz (x, coeffs) -> y_pred
        n_iterations: Número de iterações RANSAC
        sample_ratio: Fração dos dados para cada sample (0.3-0.7)
        inlier_threshold: Threshold para considerar inlier (None = auto)
        min_inliers_ratio: Mínimo de inliers necessários
    
    Returns:
        Tuple (máscara_inliers, lista_índices_outliers)
    """
    n_samples = len(x)
    sample_size = max(3, int(n_samples * sample_ratio))
    min_inliers = int(n_samples * min_inliers_ratio)
    
    # Se poucos pontos, não faz sentido usar RANSAC
    if n_samples < 6:
        return np.ones(n_samples, dtype=bool), []
    
    # Estimativa inicial do threshold se não fornecido
    if inlier_threshold is None:
        # Ajuste inicial com todos os dados para estimar escala
        try:
            initial_coeffs = fit_func(x, y)
            initial_pred = predict_func(x, initial_coeffs)
            initial_residuals = np.abs(y - initial_pred)
            # Usar MAD (Median Absolute Deviation) como estimativa robusta
            mad = np.median(initial_residuals)
            inlier_threshold = max(3 * mad, np.std(y) * 0.5)
        except:
            inlier_threshold = np.std(y) * 0.5
    
    best_inliers_mask = np.ones(n_samples, dtype=bool)
    best_n_inliers = 0
    best_score = -np.inf
    
    rng = np.random.default_rng(42)  # Seed para reprodutibilidade
    
    for _ in range(n_iterations):
        # 1. Selecionar amostra aleatória
        sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]
        
        try:
            # 2. Ajustar modelo na amostra
            coeffs = fit_func(x_sample, y_sample)
            
            # 3. Calcular resíduos em todos os pontos
            y_pred = predict_func(x, coeffs)
            residuals = np.abs(y - y_pred)
            
            # 4. Identificar inliers
            inliers_mask = residuals <= inlier_threshold
            n_inliers = np.sum(inliers_mask)
            
            # 5. Verificar se é o melhor modelo
            if n_inliers >= min_inliers and n_inliers > best_n_inliers:
                # Calcular R² nos inliers como score
                if n_inliers > 2:
                    y_inliers = y[inliers_mask]
                    y_pred_inliers = y_pred[inliers_mask]
                    ss_res = np.sum((y_inliers - y_pred_inliers) ** 2)
                    ss_tot = np.sum((y_inliers - np.mean(y_inliers)) ** 2)
                    score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                else:
                    score = 0
                
                if n_inliers > best_n_inliers or (n_inliers == best_n_inliers and score > best_score):
                    best_n_inliers = n_inliers
                    best_inliers_mask = inliers_mask
                    best_score = score
                    
        except:
            continue
    
    # Se RANSAC não encontrou bom modelo, usar todos os dados
    if best_n_inliers < min_inliers:
        return np.ones(n_samples, dtype=bool), []
    
    outlier_indices = np.where(~best_inliers_mask)[0].tolist()
    return best_inliers_mask, outlier_indices


def remove_outliers(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "ransac",
    regression_type: str = "linear",
    **kwargs
) -> tuple[np.ndarray, np.ndarray, list[int], str]:
    """
    Remove outliers dos dados usando o método especificado.
    
    Args:
        x: Array de features
        y: Array de targets  
        method: Método de detecção ("none", "ransac", "iqr", "zscore")
        regression_type: Tipo de regressão (usado pelo RANSAC)
        **kwargs: Argumentos adicionais do método
    
    Returns:
        Tuple (x_clean, y_clean, outlier_indices, warning_msg)
    """
    if method == "none" or len(x) < 4:
        return x, y, [], ""
    
    warning_msg = ""
    n_total = len(x)
    
    if method == "ransac":
        # RANSAC simplificado: usar detecção combinada conservadora
        # Foco em remover apenas pontos com X muito fora do cluster principal
        inliers_mask = _detect_outliers_combined(x, y, multiplier=2.0)
        outlier_indices = np.where(~inliers_mask)[0].tolist()
        
    elif method == "iqr":
        # Abordagem combinada: outliers em X e resíduos
        inliers_mask = _detect_outliers_combined(
            x, y,
            multiplier=kwargs.get("iqr_multiplier", 2.0)
        )
        outlier_indices = np.where(~inliers_mask)[0].tolist()
        
    elif method == "zscore":
        # Abordagem combinada: detectar outliers em X, Y e depois refinar com resíduos
        # Primeiro, detectar outliers óbvios em X e Y usando Z-score
        x_zscore = np.abs((x - np.mean(x)) / max(np.std(x), 1e-10))
        y_zscore = np.abs((y - np.mean(y)) / max(np.std(y), 1e-10))
        
        threshold = kwargs.get("zscore_threshold", 2.5)
        xy_inliers = (x_zscore <= threshold) & (y_zscore <= threshold)
        
        # Se temos pontos "bons" suficientes, ajustar e verificar resíduos
        if np.sum(xy_inliers) >= 3:
            # Ajuste robusto com Theil-Sen nos pontos iniciais
            slope, intercept = _theil_sen_fit(x[xy_inliers], y[xy_inliers])
            y_pred = slope * x + intercept
            residuals = y - y_pred
            
            # Z-score nos resíduos
            res_zscore = np.abs((residuals - np.mean(residuals)) / max(np.std(residuals), 1e-10))
            res_inliers = res_zscore <= threshold
            
            inliers_mask = xy_inliers & res_inliers
        else:
            inliers_mask = _detect_outliers_zscore(y, threshold)
        
        outlier_indices = np.where(~inliers_mask)[0].tolist()
        
    else:
        return x, y, [], f"Método de outlier não reconhecido: {method}"
    
    n_outliers = len(outlier_indices)
    n_total = len(x)
    
    # Não remover mais que 30% dos dados
    if n_outliers > 0.3 * n_total:
        warning_msg = f"Muitos outliers detectados ({n_outliers}/{n_total}). Limitando remoção a 30%."
        # Manter apenas os outliers mais extremos
        if method in ("iqr", "zscore"):
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x)
            residuals = np.abs(y - y_pred)
            # Ordenar por resíduo e manter top 30%
            sorted_indices = np.argsort(residuals)[::-1]
            max_remove = int(0.3 * n_total)
            outlier_indices = sorted(sorted_indices[:max_remove].tolist())
            inliers_mask = np.ones(n_total, dtype=bool)
            inliers_mask[outlier_indices] = False
    
    x_clean = x[inliers_mask]
    y_clean = y[inliers_mask]
    
    return x_clean, y_clean, outlier_indices, warning_msg


# =============================================================================
# Métodos de Regressão Robusta
# =============================================================================

# Métodos robustos suportados (ordenados por robustez crescente)
# - ols: Ordinary Least Squares (nenhuma robustez, sensível a outliers)
# - huber: Pesos de Huber - bom para outliers moderados
# - bisquare: Pesos de Tukey - mais robusto que Huber para outliers extremos
# - theil_sen: Mediana das inclinações - muito robusto e simples
# - lad: Least Absolute Deviations (L1) - mediana condicional
# - mm: MM-estimator - combina alta eficiência + alta robustez
# - lts: Least Trimmed Squares - extremamente robusto para alta contaminação
# - ransac_fit: RANSAC - muito robusto contra outliers dispersos
# - welsch: IRLS com pesos Welsch (exponential decay)
# - cauchy: IRLS com pesos Cauchy (heavy-tailed)
SUPPORTED_ROBUST_METHODS = [
    "ols", "huber", "bisquare", "theil_sen", 
    "lad", "mm", "lts", "ransac_fit",
    "welsch", "cauchy"
]


def _mad_scale(residuals: np.ndarray) -> float:
    """Calcula MAD (Median Absolute Deviation) como estimativa robusta de escala."""
    mad = np.median(np.abs(residuals - np.median(residuals)))
    return mad * 1.4826  # fator de consistência para distribuição normal


def _huber_weights(residuals: np.ndarray, k: float = 1.345) -> np.ndarray:
    """
    Calcula pesos de Huber para regressão robusta.
    
    k=1.345 dá 95% de eficiência para dados normais.
    Pontos com |residual| > k*MAD recebem peso menor.
    
    Função ψ de Huber:
    - Linear para |r| <= k
    - Constante para |r| > k (limita influência de outliers)
    """
    # Usar MAD (Median Absolute Deviation) como estimativa robusta de escala
    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad < 1e-10:
        return np.ones(len(residuals))
    
    # Normalizar resíduos
    scaled_residuals = residuals / (mad * 1.4826)  # 1.4826 = fator de consistência
    
    # Pesos de Huber
    weights = np.ones(len(residuals))
    mask = np.abs(scaled_residuals) > k
    weights[mask] = k / np.abs(scaled_residuals[mask])
    
    return weights


def _bisquare_weights(residuals: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Calcula pesos Bisquare (Tukey) para regressão robusta.
    
    Mais robusto que Huber - outliers extremos recebem peso ZERO.
    c=4.685 dá 95% de eficiência para dados normais.
    
    Função ψ Bisquare:
    - Suave para |r| <= c  
    - ZERO para |r| > c (rejeita completamente outliers extremos)
    
    Ideal para dados com outliers severos ou alta contaminação.
    """
    scale = _mad_scale(residuals)
    if scale < 1e-10:
        return np.ones(len(residuals))
    
    u = residuals / scale
    weights = np.zeros(len(residuals))
    mask = np.abs(u) <= c
    weights[mask] = (1 - (u[mask] / c) ** 2) ** 2
    
    return weights


def _welsch_weights(residuals: np.ndarray, c: float = 2.985) -> np.ndarray:
    """
    Calcula pesos de Welsch para regressão robusta.
    
    Ainda mais suave que Bisquare - decaimento exponencial.
    c=2.985 dá 95% de eficiência para dados normais.
    
    Todos os pontos recebem algum peso, mas outliers extremos
    recebem peso negligenciável.
    """
    scale = _mad_scale(residuals)
    if scale < 1e-10:
        return np.ones(len(residuals))
    
    u = residuals / scale
    weights = np.exp(-(u / c) ** 2)
    
    return weights


def _cauchy_weights(residuals: np.ndarray, c: float = 2.385) -> np.ndarray:
    """
    Calcula pesos de Cauchy para regressão robusta.
    
    Peso decresce gradualmente - bom para dados com caudas pesadas.
    c=2.385 dá 95% de eficiência para dados normais.
    """
    scale = _mad_scale(residuals)
    if scale < 1e-10:
        return np.ones(len(residuals))
    
    u = residuals / scale
    weights = 1 / (1 + (u / c) ** 2)
    
    return weights


def _irls_linear(x: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> tuple[float, float]:
    """
    Iteratively Reweighted Least Squares (IRLS) para regressão linear robusta.
    
    Usa função de peso de Huber para reduzir influência de outliers.
    """
    n = len(x)
    
    # Inicializar com OLS
    slope, intercept = np.polyfit(x, y, 1)
    
    for iteration in range(max_iter):
        # Calcular resíduos
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        # Calcular pesos de Huber
        weights = _huber_weights(residuals)
        
        # Regressão ponderada (Weighted Least Squares)
        W = np.diag(weights)
        X_mat = np.column_stack([x, np.ones(n)])
        
        try:
            # Solução WLS: (X'WX)^-1 X'Wy
            XtW = X_mat.T @ W
            beta = np.linalg.solve(XtW @ X_mat, XtW @ y)
            
            new_slope, new_intercept = beta[0], beta[1]
        except np.linalg.LinAlgError:
            break
        
        # Verificar convergência
        if abs(new_slope - slope) < tol and abs(new_intercept - intercept) < tol:
            slope, intercept = new_slope, new_intercept
            break
        
        slope, intercept = new_slope, new_intercept
    
    return slope, intercept


def _irls_generic_linear(x: np.ndarray, y: np.ndarray, 
                         weight_func: callable,
                         max_iter: int = 50, tol: float = 1e-6) -> tuple[float, float]:
    """
    IRLS genérico para regressão linear robusta com qualquer função de peso.
    
    Args:
        x, y: Dados
        weight_func: Função que recebe resíduos e retorna pesos
        max_iter, tol: Parâmetros de convergência
    """
    n = len(x)
    if n < 2:
        return 0.0, np.mean(y)
    
    # Inicializar com estimativa robusta
    slope, intercept = _theil_sen_fit(x, y)
    
    for iteration in range(max_iter):
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        # Calcular pesos usando a função fornecida
        weights = weight_func(residuals)
        
        if np.sum(weights > 0.01) < 2:
            break
        
        # WLS
        W = np.diag(weights)
        X_mat = np.column_stack([x, np.ones(n)])
        
        try:
            XtW = X_mat.T @ W
            beta = np.linalg.solve(XtW @ X_mat + 1e-10 * np.eye(2), XtW @ y)
            new_slope, new_intercept = beta[0], beta[1]
        except np.linalg.LinAlgError:
            break
        
        if abs(new_slope - slope) < tol and abs(new_intercept - intercept) < tol:
            slope, intercept = new_slope, new_intercept
            break
        
        slope, intercept = new_slope, new_intercept
    
    return slope, intercept


def _ransac_fit_linear(x: np.ndarray, y: np.ndarray, 
                       n_iterations: int = 200, 
                       threshold_ratio: float = 2.0) -> tuple[float, float]:
    """
    RANSAC para ajuste linear robusto.
    
    Seleciona aleatoriamente pontos, ajusta linha, e mantém o modelo
    que tem mais inliers.
    """
    n = len(x)
    if n < 3:
        return np.polyfit(x, y, 1)
    
    best_slope, best_intercept = 0.0, np.mean(y)
    best_n_inliers = 0
    
    # Estimativa inicial do threshold usando MAD
    initial_slope, initial_intercept = np.polyfit(x, y, 1)
    initial_residuals = y - (initial_slope * x + initial_intercept)
    mad = np.median(np.abs(initial_residuals))
    threshold = max(threshold_ratio * mad * 1.4826, np.std(y) * 0.3)
    
    rng = np.random.default_rng(42)
    
    for _ in range(n_iterations):
        # Selecionar 2 pontos aleatórios (mínimo para linha)
        idx = rng.choice(n, size=min(2, n), replace=False)
        
        if len(idx) < 2:
            continue
            
        x_sample = x[idx]
        y_sample = y[idx]
        
        # Evitar divisão por zero
        if abs(x_sample[1] - x_sample[0]) < 1e-10:
            continue
        
        # Ajustar linha nos 2 pontos
        slope = (y_sample[1] - y_sample[0]) / (x_sample[1] - x_sample[0])
        intercept = y_sample[0] - slope * x_sample[0]
        
        # Contar inliers
        residuals = np.abs(y - (slope * x + intercept))
        n_inliers = np.sum(residuals <= threshold)
        
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_slope = slope
            best_intercept = intercept
    
    # Reajustar usando todos os inliers do melhor modelo
    residuals = np.abs(y - (best_slope * x + best_intercept))
    inlier_mask = residuals <= threshold
    
    if np.sum(inlier_mask) >= 2:
        x_inliers = x[inlier_mask]
        y_inliers = y[inlier_mask]
        best_slope, best_intercept = np.polyfit(x_inliers, y_inliers, 1)
    
    return best_slope, best_intercept


def _irls_bisquare_linear(x: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> tuple[float, float]:
    """
    IRLS com pesos Bisquare (Tukey) para regressão linear extremamente robusta.
    
    Diferente de Huber, outliers extremos recebem peso ZERO,
    efetivamente sendo ignorados no ajuste.
    """
    n = len(x)
    if n < 2:
        return 0.0, np.mean(y)
    
    # Inicializar com estimativa robusta (mediana)
    slope, intercept = _theil_sen_fit(x, y)
    
    for iteration in range(max_iter):
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        # Pesos Bisquare
        weights = _bisquare_weights(residuals)
        
        # Verificar se há pesos suficientes
        if np.sum(weights > 0.01) < 2:
            break
        
        # WLS
        W = np.diag(weights)
        X_mat = np.column_stack([x, np.ones(n)])
        
        try:
            XtW = X_mat.T @ W
            beta = np.linalg.solve(XtW @ X_mat + 1e-10 * np.eye(2), XtW @ y)
            new_slope, new_intercept = beta[0], beta[1]
        except np.linalg.LinAlgError:
            break
        
        if abs(new_slope - slope) < tol and abs(new_intercept - intercept) < tol:
            slope, intercept = new_slope, new_intercept
            break
        
        slope, intercept = new_slope, new_intercept
    
    return slope, intercept


def _lad_linear(x: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> tuple[float, float]:
    """
    Least Absolute Deviations (LAD) - Regressão L1 / Mediana condicional.
    
    Minimiza Σ|y - ŷ| em vez de Σ(y - ŷ)².
    Muito mais robusto que OLS para outliers porque erro não é quadrático.
    
    Implementação via IRLS com pesos inversamente proporcionais ao resíduo.
    """
    n = len(x)
    if n < 2:
        return 0.0, np.median(y)
    
    # Inicializar com Theil-Sen
    slope, intercept = _theil_sen_fit(x, y)
    
    for iteration in range(max_iter):
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        # Pesos LAD: 1/|residual| (com regularização para evitar divisão por zero)
        abs_res = np.abs(residuals)
        weights = 1 / (abs_res + 1e-8)
        
        # Normalizar pesos
        weights = weights / np.sum(weights) * n
        
        # WLS
        W = np.diag(weights)
        X_mat = np.column_stack([x, np.ones(n)])
        
        try:
            XtW = X_mat.T @ W
            beta = np.linalg.solve(XtW @ X_mat + 1e-10 * np.eye(2), XtW @ y)
            new_slope, new_intercept = beta[0], beta[1]
        except np.linalg.LinAlgError:
            break
        
        if abs(new_slope - slope) < tol and abs(new_intercept - intercept) < tol:
            slope, intercept = new_slope, new_intercept
            break
        
        slope, intercept = new_slope, new_intercept
    
    return slope, intercept


def _mm_estimator_linear(x: np.ndarray, y: np.ndarray, 
                         max_iter: int = 50, tol: float = 1e-6) -> tuple[float, float]:
    """
    MM-Estimator para regressão linear.
    
    Combina:
    1. Alta breakdown point (robustez) - usa estimativa inicial S-estimator
    2. Alta eficiência - refina com Bisquare
    
    Procedimento:
    1. Estimar escala robustamente com MAD dos resíduos de Theil-Sen
    2. Usar essa escala fixa durante IRLS com pesos Bisquare
    
    Muito robusto (breakdown ~50%) mas também eficiente (~95% para normais).
    """
    n = len(x)
    if n < 2:
        return 0.0, np.mean(y)
    
    # Fase 1: Estimativa inicial altamente robusta (Theil-Sen)
    slope, intercept = _theil_sen_fit(x, y)
    
    # Calcular escala robusta e FIXAR (não reestimar a cada iteração)
    initial_residuals = y - (slope * x + intercept)
    fixed_scale = _mad_scale(initial_residuals)
    
    if fixed_scale < 1e-10:
        return slope, intercept
    
    # Fase 2: IRLS com escala fixa (MM-estimation)
    c = 4.685  # Tuning constant para 95% eficiência
    
    for iteration in range(max_iter):
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        # Pesos Bisquare com escala FIXA
        u = residuals / fixed_scale
        weights = np.zeros(n)
        mask = np.abs(u) <= c
        weights[mask] = (1 - (u[mask] / c) ** 2) ** 2
        
        if np.sum(weights > 0.01) < 2:
            break
        
        # WLS
        W = np.diag(weights)
        X_mat = np.column_stack([x, np.ones(n)])
        
        try:
            XtW = X_mat.T @ W
            beta = np.linalg.solve(XtW @ X_mat + 1e-10 * np.eye(2), XtW @ y)
            new_slope, new_intercept = beta[0], beta[1]
        except np.linalg.LinAlgError:
            break
        
        if abs(new_slope - slope) < tol and abs(new_intercept - intercept) < tol:
            slope, intercept = new_slope, new_intercept
            break
        
        slope, intercept = new_slope, new_intercept
    
    return slope, intercept


def _lts_linear(x: np.ndarray, y: np.ndarray, 
                h_ratio: float = 0.75,
                n_subsets: int = 500) -> tuple[float, float]:
    """
    Least Trimmed Squares (LTS) para regressão linear.
    
    Minimiza a soma dos h menores resíduos quadrados (ignora os maiores).
    
    Extremamente robusto - breakdown point de (n-h)/n.
    Com h_ratio=0.75, até 25% dos dados podem ser outliers.
    
    Args:
        x, y: Dados
        h_ratio: Fração dos dados a usar (0.5-0.9)
        n_subsets: Número de subconjuntos aleatórios a testar
    
    Procedimento:
    1. Amostrar subconjuntos aleatórios
    2. Para cada subconjunto, ajustar regressão
    3. Calcular soma dos h menores resíduos²
    4. Manter o melhor modelo
    """
    n = len(x)
    if n < 3:
        return np.polyfit(x, y, 1) if n >= 2 else (0.0, np.mean(y))
    
    h = max(3, int(n * h_ratio))  # Número de pontos a usar
    
    best_slope, best_intercept = np.polyfit(x, y, 1)
    best_trimmed_ss = np.inf
    
    rng = np.random.default_rng(42)
    
    for _ in range(n_subsets):
        # Selecionar subconjunto aleatório (p+1 pontos mínimo para regressão)
        sample_size = min(max(3, h // 2), n)
        idx = rng.choice(n, size=sample_size, replace=False)
        
        try:
            # Ajustar no subconjunto
            slope, intercept = np.polyfit(x[idx], y[idx], 1)
            
            # Calcular todos os resíduos²
            residuals_sq = (y - (slope * x + intercept)) ** 2
            
            # Soma dos h menores resíduos²
            sorted_res_sq = np.sort(residuals_sq)
            trimmed_ss = np.sum(sorted_res_sq[:h])
            
            if trimmed_ss < best_trimmed_ss:
                best_trimmed_ss = trimmed_ss
                best_slope = slope
                best_intercept = intercept
        except:
            continue
    
    # C-step refinamento: usar os h pontos com menores resíduos e reajustar
    for _ in range(10):  # Algumas iterações de refinamento
        residuals_sq = (y - (best_slope * x + best_intercept)) ** 2
        order = np.argsort(residuals_sq)
        h_best_idx = order[:h]
        
        try:
            new_slope, new_intercept = np.polyfit(x[h_best_idx], y[h_best_idx], 1)
            new_residuals_sq = (y - (new_slope * x + new_intercept)) ** 2
            new_trimmed_ss = np.sum(np.sort(new_residuals_sq)[:h])
            
            if new_trimmed_ss < best_trimmed_ss - 1e-10:
                best_trimmed_ss = new_trimmed_ss
                best_slope = new_slope
                best_intercept = new_intercept
            else:
                break
        except:
            break
    
    return best_slope, best_intercept


# =============================================================================
# Funções de Ajuste (Fit)
# =============================================================================

def fit_linear(x: np.ndarray, y: np.ndarray, robust_method: str = "ols") -> RegressionResult:
    """
    Ajusta regressão linear: y = a*x + b
    
    Args:
        x: Array de features
        y: Array de targets
        robust_method: Método de ajuste
            - "ols": Ordinary Least Squares (padrão, sensível a outliers)
            - "theil_sen": Estimador Theil-Sen (mediana das inclinações)
            - "huber": IRLS com pesos de Huber (balanceado)
            - "bisquare": IRLS com pesos Bisquare/Tukey (outliers = peso ZERO)
            - "welsch": IRLS com pesos Welsch (exponential decay)
            - "cauchy": IRLS com pesos Cauchy (heavy-tailed)
            - "lad": Least Absolute Deviations / L1 (mediana condicional)
            - "mm": MM-Estimator (alta robustez + alta eficiência)
            - "lts": Least Trimmed Squares (extremamente robusto, ignora 25% piores)
            - "ransac_fit": RANSAC (muito robusto contra outliers)
    """
    warnings_list = []
    
    try:
        if robust_method == "theil_sen":
            slope, intercept = _theil_sen_fit(x, y)
            warnings_list.append("Método: Theil-Sen (robusto)")
        elif robust_method == "huber":
            slope, intercept = _irls_linear(x, y)
            warnings_list.append("Método: Huber/IRLS (robusto)")
        elif robust_method == "bisquare":
            slope, intercept = _irls_bisquare_linear(x, y)
            warnings_list.append("Método: Bisquare/Tukey (outliers ignorados)")
        elif robust_method == "welsch":
            slope, intercept = _irls_generic_linear(x, y, _welsch_weights)
            warnings_list.append("Método: Welsch (exponential decay)")
        elif robust_method == "cauchy":
            slope, intercept = _irls_generic_linear(x, y, _cauchy_weights)
            warnings_list.append("Método: Cauchy (heavy-tailed)")
        elif robust_method == "lad":
            slope, intercept = _lad_linear(x, y)
            warnings_list.append("Método: LAD/L1 (mediana condicional)")
        elif robust_method == "mm":
            slope, intercept = _mm_estimator_linear(x, y)
            warnings_list.append("Método: MM-Estimator (alta robustez + eficiência)")
        elif robust_method == "lts":
            slope, intercept = _lts_linear(x, y)
            warnings_list.append("Método: LTS (extremamente robusto)")
        elif robust_method == "ransac_fit":
            slope, intercept = _ransac_fit_linear(x, y)
            warnings_list.append("Método: RANSAC (muito robusto)")
        else:
            # OLS padrão
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        coefficients = {"a": float(slope), "b": float(intercept)}
        y_pred = slope * x + intercept
        r2, rmse, mae = _compute_metrics(y, y_pred)
        
        return RegressionResult(
            regression_type="linear",
            coefficients=coefficients,
            equation=f"y = {slope:.6g}*x + {intercept:.6g}",
            r2_score=r2,
            rmse=rmse,
            mae=mae,
            n_samples=len(x),
            x_range=(float(x.min()), float(x.max())),
            warnings=warnings_list,
        )
    except Exception as e:
        return RegressionResult(
            regression_type="linear",
            coefficients={},
            equation="",
            r2_score=0.0,
            rmse=float("inf"),
            mae=float("inf"),
            n_samples=len(x),
            success=False,
            error=str(e),
        )


def _irls_polynomial(x: np.ndarray, y: np.ndarray, degree: int = 2,
                     weight_func: str = "bisquare",
                     max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    IRLS para regressão polinomial robusta.
    
    Args:
        x, y: Dados
        degree: Grau do polinômio
        weight_func: "huber", "bisquare", "cauchy", "welsch"
        max_iter, tol: Parâmetros de convergência
    
    Returns:
        Coeficientes do polinômio [a_n, ..., a_1, a_0] (ordem decrescente)
    """
    n = len(x)
    if n < degree + 1:
        return np.polyfit(x, y, degree) if n >= 2 else np.zeros(degree + 1)
    
    # Matriz de design do polinômio
    X_mat = np.column_stack([x ** i for i in range(degree, -1, -1)])
    
    # Inicializar com OLS
    coeffs = np.polyfit(x, y, degree)
    
    weight_functions = {
        "huber": _huber_weights,
        "bisquare": _bisquare_weights,
        "cauchy": _cauchy_weights,
        "welsch": _welsch_weights,
    }
    weight_fn = weight_functions.get(weight_func, _bisquare_weights)
    
    for iteration in range(max_iter):
        # Predição atual
        y_pred = np.polyval(coeffs, x)
        residuals = y - y_pred
        
        # Calcular pesos
        weights = weight_fn(residuals)
        
        if np.sum(weights > 0.01) < degree + 1:
            break
        
        # WLS
        W = np.diag(weights)
        try:
            XtW = X_mat.T @ W
            new_coeffs = np.linalg.solve(XtW @ X_mat + 1e-10 * np.eye(degree + 1), XtW @ y)
        except np.linalg.LinAlgError:
            break
        
        if np.allclose(new_coeffs, coeffs, atol=tol):
            coeffs = new_coeffs
            break
        
        coeffs = new_coeffs
    
    return coeffs


def _lts_polynomial(x: np.ndarray, y: np.ndarray, degree: int = 2,
                    h_ratio: float = 0.75, n_subsets: int = 300) -> np.ndarray:
    """
    LTS para regressão polinomial - extremamente robusto.
    """
    n = len(x)
    h = max(degree + 2, int(n * h_ratio))
    
    best_coeffs = np.polyfit(x, y, degree)
    best_trimmed_ss = np.inf
    
    rng = np.random.default_rng(42)
    
    for _ in range(n_subsets):
        sample_size = min(max(degree + 2, h // 2), n)
        idx = rng.choice(n, size=sample_size, replace=False)
        
        try:
            coeffs = np.polyfit(x[idx], y[idx], degree)
            y_pred = np.polyval(coeffs, x)
            residuals_sq = (y - y_pred) ** 2
            trimmed_ss = np.sum(np.sort(residuals_sq)[:h])
            
            if trimmed_ss < best_trimmed_ss:
                best_trimmed_ss = trimmed_ss
                best_coeffs = coeffs
        except:
            continue
    
    # Refinamento
    for _ in range(10):
        y_pred = np.polyval(best_coeffs, x)
        residuals_sq = (y - y_pred) ** 2
        h_best_idx = np.argsort(residuals_sq)[:h]
        
        try:
            new_coeffs = np.polyfit(x[h_best_idx], y[h_best_idx], degree)
            new_trimmed_ss = np.sum(np.sort((y - np.polyval(new_coeffs, x)) ** 2)[:h])
            
            if new_trimmed_ss < best_trimmed_ss - 1e-10:
                best_trimmed_ss = new_trimmed_ss
                best_coeffs = new_coeffs
            else:
                break
        except:
            break
    
    return best_coeffs


def _robust_curve_fit(
    func: callable,
    x: np.ndarray, 
    y: np.ndarray,
    p0: list,
    bounds: tuple = (-np.inf, np.inf),
    robust_method: str = "ols",
    max_iter: int = 30,
    maxfev: int = 5000
) -> tuple[np.ndarray, list[str]]:
    """
    curve_fit com suporte a métodos robustos.
    
    Para métodos robustos, usa IRLS (Iteratively Reweighted Least Squares)
    onde os pesos são calculados com base nos resíduos.
    
    Args:
        func: Função de modelo f(x, *params)
        x, y: Dados
        p0: Estimativas iniciais dos parâmetros
        bounds: Limites dos parâmetros
        robust_method: "ols", "huber", "bisquare", "cauchy", "welsch", "lts"
        
    Returns:
        (parâmetros otimizados, warnings)
    """
    warnings_list = []
    
    weight_functions = {
        "huber": _huber_weights,
        "bisquare": _bisquare_weights,
        "cauchy": _cauchy_weights,
        "welsch": _welsch_weights,
    }
    
    if robust_method == "ols" or robust_method not in weight_functions and robust_method != "lts":
        # OLS padrão via curve_fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = optimize.curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
        return popt, warnings_list
    
    if robust_method == "lts":
        # LTS: usar apenas os h melhores pontos
        n = len(x)
        h = max(len(p0) + 1, int(n * 0.75))
        
        # Fit inicial
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = optimize.curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
        
        # Refinar com LTS
        for _ in range(5):
            y_pred = func(x, *popt)
            residuals_sq = (y - y_pred) ** 2
            h_best_idx = np.argsort(residuals_sq)[:h]
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = optimize.curve_fit(
                        func, x[h_best_idx], y[h_best_idx], 
                        p0=popt, bounds=bounds, maxfev=maxfev
                    )
            except:
                break
        
        warnings_list.append("Método: LTS (extremamente robusto)")
        return popt, warnings_list
    
    # IRLS para outros métodos robustos
    weight_fn = weight_functions[robust_method]
    
    # Fit inicial
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = optimize.curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
    
    # IRLS iterations
    for iteration in range(max_iter):
        y_pred = func(x, *popt)
        residuals = y - y_pred
        weights = weight_fn(residuals)
        
        if np.sum(weights > 0.01) < len(p0):
            break
        
        # Weighted curve_fit usando sigma
        # sigma = 1/sqrt(weight) para que peso alto = baixa variância
        sigma = 1.0 / np.sqrt(np.maximum(weights, 1e-10))
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_popt, _ = optimize.curve_fit(
                    func, x, y, p0=popt, 
                    sigma=sigma, absolute_sigma=True,
                    bounds=bounds, maxfev=maxfev
                )
            
            if np.allclose(new_popt, popt, rtol=1e-4):
                popt = new_popt
                break
            popt = new_popt
        except:
            break
    
    method_names = {
        "huber": "Huber/IRLS",
        "bisquare": "Bisquare/Tukey",
        "cauchy": "Cauchy",
        "welsch": "Welsch",
    }
    warnings_list.append(f"Método: {method_names.get(robust_method, robust_method)} (robusto)")
    
    return popt, warnings_list


def fit_quadratic(x: np.ndarray, y: np.ndarray, robust_method: str = "ols") -> RegressionResult:
    """
    Ajusta regressão quadrática: y = a*x² + b*x + c
    
    Args:
        x: Array de features
        y: Array de targets
        robust_method: Método de ajuste
            - "ols": Ordinary Least Squares (padrão)
            - "huber": IRLS com pesos Huber
            - "bisquare": IRLS com pesos Bisquare (outliers ignorados)
            - "lts": Least Trimmed Squares (extremamente robusto)
    """
    warnings_list = []
    try:
        if robust_method in ["huber", "bisquare", "cauchy", "welsch"]:
            coeffs = _irls_polynomial(x, y, degree=2, weight_func=robust_method)
            warnings_list.append(f"Método: {robust_method.upper()}/IRLS (robusto)")
        elif robust_method == "lts":
            coeffs = _lts_polynomial(x, y, degree=2)
            warnings_list.append("Método: LTS (extremamente robusto)")
        else:
            # OLS padrão
            coeffs = np.polyfit(x, y, 2)
        
        a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        
        coefficients = {"a": a, "b": b, "c": c}
        y_pred = a * x**2 + b * x + c
        r2, rmse, mae = _compute_metrics(y, y_pred)
        
        return RegressionResult(
            regression_type="quadratic",
            coefficients=coefficients,
            equation=f"y = {a:.6g}*x² + {b:.6g}*x + {c:.6g}",
            r2_score=r2,
            rmse=rmse,
            mae=mae,
            n_samples=len(x),
            x_range=(float(x.min()), float(x.max())),
            warnings=warnings_list,
        )
    except Exception as e:
        return RegressionResult(
            regression_type="quadratic",
            coefficients={},
            equation="",
            r2_score=0.0,
            rmse=float("inf"),
            mae=float("inf"),
            n_samples=len(x),
            success=False,
            error=str(e),
        )


def fit_exponential(x: np.ndarray, y: np.ndarray, robust_method: str = "ols") -> RegressionResult:
    """
    Ajusta regressão exponencial: y = a * exp(b*x) + c
    
    Args:
        x: Array de features
        y: Array de targets
        robust_method: Método de ajuste
            - "ols": Ordinary Least Squares (padrão)
            - "huber", "bisquare", "cauchy", "welsch": IRLS robusto
            - "lts": Least Trimmed Squares
    """
    warnings_list = []
    
    try:
        # Estimativas iniciais
        y_min, y_max = y.min(), y.max()
        c_init = y_min * 0.9  # offset inicial
        a_init = (y_max - c_init)
        b_init = 0.1  # taxa de crescimento inicial
        
        def exp_func(x, a, b, c):
            return a * np.exp(b * x) + c
        
        # Bounds para evitar overflow
        bounds = (
            [-np.inf, -10, -np.inf],  # lower bounds
            [np.inf, 10, np.inf]       # upper bounds
        )
        
        popt, robust_warnings = _robust_curve_fit(
            exp_func, x, y,
            p0=[a_init, b_init, c_init],
            bounds=bounds,
            robust_method=robust_method
        )
        warnings_list.extend(robust_warnings)
        
        a, b, c = float(popt[0]), float(popt[1]), float(popt[2])
        coefficients = {"a": a, "b": b, "c": c}
        
        y_pred = exp_func(x, a, b, c)
        r2, rmse, mae = _compute_metrics(y, y_pred)
        
        # Verificar qualidade do ajuste
        if r2 < 0:
            warnings_list.append("R² negativo - ajuste exponencial pode não ser adequado")
        
        return RegressionResult(
            regression_type="exponential",
            coefficients=coefficients,
            equation=f"y = {a:.6g} * exp({b:.6g}*x) + {c:.6g}",
            r2_score=r2,
            rmse=rmse,
            mae=mae,
            n_samples=len(x),
            x_range=(float(x.min()), float(x.max())),
            warnings=warnings_list,
        )
    except Exception as e:
        return RegressionResult(
            regression_type="exponential",
            coefficients={},
            equation="",
            r2_score=0.0,
            rmse=float("inf"),
            mae=float("inf"),
            n_samples=len(x),
            success=False,
            error=str(e),
        )


def fit_logarithmic(x: np.ndarray, y: np.ndarray, robust_method: str = "ols") -> RegressionResult:
    """
    Ajusta regressão logarítmica: y = a * ln(x) + b
    
    Args:
        x: Array de features (deve ser > 0)
        y: Array de targets
        robust_method: Método de ajuste
            - "ols": Ordinary Least Squares (padrão)
            - "theil_sen", "huber", "bisquare", etc: Métodos robustos
    """
    warnings_list = []
    
    try:
        # Filtrar valores x <= 0
        mask = x > 0
        if not np.all(mask):
            warnings_list.append(f"Removidos {(~mask).sum()} pontos com x <= 0")
            x = x[mask]
            y = y[mask]
        
        if len(x) < 2:
            raise ValueError("Dados insuficientes após filtrar x <= 0")
        
        # Transformar para espaço linear: y = a*log(x) + b
        # É uma regressão linear em log_x
        log_x = np.log(x)
        
        # Usar fit_linear com o método robusto escolhido
        linear_result = fit_linear(log_x, y, robust_method=robust_method)
        
        a = linear_result.coefficients.get("a", 0)
        b = linear_result.coefficients.get("b", 0)
        
        if linear_result.warnings:
            warnings_list.extend(linear_result.warnings)
        coefficients = {"a": a, "b": b}
        
        y_pred = a * np.log(x) + b
        r2, rmse, mae = _compute_metrics(y, y_pred)
        
        return RegressionResult(
            regression_type="logarithmic",
            coefficients=coefficients,
            equation=f"y = {a:.6g} * ln(x) + {b:.6g}",
            r2_score=r2,
            rmse=rmse,
            mae=mae,
            n_samples=len(x),
            x_range=(float(x.min()), float(x.max())),
            warnings=warnings_list,
        )
    except Exception as e:
        return RegressionResult(
            regression_type="logarithmic",
            coefficients={},
            equation="",
            r2_score=0.0,
            rmse=float("inf"),
            mae=float("inf"),
            n_samples=len(x) if isinstance(x, np.ndarray) else 0,
            success=False,
            error=str(e),
        )


def fit_power(x: np.ndarray, y: np.ndarray, robust_method: str = "ols") -> RegressionResult:
    """
    Ajusta regressão de potência: y = a * x^b + c
    
    Args:
        x: Array de features (deve ser > 0)
        y: Array de targets
        robust_method: Método de ajuste
            - "ols": Ordinary Least Squares (padrão)
            - "huber", "bisquare", "cauchy", "welsch": IRLS robusto
            - "lts": Least Trimmed Squares
    """
    warnings_list = []
    
    try:
        # Filtrar valores x <= 0
        mask = x > 0
        if not np.all(mask):
            warnings_list.append(f"Removidos {(~mask).sum()} pontos com x <= 0")
            x = x[mask]
            y = y[mask]
        
        if len(x) < 3:
            raise ValueError("Dados insuficientes (mínimo 3 pontos)")
        
        # Estimativas iniciais usando log-log linear
        log_x = np.log(x)
        log_y_shifted = np.log(np.maximum(y - y.min() + 1, 1e-10))
        slope, intercept = np.polyfit(log_x, log_y_shifted, 1)
        
        a_init = np.exp(intercept)
        b_init = slope
        c_init = y.min() * 0.5
        
        def power_func(x, a, b, c):
            return a * np.power(x, b) + c
        
        popt, robust_warnings = _robust_curve_fit(
            power_func, x, y,
            p0=[a_init, b_init, c_init],
            robust_method=robust_method
        )
        warnings_list.extend(robust_warnings)
        
        a, b, c = float(popt[0]), float(popt[1]), float(popt[2])
        coefficients = {"a": a, "b": b, "c": c}
        
        y_pred = power_func(x, a, b, c)
        r2, rmse, mae = _compute_metrics(y, y_pred)
        
        return RegressionResult(
            regression_type="power",
            coefficients=coefficients,
            equation=f"y = {a:.6g} * x^{b:.6g} + {c:.6g}",
            r2_score=r2,
            rmse=rmse,
            mae=mae,
            n_samples=len(x),
            x_range=(float(x.min()), float(x.max())),
            warnings=warnings_list,
        )
    except Exception as e:
        return RegressionResult(
            regression_type="power",
            coefficients={},
            equation="",
            r2_score=0.0,
            rmse=float("inf"),
            mae=float("inf"),
            n_samples=len(x) if isinstance(x, np.ndarray) else 0,
            success=False,
            error=str(e),
        )


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int = 3, robust_method: str = "ols") -> RegressionResult:
    """
    Ajusta regressão polinomial: y = aₙxⁿ + ... + a₁x + a₀
    
    Args:
        x: Array de features
        y: Array de targets
        degree: Grau do polinômio (1-10)
        robust_method: Método de ajuste
            - "ols": Ordinary Least Squares (padrão)
            - "huber", "bisquare", "cauchy", "welsch": IRLS robusto
            - "lts": Least Trimmed Squares
    """
    warnings_list = []
    
    try:
        degree = min(max(1, degree), 10)  # Limitar grau entre 1 e 10
        
        # Selecionar método de ajuste
        if robust_method in ["huber", "bisquare", "cauchy", "welsch"]:
            coeffs = _irls_polynomial(x, y, degree=degree, weight_func=robust_method)
            warnings_list.append(f"Método: {robust_method.upper()}/IRLS (robusto)")
        elif robust_method == "lts":
            coeffs = _lts_polynomial(x, y, degree=degree)
            warnings_list.append("Método: LTS (extremamente robusto)")
        else:
            # OLS padrão
            coeffs = np.polyfit(x, y, degree)
        
        coefficients = {f"a{i}": float(c) for i, c in enumerate(reversed(coeffs))}
        
        y_pred = np.polyval(coeffs, x)
        r2, rmse, mae = _compute_metrics(y, y_pred)
        
        # Montar equação
        terms = []
        for i, c in enumerate(coeffs):
            exp = degree - i
            if abs(c) < 1e-10:
                continue
            if exp == 0:
                terms.append(f"{c:.6g}")
            elif exp == 1:
                terms.append(f"{c:.6g}*x")
            else:
                terms.append(f"{c:.6g}*x^{exp}")
        equation = "y = " + " + ".join(terms).replace("+ -", "- ")
        
        return RegressionResult(
            regression_type="polynomial",
            coefficients=coefficients,
            equation=equation,
            r2_score=r2,
            rmse=rmse,
            mae=mae,
            n_samples=len(x),
            x_range=(float(x.min()), float(x.max())),
        )
    except Exception as e:
        return RegressionResult(
            regression_type="polynomial",
            coefficients={},
            equation="",
            r2_score=0.0,
            rmse=float("inf"),
            mae=float("inf"),
            n_samples=len(x),
            success=False,
            error=str(e),
        )


# =============================================================================
# Funções de Predição
# =============================================================================

def predict_regression(
    x: float | np.ndarray,
    regression_type: str,
    coefficients: dict[str, float]
) -> float | np.ndarray:
    """
    Aplica a equação de regressão para predição.
    
    Args:
        x: Valor(es) de entrada
        regression_type: Tipo da regressão
        coefficients: Coeficientes da equação
    
    Returns:
        Valor(es) predito(s)
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    
    if regression_type == "linear":
        a = coefficients.get("a", 0)
        b = coefficients.get("b", 0)
        result = a * x_arr + b
        
    elif regression_type == "quadratic":
        a = coefficients.get("a", 0)
        b = coefficients.get("b", 0)
        c = coefficients.get("c", 0)
        result = a * x_arr**2 + b * x_arr + c
        
    elif regression_type == "exponential":
        a = coefficients.get("a", 1)
        b = coefficients.get("b", 0)
        c = coefficients.get("c", 0)
        # Limitar b*x para evitar overflow
        bx = np.clip(b * x_arr, -700, 700)
        result = a * np.exp(bx) + c
        
    elif regression_type == "logarithmic":
        a = coefficients.get("a", 0)
        b = coefficients.get("b", 0)
        # Proteger contra log de valores <= 0
        x_safe = np.maximum(x_arr, 1e-10)
        result = a * np.log(x_safe) + b
        
    elif regression_type == "power":
        a = coefficients.get("a", 1)
        b = coefficients.get("b", 1)
        c = coefficients.get("c", 0)
        # Proteger contra x <= 0
        x_safe = np.maximum(x_arr, 1e-10)
        result = a * np.power(x_safe, b) + c
        
    elif regression_type == "polynomial":
        # Reconstruir coeficientes na ordem correta
        max_degree = max(int(k[1:]) for k in coefficients.keys() if k.startswith("a"))
        coeffs = [coefficients.get(f"a{i}", 0) for i in range(max_degree, -1, -1)]
        result = np.polyval(coeffs, x_arr)
        
    else:
        raise ValueError(f"Tipo de regressão não suportado: {regression_type}")
    
    # Retornar escalar se entrada foi escalar
    if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
        return float(result[0])
    return result


# =============================================================================
# Função Principal de Ajuste
# =============================================================================

def fit_regression(
    x: np.ndarray,
    y: np.ndarray,
    regression_type: str,
    outlier_method: str = "none",
    robust_method: str = "ols",
    **kwargs
) -> RegressionResult:
    """
    Ajusta uma regressão do tipo especificado.
    
    Args:
        x: Array de features
        y: Array de targets
        regression_type: Tipo da regressão
        outlier_method: Método de remoção de outliers ("none", "ransac", "iqr", "zscore")
        robust_method: Método robusto para linear ("ols", "theil_sen", "huber", "ransac_fit")
        **kwargs: Argumentos adicionais (ex: degree para polynomial)
    
    Returns:
        RegressionResult com coeficientes e métricas
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    
    if len(x) != len(y):
        return RegressionResult(
            regression_type=regression_type,
            coefficients={},
            equation="",
            r2_score=0.0,
            rmse=float("inf"),
            mae=float("inf"),
            n_samples=0,
            success=False,
            error="Arrays x e y devem ter o mesmo tamanho",
        )
    
    if len(x) < 2:
        return RegressionResult(
            regression_type=regression_type,
            coefficients={},
            equation="",
            r2_score=0.0,
            rmse=float("inf"),
            mae=float("inf"),
            n_samples=len(x),
            success=False,
            error="Dados insuficientes (mínimo 2 pontos)",
        )
    
    # Remover NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]
    
    # Aplicar remoção de outliers
    outlier_indices = []
    outlier_warning = ""
    n_original = len(x)
    
    if outlier_method != "none" and len(x) >= 4:
        x, y, outlier_indices, outlier_warning = remove_outliers(
            x, y, 
            method=outlier_method,
            regression_type=regression_type,
            **kwargs
        )
    
    # Ajustar regressão
    if regression_type == "linear":
        result = fit_linear(x, y, robust_method=robust_method)
    elif regression_type == "quadratic":
        result = fit_quadratic(x, y, robust_method=robust_method)
    elif regression_type == "exponential":
        result = fit_exponential(x, y, robust_method=robust_method)
    elif regression_type == "logarithmic":
        result = fit_logarithmic(x, y, robust_method=robust_method)
    elif regression_type == "power":
        result = fit_power(x, y, robust_method=robust_method)
    elif regression_type == "polynomial":
        degree = kwargs.get("degree", 3)
        result = fit_polynomial(x, y, degree=degree, robust_method=robust_method)
    else:
        return RegressionResult(
            regression_type=regression_type,
            coefficients={},
            equation="",
            r2_score=0.0,
            rmse=float("inf"),
            mae=float("inf"),
            n_samples=len(x),
            success=False,
            error=f"Tipo de regressão não suportado: {regression_type}. Use: {SUPPORTED_REGRESSIONS}",
        )
    
    # Adicionar informações de outliers ao resultado
    result.outlier_method = outlier_method
    result.n_outliers_removed = len(outlier_indices)
    result.outlier_indices = outlier_indices
    
    if outlier_warning:
        result.warnings.append(outlier_warning)
    
    if len(outlier_indices) > 0:
        result.warnings.append(
            f"Removidos {len(outlier_indices)} outliers de {n_original} pontos ({outlier_method.upper()})"
        )
    
    return result


def fit_best_regression(
    x: np.ndarray,
    y: np.ndarray,
    types_to_try: list[str] | None = None,
    metric: str = "r2",
    outlier_method: str = "none",
    robust_method: str = "ols",
    **kwargs
) -> tuple[RegressionResult, dict[str, RegressionResult]]:
    """
    Testa múltiplos tipos de regressão e retorna o melhor.
    
    Args:
        x: Array de features
        y: Array de targets
        types_to_try: Lista de tipos a testar (None = todos)
        metric: Métrica para comparação ("r2", "rmse", "mae")
        outlier_method: Método de remoção de outliers
        robust_method: Método robusto para regressão linear
    
    Returns:
        Tuple (melhor_resultado, dicionário_com_todos_resultados)
    """
    if types_to_try is None:
        types_to_try = SUPPORTED_REGRESSIONS
    
    results: dict[str, RegressionResult] = {}
    
    for reg_type in types_to_try:
        results[reg_type] = fit_regression(
            x, y, reg_type, 
            outlier_method=outlier_method, 
            robust_method=robust_method,
            **kwargs
        )
    
    # Encontrar o melhor
    valid_results = {k: v for k, v in results.items() if v.success}
    
    if not valid_results:
        # Retornar o primeiro com erro
        return results[types_to_try[0]], results
    
    if metric == "r2":
        best_key = max(valid_results, key=lambda k: valid_results[k].r2_score)
    elif metric == "rmse":
        best_key = min(valid_results, key=lambda k: valid_results[k].rmse)
    else:  # mae
        best_key = min(valid_results, key=lambda k: valid_results[k].mae)
    
    return results[best_key], results


# =============================================================================
# Funções para Visualização
# =============================================================================

def generate_curve_points(
    regression_result: RegressionResult,
    n_points: int = 100,
    x_range: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera pontos da curva para plotagem.
    
    Args:
        regression_result: Resultado da regressão
        n_points: Número de pontos a gerar
        x_range: Range de x (None = usar do resultado)
    
    Returns:
        Tuple (x_curve, y_curve)
    """
    if x_range is None:
        x_range = regression_result.x_range
    
    if x_range is None:
        x_range = (0, 1)
    
    x_min, x_max = x_range
    
    # Para logarítmica/power, começar de valor > 0
    if regression_result.regression_type in ("logarithmic", "power"):
        x_min = max(x_min, 1e-6)
    
    x_curve = np.linspace(x_min, x_max, n_points)
    y_curve = predict_regression(
        x_curve,
        regression_result.regression_type,
        regression_result.coefficients
    )
    
    return x_curve, y_curve


def regression_to_plot_data(
    regression_result: RegressionResult,
    x_data: np.ndarray | None = None,
    y_data: np.ndarray | None = None,
    x_original: np.ndarray | None = None,
    y_original: np.ndarray | None = None,
    n_curve_points: int = 100
) -> dict[str, Any]:
    """
    Prepara dados para plotagem (compatível com frontend).
    
    Args:
        regression_result: Resultado da regressão
        x_data: Dados X usados no ajuste (após remoção de outliers)
        y_data: Dados Y usados no ajuste (após remoção de outliers)
        x_original: Dados X originais (antes de remover outliers)
        y_original: Dados Y originais (antes de remover outliers)
        n_curve_points: Número de pontos na curva
    
    Returns:
        Dicionário com dados para o gráfico
    """
    x_curve, y_curve = generate_curve_points(regression_result, n_curve_points)
    
    plot_data = {
        "regression_type": regression_result.regression_type,
        "equation": regression_result.equation,
        "metrics": {
            "r2": regression_result.r2_score,
            "rmse": regression_result.rmse,
            "mae": regression_result.mae,
            "n_samples": regression_result.n_samples,
        },
        "curve": {
            "x": x_curve.tolist(),
            "y": y_curve.tolist(),
        },
        "outlier_indices": regression_result.outlier_indices,
        "n_outliers_removed": regression_result.n_outliers_removed,
    }
    
    # Adicionar dados usados no ajuste (inliers)
    if x_data is not None and y_data is not None:
        plot_data["data_points"] = {
            "x": np.asarray(x_data).tolist(),
            "y": np.asarray(y_data).tolist(),
        }
    
    # Adicionar dados originais (incluindo outliers) se fornecidos
    if x_original is not None and y_original is not None:
        plot_data["original_data"] = {
            "x": np.asarray(x_original).tolist(),
            "y": np.asarray(y_original).tolist(),
        }
    
    return plot_data
