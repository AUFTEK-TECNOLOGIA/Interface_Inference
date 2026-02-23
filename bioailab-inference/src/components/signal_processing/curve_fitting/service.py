"""
Serviço de ajuste de curvas de crescimento.

Responsabilidade única: ajustar modelo matemático aos dados e retornar parâmetros.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Optional

from .base import MathModel, ModelRegistry, CurveFitConfig, CurveFitResult
from .utils import dedupe_and_sort, determine_window


class CurveFittingService:
    """Serviço para ajustar modelos matemáticos a curvas de crescimento."""
    
    def __init__(self, config: CurveFitConfig = None):
        self.config = config or CurveFitConfig()
    
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        model_name: str = "richards",
        config: CurveFitConfig = None
    ) -> CurveFitResult:
        """
        Ajusta um modelo matemático aos dados.
        
        Args:
            x: Array de timestamps normalizados
            y: Array de valores normalizados
            model_name: Nome do modelo ("richards", "gompertz", "logistic", "baranyi")
            config: Configuração opcional
        
        Returns:
            CurveFitResult com parâmetros ajustados e curvas calculadas
        """
        cfg = config or self.config
        
        # Obter modelo do registry
        try:
            model = ModelRegistry.create(model_name)
        except ValueError:
            return CurveFitResult.failed(model_name)
        
        # Preparar dados
        x = np.array(x, dtype=float).flatten()
        y = np.array(y, dtype=float).flatten()
        
        if len(x) < cfg.min_window_size:
            return CurveFitResult.failed(model_name)
        
        # Determinar janela de interpolação
        window_start, window_end = determine_window(
            x, y,
            threshold_start=cfg.window_threshold_start,
            threshold_end=cfg.window_threshold_end,
            smooth_sigma=cfg.smooth_sigma,
            min_window_size=cfg.min_window_size
        )
        
        if window_start is None:
            return CurveFitResult.failed(model_name)
        
        # Extrair dados da janela
        mask = (x >= window_start) & (x <= window_end)
        x_window = x[mask]
        y_window = y[mask]
        
        if len(x_window) < 2:
            return CurveFitResult.failed(model_name)
        
        # Ajustar modelo
        best_params, best_error = self._fit_model(
            x_window, y_window, model, cfg
        )
        
        if best_params is None:
            return CurveFitResult.failed(model_name)
        
        # Calcular curvas ajustadas
        y_fitted = model.equation(x, **best_params)
        dy_fitted = model.derivative1(x, **best_params)
        
        # Calcular segunda derivada numericamente a partir de dy_fitted
        # Isso garante consistência de sinais para curvas de decaimento
        # (a fórmula analítica pode ter inversão dependendo do sinal de K)
        ddy_fitted = np.gradient(dy_fitted, x)
        
        return CurveFitResult(
            success=True,
            model_name=model_name,
            params=best_params,
            error=best_error,
            window_start=window_start,
            window_end=window_end,
            x_fitted=x,
            y_fitted=y_fitted,
            dy_fitted=dy_fitted,
            ddy_fitted=ddy_fitted,
        )
    
    def _fit_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        model: MathModel,
        cfg: CurveFitConfig
    ) -> tuple[Optional[dict], float]:
        """Ajusta o modelo usando múltiplas inicializações."""
        best_params = None
        best_error = float('inf')
        
        lower, upper = model.bounds()
        
        for _ in range(cfg.max_attempts):
            # Gerar chute inicial
            p0 = model.initial_guess(x, y)
            p0_list = [p0[name] for name in model.param_names]
            
            try:
                params, _ = curve_fit(
                    model,
                    x, y,
                    p0=p0_list,
                    bounds=(lower, upper),
                    maxfev=5000
                )
                
                # Calcular erro
                y_pred = model(x, *params)
                residuals = y - y_pred
                mse = np.mean(residuals ** 2)
                
                if mse < best_error:
                    best_error = mse
                    best_params = dict(zip(model.param_names, params))
                    
                    if mse <= cfg.tolerance:
                        break
                        
            except Exception:
                continue
        
        return best_params, best_error
    
    def fit_best(
        self,
        x: np.ndarray,
        y: np.ndarray,
        models: list[str] = None,
        config: CurveFitConfig = None
    ) -> CurveFitResult:
        """
        Tenta múltiplos modelos e retorna o melhor ajuste.
        
        Args:
            x: Array de timestamps
            y: Array de valores
            models: Lista de nomes de modelos para tentar (default: todos)
            config: Configuração opcional
        
        Returns:
            CurveFitResult do melhor modelo
        """
        if models is None:
            models = ModelRegistry.list_models()
        
        best_result = CurveFitResult.failed()
        
        for model_name in models:
            result = self.fit(x, y, model_name, config)
            if result.success and result.error < best_result.error:
                best_result = result
        
        return best_result
