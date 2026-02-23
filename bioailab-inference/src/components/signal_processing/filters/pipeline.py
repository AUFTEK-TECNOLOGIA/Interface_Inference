"""
Pipeline de Filtros.

Permite encadear múltiplos filtros em sequência.
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional
from .base import SignalFilter, FilterRegistry


class FilterPipeline:
    """
    Pipeline para aplicação sequencial de múltiplos filtros.
    
    Permite configurar uma cadeia de filtros que serão aplicados
    em ordem ao sinal de entrada.
    
    Uso:
        pipeline = FilterPipeline([
            {"name": "outlier_mad", "threshold": 3.0},
            {"name": "median", "window": 5},
            {"name": "savgol", "window": 11, "polyorder": 3}
        ])
        filtered = pipeline.apply(signal)
    """
    
    def __init__(self, filters: Optional[List[Dict[str, Any]]] = None):
        """
        Inicializa o pipeline.
        
        Args:
            filters: Lista de configurações de filtros.
                     Cada item é um dict com "name" e parâmetros opcionais.
        """
        self.filter_configs = filters or []
        self.filters: List[SignalFilter] = []
        self._build_filters()
    
    def _build_filters(self) -> None:
        """Constrói instâncias dos filtros a partir das configurações."""
        self.filters = []
        
        for config in self.filter_configs:
            if isinstance(config, str):
                # Apenas nome do filtro, sem parâmetros
                filter_instance = FilterRegistry.create(config)
            elif isinstance(config, dict):
                name = config.get("name")
                if not name:
                    raise ValueError("Configuração de filtro deve ter 'name'")
                
                # Extrair parâmetros (tudo exceto 'name')
                params = {k: v for k, v in config.items() if k != "name"}
                filter_instance = FilterRegistry.create(name, **params)
            elif isinstance(config, SignalFilter):
                # Já é uma instância
                filter_instance = config
            else:
                raise ValueError(f"Configuração inválida: {config}")
            
            self.filters.append(filter_instance)
    
    def add(self, filter_name: str, **kwargs) -> "FilterPipeline":
        """
        Adiciona um filtro ao pipeline.
        
        Args:
            filter_name: Nome do filtro
            **kwargs: Parâmetros do filtro
        
        Returns:
            self para encadeamento
        """
        config = {"name": filter_name, **kwargs}
        self.filter_configs.append(config)
        self.filters.append(FilterRegistry.create(filter_name, **kwargs))
        return self
    
    def remove(self, index: int) -> "FilterPipeline":
        """Remove filtro pelo índice."""
        if 0 <= index < len(self.filters):
            del self.filter_configs[index]
            del self.filters[index]
        return self
    
    def clear(self) -> "FilterPipeline":
        """Remove todos os filtros."""
        self.filter_configs = []
        self.filters = []
        return self
    
    def apply(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """
        Aplica todos os filtros em sequência.
        
        Args:
            signal: Sinal de entrada
            **kwargs: Argumentos extras passados para cada filtro
        
        Returns:
            Sinal filtrado
        """
        if len(signal) == 0:
            return signal
        
        result = signal.copy()
        
        for filter_instance in self.filters:
            result = filter_instance.apply(result, **kwargs)
        
        return result
    
    def apply_with_history(
        self,
        signal: np.ndarray,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Aplica filtros e retorna resultado de cada etapa.
        
        Útil para debug e visualização do efeito de cada filtro.
        
        Returns:
            Dict com "original", resultados intermediários e "final"
        """
        history = {"original": signal.copy()}
        
        result = signal.copy()
        for i, filter_instance in enumerate(self.filters):
            result = filter_instance.apply(result, **kwargs)
            history[f"step_{i}_{filter_instance.name}"] = result.copy()
        
        history["final"] = result
        return history
    
    def describe(self) -> List[Dict[str, Any]]:
        """Retorna descrição de todos os filtros no pipeline."""
        return [
            {
                "name": f.name,
                "description": f.description,
                "params": f.params
            }
            for f in self.filters
        ]
    
    @classmethod
    def from_preset(cls, preset_config: Dict[str, Any]) -> "FilterPipeline":
        """
        Cria pipeline a partir de configuração de preset.
        
        Args:
            preset_config: Dict com "filters" contendo lista de configs
        
        Returns:
            FilterPipeline configurado
        """
        filters = preset_config.get("filters", [])
        return cls(filters)
    
    def __len__(self) -> int:
        return len(self.filters)
    
    def __repr__(self) -> str:
        filter_names = [f.name for f in self.filters]
        return f"FilterPipeline([{' -> '.join(filter_names)}])"


# Conveniência: pipeline pré-configurados
def create_denoising_pipeline(aggressive: bool = False) -> FilterPipeline:
    """
    Cria pipeline padrão para remoção de ruído.
    
    Args:
        aggressive: Se True, usa parâmetros mais agressivos
    """
    if aggressive:
        return FilterPipeline([
            {"name": "outlier_mad", "threshold": 2.5},
            {"name": "median", "window": 7},
            {"name": "savgol", "window": 21, "polyorder": 3}
        ])
    else:
        return FilterPipeline([
            {"name": "outlier_std", "threshold": 3.0},
            {"name": "median", "window": 5},
            {"name": "ema", "alpha": 0.3}
        ])


def create_smoothing_pipeline(method: str = "savgol") -> FilterPipeline:
    """
    Cria pipeline para suavização de sinal.
    
    Args:
        method: Método principal ('savgol', 'lowpass', 'moving_average')
    """
    if method == "savgol":
        return FilterPipeline([
            {"name": "savgol", "window": 11, "polyorder": 3}
        ])
    elif method == "lowpass":
        return FilterPipeline([
            {"name": "butterworth", "cutoff_freq": 0.1, "order": 4}
        ])
    elif method == "moving_average":
        return FilterPipeline([
            {"name": "moving_average", "window": 10}
        ])
    else:
        raise ValueError(f"Método desconhecido: {method}")


def create_growth_analysis_pipeline() -> FilterPipeline:
    """
    Pipeline otimizado para análise de curvas de crescimento.
    
    Remove outliers, suaviza mantendo características importantes
    como lag phase, exponential phase e stationary phase.
    """
    return FilterPipeline([
        {"name": "outlier_iqr", "k": 2.0, "method": "interpolate"},
        {"name": "savgol", "window": 15, "polyorder": 3}
    ])
