"""
Serviço de inferência ML.

Orquestra a execução de modelos usando diferentes adaptadores.
"""

from typing import Optional, List
import numpy as np

from .base import MLAdapter, AdapterRegistry, InferenceResult


class InferenceService:
    """
    Serviço para execução de inferência ML.
    
    Permite usar diferentes backends e estratégias de fallback.
    """
    
    def __init__(
        self,
        default_adapter: str = "onnx",
        fallback_adapters: List[str] = None
    ):
        """
        Inicializa o serviço.
        
        Args:
            default_adapter: Nome do adaptador padrão
            fallback_adapters: Lista de adaptadores para fallback
        """
        self.default_adapter = default_adapter
        self.fallback_adapters = fallback_adapters or []
    
    def predict(
        self,
        model_path: str,
        features: np.ndarray,
        adapter_name: str = None,
        **kwargs
    ) -> InferenceResult:
        """
        Executa predição com fallback automático.
        
        Args:
            model_path: Caminho do modelo
            features: Array de features
            adapter_name: Nome do adaptador (usa default se None)
            **kwargs: Parâmetros adicionais (scaler_path, etc.)
        
        Returns:
            InferenceResult com predição
        """
        adapters_to_try = [adapter_name or self.default_adapter] + self.fallback_adapters
        
        for name in adapters_to_try:
            adapter = AdapterRegistry.get(name)
            if adapter is None:
                continue
            
            result = adapter.predict(model_path, features, **kwargs)
            if result.success:
                return result
        
        return InferenceResult.failed("Todos os adaptadores falharam")
    
    def predict_single(
        self,
        model_path: str,
        feature_value: float,
        scaler_path: str = None,
        adapter_name: str = None
    ) -> Optional[float]:
        """
        Predição simplificada para uma única feature.
        
        Args:
            model_path: Caminho do modelo
            feature_value: Valor da feature
            scaler_path: Caminho do scaler
            adapter_name: Nome do adaptador
        
        Returns:
            Valor predito ou None
        """
        features = np.array([[feature_value]], dtype=np.float32)
        result = self.predict(
            model_path, features,
            adapter_name=adapter_name,
            scaler_path=scaler_path
        )
        return result.value if result.success else None
    
    def batch_predict(
        self,
        model_path: str,
        features_list: List[np.ndarray],
        **kwargs
    ) -> List[InferenceResult]:
        """
        Executa predições em lote.
        
        Args:
            model_path: Caminho do modelo
            features_list: Lista de arrays de features
            **kwargs: Parâmetros adicionais
        
        Returns:
            Lista de resultados
        """
        return [
            self.predict(model_path, features, **kwargs)
            for features in features_list
        ]
    
    def preload_models(self, model_configs: List[dict]):
        """
        Pré-carrega modelos para melhor performance.
        
        Args:
            model_configs: Lista de dicts com model_path e outros params
        """
        adapter = AdapterRegistry.get(self.default_adapter)
        if adapter is None:
            return
        
        for config in model_configs:
            adapter.load_model(**config)
