"""
Adaptador para inferência ONNX.

Responsabilidade única: carregar e executar modelos ONNX.
"""

from typing import Dict, Any, Optional
import numpy as np

from .base import MLAdapter, AdapterRegistry, InferenceResult


@AdapterRegistry.register
class OnnxAdapter(MLAdapter):
    """
    Adaptador para modelos ONNX Runtime.
    
    Suporta modelos no formato .onnx com scalers opcionais.
    """
    
    name = "onnx"
    description = "ONNX Runtime para modelos de ML"
    supported_formats = [".onnx"]
    
    def __init__(self):
        super().__init__()
        self._ort = None
        self._joblib = None
    
    def _ensure_imports(self):
        """Importa dependências sob demanda."""
        if self._ort is None:
            import onnxruntime as ort
            self._ort = ort
        if self._joblib is None:
            import joblib
            self._joblib = joblib
    
    def load_model(
        self,
        model_path: str,
        scaler_path: str = None,
        **kwargs
    ) -> bool:
        """
        Carrega modelo ONNX e scaler opcional.
        
        Args:
            model_path: Caminho para arquivo .onnx
            scaler_path: Caminho para scaler .joblib (opcional)
        
        Returns:
            True se carregou com sucesso
        """
        cache_key = self._make_cache_key(model_path, scaler_path)
        
        if cache_key in self._model_cache:
            return True
        
        try:
            self._ensure_imports()
            
            available = []
            if hasattr(self._ort, "get_available_providers"):
                available = self._ort.get_available_providers()

            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            session = self._ort.InferenceSession(model_path, providers=providers)
            
            scaler = None
            if scaler_path:
                scaler = self._joblib.load(scaler_path)
            
            self._model_cache[cache_key] = {
                "session": session,
                "input_name": session.get_inputs()[0].name,
                "output_name": session.get_outputs()[0].name,
                "scaler": scaler,
            }
            return True
            
        except Exception as e:
            print(f"Erro ao carregar modelo ONNX: {e}")
            return False
    
    def predict(
        self,
        model_path: str,
        features: np.ndarray,
        scaler_path: str = None,
        **kwargs
    ) -> InferenceResult:
        """
        Executa predição com modelo ONNX.
        
        Args:
            model_path: Caminho do modelo
            features: Array de features (pode ser 1D ou 2D)
            scaler_path: Caminho do scaler (opcional)
        
        Returns:
            InferenceResult com predição
        """
        cache_key = self._make_cache_key(model_path, scaler_path)
        
        # Carrega se necessário
        if cache_key not in self._model_cache:
            if not self.load_model(model_path, scaler_path):
                return InferenceResult.failed("Falha ao carregar modelo")
        
        model = self._model_cache[cache_key]
        
        try:
            # Garantir formato 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            features = features.astype(np.float32)
            
            # Aplicar scaler se disponível
            if model["scaler"] is not None:
                features = model["scaler"].transform(features)
                features = features.astype(np.float32)
            
            # Executar inferência
            input_feed = {model["input_name"]: features}
            output = model["session"].run([model["output_name"]], input_feed)[0]
            
            value = float(output[0][0]) if output.ndim > 1 else float(output[0])
            
            return InferenceResult.ok(value, confidence=1.0, backend="onnx")
            
        except Exception as e:
            return InferenceResult.failed(f"Erro na inferência: {e}")

    def predict_raw(
        self,
        model_path: str,
        features: np.ndarray,
        scaler_path: str = None,
        **kwargs
    ) -> tuple[bool, Optional[np.ndarray], Optional[str]]:
        """
        Executa inferência e retorna o output bruto (np.ndarray).

        Útil para modelos que retornam vetores/séries (ex: embeddings, séries tratadas, probabilidades).
        """
        cache_key = self._make_cache_key(model_path, scaler_path)

        if cache_key not in self._model_cache:
            if not self.load_model(model_path, scaler_path):
                return False, None, "Falha ao carregar modelo"

        model = self._model_cache[cache_key]

        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)

            features = features.astype(np.float32)

            if model["scaler"] is not None:
                features = model["scaler"].transform(features)
                features = features.astype(np.float32)

            input_feed = {model["input_name"]: features}
            output = model["session"].run([model["output_name"]], input_feed)[0]
            return True, output, None
        except Exception as e:
            return False, None, f"Erro na inferência: {e}"
    
    def _make_cache_key(self, model_path: str, scaler_path: str = None) -> str:
        """Gera chave de cache."""
        return f"{model_path}:{scaler_path or 'no_scaler'}"


# Manter compatibilidade com código existente
class OnnxInferenceAdapter:
    """
    Wrapper de compatibilidade para o adaptador ONNX.
    
    Mantém a interface antiga enquanto usa o novo sistema.
    """
    
    def __init__(self):
        self._adapter = AdapterRegistry.get("onnx") or OnnxAdapter()
    
    def load_model(self, model_path: str, scaler_path: str) -> bool:
        return self._adapter.load_model(model_path, scaler_path)
    
    def predict(
        self,
        model_path: str,
        scaler_path: str,
        feature_value: float
    ) -> Optional[float]:
        """Interface de compatibilidade com código existente."""
        features = np.array([[feature_value]], dtype=np.float32)
        result = self._adapter.predict(model_path, features, scaler_path)
        return result.value if result.success else None

    def predict_raw(
        self,
        model_path: str,
        features: np.ndarray,
        scaler_path: str = None,
    ) -> np.ndarray:
        ok, output, error = getattr(self._adapter, "predict_raw")(model_path, features, scaler_path)
        if not ok or output is None:
            raise RuntimeError(error or "Falha ao executar inferência")
        return output
    
    def clear_cache(self):
        self._adapter.clear_cache()


# Singleton para compatibilidade
_inference_adapter = None


def get_inference_adapter() -> OnnxInferenceAdapter:
    """Retorna instância singleton do adaptador (compatibilidade)."""
    global _inference_adapter
    if _inference_adapter is None:
        _inference_adapter = OnnxInferenceAdapter()
    return _inference_adapter
