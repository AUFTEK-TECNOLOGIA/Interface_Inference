"""
Model Registry - Gerenciamento de modelos ML.

Este módulo fornece:
- Carregamento de modelos do registry JSON
- Validação de modelos
- Descoberta automática de modelos
- Versionamento de modelos
"""

from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ModelInfo:
    """Informações de um modelo registrado."""
    id: str
    name: str
    description: str
    model_path: str
    scaler_path: str
    
    # Metadata
    training_date: Optional[str] = None
    algorithm: Optional[str] = None
    framework: Optional[str] = None
    
    # Input
    recommended_feature: str = "inflection_time"
    supported_features: list[str] = field(default_factory=list)
    feature_ranges: dict = field(default_factory=dict)
    
    # Output
    output_type: str = "regression"
    output_unit: str = ""
    output_range: dict = field(default_factory=dict)
    
    # Status
    status: str = "active"
    tags: list[str] = field(default_factory=list)
    
    # Computed
    model_exists: bool = False
    scaler_exists: bool = False
    model_hash: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model_path": self.model_path,
            "scaler_path": self.scaler_path,
            "training_date": self.training_date,
            "algorithm": self.algorithm,
            "recommended_feature": self.recommended_feature,
            "supported_features": self.supported_features,
            "output_unit": self.output_unit,
            "status": self.status,
            "model_exists": self.model_exists,
            "scaler_exists": self.scaler_exists,
        }


# =============================================================================
# MODEL REGISTRY
# =============================================================================

class ModelRegistry:
    """
    Registry central de modelos ML.
    
    Carrega modelos de um arquivo JSON e fornece métodos para
    buscar, validar e obter informações sobre modelos.
    """
    
    _instance: Optional["ModelRegistry"] = None
    _registry_path: Optional[Path] = None
    _models: dict[str, ModelInfo] = {}
    _defaults: dict = {}
    _feature_definitions: dict = {}
    _loaded: bool = False
    _last_load: Optional[datetime] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        """Retorna instância singleton."""
        return cls()
    
    def load(self, registry_path: str | Path = None, force: bool = False) -> bool:
        """
        Carrega o registry de modelos.
        
        Args:
            registry_path: Caminho para o JSON do registry
            force: Se True, recarrega mesmo se já carregado
        
        Returns:
            True se carregou com sucesso
        """
        if self._loaded and not force:
            return True
        
        # Determinar path do registry
        if registry_path:
            path = Path(registry_path)
        else:
            # Tentar encontrar automaticamente
            current = Path(__file__).parent
            candidates = [
                current.parent.parent.parent / "resources" / "model_registry.json",
                current.parent.parent / "resources" / "model_registry.json",
                Path("resources/model_registry.json"),
            ]
            path = None
            for c in candidates:
                if c.exists():
                    path = c
                    break
            
            if path is None:
                logger.warning("model_registry.json não encontrado")
                return False
        
        if not path.exists():
            logger.error(f"Registry não encontrado: {path}")
            return False
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._registry_path = path
            self._defaults = data.get("defaults", {})
            self._feature_definitions = data.get("feature_definitions", {})
            
            # Carregar modelos
            self._models = {}
            models_data = data.get("models", {})
            
            project_root = path.parent.parent  # resources/ -> project root
            
            for model_id, model_data in models_data.items():
                model_info = self._parse_model(model_id, model_data, project_root)
                self._models[model_id] = model_info
            
            self._loaded = True
            self._last_load = datetime.now()
            
            logger.info(f"Registry carregado: {len(self._models)} modelos de {path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar registry: {e}")
            return False
    
    def _parse_model(self, model_id: str, data: dict, project_root: Path) -> ModelInfo:
        """Parse dados de um modelo do JSON."""
        model_path = data.get("model_path", "")
        scaler_path = data.get("scaler_path", "")
        
        # Resolver paths
        model_full_path = project_root / model_path if model_path else None
        scaler_full_path = project_root / scaler_path if scaler_path else None
        
        # Verificar existência
        model_exists = model_full_path.exists() if model_full_path else False
        scaler_exists = scaler_full_path.exists() if scaler_full_path else False
        
        # Calcular hash do modelo (para versionamento)
        model_hash = None
        if model_exists:
            try:
                model_hash = self._file_hash(model_full_path)
            except Exception:
                pass
        
        # Extrair metadados
        metadata = data.get("metadata", {})
        input_config = data.get("input", {})
        output_config = data.get("output", {})
        
        return ModelInfo(
            id=model_id,
            name=data.get("name", model_id),
            description=data.get("description", ""),
            model_path=str(model_full_path) if model_full_path else model_path,
            scaler_path=str(scaler_full_path) if scaler_full_path else scaler_path,
            training_date=metadata.get("training_date"),
            algorithm=metadata.get("algorithm"),
            framework=metadata.get("framework"),
            recommended_feature=input_config.get("recommended_feature", "inflection_time"),
            supported_features=input_config.get("supported_features", []),
            feature_ranges=input_config.get("feature_ranges", {}),
            output_type=output_config.get("type", "regression"),
            output_unit=output_config.get("unit", ""),
            output_range=output_config.get("range", {}),
            status=data.get("status", "active"),
            tags=data.get("tags", []),
            model_exists=model_exists,
            scaler_exists=scaler_exists,
            model_hash=model_hash,
        )
    
    @staticmethod
    def _file_hash(path: Path, algorithm: str = "md5") -> str:
        """Calcula hash de um arquivo."""
        h = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Retorna informações de um modelo."""
        self.load()
        return self._models.get(model_id)
    
    def get_model_or_default(self, model_id: str = None) -> Optional[ModelInfo]:
        """Retorna modelo especificado ou o default."""
        self.load()
        
        if model_id:
            model = self._models.get(model_id)
            if model:
                return model
        
        # Usar default
        default_id = self._defaults.get("resource")
        if default_id:
            return self._models.get(default_id)
        
        # Primeiro disponível
        if self._models:
            return next(iter(self._models.values()))
        
        return None
    
    def list_models(
        self,
        status: str = None,
        tags: list[str] = None,
        only_available: bool = False
    ) -> list[ModelInfo]:
        """
        Lista modelos com filtros opcionais.
        
        Args:
            status: Filtrar por status (ex: "active")
            tags: Filtrar por tags (AND)
            only_available: Se True, apenas modelos cujos arquivos existem
        
        Returns:
            Lista de ModelInfo
        """
        self.load()
        
        result = []
        for model in self._models.values():
            # Filtrar por status
            if status and model.status != status:
                continue
            
            # Filtrar por tags
            if tags:
                if not all(t in model.tags for t in tags):
                    continue
            
            # Filtrar por disponibilidade
            if only_available:
                if not model.model_exists or not model.scaler_exists:
                    continue
            
            result.append(model)
        
        return result
    
    def list_model_ids(self) -> list[str]:
        """Retorna lista de IDs de modelos."""
        self.load()
        return list(self._models.keys())
    
    def get_default_feature(self, model_id: str = None) -> str:
        """Retorna feature padrão para um modelo."""
        model = self.get_model(model_id) if model_id else None
        
        if model and model.recommended_feature:
            return model.recommended_feature
        
        return self._defaults.get("input_feature", "inflection_time")
    
    def get_feature_definition(self, feature_name: str) -> dict:
        """Retorna definição de uma feature."""
        self.load()
        return self._feature_definitions.get(feature_name, {})
    
    def validate_model(self, model_id: str) -> tuple[bool, list[str]]:
        """
        Valida se um modelo está pronto para uso.
        
        Returns:
            Tuple de (is_valid, errors)
        """
        self.load()
        errors = []
        
        model = self._models.get(model_id)
        if not model:
            errors.append(f"Modelo '{model_id}' não encontrado no registry")
            return False, errors
        
        if not model.model_exists:
            errors.append(f"Arquivo de modelo não encontrado: {model.model_path}")
        
        if not model.scaler_exists:
            errors.append(f"Arquivo de scaler não encontrado: {model.scaler_path}")
        
        if model.status != "active":
            errors.append(f"Modelo não está ativo (status: {model.status})")
        
        return len(errors) == 0, errors
    
    def to_available_resources_format(self) -> dict:
        """
        Converte para formato AVAILABLE_RESOURCES (compatibilidade).
        
        Retorna dict no formato usado pelo MLInferenceBlock.
        """
        self.load()
        
        result = {}
        for model_id, model in self._models.items():
            if model.status != "active":
                continue
            
            result[model_id] = {
                "model": model.model_path,
                "scaler": model.scaler_path,
                "description": model.description,
                "output_unit": model.output_unit,
                "recommended_feature": model.recommended_feature,
            }
        
        return result
    
    def reload(self) -> bool:
        """Força recarregamento do registry."""
        return self.load(force=True)


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

def get_model_registry() -> ModelRegistry:
    """Retorna instância singleton do registry."""
    return ModelRegistry.get_instance()


def get_model(model_id: str) -> Optional[ModelInfo]:
    """Atalho para obter um modelo."""
    return get_model_registry().get_model(model_id)


def list_available_models() -> list[str]:
    """Lista IDs de modelos disponíveis."""
    return get_model_registry().list_model_ids()


def get_available_resources() -> dict:
    """
    Retorna dict de recursos no formato AVAILABLE_RESOURCES.
    
    Compatível com o formato usado pelo MLInferenceBlock.
    """
    return get_model_registry().to_available_resources_format()
