"""
Framework de Pipeline em Blocos.

Este módulo fornece a infraestrutura para criar pipelines modulares
onde componentes podem ser conectados dinamicamente.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Type, Union
from datetime import datetime
import time


@dataclass
class BlockContext:
    """Contexto de execução de um bloco."""
    block_id: str
    pipeline_id: str
    execution_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_success(self):
        """Marca execução como bem-sucedida."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = True

    def mark_failure(self, error: str):
        """Marca execução como falha."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = False
        self.error_message = error


@dataclass
class BlockInput:
    """Entrada de dados para um bloco."""
    data: Dict[str, Any]
    context: BlockContext
    previous_outputs: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Obtém valor da entrada."""
        return self.data.get(key, default)

    def get_required(self, key: str) -> Any:
        """Obtém valor obrigatório da entrada."""
        if key not in self.data:
            raise ValueError(f"Campo obrigatório '{key}' não encontrado na entrada")
        return self.data[key]


@dataclass
class BlockOutput:
    """Saída de dados de um bloco."""
    data: Dict[str, Any]
    context: BlockContext
    next_blocks: List[str] = field(default_factory=list)

    def set(self, key: str, value: Any):
        """Define valor na saída."""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Obtém valor da saída."""
        return self.data.get(key, default)


class Block(ABC):
    """
    Interface base para blocos de pipeline.

    Cada bloco representa uma unidade de processamento independente
    que pode ser conectada a outros blocos.
    """

    name: str = "base_block"
    description: str = "Bloco base"
    version: str = "1.0.0"

    input_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    output_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __init__(self, **config):
        """Inicializa o bloco com configuração."""
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Valida configuração do bloco."""
        pass  # Override em subclasses

    @abstractmethod
    def execute(self, input_data: BlockInput) -> BlockOutput:
        """
        Executa o processamento do bloco.

        Args:
            input_data: Dados de entrada com contexto

        Returns:
            Dados de saída com contexto atualizado
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o bloco."""
        # data_inputs: lista explícita de inputs para handles no frontend
        # Se não definida, o frontend usa as keys do input_schema
        data_inputs = getattr(self, 'data_inputs', None)
        if data_inputs is None:
            # Gerar automaticamente a partir do input_schema
            data_inputs = list(self.input_schema.keys()) if self.input_schema else []
        
        # data_outputs: lista explícita de outputs para handles no frontend
        data_outputs = getattr(self, 'data_outputs', None)
        if data_outputs is None:
            data_outputs = list(self.output_schema.keys()) if self.output_schema else []
        
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "config": self.config,
            "config_inputs": getattr(self, 'config_inputs', []),
            "config_schema": getattr(self, 'config_schema', {}),
            "data_inputs": data_inputs,
            "data_outputs": data_outputs,
        }


class BlockRegistry:
    """
    Registry para blocos de pipeline.

    Permite registrar e recuperar blocos por nome.
    """

    _blocks: Dict[str, Type[Block]] = {}

    @classmethod
    def register(cls, block_class: Type[Block]) -> Type[Block]:
        """
        Decorator para registrar um bloco.

        Uso:
            @BlockRegistry.register
            class MyBlock(Block):
                name = "my_block"
                ...
        """
        cls._blocks[block_class.name] = block_class
        return block_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[Block]]:
        """Retorna classe do bloco pelo nome."""
        return cls._blocks.get(name)

    @classmethod
    def create(cls, name: str, **config) -> Block:
        """
        Cria instância de um bloco pelo nome.

        Args:
            name: Nome do bloco registrado
            **config: Parâmetros de configuração

        Returns:
            Instância do bloco

        Raises:
            ValueError: Se bloco não encontrado
        """
        block_class = cls._blocks.get(name)
        if block_class is None:
            available = list(cls._blocks.keys())
            raise ValueError(f"Bloco '{name}' não encontrado. Disponíveis: {available}")
        return block_class(**config)

    @classmethod
    def list_blocks(cls) -> List[str]:
        """Lista todos os blocos registrados."""
        return list(cls._blocks.keys())

    @classmethod
    def get_info(cls) -> List[Dict[str, Any]]:
        """Retorna informações de todos os blocos."""
        return [block_class().get_info() for block_class in cls._blocks.values()]