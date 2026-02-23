"""
Engine de Pipeline.

Responsável por executar pipelines de blocos de forma sequencial ou paralela.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import uuid
import time

from .base import Block, BlockInput, BlockOutput, BlockContext, BlockRegistry


@dataclass
class PipelineStep:
    """Passo de um pipeline."""
    block_name: str
    block_config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # IDs dos passos que devem executar antes
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_mapping: Dict[str, str] = field(default_factory=dict)  # Mapeia input_name -> "step_id.output_name"


@dataclass
class PipelineConfig:
    """Configuração de um pipeline."""
    name: str
    description: str = ""
    steps: List[PipelineStep] = field(default_factory=list)
    max_parallel: int = 1  # Máximo de blocos executando em paralelo
    timeout_seconds: float = 300.0  # Timeout total do pipeline
    fail_fast: bool = True  # Parar no primeiro erro
    generate_output_graphs: bool = False  # Gerar gráficos de saída para cada bloco


@dataclass
class PipelineResult:
    """Resultado da execução de um pipeline."""
    pipeline_id: str
    success: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    step_results: Dict[str, BlockOutput] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_completed(self, success: bool = True):
        """Marca pipeline como concluído."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success


class PipelineEngine:
    """
    Engine para executar pipelines de blocos.

    Permite executar sequências de blocos com dependências,
    controle de fluxo e tratamento de erros.
    """

    def __init__(self, config: PipelineConfig):
        """
        Inicializa o engine com configuração.

        Args:
            config: Configuração do pipeline
        """
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Valida configuração do pipeline."""
        if not self.config.steps:
            raise ValueError("Pipeline deve ter pelo menos um passo")

        # Validar dependências
        step_ids = {step.step_id for step in self.config.steps}
        for step in self.config.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise ValueError(f"Dependência '{dep}' não encontrada para passo '{step.step_id}'")

        # Verificar ciclos (simples - não detecta ciclos complexos)
        # TODO: Implementar detecção de ciclos mais sofisticada

    def execute(self, initial_data: Dict[str, Any]) -> PipelineResult:
        """
        Executa o pipeline.

        Args:
            initial_data: Dados iniciais para o primeiro bloco

        Returns:
            Resultado da execução
        """
        pipeline_id = str(uuid.uuid4())
        result = PipelineResult(
            pipeline_id=pipeline_id,
            success=False,
            start_time=datetime.now()
        )
        state: Dict[str, Any] = initial_data.copy()

        try:
            # Preparar contexto inicial
            context = BlockContext(
                block_id="pipeline_start",
                pipeline_id=pipeline_id,
                execution_id=pipeline_id
            )

            # Executar passos em ordem topológica
            executed_steps: Dict[str, BlockOutput] = {}
            failed_steps: Set[str] = set()
            remaining_steps = self.config.steps.copy()

            start_time = time.time()
            
            # DEBUG: contar iterações
            iteration = 0

            while remaining_steps and (time.time() - start_time) < self.config.timeout_seconds:
                iteration += 1
                # Encontrar passos prontos para executar
                ready_steps = []
                for step in remaining_steps:
                    # Verificar se todas as dependências foram executadas
                    deps_satisfied = all(dep in executed_steps for dep in step.depends_on)
                    if deps_satisfied:
                        ready_steps.append(step)

                if not ready_steps and remaining_steps:
                    # Pode ser deadlock real (ciclo) ou bloqueio por falha em dependências.
                    newly_blocked: List[PipelineStep] = []
                    for step in remaining_steps:
                        failed_deps = [dep for dep in step.depends_on if dep in failed_steps]
                        if not failed_deps:
                            continue

                        blocked_context = BlockContext(
                            block_id=step.step_id,
                            pipeline_id=pipeline_id,
                            execution_id=f"{pipeline_id}_{step.step_id}"
                        )
                        blocked_context.metadata["blocked"] = True
                        blocked_context.metadata["blocked_by"] = failed_deps
                        blocked_context.mark_failure(f"Bloqueado por falha em dependências: {failed_deps}")
                        result.step_results[step.step_id] = BlockOutput(
                            data={"_blocked": True, "_blocked_by": failed_deps},
                            context=blocked_context
                        )
                        result.errors.append(f"Passo '{step.step_id}' bloqueado: dependências falharam {failed_deps}")
                        newly_blocked.append(step)

                    if newly_blocked:
                        for step in newly_blocked:
                            if step in remaining_steps:
                                remaining_steps.remove(step)
                        continue

                    # Deadlock - dependências circulares ou passos faltando
                    result.errors.append("Deadlock detectado: dependências não podem ser satisfeitas (possível ciclo)")
                    break

                # Executar passos prontos (por enquanto sequencial)
                for step in ready_steps:
                    step_result = self._execute_step(step, state, executed_steps, pipeline_id)

                    # Sempre registrar resultado do step (sucesso, skip, falha)
                    result.step_results[step.step_id] = step_result

                    if step_result.context.success:
                        executed_steps[step.step_id] = step_result
                        # NÃO atualizar state global com outputs - evita vazamento entre branches paralelas
                        # Os outputs são acessados via input_mapping específico de cada step
                    else:
                        failed_steps.add(step.step_id)
                        result.errors.append(f"Passo '{step.step_id}' falhou: {step_result.context.error_message}")
                        if self.config.fail_fast:
                            break

                    remaining_steps.remove(step)

                if self.config.fail_fast and result.errors:
                    break

            # Verificar se todos os passos foram executados
            if not remaining_steps and not result.errors:
                result.mark_completed(success=True)
            else:
                result.mark_completed(success=False)

        except Exception as e:
            result.errors.append(f"Erro geral no pipeline: {str(e)}")
            result.mark_completed(success=False)

        return result

    def _execute_step(
        self,
        step: PipelineStep,
        state: Dict[str, Any],
        previous_outputs: Dict[str, BlockOutput],
        pipeline_id: str
    ) -> BlockOutput:
        """
        Executa um passo individual do pipeline.

        Args:
            step: Configuração do passo
            initial_data: Dados iniciais
            previous_outputs: Resultados dos passos anteriores
            pipeline_id: ID do pipeline

        Returns:
            Resultado da execução do passo
        """
        context = BlockContext(
            block_id=step.step_id,
            pipeline_id=pipeline_id,
            execution_id=f"{pipeline_id}_{step.step_id}"
        )
        context.metadata["block_config"] = step.block_config
        # Suporte a override por bloco: se o bloco fornecer 'generate_output_graphs', usar esse valor
        if isinstance(step.block_config, dict) and "generate_output_graphs" in step.block_config:
            context.metadata["generate_output_graphs"] = bool(step.block_config.get("generate_output_graphs"))
        else:
            context.metadata["generate_output_graphs"] = self.config.generate_output_graphs

        try:
            # Criar bloco
            block = BlockRegistry.create(step.block_name, **step.block_config)

            # Preparar entrada - usar APENAS dados do input_mapping para evitar vazamento entre branches
            # O state global só é usado para dados iniciais do pipeline (não de outros steps)
            step_input_data = {}
            
            # Aplicar input_mapping: mapeia outputs específicos de passos anteriores para inputs deste passo
            # Formato: {"experiment_data": "experiment_fetch_1.experiment_data"}
            if step.input_mapping:
                for input_name, source_ref in step.input_mapping.items():
                    if "." in source_ref:
                        source_step, output_name = source_ref.rsplit(".", 1)
                        if source_step in previous_outputs:
                            source_data = previous_outputs[source_step].data
                            if output_name in source_data:
                                step_input_data[input_name] = source_data[output_name]
                    else:
                        # Referência direta do state inicial (não de outros steps)
                        if source_ref in state:
                            step_input_data[input_name] = state[source_ref]
            
            # Se não há input_mapping (primeiro bloco), usar state inicial
            if not step.input_mapping:
                step_input_data = state.copy()
            
            # Verificar se algum input obrigatório está inativo (controle de fluxo condicional)
            # Se estiver, marcar este bloco como pulado e propagar a inatividade para TODOS os outputs
            is_inactive = self._check_inactive_inputs(step_input_data, block)
            
            if is_inactive:
                context.mark_success()
                context.metadata["skipped"] = True
                context.metadata["skip_reason"] = "Input inativo (fluxo condicional)"
                
                # Propagar inatividade para TODOS os outputs do bloco
                # Isso permite que blocos subsequentes encontrem as chaves esperadas
                inactive_marker = {"_inactive": True, "_reason": "Input from inactive branch"}
                output_data = {}
                output_schema = getattr(block, 'output_schema', {})
                for output_name in output_schema.keys():
                    output_data[output_name] = inactive_marker.copy()
                
                # Se não houver output_schema definido, usar marcador genérico
                if not output_data:
                    output_data = {"_inactive": True, "_reason": "Input from inactive branch"}
                
                return BlockOutput(
                    data=output_data,
                    context=context
                )
            
            input_data = BlockInput(
                data=step_input_data,
                context=context,
                previous_outputs={k: v.data for k, v in previous_outputs.items()}
            )

            # Executar bloco
            output = block.execute(input_data)

            # Atualizar contexto
            output.context.mark_success()

            return output

        except Exception as e:
            context.mark_failure(str(e))

            # Retornar output vazio em caso de erro
            return BlockOutput(
                data={},
                context=context
            )
    
    def _check_inactive_inputs(self, input_data: Dict[str, Any], block: Block) -> bool:
        """
        Verifica se algum input obrigatório do bloco está inativo.
        
        Args:
            input_data: Dados de entrada
            block: Instância do bloco
            
        Returns:
            True se bloco deve ser pulado (input inativo)
        """
        input_schema = getattr(block, 'input_schema', {})
        
        for input_name, schema in input_schema.items():
            is_required = schema.get("required", False)
            if not is_required:
                continue
                
            value = input_data.get(input_name)
            
            # Verificar se é um objeto inativo
            if isinstance(value, dict) and value.get("_inactive") == True:
                return True
        
        return False

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o pipeline."""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "steps": [
                {
                    "id": step.step_id,
                    "block": step.block_name,
                    "depends_on": step.depends_on,
                    "config": step.block_config
                }
                for step in self.config.steps
            ],
            "max_parallel": self.config.max_parallel,
            "timeout_seconds": self.config.timeout_seconds,
            "fail_fast": self.config.fail_fast
        }
