"""
Orquestrador de Treinamento de Modelos ML.

Este módulo gerencia:
1. Ordem de treinamento baseada em dependências do pipeline
2. Grid Search com exposição de candidatos para seleção manual
3. Validação de pré-requisitos antes de treinar cada modelo

Autor: BioAILab
Versão: 1.0.0
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class TrainingStatus(str, Enum):
    """Status de treinamento de um modelo."""
    PENDING = "pending"           # Aguardando pré-requisitos
    READY = "ready"               # Pronto para treinar
    TRAINING = "training"         # Em treinamento
    AWAITING_SELECTION = "awaiting_selection"  # Grid search concluído, aguardando seleção
    TRAINED = "trained"           # Treinado e salvo
    FAILED = "failed"             # Falhou
    SKIPPED = "skipped"           # Pulado (sem dados ou desabilitado)


@dataclass
class ModelCandidate:
    """Um candidato de modelo gerado pelo grid search."""
    algorithm: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    score: float
    rank: int
    
    # Se selecionado, estes são preenchidos
    selected: bool = False
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None
    metadata_path: Optional[str] = None


@dataclass
class TrainingTask:
    """Representa uma tarefa de treinamento para um step do pipeline."""
    step_id: str
    block_name: str
    label: str
    unit: str
    status: TrainingStatus = TrainingStatus.PENDING
    
    # Dependências (steps que precisam estar treinados antes)
    depends_on: List[str] = field(default_factory=list)
    
    # Dados de treino (preenchidos quando status == READY)
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    n_samples: int = 0
    
    # Candidatos do grid search (preenchidos após treino)
    candidates: List[ModelCandidate] = field(default_factory=list)
    selected_candidate_index: Optional[int] = None
    
    # Resultado final (após seleção)
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None
    metadata_path: Optional[str] = None
    
    # Mensagens de erro/aviso
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TrainingSession:
    """Sessão de treinamento com múltiplos modelos."""
    session_id: str
    tenant: str
    pipeline_name: str
    created_at: str
    
    tasks: Dict[str, TrainingTask] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    
    # Estado geral
    current_task: Optional[str] = None
    completed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a sessão para JSON."""
        return {
            "session_id": self.session_id,
            "tenant": self.tenant,
            "pipeline_name": self.pipeline_name,
            "created_at": self.created_at,
            "tasks": {
                k: {
                    "step_id": v.step_id,
                    "block_name": v.block_name,
                    "label": v.label,
                    "unit": v.unit,
                    "status": v.status.value,
                    "depends_on": v.depends_on,
                    "n_samples": v.n_samples,
                    "candidates": [
                        {
                            "algorithm": c.algorithm,
                            "params": c.params,
                            "metrics": c.metrics,
                            "score": c.score,
                            "rank": c.rank,
                            "selected": c.selected,
                        }
                        for c in v.candidates
                    ],
                    "selected_candidate_index": v.selected_candidate_index,
                    "model_path": v.model_path,
                    "scaler_path": v.scaler_path,
                    "metadata_path": v.metadata_path,
                    "errors": v.errors,
                    "warnings": v.warnings,
                }
                for k, v in self.tasks.items()
            },
            "execution_order": self.execution_order,
            "current_task": self.current_task,
            "completed": self.completed,
        }


class TrainingOrchestrator:
    """
    Orquestra o treinamento de múltiplos modelos ML em um pipeline.
    
    Responsabilidades:
    1. Analisar dependências entre blocos ML
    2. Determinar ordem correta de treinamento
    3. Executar grid search e expor candidatos
    4. Permitir seleção manual de modelo
    5. Validar que predecessores estão treinados
    """
    
    # Blocos ML que precisam de treinamento
    ML_BLOCKS = {
        "ml_inference",
        "ml_inference_series", 
        "ml_inference_multichannel",
        "ml_transform_series",
        "ml_forecaster_series",
        "ml_detector",
    }
    
    def __init__(self, tenant: str, pipeline_config: Dict[str, Any]):
        self.tenant = tenant
        self.pipeline_config = pipeline_config
        self.sessions_dir = self._get_sessions_dir()
        
    def _get_sessions_dir(self) -> Path:
        """Diretório para salvar sessões de treinamento."""
        from pathlib import Path
        base = Path(__file__).parent.parent.parent.parent / "resources" / self.tenant / "training_sessions"
        base.mkdir(parents=True, exist_ok=True)
        return base
    
    def analyze_dependencies(self) -> Dict[str, List[str]]:
        """
        Analisa o pipeline e retorna dependências entre blocos ML.
        
        Returns:
            Dict mapeando step_id -> list de step_ids que precisam estar treinados antes
        """
        steps = self.pipeline_config.get("steps", [])
        dependencies: Dict[str, List[str]] = {}
        
        # Mapeia step_id -> block_name
        step_blocks: Dict[str, str] = {}
        for step in steps:
            sid = str(step.get("step_id", ""))
            block = str(step.get("block_name", ""))
            step_blocks[sid] = block
        
        # Para cada bloco ML, verifica se depende de outro bloco ML
        for step in steps:
            sid = str(step.get("step_id", ""))
            block = str(step.get("block_name", ""))
            
            if block not in self.ML_BLOCKS:
                continue
                
            dependencies[sid] = []
            
            # Analisa input_mapping para ver de onde vêm os dados
            input_mapping = step.get("input_mapping", {})
            depends_on = step.get("depends_on", [])
            
            for dep in depends_on:
                dep_block = step_blocks.get(dep, "")
                if dep_block in self.ML_BLOCKS:
                    # Este bloco ML depende de outro bloco ML!
                    dependencies[sid].append(dep)
            
            # Também verifica input_mapping
            for key, source in input_mapping.items():
                if "." in source:
                    src_step = source.split(".")[0]
                    src_block = step_blocks.get(src_step, "")
                    if src_block in self.ML_BLOCKS and src_step not in dependencies[sid]:
                        dependencies[sid].append(src_step)
        
        return dependencies
    
    def compute_execution_order(self) -> List[str]:
        """
        Calcula a ordem de execução respeitando dependências (ordenação topológica).
        
        Returns:
            Lista de step_ids na ordem correta de treinamento
        """
        deps = self.analyze_dependencies()
        
        # Ordenação topológica (Kahn's algorithm)
        in_degree: Dict[str, int] = {k: 0 for k in deps.keys()}
        for step_id, step_deps in deps.items():
            for dep in step_deps:
                if dep in in_degree:
                    in_degree[step_id] += 1
        
        # Steps sem dependências
        queue = [s for s, d in in_degree.items() if d == 0]
        order: List[str] = []
        
        while queue:
            step = queue.pop(0)
            order.append(step)
            
            # Reduz in_degree dos dependentes
            for s, d in deps.items():
                if step in d:
                    in_degree[s] -= 1
                    if in_degree[s] == 0:
                        queue.append(s)
        
        # Verifica ciclos
        if len(order) != len(deps):
            remaining = set(deps.keys()) - set(order)
            raise ValueError(f"Ciclo de dependência detectado entre: {remaining}")
        
        return order
    
    def create_session(self) -> TrainingSession:
        """
        Cria uma nova sessão de treinamento.
        
        Returns:
            TrainingSession configurada com todas as tasks
        """
        import uuid
        
        session_id = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        deps = self.analyze_dependencies()
        order = self.compute_execution_order()
        
        session = TrainingSession(
            session_id=session_id,
            tenant=self.tenant,
            pipeline_name=self.pipeline_config.get("name", ""),
            created_at=datetime.utcnow().isoformat() + "Z",
            execution_order=order,
        )
        
        # Cria tasks para cada bloco ML
        steps = self.pipeline_config.get("steps", [])
        for step in steps:
            sid = str(step.get("step_id", ""))
            block = str(step.get("block_name", ""))
            
            if block not in self.ML_BLOCKS:
                continue
            
            task = TrainingTask(
                step_id=sid,
                block_name=block,
                label="",  # Preenchido durante coleta de dados
                unit="",
                depends_on=deps.get(sid, []),
                status=TrainingStatus.PENDING if deps.get(sid) else TrainingStatus.READY,
            )
            session.tasks[sid] = task
        
        # Salva sessão
        self._save_session(session)
        
        return session
    
    def _save_session(self, session: TrainingSession) -> Path:
        """Salva sessão em arquivo JSON."""
        path = self.sessions_dir / f"{session.session_id}.json"
        path.write_text(json.dumps(session.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    
    def load_session(self, session_id: str) -> Optional[TrainingSession]:
        """Carrega uma sessão existente."""
        path = self.sessions_dir / f"{session_id}.json"
        if not path.exists():
            return None
        
        data = json.loads(path.read_text(encoding="utf-8"))
        
        session = TrainingSession(
            session_id=data["session_id"],
            tenant=data["tenant"],
            pipeline_name=data["pipeline_name"],
            created_at=data["created_at"],
            execution_order=data["execution_order"],
            current_task=data.get("current_task"),
            completed=data.get("completed", False),
        )
        
        for sid, t in data.get("tasks", {}).items():
            task = TrainingTask(
                step_id=t["step_id"],
                block_name=t["block_name"],
                label=t.get("label", ""),
                unit=t.get("unit", ""),
                status=TrainingStatus(t["status"]),
                depends_on=t.get("depends_on", []),
                n_samples=t.get("n_samples", 0),
                selected_candidate_index=t.get("selected_candidate_index"),
                model_path=t.get("model_path"),
                scaler_path=t.get("scaler_path"),
                metadata_path=t.get("metadata_path"),
                errors=t.get("errors", []),
                warnings=t.get("warnings", []),
            )
            
            for c in t.get("candidates", []):
                task.candidates.append(ModelCandidate(
                    algorithm=c["algorithm"],
                    params=c["params"],
                    metrics=c["metrics"],
                    score=c["score"],
                    rank=c["rank"],
                    selected=c.get("selected", False),
                ))
            
            session.tasks[sid] = task
        
        return session
    
    def validate_can_train(self, session: TrainingSession, step_id: str) -> Tuple[bool, List[str]]:
        """
        Valida se um step pode ser treinado.
        
        Returns:
            (pode_treinar: bool, erros: List[str])
        """
        task = session.tasks.get(step_id)
        if not task:
            return False, [f"Task {step_id} não encontrada"]
        
        errors: List[str] = []
        
        # Verifica dependências
        for dep_id in task.depends_on:
            dep_task = session.tasks.get(dep_id)
            if not dep_task:
                errors.append(f"Dependência {dep_id} não encontrada")
                continue
            
            if dep_task.status != TrainingStatus.TRAINED:
                errors.append(
                    f"Dependência '{dep_id}' precisa estar treinada primeiro "
                    f"(status atual: {dep_task.status.value})"
                )
        
        return len(errors) == 0, errors
    
    def get_next_trainable(self, session: TrainingSession) -> Optional[str]:
        """
        Retorna o próximo step que pode ser treinado na ordem correta.
        
        Returns:
            step_id ou None se não houver mais steps para treinar
        """
        for step_id in session.execution_order:
            task = session.tasks.get(step_id)
            if not task:
                continue
            
            if task.status in (TrainingStatus.PENDING, TrainingStatus.READY):
                can_train, _ = self.validate_can_train(session, step_id)
                if can_train:
                    return step_id
        
        return None
    
    def get_session_summary(self, session: TrainingSession) -> Dict[str, Any]:
        """
        Retorna um resumo da sessão de treinamento.
        """
        by_status: Dict[str, int] = {}
        for task in session.tasks.values():
            by_status[task.status.value] = by_status.get(task.status.value, 0) + 1
        
        awaiting_selection = [
            {
                "step_id": t.step_id,
                "block_name": t.block_name,
                "label": t.label,
                "candidates_count": len(t.candidates),
                "top_3": [
                    {
                        "rank": c.rank,
                        "algorithm": c.algorithm,
                        "score": c.score,
                        "metrics": {k: v for k, v in c.metrics.items() if k.startswith("val_") or k.startswith("test_")},
                    }
                    for c in sorted(t.candidates, key=lambda x: x.rank)[:3]
                ]
            }
            for t in session.tasks.values()
            if t.status == TrainingStatus.AWAITING_SELECTION
        ]
        
        blocked = [
            {
                "step_id": t.step_id,
                "blocked_by": [
                    d for d in t.depends_on
                    if session.tasks.get(d, TrainingTask(step_id="", block_name="", label="", unit="")).status != TrainingStatus.TRAINED
                ]
            }
            for t in session.tasks.values()
            if t.status == TrainingStatus.PENDING and t.depends_on
        ]
        
        return {
            "session_id": session.session_id,
            "tenant": session.tenant,
            "created_at": session.created_at,
            "completed": session.completed,
            "status_summary": by_status,
            "execution_order": session.execution_order,
            "next_trainable": self.get_next_trainable(session),
            "awaiting_selection": awaiting_selection,
            "blocked_tasks": blocked,
        }


# =============================================================================
# Funções auxiliares para integração com o router
# =============================================================================

def create_training_session(tenant: str, pipeline_config: Dict[str, Any]) -> TrainingSession:
    """Cria uma nova sessão de treinamento."""
    orchestrator = TrainingOrchestrator(tenant, pipeline_config)
    return orchestrator.create_session()


def get_training_session(tenant: str, session_id: str) -> Optional[TrainingSession]:
    """Carrega uma sessão de treinamento existente."""
    orchestrator = TrainingOrchestrator(tenant, {})
    return orchestrator.load_session(session_id)


def list_training_sessions(tenant: str) -> List[Dict[str, Any]]:
    """Lista todas as sessões de treinamento do tenant."""
    from pathlib import Path
    base = Path(__file__).parent.parent.parent.parent / "resources" / tenant / "training_sessions"
    if not base.exists():
        return []
    
    sessions = []
    for f in sorted(base.glob("train_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            by_status: Dict[str, int] = {}
            for t in data.get("tasks", {}).values():
                st = t.get("status", "unknown")
                by_status[st] = by_status.get(st, 0) + 1
            
            sessions.append({
                "session_id": data.get("session_id"),
                "created_at": data.get("created_at"),
                "completed": data.get("completed", False),
                "status_summary": by_status,
            })
        except Exception:
            continue
    
    return sessions
