"""
Schemas de request/response para a API.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    """Request para processamento de experimento."""
    experimentId: str = Field(..., description="ID do experimento (MongoDB)")
    analysisId: str = Field(..., description="ID da análise dentro do protocolo")
    tenant: str = Field(..., description="Identificador do tenant/cliente")
    debug_mode: Optional[bool] = Field(None, description="Se True, retorna dados intermediários. Se None, usa configuração do tenant")


class PredictRequest(BaseModel):
    """Request para predição direta com features."""
    Amplitude: float
    TempoPontoInflexao: float
    PontoInflexao: float
    TempoPicoPrimeiraDerivada: float
    PicoPrimeiraDerivada: float
    TempoPicoSegundaDerivada: float
    PicoSegundaDerivada: float


class PredictionResponse(BaseModel):
    """Response com resultados de predição."""
    analysis_mode: Optional[str] = Field(None, description="Modo de análise usado")
    
    class Config:
        extra = "allow"


class PipelineStepSchema(BaseModel):
    """Representa um passo declarativo do pipeline."""
    step_id: str = Field(..., description="Identificador único do passo")
    block_name: str = Field(..., description="Nome do bloco registrado")
    depends_on: List[str] = Field(default_factory=list, description="IDs de passos que precisam executar antes")
    block_config: Dict[str, Any] = Field(default_factory=dict, description="Configuração específica do bloco")
    input_mapping: Dict[str, str] = Field(default_factory=dict, description="Mapeia input_name -> step_id.output_name")


class PipelineRunRequest(BaseModel):
    """Request para simulação visual do pipeline."""
    name: str
    description: Optional[str] = ""
    steps: List[PipelineStepSchema]
    initial_state: Dict[str, Any] = Field(default_factory=dict)
    max_parallel: int = 1
    timeout_seconds: float = 300.0
    fail_fast: bool = True
    generate_output_graphs: bool = False


class PipelineRunResponse(BaseModel):
    """Resultado serializado da execução do pipeline."""
    pipeline_id: Optional[str] = Field(None, description="Identificador da execução do pipeline")
    success: bool
    duration_ms: float
    errors: List[str] = Field(default_factory=list)
    step_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Metadados por step (status, tempos, erro, etc.)")


class TenantPipelineExecuteRequest(BaseModel):
    """Request para executar pipeline padrão de um tenant (como API)."""
    experimentId: str = Field(..., description="ID do experimento (MongoDB)")
    analysisId: str = Field(..., description="ID da análise dentro do protocolo")
    tenant: str = Field(..., description="Identificador do tenant/cliente")
    debug_mode: Optional[bool] = Field(None, description="Se True, inclui dados de debug dos blocos")
    include_steps: bool = Field(False, description="Se True, inclui metadados de execução por step (telemetria)")
    generate_output_graphs: bool = Field(False, description="Se True, permite que blocos gerem gráficos (data URIs)")
    timeout_seconds: float = Field(300.0, description="Timeout total do pipeline")
    fail_fast: bool = Field(True, description="Parar no primeiro erro")


class TenantPipelineExecuteResponse(BaseModel):
    """Response de execução do pipeline padrão do tenant."""
    pipeline_id: str
    success: bool
    duration_ms: float
    errors: List[str] = Field(default_factory=list)
    response: Dict[str, Any] = Field(default_factory=dict, description="Resposta final do response_builder")
    steps: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Metadados por step (status, tempos, erro, etc.)",
    )


class WorkspacePipelineRef(BaseModel):
    tenant: str = Field(..., description="Identificador do tenant/cliente")
    pipeline: str = Field(..., description="Nome do pipeline (sem extensões)")


class WorkspacePipelineInfo(BaseModel):
    tenant: str
    pipeline: str
    file: str = Field(..., description="Caminho relativo dentro de resources/")
    updated_at: Optional[str] = Field(None, description="ISO datetime da última modificação")
    title: Optional[str] = Field(None, description="Título amigável para exibição")
    logo: Optional[str] = Field(None, description="URL ou data URI do logo")
    accent_color: Optional[str] = Field(None, description="Cor de destaque (#RRGGBB)")
    active_version: Optional[str] = Field(None, description="Versão ativa do pipeline (vN)")
    versions_count: Optional[int] = Field(None, description="Quantidade de versões registradas")


class WorkspaceListResponse(BaseModel):
    pipelines: List[WorkspacePipelineInfo] = Field(default_factory=list)


class WorkspaceCreateRequest(BaseModel):
    tenant: str = Field(..., description="Identificador do tenant/cliente (nome da pasta em resources/)")
    pipeline: Optional[str] = Field(None, description="Nome do pipeline (padrão: igual ao tenant)")
    overwrite: bool = Field(False, description="Se true, sobrescreve pipeline existente")


class WorkspaceSaveRequest(BaseModel):
    tenant: str = Field(..., description="Identificador do tenant/cliente")
    pipeline: Optional[str] = Field(None, description="Nome do pipeline (padrão: igual ao tenant)")
    workspace_version: Optional[str] = Field(
        None,
        description="ID da versão do pipeline a ser salva (se omitido, salva na versão ativa)",
    )
    data: Dict[str, Any] = Field(..., description="Conteúdo completo do arquivo do pipeline (formato do Pipeline Studio)")
    change_reason: Optional[str] = Field(
        None,
        description="Razão da modificação (registrada no histórico de versões, se aplicável)",
    )


class WorkspaceLogoUrlRequest(BaseModel):
    tenant: str = Field(..., description="Identificador do tenant/cliente")
    pipeline: Optional[str] = Field(None, description="Nome do pipeline (padrão: igual ao tenant)")
    url: str = Field(..., description="URL da imagem do logo")


class WorkspaceDuplicateRequest(BaseModel):
    """Duplica um pipeline existente para um novo tenant/pipeline."""
    source_tenant: str = Field(..., description="Tenant de origem")
    source_pipeline: str = Field(..., description="Pipeline de origem (sem extensões)")
    target_tenant: str = Field(..., description="Tenant de destino (será criado em resources/<tenant>/)")
    target_pipeline: Optional[str] = Field(None, description="Pipeline de destino (padrão: igual ao target_tenant)")
    target_title: Optional[str] = Field(None, description="Título amigável (padrão: igual ao pipeline)")
    overwrite: bool = Field(False, description="Se true, sobrescreve pipeline existente no destino")


class PipelineTrainModelSpec(BaseModel):
    step_id: str = Field(..., description="ID do step/nó a ser treinado (ex: ml_inference_20)")
    algorithm: str = Field(
        "ridge",
        description="Algoritmo (ridge, rf, gbm, svr, mlp, cnn, lstm, xgb, lgbm, catboost)",
    )
    params: Dict[str, Any] = Field(default_factory=dict, description="Hiperparâmetros do algoritmo")
    grid_search: bool = Field(False, description="Se true, faz varredura (grid search) de modelos/parâmetros")
    algorithms: Optional[List[str]] = Field(
        None,
        description="Lista de algoritmos candidatos para grid search (ex: ['ridge','rf']). Se omitido, usa apenas 'algorithm'.",
    )
    param_grid: Optional[Dict[str, List[Any]]] = Field(
        None,
        description="Grade de hiperparâmetros para grid search (ex: {'alpha':[0.1,1,10]}).",
    )
    params_by_algorithm: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Parâmetros fixos por algoritmo (ex: {'ridge': {'alpha': 1.0}}).",
    )
    param_grid_by_algorithm: Optional[Dict[str, Dict[str, List[Any]]]] = Field(
        None,
        description="Grade por algoritmo (evita produto cartesiano entre modelos), ex: {'rf': {'n_estimators':[200,400]}}.",
    )
    selection_metric: Optional[str] = Field(
        None,
        description="Métrica para escolher o melhor trial (rmse, mae, r2). Se omitido, usa o default do request.",
    )
    max_trials: Optional[int] = Field(
        None,
        ge=1,
        le=500,
        description="Limite de trials da varredura (por bloco). Se omitido, usa o default do request.",
    )
    enabled: bool = Field(True, description="Se false, ignora este bloco no treinamento")


class TenantPipelineTrainRequest(BaseModel):
    tenant: str = Field(..., description="Identificador do tenant/cliente")
    protocolId: str = Field(..., description="ID do protocolo usado para roteamento do pipeline")
    experimentIds: List[str] = Field(..., min_length=1, description="Lista de experimentIds para treinamento")
    version: Optional[str] = Field(
        None,
        description="Versão (vN) do pipeline a ser usada como base para treinamento. Se omitido, usa a ativa.",
    )
    targets_map: Optional[Dict[str, Dict[str, str]]] = Field(
        None,
        description="Mapeamento label -> unit -> campo em lab_results (ex: ecoliNmp)",
    )
    models: Optional[List[PipelineTrainModelSpec]] = Field(
        None,
        description="Configuração por bloco trainável (algoritmo/params). Se omitido, treina todos com padrão.",
    )
    y_transform: str = Field("log10p", description="Transformação do alvo (none, log10p)")
    selection_metric: str = Field("rmse", description="Métrica padrão para seleção no grid search (rmse, mae, r2).")
    max_trials: int = Field(60, ge=1, le=500, description="Limite padrão de trials por bloco no grid search.")
    test_size: float = Field(
        0.2,
        ge=0.0,
        le=0.8,
        description="Proporção do conjunto de validação (0 = sem validação).",
    )
    random_state: int = Field(42, description="Semente para reprodutibilidade (split/treino).")
    perm_importance: bool = Field(False, description="Se true, calcula importância por permutação (validação).")
    perm_repeats: int = Field(10, ge=1, le=50, description="Repetições da importância por permutação.")
    skip_missing: bool = Field(True, description="Se true, pula experimentos sem lab_results/target; se false, falha")
    apply_to_pipeline: bool = Field(True, description="Se true, cria/ativa uma nova versão com model_path/scaler_path atualizados")
    change_reason: Optional[str] = Field(None, description="Razão registrada no histórico de versões")


class TrainedModelInfo(BaseModel):
    step_id: str
    block_name: str
    label: str
    unit: str
    model_path: str
    scaler_path: str
    n_samples: int
    skipped: int
    metrics: Dict[str, Any] = Field(default_factory=dict)


class TenantPipelineTrainResponse(BaseModel):
    success: bool
    trained: List[TrainedModelInfo] = Field(default_factory=list)
    skipped_experiments: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    version: Optional[str] = Field(None, description="Versão ativada (vN) se apply_to_pipeline=true")


# =============================================================================
# Orquestrador de Treinamento (Grid Search + Dependências)
# =============================================================================

class TrainingSessionCreateRequest(BaseModel):
    """Request para criar uma nova sessão de treinamento."""
    tenant: str = Field(..., description="Identificador do tenant/cliente")
    version: Optional[str] = Field(None, description="Versão do pipeline a usar (padrão: ativa)")


class TrainingSessionSummary(BaseModel):
    """Resumo de uma sessão de treinamento."""
    session_id: str
    created_at: str
    completed: bool
    status_summary: Dict[str, int] = Field(default_factory=dict)


class TrainingCandidateInfo(BaseModel):
    """Informações de um candidato de modelo."""
    rank: int
    algorithm: str
    params: Dict[str, Any] = Field(default_factory=dict)
    score: float
    metrics: Dict[str, Any] = Field(default_factory=dict)
    selected: bool = False


class TrainingTaskInfo(BaseModel):
    """Informações de uma task de treinamento."""
    step_id: str
    block_name: str
    label: str
    unit: str
    status: str
    depends_on: List[str] = Field(default_factory=list)
    n_samples: int = 0
    candidates: List[TrainingCandidateInfo] = Field(default_factory=list)
    selected_candidate_index: Optional[int] = None
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None
    metadata_path: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class TrainingSessionDetail(BaseModel):
    """Detalhes completos de uma sessão de treinamento."""
    session_id: str
    tenant: str
    pipeline_name: str
    created_at: str
    completed: bool
    execution_order: List[str] = Field(default_factory=list)
    next_trainable: Optional[str] = None
    status_summary: Dict[str, int] = Field(default_factory=dict)
    tasks: Dict[str, TrainingTaskInfo] = Field(default_factory=dict)
    awaiting_selection: List[Dict[str, Any]] = Field(default_factory=list)
    blocked_tasks: List[Dict[str, Any]] = Field(default_factory=list)


class TrainingSessionListResponse(BaseModel):
    """Lista de sessões de treinamento."""
    sessions: List[TrainingSessionSummary] = Field(default_factory=list)


class TrainingRunStepRequest(BaseModel):
    """Request para treinar um step específico."""
    session_id: str = Field(..., description="ID da sessão de treinamento")
    step_id: str = Field(..., description="ID do step a treinar")
    analysisId: str = Field(..., description="ID da análise")
    experimentIds: List[str] = Field(..., min_length=1, description="Lista de experimentIds")
    algorithm: str = Field("ridge", description="Algoritmo padrão")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parâmetros do algoritmo")
    grid_search: bool = Field(True, description="Se true, faz grid search")
    algorithms: Optional[List[str]] = Field(None, description="Algoritmos para grid search")
    param_grid: Optional[Dict[str, List[Any]]] = Field(None, description="Grid de parâmetros")
    selection_metric: str = Field("rmse", description="Métrica para seleção")
    max_trials: int = Field(60, ge=1, le=500, description="Máximo de trials")
    test_size: float = Field(0.2, ge=0.0, le=0.8, description="Proporção de validação")
    y_transform: str = Field("log10p", description="Transformação do y")
    targets_map: Optional[Dict[str, Dict[str, str]]] = Field(None, description="Mapeamento de targets")
    skip_missing: bool = Field(True, description="Pular experimentos sem dados")
    auto_select_best: bool = Field(False, description="Se true, seleciona automaticamente o melhor modelo")


class TrainingSelectModelRequest(BaseModel):
    """Request para selecionar um modelo candidato."""
    session_id: str = Field(..., description="ID da sessão de treinamento")
    step_id: str = Field(..., description="ID do step")
    candidate_index: int = Field(..., ge=0, description="Índice do candidato a selecionar (0-based)")


class TrainingApplyRequest(BaseModel):
    """Request para aplicar modelos selecionados ao pipeline."""
    session_id: str = Field(..., description="ID da sessão de treinamento")
    change_reason: Optional[str] = Field(None, description="Razão da modificação")

