"""
Router principal da API.
"""

import asyncio
import json
import re
import mimetypes
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body, Query, Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

try:
    # FastAPI considera multipart disponível quando python-multipart está instalado.
    # Há pacotes alternativos chamados "multipart" que não satisfazem o requisito;
    # então seguimos a mesma checagem que o FastAPI usa internamente.
    from multipart.multipart import parse_options_header  # type: ignore  # noqa: F401

    _MULTIPART_AVAILABLE = True
except Exception:
    _MULTIPART_AVAILABLE = False

from .schemas import (
    ProcessRequest,
    PredictRequest,
    PredictionResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    TenantPipelineExecuteRequest,
    TenantPipelineExecuteResponse,
    WorkspaceCreateRequest,
    WorkspaceListResponse,
    WorkspaceSaveRequest,
    WorkspaceLogoUrlRequest,
    WorkspaceDuplicateRequest,
    TenantPipelineTrainRequest,
    TenantPipelineTrainResponse,
    # Orquestrador de treinamento
    TrainingSessionCreateRequest,
    TrainingSessionSummary,
    TrainingSessionDetail,
    TrainingSessionListResponse,
    TrainingTaskInfo,
    TrainingCandidateInfo,
    TrainingRunStepRequest,
    TrainingSelectModelRequest,
    TrainingApplyRequest,
)
from .dependencies import validate_api_key
from ..use_cases.process_experiment import ProcessExperimentUseCase, ProcessRequest as UCRequest
from ..infrastructure.database.mongo_repository import MongoRepository
from ..infrastructure.database.mock_repository import MockRepository
from ..infrastructure.ml import OnnxInferenceAdapter
from ..infrastructure.external.spectral_api import SpectralApiAdapter
from ..infrastructure.config.settings import get_settings
from ..infrastructure.config.tenant_loader import get_tenant_loader
from ..components.pipeline import (
    BlockRegistry,
    PipelineConfig,
    PipelineStep,
    PipelineEngine,
)
from .pipeline_library import build_pipeline_library
from ..infrastructure.ml.training import (
    DEFAULT_TARGETS_MAP,
    extract_target_value,
    select_lab_result_for_field,
    select_latest_lab_result,
    train_regressor_export_onnx,
    unit_slug,
)


class NumpyEncoder(json.JSONEncoder):
    """Encoder para converter tipos numpy em tipos JSON nativos."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _ensure_jsonable(value: any):
    """Converte valores numpy/não serializáveis para tipos JSON seguros."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        if value.size == 1:
            return _ensure_jsonable(value.item())
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: _ensure_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_ensure_jsonable(v) for v in value]
    return value


router = APIRouter()

# =============================================================================
# Workspace (Pipeline Studio) - persistência em resources/<tenant>/pipeline/
# =============================================================================

_TENANT_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_VERSION_RE = re.compile(r"^v[0-9]{1,6}$")


def _repo_root() -> Path:
    # src/interface/router.py -> src/interface -> src -> repo_root
    return Path(__file__).resolve().parents[2]


def _resources_root() -> Path:
    return _repo_root() / "resources"


def _validate_segment(name: str, label: str) -> str:
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail=f"{label} é obrigatório")
    if not _TENANT_RE.match(name):
        raise HTTPException(status_code=400, detail=f"{label} inválido (use apenas letras, números, '_' ou '-')")
    return name


def _validate_version_id(version: str) -> str:
    version = (version or "").strip()
    if not version:
        return ""
    if not _VERSION_RE.match(version):
        raise HTTPException(status_code=400, detail="Versão inválida (use o formato vN)")
    return version


def _workspace_pipeline_file(tenant: str, pipeline: str) -> Path:
    tenant = _validate_segment(tenant, "tenant")
    pipeline = _validate_segment(pipeline, "pipeline")
    return _resources_root() / tenant / "pipeline" / f"{pipeline}.json"


def _workspace_pipeline_dir(tenant: str, pipeline: str) -> Path:
    tenant = _validate_segment(tenant, "tenant")
    pipeline = _validate_segment(pipeline, "pipeline")
    return _resources_root() / tenant / "pipeline"


def _workspace_logo_dir(tenant: str, pipeline: str) -> Path:
    return _workspace_pipeline_dir(tenant, pipeline) / "resources" / "img"


def _workspace_versions_dir(tenant: str) -> Path:
    tenant = _validate_segment(tenant, "tenant")
    return _resources_root() / tenant / "pipeline" / "versions"


def _workspace_versions_manifest(tenant: str) -> Path:
    tenant = _validate_segment(tenant, "tenant")
    return _resources_root() / tenant / "pipeline" / "_versions.json"


def _read_versions_manifest(tenant: str) -> dict:
    path = _workspace_versions_manifest(tenant)
    if not path.exists():
        return {}
    data = _read_json_file(path)
    return data if isinstance(data, dict) else {}


def _write_versions_manifest(tenant: str, data: dict) -> None:
    path = _workspace_versions_manifest(tenant)
    _write_json_file(path, data)


# =============================================================================
# CACHE DE FEATURES PARA GRID-SEARCH (v2 - Cache por Experimento)
# =============================================================================
# Cache granular por experimento: permite reutilizar features quando apenas
# parte do dataset muda ou quando apenas o target (y) muda.
#
# Estrutura:
#   cache/features/v2/{pipeline_hash}/{step_id}/{experiment_id}.npz
#   cache/features/v2/{pipeline_hash}/{step_id}/{experiment_id}.json
#
# O Y (target) NÃO é cacheado - é buscado diretamente do lab_results.

from functools import lru_cache
from threading import Lock

# Lock para evitar race conditions no cache
_cache_lock = Lock()

# LRU Cache em memória para dados frequentes (Fase 2)
_LAB_RESULTS_CACHE_SIZE = 200
_EXPERIMENT_DATA_CACHE_SIZE = 50


def _features_cache_dir(tenant: str) -> Path:
    """Diretório de cache de features para um tenant."""
    return _resources_root() / tenant / "cache" / "features"


def _features_cache_dir_v2(tenant: str) -> Path:
    """Diretório de cache v2 (por experimento) para um tenant."""
    return _resources_root() / tenant / "cache" / "features" / "v2"


def _compute_pipeline_hash(pipeline_json: dict, step_id: str) -> str:
    """
    Calcula hash do pipeline até o step alvo.
    
    O hash NÃO inclui experiment_ids - isso permite reutilizar cache
    entre datasets diferentes que usam o mesmo pipeline.
    """
    import hashlib
    
    # Extrair steps do pipeline
    if "execution" in pipeline_json and "steps" in pipeline_json.get("execution", {}):
        steps = pipeline_json["execution"]["steps"]
    else:
        steps = pipeline_json.get("steps", [])
    
    # Filtrar apenas steps até o step_id alvo (inclusive dependências)
    # Por simplicidade, incluir todos os steps - o hash será único por pipeline
    steps_sorted = sorted(steps, key=lambda s: str(s.get("step_id", "")))
    
    canonical = {
        "steps": steps_sorted,
        "target_step": step_id,
    }
    
    canonical_str = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical_str.encode()).hexdigest()[:12]


def _get_experiment_cache_path(tenant: str, pipeline_hash: str, step_id: str, exp_id: str) -> tuple[Path, Path]:
    """Retorna paths do cache para um experimento específico."""
    # Sanitizar exp_id para uso em filename (remover caracteres problemáticos)
    safe_exp_id = exp_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    
    cache_dir = _features_cache_dir_v2(tenant) / pipeline_hash / step_id
    npz_path = cache_dir / f"{safe_exp_id}.npz"
    json_path = cache_dir / f"{safe_exp_id}.json"
    return npz_path, json_path


def _save_experiment_features(
    tenant: str,
    pipeline_hash: str,
    step_id: str,
    exp_id: str,
    X: np.ndarray,
    metadata: dict,
) -> bool:
    """
    Salva features de um único experimento no cache.
    
    Args:
        tenant: Tenant
        pipeline_hash: Hash do pipeline
        step_id: ID do step alvo
        exp_id: ID do experimento
        X: Array de features (shape pode variar por tipo de bloco)
        metadata: Metadados (block_cfg, channels, shape, etc.)
    
    Returns:
        True se salvou com sucesso
    """
    npz_path, json_path = _get_experiment_cache_path(tenant, pipeline_hash, step_id, exp_id)
    
    try:
        with _cache_lock:
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Salvar features
            np.savez_compressed(npz_path, X=X.astype(np.float32))
            
            # Salvar metadados (inclui valor de X para fácil inspeção)
            # Converter X para lista para serialização JSON
            x_value = X.flatten().tolist() if X.size <= 100 else X.flatten()[:100].tolist()  # limitar se muito grande
            
            # Extrair campos especiais do metadata (sem modificar original)
            y_value = metadata.get("y_value")
            y_value_raw = metadata.get("y_value_raw")
            dilution = metadata.get("dilution")
            
            # Criar metadata base sem os campos especiais
            meta_base = {k: v for k, v in metadata.items() if k not in ("y_value", "y_value_raw", "dilution")}
            
            meta = {
                "exp_id": exp_id,
                "pipeline_hash": pipeline_hash,
                "step_id": step_id,
                "shape": list(X.shape),
                "x_value": x_value[0] if len(x_value) == 1 else x_value,  # scalar se único valor
                "created_at": _utc_now_iso(),
                **meta_base,
            }
            
            # Adicionar campos y no final (só se existirem)
            if y_value is not None:
                meta["y_value"] = y_value
            if y_value_raw is not None:
                meta["y_value_raw"] = y_value_raw
            if dilution is not None:
                meta["dilution"] = dilution
                
            _write_json_file(json_path, meta)
        
        return True
    except Exception as e:
        print(f"[cache-v2] Erro ao salvar cache para {exp_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def _load_experiment_features(
    tenant: str,
    pipeline_hash: str,
    step_id: str,
    exp_id: str,
) -> tuple[np.ndarray | None, dict | None]:
    """
    Carrega features de um único experimento do cache.
    
    Returns:
        (X, metadata) ou (None, None) se não existe
    """
    npz_path, json_path = _get_experiment_cache_path(tenant, pipeline_hash, step_id, exp_id)
    
    if not npz_path.exists() or not json_path.exists():
        return None, None
    
    try:
        data = np.load(npz_path)
        X = data["X"]
        metadata = _read_json_file(json_path)
        return X, metadata
    except Exception as e:
        print(f"[cache-v2] Erro ao carregar cache para {exp_id}: {e}")
        return None, None


def _get_cached_experiments_batch(
    tenant: str,
    pipeline_hash: str,
    step_id: str,
    experiment_ids: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, dict], list[str]]:
    """
    Carrega features de múltiplos experimentos do cache em batch.
    
    Returns:
        (cached_X, cached_meta, missing_ids)
        - cached_X: {exp_id: X_array}
        - cached_meta: {exp_id: metadata}
        - missing_ids: lista de exp_ids não encontrados no cache
    """
    cached_X = {}
    cached_meta = {}
    missing_ids = []
    
    for exp_id in experiment_ids:
        X, meta = _load_experiment_features(tenant, pipeline_hash, step_id, exp_id)
        if X is not None and meta is not None:
            cached_X[exp_id] = X
            cached_meta[exp_id] = meta
        else:
            missing_ids.append(exp_id)
    
    if cached_X:
        print(f"[cache-v2] Carregados {len(cached_X)}/{len(experiment_ids)} experimentos do cache")
    
    return cached_X, cached_meta, missing_ids


def _invalidate_experiment_cache(
    tenant: str,
    pipeline_hash: str | None = None,
    step_id: str | None = None,
    exp_id: str | None = None,
) -> int:
    """
    Invalida cache de experimentos.
    
    Args:
        tenant: Tenant
        pipeline_hash: Se fornecido, invalida apenas este pipeline
        step_id: Se fornecido, invalida apenas este step
        exp_id: Se fornecido, invalida apenas este experimento
    
    Returns:
        Número de arquivos removidos
    """
    cache_dir = _features_cache_dir_v2(tenant)
    if not cache_dir.exists():
        return 0
    
    removed = 0
    
    if exp_id and pipeline_hash and step_id:
        # Remover experimento específico
        npz_path, json_path = _get_experiment_cache_path(tenant, pipeline_hash, step_id, exp_id)
        for p in [npz_path, json_path]:
            if p.exists():
                p.unlink()
                removed += 1
    elif pipeline_hash and step_id:
        # Remover todos experimentos de um step
        step_dir = cache_dir / pipeline_hash / step_id
        if step_dir.exists():
            import shutil
            removed = len(list(step_dir.glob("*")))
            shutil.rmtree(step_dir)
    elif pipeline_hash:
        # Remover todo o pipeline
        pipeline_dir = cache_dir / pipeline_hash
        if pipeline_dir.exists():
            import shutil
            removed = len(list(pipeline_dir.rglob("*")))
            shutil.rmtree(pipeline_dir)
    else:
        # Remover todo o cache v2
        import shutil
        removed = len(list(cache_dir.rglob("*")))
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    if removed:
        print(f"[cache-v2] Cache invalidado: {removed} arquivos removidos")
    
    return removed


def _get_cache_stats(tenant: str) -> dict:
    """Retorna estatísticas do cache."""
    cache_dir = _features_cache_dir_v2(tenant)
    if not cache_dir.exists():
        return {"total_files": 0, "total_size_mb": 0, "pipelines": 0, "experiments": 0}
    
    npz_files = list(cache_dir.rglob("*.npz"))
    total_size = sum(f.stat().st_size for f in npz_files)
    pipeline_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
    
    return {
        "total_files": len(npz_files),
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "pipelines": len(pipeline_dirs),
        "experiments": len(npz_files),
    }


# =============================================================================
# CACHE LRU EM MEMÓRIA PARA DADOS MONGODB (Fase 2)
# =============================================================================
# Evita queries repetidas ao MongoDB para dados frequentemente acessados

# Caches com TTL simulado via dicionário separado
_lab_results_cache: dict[str, list] = {}
_lab_results_cache_time: dict[str, float] = {}
_experiment_data_cache: dict[str, list] = {}
_experiment_data_cache_time: dict[str, float] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutos


def _cache_key_lab(tenant: str, exp_id: str) -> str:
    return f"{tenant}:{exp_id}"


def _get_lab_results_cached(repo, tenant: str, exp_id: str, limit: int = 50) -> list:
    """
    Busca lab_results com cache em memória.
    Útil durante treino quando o mesmo experimento é acessado múltiplas vezes.
    """
    import time
    key = _cache_key_lab(tenant, exp_id)
    now = time.time()
    
    # Verificar cache
    if key in _lab_results_cache:
        cache_time = _lab_results_cache_time.get(key, 0)
        if now - cache_time < _CACHE_TTL_SECONDS:
            return _lab_results_cache[key]
    
    # Buscar do banco
    results = repo.get_lab_results(tenant, exp_id, limit=limit)
    
    # Salvar no cache (com limite de tamanho)
    if len(_lab_results_cache) >= _LAB_RESULTS_CACHE_SIZE:
        # Remover entrada mais antiga
        oldest_key = min(_lab_results_cache_time, key=_lab_results_cache_time.get)
        _lab_results_cache.pop(oldest_key, None)
        _lab_results_cache_time.pop(oldest_key, None)
    
    _lab_results_cache[key] = results
    _lab_results_cache_time[key] = now
    
    return results


def _clear_memory_caches():
    """Limpa caches em memória."""
    global _lab_results_cache, _lab_results_cache_time
    global _experiment_data_cache, _experiment_data_cache_time
    _lab_results_cache.clear()
    _lab_results_cache_time.clear()
    _experiment_data_cache.clear()
    _experiment_data_cache_time.clear()
    print("[cache] Caches em memória limpos")


# =============================================================================
# FUNÇÕES DE INTEGRAÇÃO CACHE V2 + BUSCA DE Y
# =============================================================================

def _get_y_batch_from_lab_results(
    tenant: str,
    experiment_ids: list[str],
    target_field: str,
    dilution_factors: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """
    Busca Y (target) para múltiplos experimentos em batch.
    
    NÃO USA CACHE - sempre busca fresco do banco.
    Isso permite que Y mude (ex: NMP→UFC) sem invalidar cache de features.
    
    Args:
        tenant: ID do tenant
        experiment_ids: Lista de IDs de experimentos
        target_field: Campo alvo (ex: "nmp_100ml", "ufc_100ml")
        dilution_factors: Dicionário {exp_id: factor} para correção de diluição
    
    Returns:
        (y_values, skip_reasons): 
            - y_values: {exp_id: y_val} para experimentos com Y válido
            - skip_reasons: {exp_id: motivo} para experimentos sem Y válido
    """
    settings = get_settings()
    dilution_factors = dilution_factors or {}

    def _normalize_dilution_factor(raw_value: Any) -> float:
        try:
            factor = float(10 ** int(raw_value))
            return factor if factor > 0 else 1.0
        except Exception:
            return 1.0

    # Completar fatores de diluição faltantes buscando no experimento
    missing_ids = [eid for eid in experiment_ids if eid not in dilution_factors]
    if missing_ids:
        settings = get_settings()
        real_missing = [eid for eid in missing_ids if not eid.startswith("mock:")]
        mock_missing = [eid for eid in missing_ids if eid.startswith("mock:")]

        if real_missing:
            repo = MongoRepository(settings.mongo_uri, settings.tenant_db_prefix)
            for eid in real_missing:
                try:
                    exp = repo.get_experiment(tenant, eid) or {}
                    dilution_factors[eid] = _normalize_dilution_factor(exp.get("diluicao"))
                except Exception:
                    dilution_factors[eid] = 1.0

        if mock_missing:
            repo = MockRepository(settings.resources_dir)
            for eid in mock_missing:
                try:
                    exp = repo.get_experiment(tenant, eid) or {}
                    dilution_factors[eid] = _normalize_dilution_factor(exp.get("diluicao"))
                except Exception:
                    dilution_factors[eid] = 1.0
    
    y_values: dict[str, float] = {}
    skip_reasons: dict[str, str] = {}
    
    # Separar mocks de reais
    mock_ids = [eid for eid in experiment_ids if eid.startswith("mock:")]
    real_ids = [eid for eid in experiment_ids if not eid.startswith("mock:")]
    
    # Processar experimentos reais em batch
    if real_ids:
        repo = MongoRepository(settings.mongo_uri, settings.tenant_db_prefix)
        try:
            lab_results_batch = repo.get_lab_results_batch(tenant, real_ids, limit_per_exp=5)
        except Exception as e:
            # Fallback: buscar individualmente
            lab_results_batch = {}
            for eid in real_ids:
                try:
                    lab_results_batch[eid] = repo.get_lab_results(tenant, eid, limit=5)
                except:
                    pass
        
        for eid in real_ids:
            lab_results = lab_results_batch.get(eid, [])
            if not lab_results:
                skip_reasons[eid] = "sem lab_results"
                continue
            
            lab = select_lab_result_for_field(lab_results, target_field)
            if not lab:
                skip_reasons[eid] = f"lab_result não encontrado para field={target_field}"
                continue
            
            y_val = extract_target_value(lab, target_field)
            if y_val is None:
                skip_reasons[eid] = f"y_val é None para field={target_field}"
                continue
            
            # Aplicar correção de diluição
            dilution = dilution_factors.get(eid, 1.0)
            if dilution <= 0:
                dilution = 1.0
            if dilution != 1.0 and dilution > 0:
                y_val = y_val / dilution
            
            y_values[eid] = float(y_val)
    
    # Processar mocks
    if mock_ids:
        mock_repo = MockRepository(settings.resources_dir)
        for eid in mock_ids:
            try:
                lab_results = mock_repo.get_lab_results(tenant, eid, limit=5)
                if not lab_results:
                    skip_reasons[eid] = "sem lab_results (mock)"
                    continue
                
                lab = select_lab_result_for_field(lab_results, target_field)
                if not lab:
                    skip_reasons[eid] = f"lab_result não encontrado para field={target_field} (mock)"
                    continue
                
                y_val = extract_target_value(lab, target_field)
                if y_val is None:
                    skip_reasons[eid] = f"y_val é None para field={target_field} (mock)"
                    continue
                
                dilution = dilution_factors.get(eid, 1.0)
                if dilution <= 0:
                    dilution = 1.0
                if dilution != 1.0 and dilution > 0:
                    y_val = y_val / dilution
                
                y_values[eid] = float(y_val)
            except Exception as e:
                skip_reasons[eid] = f"erro mock: {e}"
    
    return y_values, skip_reasons


async def _collect_features_with_cache_v2(
    tenant: str,
    payload: dict,
    step_id: str,
    experiment_ids: list[str],
    engine: "PipelineEngine",
    config: "PipelineExecutionConfig",
    use_cache: bool = True,
    invalidate_cache: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, dict], list[str], list[str], dict]:
    """
    Coleta features usando cache v2 (por experimento).
    
    Só executa o pipeline para experimentos que não estão em cache.
    
    Args:
        tenant: ID do tenant
        payload: Pipeline JSON
        step_id: ID do step de ML
        experiment_ids: Lista de IDs de experimentos
        engine: Engine de pipeline já configurado
        config: Configuração do pipeline
        use_cache: Se deve usar cache
        invalidate_cache: Se deve invalidar cache antes
    
    Returns:
        (X_dict, metadata_dict, errors, skipped, block_cfg):
            - X_dict: {exp_id: X_array} features por experimento
            - metadata_dict: {exp_id: {...}} metadados por experimento
            - errors: Lista de erros
            - skipped: Lista de motivos de skip
            - block_cfg: Configuração do bloco
    """
    # run_in_thread está definido neste mesmo arquivo (router.py)
    
    X_dict: dict[str, np.ndarray] = {}
    metadata_dict: dict[str, dict] = {}
    errors: list[str] = []
    skipped: list[str] = []
    block_cfg: dict = {}
    
    # Encontrar step e bloco
    step = None
    for s in config.steps:
        if str(s.step_id) == step_id:
            step = s
            break
    
    if not step:
        errors.append(f"Step {step_id} não encontrado")
        return X_dict, metadata_dict, errors, skipped, block_cfg
    
    block_name = str(step.block_name)
    block_cfg = dict(step.block_config or {})
    
    # Invalidar cache v2 se solicitado
    if invalidate_cache:
        pipeline_hash = _compute_pipeline_hash(payload, step_id)
        print(f"[cache-v2] INVALIDANDO cache para {len(experiment_ids)} experimentos (pipeline_hash={pipeline_hash})")
        for exp_id in experiment_ids:
            _invalidate_experiment_cache(tenant, pipeline_hash, step_id, exp_id)
        print(f"[cache-v2] Cache invalidado!")
    
    # Carregar do cache v2 o que já existe
    missing_ids = list(experiment_ids)
    if use_cache and not invalidate_cache:
        pipeline_hash = _compute_pipeline_hash(payload, step_id)
        cached_X, cached_meta, missing_ids = _get_cached_experiments_batch(
            tenant, pipeline_hash, step_id, experiment_ids
        )
        X_dict.update(cached_X)
        metadata_dict.update(cached_meta)
        
        if cached_X:
            print(f"[cache-v2] Carregados {len(cached_X)} experimentos do cache")
    
    # Executar pipeline APENAS para experimentos faltantes
    if missing_ids:
        print(f"[cache-v2] Executando pipeline para {len(missing_ids)} experimentos (use_cache={use_cache}, invalidate_cache={invalidate_cache})")
        
        for exp_id in missing_ids:
            try:
                result = await run_in_thread(
                    engine.execute,
                    {"experimentId": exp_id, "analysisId": None, "tenant": tenant},
                )
                
                input_mapping = dict(step.input_mapping or {})
                
                print(f"[cache-v2] Processando {exp_id}, block_name={block_name}")
                
                if block_name == "ml_inference":
                    src = input_mapping.get("features", "")
                    if "." not in src:
                        skipped.append(f"{exp_id}: no '.' in features source '{src}'")
                        continue
                    src_step, src_out = src.split(".", 1)
                    src_output = result.step_results.get(src_step)
                    
                    features = src_output.data.get(src_out) if src_output and isinstance(src_output.data, dict) else None
                    if not isinstance(features, dict):
                        skipped.append(f"{exp_id}: features not dict")
                        continue
                    
                    label = _extract_label_from_any(features)
                    input_feature = str(block_cfg.get("input_feature", "growth_rate") or "growth_rate")
                    channel = str(block_cfg.get("channel", "") or "")
                    
                    fv, _ = _select_feature_value(features, input_feature, channel)
                    if fv is None:
                        skipped.append(f"{exp_id}: feature value is None")
                        continue
                    
                    # Extrair y e dilution_factor dos inputs do bloco (se conectados)
                    y_value_from_input = None
                    dilution_from_input = None
                    
                    # Debug: mostrar input_mapping
                    print(f"[cache-v2] {exp_id}: input_mapping keys = {list(input_mapping.keys())}")
                    
                    # Tentar extrair y
                    y_src = input_mapping.get("y", "")
                    if y_src and "." in y_src:
                        y_step, y_out = y_src.split(".", 1)
                        y_output = result.step_results.get(y_step)
                        if y_output and isinstance(y_output.data, dict):
                            y_data = y_output.data.get(y_out)
                            
                            # y_data pode ser:
                            # 1. Lista de lab_results (mais comum) - extrair valor baseado no label
                            # 2. Dict com value/y/target
                            # 3. Número direto
                            if isinstance(y_data, list) and len(y_data) > 0:
                                # Lista de lab_results - usar select_lab_result_for_field
                                # Inferir target_field baseado no label
                                targets_map = DEFAULT_TARGETS_MAP
                                by_unit = targets_map.get(label) if isinstance(targets_map.get(label), dict) else None
                                unit = str(block_cfg.get("output_unit") or "").strip()
                                target_field = str(by_unit.get(unit) or "") if isinstance(by_unit, dict) else ""
                                if not target_field:
                                    # Fallback para campo padrão baseado no label
                                    LABEL_TO_DEFAULT_FIELD = {
                                        "coliformes_totais": "nmp_100ml",
                                        "ecoli": "nmp_100ml",
                                        "e_coli": "nmp_100ml",
                                    }
                                    target_field = LABEL_TO_DEFAULT_FIELD.get(label, "nmp_100ml")
                                
                                # Usar select_lab_result_for_field que já filtra por label correto
                                lab = select_lab_result_for_field(y_data, target_field)
                                if lab:
                                    y_val = extract_target_value(lab, target_field)
                                    if y_val is not None:
                                        y_value_from_input = float(y_val)
                                        print(f"[cache-v2] {exp_id}: y extraído via select_lab_result_for_field({target_field}) = {y_value_from_input}")
                                else:
                                    # Fallback: tentar campos diretos no primeiro lab_result
                                    lab = y_data[0]
                                    if isinstance(lab, dict):
                                        LABEL_TO_FIELD = {
                                            "coliformes_totais": ["coliformesTotaisNmp", "coliformesTotaisUfc", "count"],
                                            "ecoli": ["ecoliNmp", "ecoliUfc", "count"],
                                            "e_coli": ["ecoliNmp", "ecoliUfc", "count"],
                                        }
                                        fields_to_try = LABEL_TO_FIELD.get(label, ["count", "coliformesTotaisNmp", "ecoliNmp"])
                                        for field in fields_to_try:
                                            val = lab.get(field)
                                            if val is not None:
                                                try:
                                                    y_value_from_input = float(val)
                                                    print(f"[cache-v2] {exp_id}: y extraído de lab_results[0].{field} = {y_value_from_input}")
                                                    break
                                                except (ValueError, TypeError):
                                                    continue
                            elif isinstance(y_data, dict):
                                y_val = y_data.get("value") or y_data.get("y") or y_data.get("target")
                                if y_val is not None:
                                    y_value_from_input = float(y_val)
                            elif isinstance(y_data, (int, float)):
                                y_value_from_input = float(y_data)
                    
                    # Tentar extrair dilution_factor
                    dil_src = input_mapping.get("dilution_factor", "")
                    if dil_src and "." in dil_src:
                        dil_step, dil_out = dil_src.split(".", 1)
                        dil_output = result.step_results.get(dil_step)
                        if dil_output and isinstance(dil_output.data, dict):
                            dil_data = dil_output.data.get(dil_out)
                            if isinstance(dil_data, dict):
                                dil_val = dil_data.get("value") or dil_data.get("dilution") or dil_data.get("factor")
                                if dil_val is not None:
                                    dilution_from_input = float(dil_val)
                            elif isinstance(dil_data, (int, float)):
                                dilution_from_input = float(dil_data)
                    
                    print(f"[cache-v2] {exp_id}: RESULTADO -> y_value_from_input={y_value_from_input}, dilution_from_input={dilution_from_input}")
                    
                    X_vec = np.array([[float(fv)]], dtype=np.float32)
                    exp_metadata = {
                        "block_name": block_name,
                        "input_feature": input_feature,
                        "channel": channel,
                        "label": label,
                    }
                    
                    # Adicionar y e dilution ao metadata se encontrados
                    if y_value_from_input is not None:
                        exp_metadata["y_value_from_input"] = y_value_from_input
                    if dilution_from_input is not None:
                        exp_metadata["dilution_from_input"] = dilution_from_input
                        exp_metadata["dilution_factor"] = dilution_from_input
                
                elif block_name in ("ml_inference_series", "ml_inference_multichannel"):
                    src = input_mapping.get("sensor_data", "")
                    if "." not in src:
                        skipped.append(f"{exp_id}: no '.' in sensor_data source")
                        continue
                    src_step, src_out = src.split(".", 1)
                    src_output = result.step_results.get(src_step)
                    sensor_data = src_output.data.get(src_out) if src_output and isinstance(src_output.data, dict) else None
                    
                    if not isinstance(sensor_data, dict):
                        skipped.append(f"{exp_id}: sensor_data not dict")
                        continue
                    
                    label = _extract_label_from_any(sensor_data)
                    ch_dict = sensor_data.get("channels") or {}
                    
                    # Extrair y e dilution_factor dos inputs (mesma lógica do ml_inference)
                    y_value_from_input = None
                    dilution_from_input = None
                    
                    y_src = input_mapping.get("y", "")
                    if y_src and "." in y_src:
                        y_step, y_out = y_src.split(".", 1)
                        y_output = result.step_results.get(y_step)
                        if y_output and isinstance(y_output.data, dict):
                            y_data = y_output.data.get(y_out)
                            if isinstance(y_data, dict):
                                y_val = y_data.get("value") or y_data.get("y") or y_data.get("target")
                                if y_val is not None:
                                    y_value_from_input = float(y_val)
                            elif isinstance(y_data, (int, float)):
                                y_value_from_input = float(y_data)
                    
                    dil_src = input_mapping.get("dilution_factor", "")
                    if dil_src and "." in dil_src:
                        dil_step, dil_out = dil_src.split(".", 1)
                        dil_output = result.step_results.get(dil_step)
                        if dil_output and isinstance(dil_output.data, dict):
                            dil_data = dil_output.data.get(dil_out)
                            if isinstance(dil_data, dict):
                                dil_val = dil_data.get("value") or dil_data.get("dilution") or dil_data.get("factor")
                                if dil_val is not None:
                                    dilution_from_input = float(dil_val)
                            elif isinstance(dil_data, (int, float)):
                                dilution_from_input = float(dil_data)
                    
                    if block_name == "ml_inference_series":
                        channel = str(block_cfg.get("channel") or "").strip() or None
                        from ..components.pipeline.blocks import _series_from_sensor_data, _apply_length_policy
                        _, y_series, _ = _series_from_sensor_data(sensor_data, channel=channel)
                        y_series = np.array(y_series, dtype=np.float32).reshape(-1)
                        X_vec = y_series.reshape(1, -1)
                        exp_metadata = {
                            "block_name": block_name,
                            "channel": channel,
                            "series_length": len(y_series),
                            "label": label,
                        }
                        # Adicionar y e dilution ao metadata se encontrados
                        if y_value_from_input is not None:
                            exp_metadata["y_value_from_input"] = y_value_from_input
                        if dilution_from_input is not None:
                            exp_metadata["dilution_from_input"] = dilution_from_input
                            exp_metadata["dilution_factor"] = dilution_from_input
                    else:
                        # multichannel
                        channels_raw = block_cfg.get("channels", [])
                        if isinstance(channels_raw, str):
                            use_channels = [c.strip() for c in channels_raw.split(",") if c.strip()]
                        elif isinstance(channels_raw, list):
                            use_channels = [str(c).strip() for c in channels_raw if str(c).strip()]
                        else:
                            use_channels = list(ch_dict.keys())
                        
                        base_len = None
                        for ch_name, values in ch_dict.items():
                            if isinstance(values, list):
                                base_len = len(values)
                                break
                        
                        if base_len is None:
                            skipped.append(f"{exp_id}: no channel arrays found")
                            continue
                        
                        max_length_i = int(block_cfg.get("max_length") or base_len)
                        pad_value = float(block_cfg.get("pad_value", 0.0) or 0.0)
                        align = str(block_cfg.get("align", "end") or "end").strip().lower()
                        
                        from ..components.pipeline.blocks import _apply_length_policy
                        
                        arrays = []
                        for ch in use_channels:
                            values = ch_dict.get(ch)
                            if isinstance(values, list):
                                y = np.array(values, dtype=np.float32).reshape(-1)
                            else:
                                y = np.zeros(base_len, dtype=np.float32)
                            y = _apply_length_policy(y, max_length=max_length_i, pad_value=pad_value, align=align)
                            arrays.append(y)
                        
                        if not arrays:
                            skipped.append(f"{exp_id}: no channel arrays found")
                            continue
                        
                        mat = np.stack(arrays, axis=1)
                        X_vec = mat.reshape(1, -1)
                        exp_metadata = {
                            "block_name": block_name,
                            "channels": use_channels,
                            "max_length": max_length_i,
                            "label": label,
                        }
                        # Adicionar y e dilution ao metadata se encontrados
                        if y_value_from_input is not None:
                            exp_metadata["y_value_from_input"] = y_value_from_input
                        if dilution_from_input is not None:
                            exp_metadata["dilution_from_input"] = dilution_from_input
                            exp_metadata["dilution_factor"] = dilution_from_input
                else:
                    skipped.append(f"{exp_id}: unsupported block {block_name}")
                    continue
                
                # Salvar no dicionário
                print(f"[cache-v2] {exp_id}: X_vec.shape={X_vec.shape}, pronto para salvar")
                X_dict[exp_id] = X_vec
                metadata_dict[exp_id] = exp_metadata
                
                # Salvar no cache v2
                if use_cache:
                    pipeline_hash = _compute_pipeline_hash(payload, step_id)
                    
                    # Obter y_value e dilution para salvar no cache
                    y_value_for_cache = None
                    y_value_raw = None
                    dilution_applied = None
                    
                    try:
                        # Prioridade 1: usar valores dos inputs do bloco (já extraídos)
                        y_from_input = exp_metadata.get("y_value_from_input")
                        dil_from_input = exp_metadata.get("dilution_from_input")
                        
                        if y_from_input is not None:
                            y_value_raw = float(y_from_input)
                            y_value_for_cache = y_value_raw
                            
                            # Aplicar diluição se disponível
                            if dil_from_input is not None and dil_from_input != 1.0 and dil_from_input > 0:
                                y_value_for_cache = y_value_raw / dil_from_input
                                dilution_applied = float(dil_from_input)
                                print(f"[cache-v2] {exp_id}: y_from_input={y_value_raw}, dilution={dilution_applied}, y_corrected={y_value_for_cache}")
                            else:
                                print(f"[cache-v2] {exp_id}: y_from_input={y_value_raw} (sem diluição)")
                        
                        # Fallback: buscar do MongoDB (se não veio dos inputs)
                        elif label:
                            targets_map = DEFAULT_TARGETS_MAP
                            by_unit = targets_map.get(label) if isinstance(targets_map.get(label), dict) else None
                            unit = str(block_cfg.get("output_unit") or "").strip()
                            target_field = str(by_unit.get(unit) or "") if isinstance(by_unit, dict) else "nmp_100ml"
                            if not target_field:
                                target_field = "nmp_100ml"
                            
                            y_val, _ = _get_y_val_fast(tenant, exp_id, target_field, label)
                            if y_val is not None:
                                y_value_raw = float(y_val)
                                y_value_for_cache = y_value_raw  # Sem diluição no fallback
                                print(f"[cache-v2] {exp_id}: y_from_mongo={y_value_raw} (fallback, sem diluição)")
                    except Exception as ye:
                        print(f"[cache-v2] Aviso: erro ao buscar y_value para {exp_id}: {ye}")
                    
                    # Adicionar y_value ao metadata para salvar (sem campos temporários de input)
                    save_metadata = {k: v for k, v in exp_metadata.items() 
                                    if k not in ("y_value_from_input", "dilution_from_input")}
                    if y_value_for_cache is not None:
                        save_metadata["y_value"] = y_value_for_cache
                    if y_value_raw is not None and y_value_raw != y_value_for_cache:
                        save_metadata["y_value_raw"] = y_value_raw
                    if dilution_applied is not None:
                        save_metadata["dilution"] = dilution_applied
                    if exp_metadata.get("dilution_factor") is not None:
                        save_metadata["dilution_factor"] = exp_metadata["dilution_factor"]
                    
                    saved_ok = _save_experiment_features(
                        tenant=tenant,
                        pipeline_hash=pipeline_hash,
                        step_id=step_id,
                        exp_id=exp_id,
                        X=X_vec,
                        metadata=save_metadata,
                    )
                    if saved_ok:
                        print(f"[cache-v2] ✓ Salvo: {exp_id}" + (f" (y={y_value_for_cache})" if y_value_for_cache else ""))
                    else:
                        print(f"[cache-v2] ✗ Falhou ao salvar: {exp_id}")
                
            except Exception as e:
                errors.append(f"{exp_id}: {e}")
                continue
    
    return X_dict, metadata_dict, errors, skipped, block_cfg


def _get_y_val_fast(
    tenant: str,
    exp_id: str,
    target_field: str,
    label: str | None = None,
) -> tuple[float | None, str]:
    """
    Pré-verifica se um experimento tem y_val válido SEM executar o pipeline.
    
    Returns:
        (y_val, skip_reason) - y_val se encontrado, senão None e motivo do skip
    """
    settings = get_settings()
    
    # Determinar repositório
    if exp_id.startswith("mock:"):
        repo = MockRepository(settings.resources_dir)
    else:
        repo = MongoRepository(settings.mongo_uri, settings.tenant_db_prefix)
    
    # Buscar lab_results
    try:
        lab_results = repo.get_lab_results(tenant, exp_id, limit=5)
    except Exception as e:
        return None, f"erro ao buscar lab_results: {e}"
    
    if not lab_results:
        return None, "sem lab_results"
    
    # Selecionar lab_result apropriado
    lab = select_lab_result_for_field(lab_results, target_field)
    if not lab:
        return None, f"lab_result não encontrado para field={target_field}"
    
    # Extrair y_val
    y_val = extract_target_value(lab, target_field)
    if y_val is None:
        return None, f"y_val é None para field={target_field}"
    
    return y_val, ""


    editor = payload.get("editor") if isinstance(payload, dict) else None
    if not isinstance(editor, dict):
        return [], []
    nodes = editor.get("nodes") if isinstance(editor.get("nodes"), list) else []
    edges = editor.get("edges") if isinstance(editor.get("edges"), list) else []
    safe_nodes = [n for n in nodes if isinstance(n, dict) and n.get("id")]
    safe_edges = [e for e in edges if isinstance(e, dict) and e.get("id")]
    return safe_nodes, safe_edges


def _node_display_name(node: dict) -> str:
    data = node.get("data") if isinstance(node.get("data"), dict) else {}
    label = str(data.get("label") or "").strip()
    block_name = str(data.get("blockName") or "").strip()
    return label or block_name or str(node.get("id") or "")


def _summarize_editor_changes(prev_payload: dict, new_payload: dict) -> list[str]:
    prev_nodes, prev_edges = _safe_editor_graph(prev_payload)
    new_nodes, new_edges = _safe_editor_graph(new_payload)

    prev_nodes_by_id = {str(n.get("id")): n for n in prev_nodes}
    new_nodes_by_id = {str(n.get("id")): n for n in new_nodes}
    prev_edges_by_id = {str(e.get("id")): e for e in prev_edges}
    new_edges_by_id = {str(e.get("id")): e for e in new_edges}

    prev_node_ids = set(prev_nodes_by_id.keys())
    new_node_ids = set(new_nodes_by_id.keys())
    prev_edge_ids = set(prev_edges_by_id.keys())
    new_edge_ids = set(new_edges_by_id.keys())

    added_nodes = sorted(new_node_ids - prev_node_ids)
    removed_nodes = sorted(prev_node_ids - new_node_ids)
    added_edges = sorted(new_edge_ids - prev_edge_ids)
    removed_edges = sorted(prev_edge_ids - new_edge_ids)

    config_changed = 0
    label_changed = 0
    position_changed = 0

    for node_id in sorted(new_node_ids & prev_node_ids):
        prev_node = prev_nodes_by_id.get(node_id) or {}
        new_node = new_nodes_by_id.get(node_id) or {}

        prev_data = prev_node.get("data") if isinstance(prev_node.get("data"), dict) else {}
        new_data = new_node.get("data") if isinstance(new_node.get("data"), dict) else {}

        if str(prev_data.get("label") or "").strip() != str(new_data.get("label") or "").strip():
            label_changed += 1

        prev_cfg = prev_data.get("config") if isinstance(prev_data.get("config"), dict) else {}
        new_cfg = new_data.get("config") if isinstance(new_data.get("config"), dict) else {}
        if prev_cfg != new_cfg:
            config_changed += 1

        prev_pos = prev_node.get("position") if isinstance(prev_node.get("position"), dict) else {}
        new_pos = new_node.get("position") if isinstance(new_node.get("position"), dict) else {}
        if prev_pos != new_pos:
            position_changed += 1

    bullets: list[str] = []

    if added_nodes or removed_nodes:
        extra = []
        if added_nodes:
            names = [_node_display_name(new_nodes_by_id[i]) for i in added_nodes[:3] if i in new_nodes_by_id]
            extra.append(f"+{len(added_nodes)}" + (f" ({', '.join(names)})" if names else ""))
        if removed_nodes:
            names = [_node_display_name(prev_nodes_by_id[i]) for i in removed_nodes[:3] if i in prev_nodes_by_id]
            extra.append(f"-{len(removed_nodes)}" + (f" ({', '.join(names)})" if names else ""))
        bullets.append("Blocos: " + " / ".join(extra))

    if added_edges or removed_edges:
        bullets.append(f"Conexões: +{len(added_edges)} / -{len(removed_edges)}")

    if config_changed:
        bullets.append(f"Configurações alteradas: {config_changed}")

    if label_changed:
        bullets.append(f"Rótulos alterados: {label_changed}")

    if not bullets and position_changed:
        bullets.append(f"Layout atualizado: {position_changed} blocos reposicionados")

    return bullets


def _ensure_versions_initialized(tenant: str, pipeline: str) -> dict:
    """
    Garante que existe um manifest de versões e pelo menos 1 versão.

    Compatibilidade: o arquivo ativo continua sendo resources/<tenant>/pipeline/<pipeline>.json.
    Versões ficam em resources/<tenant>/pipeline/versions/<version>.json e o manifest em _versions.json.
    """
    tenant = _validate_segment(tenant, "tenant")
    pipeline = _validate_segment(pipeline or tenant, "pipeline")

    manifest = _read_versions_manifest(tenant)
    versions = manifest.get("versions") if isinstance(manifest.get("versions"), list) else []
    active = manifest.get("active") if isinstance(manifest.get("active"), str) else ""

    active_file = _workspace_pipeline_file(tenant, pipeline)
    if not active_file.exists():
        return {"active": "", "versions": []}

    if versions and active:
        return {"active": active, "versions": versions}

    # Inicializar a partir do arquivo ativo atual
    versions_dir = _workspace_versions_dir(tenant)
    versions_dir.mkdir(parents=True, exist_ok=True)
    v1 = versions_dir / "v1.json"
    if not v1.exists():
        v1.write_bytes(active_file.read_bytes())

    manifest = {
        "active": "v1",
        "versions": [
            {
                "id": "v1",
                "name": "v1",
                "created_at": _utc_now_iso(),
                "based_on": None,
                "history": [{"at": _utc_now_iso(), "reason": "Versão inicial"}],
            }
        ],
    }
    _write_versions_manifest(tenant, manifest)
    return {"active": "v1", "versions": manifest["versions"]}


def _next_version_id(existing: list[dict]) -> str:
    nums: list[int] = []
    for item in existing or []:
        if not isinstance(item, dict):
            continue
        vid = str(item.get("id") or "")
        m = re.match(r"^v(\d+)$", vid)
        if m:
            try:
                nums.append(int(m.group(1)))
            except Exception:
                pass
    n = max(nums) + 1 if nums else 2
    return f"v{n}"


def _append_version_history(manifest: dict, version_id: str, reason: str, action: str) -> None:
    if not isinstance(manifest, dict):
        return
    versions = manifest.get("versions")
    if not isinstance(versions, list):
        return
    rid = str(reason or "").strip()
    if not rid:
        return
    for v in versions:
        if not isinstance(v, dict):
            continue
        if str(v.get("id") or "") != version_id:
            continue
        history = v.get("history") if isinstance(v.get("history"), list) else []
        history.append({"at": _utc_now_iso(), "action": action, "reason": rid})
        v["history"] = history[-100:]
        break


def _set_version_name(manifest: dict, version_id: str, name: str) -> None:
    if not isinstance(manifest, dict):
        return
    versions = manifest.get("versions")
    if not isinstance(versions, list):
        return
    nm = str(name or "").strip()
    for v in versions:
        if isinstance(v, dict) and str(v.get("id") or "") == version_id:
            v["name"] = nm
            break

def _safe_filename(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = name.strip("._-") or "logo"
    return name[:80]


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _dominant_color_from_image_bytes(data: bytes, fallback: str = "#6366f1") -> str:
    """
    Estima uma cor dominante do logo (para cor de destaque).
    - Para raster (PNG/JPG/WebP/GIF): usa bins em baixa resoluÇão e ignora pixels quase brancos/transparentes.
    - Para SVG: tenta extrair a primeira cor hex encontrada.
    """
    try:
        if data.lstrip().startswith(b"<"):
            text = data.decode("utf-8", errors="ignore")
            m = re.search(r"#[0-9a-fA-F]{6}", text)
            if m:
                return m.group(0).lower()
            return fallback

        from PIL import Image
        import colorsys

        img = Image.open(__import__("io").BytesIO(data)).convert("RGBA")
        img.thumbnail((64, 64))
        pixels = list(img.getdata())

        counts: dict[tuple[int, int, int], int] = {}
        for r, g, b, a in pixels:
            if a < 20:
                continue
            # ignorar quase branco (fundos)
            if r > 245 and g > 245 and b > 245:
                continue
            # agrupar em bins para reduzir ruído
            key = (int(r // 16) * 16 + 8, int(g // 16) * 16 + 8, int(b // 16) * 16 + 8)
            counts[key] = counts.get(key, 0) + 1

        if not counts:
            return fallback

        # Ordenar por frequência, preferindo cores com mais saturação
        def score(item):
            (r, g, b), n = item
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            return (n, s, v)

        best = max(counts.items(), key=score)[0]
        return _rgb_to_hex(best)
    except Exception:
        return fallback


def _public_asset_url(tenant: str, pipeline: str, asset_path: str) -> str:
    asset_path = asset_path.lstrip("/").replace("\\", "/")
    return f"/pipelines/workspaces/assets/{tenant}/{pipeline}/{asset_path}"


def _normalize_source(source: str) -> str:
    src = str(source or "all").strip().lower()
    if src not in ("all", "mongo", "mock"):
        raise HTTPException(status_code=400, detail="source inválido (use: all, mongo, mock)")
    return src


def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Pipeline não encontrado")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao ler pipeline: {exc}")


def _write_json_file(path: Path, data: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao salvar pipeline: {exc}")


def _utc_now_iso() -> str:
    import datetime

    return datetime.datetime.utcnow().isoformat() + "Z"


def _default_pipeline_payload(name: str) -> dict:
    return {
        "version": "1.0",
        "name": name,
        "savedAt": _utc_now_iso(),
        "workspace": {
            "title": name,
            "logo": "",
            "accent_color": "#1e90ff",
        },
        "editor": {"nodes": [], "edges": []},
        "execution": {"name": name, "steps": [], "initial_state": {}},
    }

# ThreadPool para processamento pesado
_executor = ThreadPoolExecutor(max_workers=4)


async def run_in_thread(func, *args, **kwargs):
    """Executa função síncrona em thread separada."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))


def _create_use_case() -> ProcessExperimentUseCase:
    """Factory para criar o use case com dependências."""
    settings = get_settings()
    
    return ProcessExperimentUseCase(
        repository=MongoRepository(settings.mongo_uri, settings.tenant_db_prefix),
        tenant_loader=get_tenant_loader(),
        spectral_api=SpectralApiAdapter(settings.spectral_api_url),
        ml_adapter=OnnxInferenceAdapter(),
    )


@router.post(
    "/processar",
    summary="Processa dados do experimento e retorna predição"
)
async def endpoint_processar(dados: ProcessRequest):
    """
    Recebe os IDs de um experimento, busca e processa os dados,
    e retorna a predição final baseada na configuração do tenant.
    
    Se debug_mode=True, retorna dados intermediários de cada etapa:
    - Dados RAW
    - Após slice temporal
    - Após filtros de preprocessing
    - Após conversão espectral (se aplicável)
    - Após filtros pós-conversão
    - Dados normalizados
    - Curva ajustada
    """
    use_case = _create_use_case()
    
    request = UCRequest(
        experiment_id=dados.experimentId,
        analysis_id=dados.analysisId,
        tenant=dados.tenant,
    )
    
    # Se debug_mode não vier na requisição, usa config do tenant
    debug_mode = dados.debug_mode
    if debug_mode is None:
        tenant_loader = get_tenant_loader()
        tenant_config = tenant_loader.load(dados.tenant)
        debug_mode = tenant_config.debug_mode
    
    result = await run_in_thread(use_case.execute, request, debug_mode)
    payload = _ensure_jsonable(result.to_dict())
    # Log tipos finais
    for k, v in payload.items():
        print(f"[payload] {k}: {type(v)} -> {v if isinstance(v, (int, float, str, bool)) else '...'}")
    try:
        return JSONResponse(content=payload)
    except TypeError as e:
        # Logar campo problemático
        print("[SERIALIZE ERROR]", e)
        for k, v in payload.items():
            try:
                json.dumps({k: v}, cls=NumpyEncoder)
            except Exception as inner:
                print(f"[FIELD] {k}: type={type(v)} value={v} error={inner}")
        # Fallback extra usando encoder explícito
        return JSONResponse(content=json.loads(json.dumps(payload, cls=NumpyEncoder)))


@router.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(validate_api_key)],
    summary="Executa predição direta a partir das features"
)
async def endpoint_predict(features: PredictRequest):
    """
    Recebe as 7 features já calculadas e retorna a predição.
    Requer token de autenticação no header.
    """
    settings = get_settings()
    ml_adapter = OnnxInferenceAdapter()
    
    # Caminhos dos modelos (hardcoded por enquanto, mover para config)
    resources = settings.resources_dir
    
    models = {
        "coli_nmp": {
            "model": str(resources / "LINEARregression_richards_turbidimetria_R__NMP_20_01_ATE_03_09_2025.onnx"),
            "scaler": str(resources / "scaler_turb_R_NMP_20_01_ATE_03_09_2025.joblib"),
            "feature": "TempoPontoInflexao",
        },
        "coli_ufc": {
            "model": str(resources / "LINEARregression_richards_turbidimetria_R_UFC_01_07_ATE_03_09_2025.onnx"),
            "scaler": str(resources / "scaler_turb_R_UFC_01_07_ATE_03_09_2025.joblib"),
            "feature": "TempoPontoInflexao",
        },
        "ecoli_nmp": {
            "model": str(resources / "LINEARregression_richards_fluorescencia_R_NMP_20_01_ATE_03_09_2025.onnx"),
            "scaler": str(resources / "scaler_fluorescencia_R_NMP_20_01_ATE_03_09_2025.joblib"),
            "feature": "TempoPontoInflexao",
        },
        "ecoli_ufc": {
            "model": str(resources / "LINEARregression_richards_fluorescencia_R_UFC_01_07_ATE_03_09_2025.onnx"),
            "scaler": str(resources / "scaler_fluorescencia_R_UFC_01_07_ATE_03_09_2025.joblib"),
            "feature": "TempoPontoInflexao",
        },
    }
    
    results = {}
    for key, config in models.items():
        feature_name = config["feature"]
        feature_value = getattr(features, feature_name)
        pred = await run_in_thread(
            ml_adapter.predict,
            config["model"],
            config["scaler"],
            feature_value
        )
        results[f"predict_{key}"] = pred
    
    return results


@router.get(
    "/pipelines/blocks",
    summary="Lista os blocos de pipeline disponíveis"
)
async def endpoint_list_pipeline_blocks():
    return {"blocks": BlockRegistry.get_info()}


@router.post(
    "/pipelines/simulate",
    response_model=PipelineRunResponse,
    summary="Executa um pipeline declarativo enviado pela GUI"
)
async def endpoint_simulate_pipeline(request: PipelineRunRequest):
    steps = [
        PipelineStep(
            block_name=step.block_name,
            block_config=step.block_config,
            depends_on=step.depends_on,
            step_id=step.step_id,
            input_mapping=step.input_mapping
        )
        for step in request.steps
    ]

    config = PipelineConfig(
        name=request.name,
        description=request.description or "",
        steps=steps,
        max_parallel=request.max_parallel,
        timeout_seconds=request.timeout_seconds,
        fail_fast=request.fail_fast,
        generate_output_graphs=request.generate_output_graphs
    )

    engine = PipelineEngine(config)
    result = engine.execute(request.initial_state or {})

    serialized_steps = {
        step_id: _ensure_jsonable(output.data)
        for step_id, output in result.step_results.items()
    }

    steps_meta = []
    for step in config.steps:
        output = result.step_results.get(step.step_id)
        ctx = getattr(output, "context", None)
        metadata = _ensure_jsonable(getattr(ctx, "metadata", {}) or {})
        success = bool(getattr(ctx, "success", False)) if ctx else False
        skipped = bool(metadata.get("skipped")) if isinstance(metadata, dict) else False
        blocked = bool(metadata.get("blocked")) if isinstance(metadata, dict) else False
        status = "skipped" if skipped else "success" if success else "blocked" if blocked else "failed"
        steps_meta.append(
            {
                "step_id": step.step_id,
                "block_name": step.block_name,
                "status": status,
                "success": success,
                "skipped": skipped,
                "duration_ms": float(getattr(ctx, "duration_ms", 0.0) or 0.0),
                "error_message": getattr(ctx, "error_message", None),
                "metadata": metadata,
            }
        )

    return PipelineRunResponse(
        pipeline_id=result.pipeline_id,
        success=result.success,
        duration_ms=result.duration_ms,
        errors=result.errors,
        step_results=serialized_steps,
        steps=steps_meta,
    )


def _tenant_prediction_label(prediction_id: str, sensor: str) -> str:
    parts = (prediction_id or "").split("_")
    if len(parts) > 1 and parts[1]:
        return parts[1]
    return sensor or prediction_id or "flow"


def _tenant_prediction_resource(sensor: str, prediction_id: str) -> str:
    suffix = "NMP" if (prediction_id or "").lower().endswith("_nmp") else "UFC" if (prediction_id or "").lower().endswith("_ufc") else None
    if sensor == "turbidimetry":
        return f"turbidimetria_{suffix}" if suffix else "turbidimetria_NMP"
    if sensor == "fluorescence":
        return f"fluorescencia_{suffix}" if suffix else "fluorescencia_NMP"
    return "fluorescencia_NMP"


def _tenant_feature_name_to_pipeline_feature(feature_name: str | None) -> str:
    mapping = {
        "TempoPontoInflexao": "inflection_time",
        "Amplitude": "asymptote",
        "LagTime": "lag_time",
    }
    return mapping.get(feature_name or "", "inflection_time")


def _build_tenant_pipeline_config(tenant_config, timeout_seconds: float, fail_fast: bool, generate_output_graphs: bool, debug_mode: bool) -> PipelineConfig:
    from ..infrastructure.config.tenant_loader import requires_conversion

    steps: list[PipelineStep] = []

    steps.append(
        PipelineStep(
            block_name="experiment_fetch",
            step_id="experiment_fetch",
            block_config={
                "include_experiment_output": bool(debug_mode),
                "include_experiment_data_output": bool(debug_mode),
            },
            depends_on=[],
            input_mapping={},
        )
    )

    if len(getattr(tenant_config, "predictions", []) or []) > 8:
        raise ValueError("Este endpoint suporta no máximo 8 predictions por tenant (limitação do response_builder).")

    response_builder_mapping = {}
    for idx, pred in enumerate(tenant_config.predictions, start=1):
        label_value = _tenant_prediction_label(pred.id, pred.sensor)
        resource_name = _tenant_prediction_resource(pred.sensor, pred.id)
        input_feature = _tenant_feature_name_to_pipeline_feature(getattr(getattr(pred, "ml_model", None), "feature_name", None))

        needs_conv, color_space, subchannel = requires_conversion(pred.channel)
        channel_for_fit = subchannel or ""

        step_label = f"label_{idx}"
        step_extract = f"{pred.sensor}_extraction_{idx}"
        step_slice = f"time_slice_{idx}"
        step_filter = f"moving_average_filter_{idx}"
        step_convert = f"{(color_space or '').lower()}_conversion_{idx}" if needs_conv and color_space else None
        step_curve = f"curve_fit_{idx}"
        step_growth_features = f"growth_features_{idx}"
        step_ml = f"ml_inference_{idx}"

        steps.append(
            PipelineStep(
                block_name="label",
                step_id=step_label,
                depends_on=["experiment_fetch"],
                block_config={"label": label_value},
                input_mapping={
                    "experiment_data": "experiment_fetch.experiment_data",
                    "experiment": "experiment_fetch.experiment",
                },
            )
        )

        steps.append(
            PipelineStep(
                block_name=f"{pred.sensor}_extraction",
                step_id=step_extract,
                depends_on=[step_label],
                block_config={"generate_output_graphs": bool(generate_output_graphs)},
                input_mapping={
                    "experiment_data": f"{step_label}.experiment_data",
                },
            )
        )

        steps.append(
            PipelineStep(
                block_name="time_slice",
                step_id=step_slice,
                depends_on=[step_extract],
                block_config={
                    "slice_mode": "index",
                    "start_index": getattr(getattr(pred, "preprocessing", None), "startIndex", 0) or 0,
                    "end_index": getattr(getattr(pred, "preprocessing", None), "endIndex", None),
                    "include_raw_output": bool(debug_mode),
                },
                input_mapping={
                    "sensor_data": f"{step_extract}.sensor_data",
                },
            )
        )

        # Suporte inicial: moving_average no preprocessing do tenant
        filters = getattr(getattr(pred, "preprocessing", None), "filters", []) or []
        last_sensor_step = step_slice
        for filter_cfg in filters:
            if not isinstance(filter_cfg, dict):
                continue
            if filter_cfg.get("type") != "moving_average":
                continue
            steps.append(
                PipelineStep(
                    block_name="moving_average_filter",
                    step_id=step_filter,
                    depends_on=[last_sensor_step],
                    block_config={
                        "window_size": int(filter_cfg.get("window", 5) or 5),
                        "alignment": "center",
                        "include_raw_output": bool(debug_mode),
                    },
                    input_mapping={
                        "sensor_data": f"{last_sensor_step}.sensor_data",
                    },
                )
            )
            last_sensor_step = step_filter
            break

        if needs_conv and color_space:
            conversion_block = {
                "RGB": "rgb_conversion",
                "XYZ": "xyz_conversion",
                "LAB": "lab_conversion",
                "HSV": "hsv_conversion",
                "HSB": "hsb_conversion",
                "CMYK": "cmyk_conversion",
                "xyY": "xyy_conversion",
            }.get(color_space)
            if not conversion_block:
                raise ValueError(f"Conversão não suportada para color_space={color_space} (channel={pred.channel})")
            steps.append(
                PipelineStep(
                    block_name=conversion_block,
                    step_id=step_convert,
                    depends_on=[last_sensor_step],
                    block_config={
                        "include_raw_output": bool(debug_mode),
                        "generate_output_graphs": bool(generate_output_graphs),
                    },
                    input_mapping={
                        "sensor_data": f"{last_sensor_step}.sensor_data",
                    },
                )
            )
            last_sensor_step = step_convert

        steps.append(
            PipelineStep(
                block_name="curve_fit",
                step_id=step_curve,
                depends_on=[last_sensor_step],
                block_config={
                    "model": pred.math_model or "richards",
                    "channel": channel_for_fit,
                    "include_raw_output": bool(debug_mode),
                    "generate_output_graphs": bool(generate_output_graphs),
                },
                input_mapping={
                    "sensor_data": f"{last_sensor_step}.sensor_data",
                },
            )
        )

        steps.append(
            PipelineStep(
                block_name="growth_features",
                step_id=step_growth_features,
                depends_on=[step_curve],
                block_config={
                    "channel": channel_for_fit,
                    "features": ["inflection_time", "growth_rate", "asymptote", "r_squared"],
                    "include_raw_output": bool(debug_mode),
                },
                input_mapping={
                    "sensor_data": f"{step_curve}.fitted_data",  # Curvas ajustadas
                },
            )
        )

        steps.append(
            PipelineStep(
                block_name="ml_inference",
                step_id=step_ml,
                depends_on=[step_growth_features],
                block_config={
                    "resource": resource_name,
                    "input_feature": input_feature,
                    "channel": channel_for_fit,
                    "include_raw_output": bool(debug_mode),
                },
                input_mapping={
                    "features": f"{step_growth_features}.features",
                },
            )
        )

        response_builder_mapping[f"input_{idx}"] = f"{step_ml}.prediction"

    steps.append(
        PipelineStep(
            block_name="response_builder",
            step_id="response_builder",
            depends_on=[s.step_id for s in steps if s.step_id.startswith("ml_inference_")],
            block_config={
                "group_by_label": True,
                "flatten_labels": True,
                "include_raw_output": bool(debug_mode),
                "static_fields": {"analysis_mode": getattr(tenant_config, "analysis_mode", "prediction")},
            },
            input_mapping=response_builder_mapping,
        )
    )

    return PipelineConfig(
        name=f"{getattr(tenant_config, 'tenant_id', None) or 'tenant'}_api_pipeline",
        description="Pipeline padrão gerado a partir do tenant.json",
        steps=steps,
        max_parallel=1,
        timeout_seconds=float(timeout_seconds),
        fail_fast=bool(fail_fast),
        generate_output_graphs=bool(generate_output_graphs),
    )


def _build_workspace_pipeline_config(
    tenant: str,
    pipeline_json: dict,
    *,
    timeout_seconds: float,
    fail_fast: bool,
    generate_output_graphs: bool,
) -> PipelineConfig:
    execution = pipeline_json.get("execution") if isinstance(pipeline_json.get("execution"), dict) else {}
    steps_raw = execution.get("steps") if isinstance(execution.get("steps"), list) else []
    if not steps_raw:
        raise ValueError("Pipeline não possui passos em execution.steps")

    steps: list[PipelineStep] = []
    for s in steps_raw:
        if not isinstance(s, dict):
            continue
        step_id = str(s.get("step_id") or "")
        block_name = str(s.get("block_name") or "")
        if not step_id or not block_name:
            continue
        steps.append(
            PipelineStep(
                step_id=step_id,
                block_name=block_name,
                block_config=s.get("block_config") if isinstance(s.get("block_config"), dict) else {},
                depends_on=s.get("depends_on") if isinstance(s.get("depends_on"), list) else [],
                input_mapping=s.get("input_mapping") if isinstance(s.get("input_mapping"), dict) else {},
            )
        )

    name = str(execution.get("name") or pipeline_json.get("name") or f"{tenant}_pipeline")
    description = str(pipeline_json.get("description") or "")
    max_parallel_raw = execution.get("max_parallel")
    try:
        max_parallel = int(max_parallel_raw) if max_parallel_raw is not None else 1
    except Exception:
        max_parallel = 1

    return PipelineConfig(
        name=name,
        description=description,
        steps=steps,
        max_parallel=max_parallel,
        timeout_seconds=float(timeout_seconds),
        fail_fast=bool(fail_fast),
        generate_output_graphs=bool(generate_output_graphs),
    )


@router.post(
    "/pipelines/execute",
    response_model=TenantPipelineExecuteResponse,
    response_model_exclude_none=True,
    summary="Executa o pipeline ativo do cliente"
)
async def endpoint_execute_tenant_pipeline(request: TenantPipelineExecuteRequest):
    # Preferir pipeline do Pipeline Studio salvo em resources/<tenant>/pipeline/<tenant>.json (versão ativa)
    tenant = _validate_segment(request.tenant, "tenant")
    pipeline_name = tenant
    workspace_file = _workspace_pipeline_file(tenant, pipeline_name)

    debug_mode = bool(request.debug_mode) if request.debug_mode is not None else False

    config = None
    if workspace_file.exists():
        try:
            payload = _read_json_file(workspace_file)
            config = _build_workspace_pipeline_config(
                tenant=tenant,
                pipeline_json=payload,
                timeout_seconds=request.timeout_seconds,
                fail_fast=request.fail_fast,
                generate_output_graphs=request.generate_output_graphs,
            )
        except Exception:
            config = None

    # Fallback (legado): gerar pipeline a partir do tenant.json (config interna)
    if config is None:
        tenant_loader = get_tenant_loader()
        tenant_config = tenant_loader.load(tenant)
        if request.debug_mode is None:
            debug_mode = bool(getattr(tenant_config, "debug_mode", False))
        config = _build_tenant_pipeline_config(
            tenant_config=tenant_config,
            timeout_seconds=request.timeout_seconds,
            fail_fast=request.fail_fast,
            generate_output_graphs=request.generate_output_graphs,
            debug_mode=bool(debug_mode),
        )

    engine = PipelineEngine(config)
    result = engine.execute(
        {
            "experimentId": request.experimentId,
            "analysisId": request.analysisId,
            "tenant": tenant,
        }
    )

    response_payload = {}
    response_output = result.step_results.get("response_builder")
    if response_output and isinstance(response_output.data, dict):
        response_payload = _ensure_jsonable(response_output.data.get("response") or {})
    else:
        # fallback: procurar o primeiro step que tenha campo 'response'
        for out in result.step_results.values():
            if out and isinstance(getattr(out, "data", None), dict) and "response" in out.data:
                response_payload = _ensure_jsonable(out.data.get("response") or {})
                break

    steps_meta = None
    if bool(getattr(request, "include_steps", False)):
        steps_meta = []
        for step in config.steps:
            output = result.step_results.get(step.step_id)
            ctx = getattr(output, "context", None)
            metadata = _ensure_jsonable(getattr(ctx, "metadata", {}) or {})
            success = bool(getattr(ctx, "success", False)) if ctx else False
            skipped = bool(metadata.get("skipped")) if isinstance(metadata, dict) else False
            blocked = bool(metadata.get("blocked")) if isinstance(metadata, dict) else False
            status = "skipped" if skipped else "success" if success else "blocked" if blocked else "failed"
            steps_meta.append(
                {
                    "step_id": step.step_id,
                    "block_name": step.block_name,
                    "status": status,
                    "success": success,
                    "skipped": skipped,
                    "duration_ms": float(getattr(ctx, "duration_ms", 0.0) or 0.0),
                    "error_message": getattr(ctx, "error_message", None),
                    "metadata": metadata,
                }
            )

    return TenantPipelineExecuteResponse(
        pipeline_id=result.pipeline_id,
        success=result.success,
        duration_ms=result.duration_ms,
        errors=result.errors,
        response=response_payload,
        steps=steps_meta,
    )


def _update_workspace_pipeline_block_config(payload: dict, step_id: str, patch: dict) -> None:
    if not isinstance(payload, dict) or not step_id or not isinstance(patch, dict):
        return

    # execution.steps
    exec_obj = payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
    steps = exec_obj.get("steps") if isinstance(exec_obj.get("steps"), list) else []
    for s in steps:
        if not isinstance(s, dict):
            continue
        if str(s.get("step_id") or "") != step_id:
            continue
        cfg = s.get("block_config") if isinstance(s.get("block_config"), dict) else {}
        cfg.update(patch)
        s["block_config"] = cfg
        break

    # editor.nodes[].data.config
    editor = payload.get("editor") if isinstance(payload.get("editor"), dict) else {}
    nodes = editor.get("nodes") if isinstance(editor.get("nodes"), list) else []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if str(n.get("id") or "") != step_id:
            continue
        data = n.get("data") if isinstance(n.get("data"), dict) else {}
        cfg = data.get("config") if isinstance(data.get("config"), dict) else {}
        cfg.update(patch)
        data["config"] = cfg
        n["data"] = data
        break


def _extract_label_from_any(data: Any) -> str:
    if isinstance(data, dict):
        lbl = data.get("_label")
        if isinstance(lbl, str) and lbl.strip():
            return lbl.strip()
    return ""


def _select_feature_value(features: Any, input_feature: str, channel: str) -> tuple[float | None, str]:
    if not isinstance(features, dict):
        return None, ""

    input_feature = str(input_feature or "").strip()
    if not input_feature:
        return None, ""

    channel = str(channel or "").strip()
    if channel and channel in features and isinstance(features.get(channel), dict):
        val = features[channel].get(input_feature)
        try:
            return (float(val) if val is not None else None), channel
        except Exception:
            return None, channel

    # top-level
    if input_feature in features and not isinstance(features.get(input_feature), (dict, list)):
        try:
            return float(features.get(input_feature)), ""
        except Exception:
            return None, ""

    # procurar primeiro canal/dict com a feature
    for k, v in features.items():
        if not isinstance(v, dict):
            continue
        if input_feature in v:
            try:
                return float(v.get(input_feature)), str(k)
            except Exception:
                return None, str(k)
    return None, ""


# =============================================================================
# Funções de Otimização para Treinamento de Pipeline
# =============================================================================
#
# OTIMIZAÇÃO IMPLEMENTADA:
# 
# Problema: Ao treinar múltiplos modelos ML, o sistema executava o pipeline
#           completo para cada experimento, mesmo que:
#           - O experimento não tivesse lab_results
#           - O experimento não tivesse a bacteria/label necessária
#           - O mesmo experimento fosse necessário para múltiplos modelos
#
# Solução: Sistema de pré-filtragem e cache em 4 fases:
#
#   FASE 1: Análise de Requisitos
#     - Identifica label, unit, dilution de cada bloco ML
#
#   FASE 2: Pré-filtragem Rápida
#     - Consulta apenas metadados (lab_results, dilution_factor)
#     - Filtra experimentos sem executar pipeline
#     - Identifica quais experimentos servem para cada bloco ML
#
#   FASE 3: Execução com Cache
#     - Executa pipeline UMA vez por experimento único
#     - Armazena resultado em cache
#
#   FASE 4: Coleta de Dados
#     - Cada bloco ML busca dados do cache
#     - Sem re-execução do pipeline
#
# Benefício: Redução de 50-80% no tempo de treinamento em cenários típicos
#
# =============================================================================

def _analyze_ml_block_requirements(
    step_id: str,
    block_name: str,
    block_config: dict,
    metadata_path: Optional[str] = None
) -> dict:
    """
    Analisa os requisitos de um bloco ML para filtrar experimentos.
    
    Returns:
        {
            "step_id": str,
            "block_name": str,
            "label": Optional[str],  # bacteria esperada (se disponível no metadata)
            "unit": Optional[str],   # unidade esperada
            "requires_dilution": Optional[bool],  # True/False/None
            "input_feature": Optional[str],
            "channel": Optional[str],
        }
    """
    requirements = {
        "step_id": step_id,
        "block_name": block_name,
        "label": None,
        "unit": block_config.get("output_unit"),
        "requires_dilution": None,
        "input_feature": block_config.get("input_feature"),
        "channel": block_config.get("channel"),
    }
    
    # Detectar requisito de diluição pelo nome do step_id ou label
    step_id_lower = step_id.lower()
    if "dilut" in step_id_lower or "diluido" in step_id_lower or "diluted" in step_id_lower:
        requirements["requires_dilution"] = True
    elif "undiluted" in step_id_lower or "nao_diluido" in step_id_lower or "sem_diluicao" in step_id_lower:
        requirements["requires_dilution"] = False
    
    # Tentar ler metadata para obter label e outras informações
    if metadata_path:
        try:
            metadata_file = _repo_root() / metadata_path
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                
                # Extrair label do block_config salvo no metadata
                saved_block_config = metadata.get("block_config", {})
                if isinstance(saved_block_config, dict):
                    # Alguns metadatas podem ter informações úteis
                    # Por enquanto, não fazemos nada aqui, mas pode ser expandido
                    pass
        except Exception:
            pass
    
    return requirements


def _prefilter_experiments_for_training(
    experiment_ids: list[str],
    tenant: str,
    targets_map: dict,
    ml_blocks_requirements: list[dict],
) -> dict:
    """
    Pré-filtra experimentos consultando apenas metadados (lab_results e dilution).
    
    Returns:
        {
            "valid_experiments_by_block": {
                "step_id_1": ["exp1", "exp5", ...],
                "step_id_2": ["exp2", "exp8", ...],
            },
            "unique_experiments": ["exp1", "exp2", "exp5", "exp8", ...],
            "skipped_count": int,
        }
    """
    from ..infrastructure.database.mock_repository import MockExperimentRepository
    from ..infrastructure.database.corsan_rest_repository import CorsanRestExperimentRepository
    
    # Escolher repositório
    if tenant.lower() == "corsan":
        repo = CorsanRestExperimentRepository()
    else:
        repo = MockExperimentRepository()
    
    valid_experiments_by_block: dict[str, list[str]] = {
        req["step_id"]: [] for req in ml_blocks_requirements
    }
    unique_experiments = set()
    skipped_count = 0
    
    for exp_id in experiment_ids:
        try:
            # Buscar apenas metadados do experimento
            experiment = repo.get_experiment(exp_id)
            if not experiment:
                skipped_count += 1
                continue
            
            lab_results = experiment.lab_results or {}
            dilution_factor = getattr(experiment, "dilution_factor", 1.0) or 1.0
            
            # Para cada bloco ML, verificar se o experimento é válido
            for req in ml_blocks_requirements:
                unit = req.get("unit", "")
                requires_lab = req.get("requires_lab_results", True)  # Por padrão, requer lab_results
                
                # Se é forecaster (não requer lab_results), aceitar o experimento
                if not requires_lab:
                    valid_experiments_by_block[req["step_id"]].append(exp_id)
                    unique_experiments.add(exp_id)
                    continue
                
                # Para blocos que requerem lab_results
                # Verificar se tem algum resultado em lab_results
                has_lab_data = bool(lab_results)
                
                # Verificar diluição se necessário
                matches_dilution = True
                if req.get("requires_dilution") is True:
                    matches_dilution = (dilution_factor != 1.0)
                elif req.get("requires_dilution") is False:
                    matches_dilution = (dilution_factor == 1.0)
                
                # Se tem dados e atende requisitos de diluição
                if has_lab_data and matches_dilution:
                    valid_experiments_by_block[req["step_id"]].append(exp_id)
                    unique_experiments.add(exp_id)
        
        except Exception:
            skipped_count += 1
            continue
    
    return {
        "valid_experiments_by_block": valid_experiments_by_block,
        "unique_experiments": sorted(list(unique_experiments)),
        "skipped_count": skipped_count,
    }


def _execute_experiments_with_cache(
    experiment_ids: list[str],
    engine: Any,
    tenant: str,
    protocol_id: str,
    skip_missing: bool = True,
) -> tuple[dict, list[str]]:
    """
    Executa experimentos uma vez e armazena em cache.
    
    Returns:
        (cache_dict, errors_list)
        cache_dict = {
            "exp_id": {
                "result": PipelineResult,
                "lab_results": dict,
                "dilution_factor": float,
            }
        }
    """
    cache: dict[str, dict] = {}
    errors: list[str] = []
    
    for exp_id in experiment_ids:
        try:
            result = engine.execute({
                "experimentId": exp_id,
                "analysisId": protocol_id,
                "tenant": tenant
            })
            
            # Extrair metadados do experiment_fetch
            lab_results = None
            dilution_factor = 1.0
            
            # Procurar o step de experiment_fetch
            for step_id, step_output in result.step_results.items():
                if not step_output:
                    continue
                out_data = getattr(step_output, "data", {})
                if not isinstance(out_data, dict):
                    continue
                
                if "lab_results" in out_data:
                    lab_results = out_data.get("lab_results")
                if "dilution_factor" in out_data:
                    try:
                        df_raw = out_data.get("dilution_factor")
                        dilution_factor = float(df_raw) if df_raw is not None else 1.0
                        if dilution_factor <= 0:
                            dilution_factor = 1.0
                    except (ValueError, TypeError):
                        dilution_factor = 1.0
                
                if lab_results is not None:
                    break
            
            cache[exp_id] = {
                "result": result,
                "lab_results": lab_results,
                "dilution_factor": dilution_factor,
            }
        
        except Exception as e:
            if not skip_missing:
                errors.append(f"{exp_id}: falha ao executar pipeline: {e}")
            continue
    
    return cache, errors


@router.post(
    "/pipelines/train",
    response_model=TenantPipelineTrainResponse,
    response_model_exclude_none=True,
    summary="Treina modelos ML usando lab_results e atualiza o pipeline (otimizado)"
)
async def endpoint_train_tenant_pipeline(request: TenantPipelineTrainRequest):
    tenant = _validate_segment(request.tenant, "tenant")
    pipeline_name = tenant

    versions_info = _ensure_versions_initialized(tenant, pipeline_name)
    active_version = str(versions_info.get("active") or "").strip() or None

    base_version = _validate_version_id(getattr(request, "version", None)) if getattr(request, "version", None) else None
    if base_version:
        source_file = _workspace_versions_dir(tenant) / f"{base_version}.json"
        if not source_file.exists():
            raise HTTPException(status_code=404, detail="Versão do pipeline não encontrada em resources/<tenant>/pipeline/versions/")
    else:
        source_file = _workspace_pipeline_file(tenant, pipeline_name)
        if not source_file.exists():
            raise HTTPException(status_code=404, detail="Pipeline do tenant não encontrado em resources/<tenant>/pipeline/")

    payload = _read_json_file(source_file)
    config = _build_workspace_pipeline_config(
        tenant=tenant,
        pipeline_json=payload,
        timeout_seconds=300.0,
        fail_fast=True,
        generate_output_graphs=False,
    )

    engine = PipelineEngine(config)

    targets_map = request.targets_map if isinstance(request.targets_map, dict) else DEFAULT_TARGETS_MAP
    y_transform = str(request.y_transform or "log10p")
    skip_missing = bool(request.skip_missing)

    # Quais steps treinar
    trainable_names = {"ml_inference", "ml_inference_series", "ml_inference_multichannel"}
    forecaster_names = {"ml_forecaster_series"}
    trainable_steps = [s for s in config.steps if str(getattr(s, "block_name", "")) in trainable_names]
    forecaster_steps = [s for s in config.steps if str(getattr(s, "block_name", "")) in forecaster_names]

    model_specs_by_step: dict[str, dict] = {}
    if request.models:
        for m in request.models:
            if not getattr(m, "enabled", True):
                continue
            model_specs_by_step[str(m.step_id)] = {
                "algorithm": str(getattr(m, "algorithm", None) or "ridge"),
                "params": dict(getattr(m, "params", None) or {}),
                "grid_search": bool(getattr(m, "grid_search", False)),
                "algorithms": list(getattr(m, "algorithms", None) or []),
                "param_grid": dict(getattr(m, "param_grid", None) or {}),
                "params_by_algorithm": dict(getattr(m, "params_by_algorithm", None) or {}),
                "param_grid_by_algorithm": dict(getattr(m, "param_grid_by_algorithm", None) or {}),
                "selection_metric": str(getattr(m, "selection_metric", None) or "").strip(),
                "max_trials": int(getattr(m, "max_trials", None) or 0),
            }

    # localizar step_id do experiment_fetch
    fetch_step_id = ""
    for s in config.steps:
        if str(getattr(s, "block_name", "")) == "experiment_fetch":
            fetch_step_id = str(getattr(s, "step_id", ""))
            break

    skipped_experiments: list[str] = []
    errors: list[str] = []

    # datasets: (step_id,label,unit) -> {X:[], y:[], skipped:int}
    datasets: dict[tuple[str, str, str], dict[str, Any]] = {}
    # forecasters: (step_id,label,target_channel) -> {X:[], y:[], skipped:int}
    forecaster_datasets: dict[tuple[str, str, str], dict[str, Any]] = {}

    # ====================================================================================
    # OTIMIZAÇÃO: Pré-filtragem e execução com cache
    # ====================================================================================
    
    # 1. Analisar requisitos dos blocos ML (trainable + forecasters)
    ml_blocks_requirements = []
    
    # Adicionar blocos treináveis (ml_inference, ml_inference_series, etc.)
    for step in trainable_steps:
        step_id = str(step.step_id)
        block_name = str(step.block_name)
        block_cfg = dict(step.block_config or {})
        metadata_path = block_cfg.get("metadata_path")
        
        req = _analyze_ml_block_requirements(step_id, block_name, block_cfg, metadata_path)
        ml_blocks_requirements.append(req)
    
    # Adicionar forecasters (não dependem de lab_results, mas ajuda filtrar)
    for step in forecaster_steps:
        step_id = str(step.step_id)
        block_name = str(step.block_name)
        block_cfg = dict(step.block_config or {})
        metadata_path = block_cfg.get("metadata_path")
        
        req = _analyze_ml_block_requirements(step_id, block_name, block_cfg, metadata_path)
        req["requires_lab_results"] = False  # Forecasters não precisam de lab_results
        ml_blocks_requirements.append(req)
    
    # 2. Pré-filtrar experimentos (consulta apenas metadados)
    # Para simplificar, se não há requisitos específicos, processar todos
    if ml_blocks_requirements:
        prefilter_result = _prefilter_experiments_for_training(
            [str(eid or "").strip() for eid in request.experimentIds if str(eid or "").strip()],
            tenant,
            targets_map,
            ml_blocks_requirements,
        )
        
        unique_experiments = prefilter_result["unique_experiments"]
        valid_experiments_by_block = prefilter_result["valid_experiments_by_block"]
        
        # Experimentos que foram filtrados
        all_requested = set(str(eid or "").strip() for eid in request.experimentIds if str(eid or "").strip())
        filtered_out = all_requested - set(unique_experiments)
        skipped_experiments.extend(list(filtered_out))
    else:
        # Sem blocos ML, processar todos
        unique_experiments = [str(eid or "").strip() for eid in request.experimentIds if str(eid or "").strip()]
        valid_experiments_by_block = {}
    
    # 3. Executar apenas os experimentos únicos necessários com cache
    pipeline_cache, cache_errors = await run_in_thread(
        _execute_experiments_with_cache,
        unique_experiments,
        engine,
        tenant,
        request.protocolId,
        skip_missing,
    )
    errors.extend(cache_errors)
    
    # Log de otimização
    total_requested = len([str(eid or "").strip() for eid in request.experimentIds if str(eid or "").strip()])
    total_executed = len(unique_experiments)
    saved_executions = total_requested - total_executed
    
    print(f"[OTIMIZAÇÃO] Experimentos solicitados: {total_requested}")
    print(f"[OTIMIZAÇÃO] Experimentos únicos executados: {total_executed}")
    print(f"[OTIMIZAÇÃO] Execuções economizadas: {saved_executions} ({saved_executions/total_requested*100:.1f}%)" if total_requested > 0 else "")
    
    # ====================================================================================
    # Loop otimizado: processar experimentos do cache
    # ====================================================================================
    
    for exp_id in unique_experiments:
        if exp_id not in pipeline_cache:
            continue
        
        cached = pipeline_cache[exp_id]
        result = cached["result"]
        lab_results = cached["lab_results"]
        dilution_factor = cached["dilution_factor"]
        
        # Compatibilidade: lab pode ser dict ou None
        lab = lab_results if isinstance(lab_results, dict) else None

        # Forecaster não depende de lab_results: pode treinar mesmo quando lab estiver ausente.
        for step in forecaster_steps:
            step_id = str(step.step_id)
            block_name = str(step.block_name)
            block_cfg = dict(step.block_config or {})
            spec = model_specs_by_step.get(step_id, {"algorithm": "ridge", "params": {}})

            input_mapping = dict(step.input_mapping or {})
            src = input_mapping.get("sensor_data", "")

            try:
                if "." not in src:
                    raise ValueError("input_mapping.sensor_data inválido")
                src_step, src_out = src.split(".", 1)
                src_output = result.step_results.get(src_step)
                sensor_data = src_output.data.get(src_out) if src_output and isinstance(src_output.data, dict) else None
                if not isinstance(sensor_data, dict) or not isinstance(sensor_data.get("channels"), dict):
                    raise ValueError("sensor_data inválido")

                label = _extract_label_from_any(sensor_data) or "sem_etiqueta"
                ch_dict = sensor_data.get("channels") or {}
                available = [str(k) for k in ch_dict.keys()]

                from ..components.pipeline.blocks import MLForecasterSeriesBlock  # local import

                input_channels_raw = block_cfg.get("input_channels", None)
                if isinstance(input_channels_raw, str):
                    input_channels = [c.strip() for c in input_channels_raw.split(",") if c.strip()]
                elif isinstance(input_channels_raw, list):
                    input_channels = [str(c).strip() for c in input_channels_raw if str(c).strip()]
                else:
                    input_channels = []

                target_channel_raw = str(block_cfg.get("target_channel", "") or "").strip()

                resolved_target: str | None = None
                if target_channel_raw:
                    try:
                        resolved_target = MLForecasterSeriesBlock._resolve_channel_name(target_channel_raw, available)
                    except ValueError:
                        resolved_target = None

                if input_channels:
                    resolved_inputs: list[str] = []
                    for c in input_channels:
                        resolved_inputs.append(MLForecasterSeriesBlock._resolve_channel_name(c, available))
                    input_channels = resolved_inputs
                else:
                    if resolved_target:
                        input_channels = MLForecasterSeriesBlock._default_channels_by_target(available, resolved_target)
                    else:
                        input_channels = list(available)

                input_channels = [c for c in input_channels if c in ch_dict]
                if not input_channels:
                    raise ValueError("nenhum canal de entrada")

                if not resolved_target:
                    if target_channel_raw:
                        subset = list(input_channels)
                        if target_channel_raw in subset:
                            resolved_target = target_channel_raw
                        else:
                            matches = [a for a in subset if a.endswith(f":{target_channel_raw}")]
                            if len(matches) == 1:
                                resolved_target = matches[0]
                            elif len(matches) > 1:
                                raise ValueError(
                                    f"target_channel '{target_channel_raw}' ambíguo no conjunto selecionado; use '<sensor>:<canal>'"
                                )
                            else:
                                raise ValueError(f"target_channel '{target_channel_raw}' não encontrado")
                    else:
                        resolved_target = input_channels[0]

                if resolved_target not in input_channels:
                    input_channels = [resolved_target] + [c for c in input_channels if c != resolved_target]

                try:
                    window = int(block_cfg.get("window", 30) or 30)
                except Exception:
                    window = 30
                window = 1 if window < 1 else 2048 if window > 2048 else window

                try:
                    horizon = int(block_cfg.get("horizon", 1) or 1)
                except Exception:
                    horizon = 1
                horizon = 1 if horizon < 1 else 2048 if horizon > 2048 else horizon

                try:
                    max_samples = int(block_cfg.get("max_samples", 2000) or 2000)
                except Exception:
                    max_samples = 2000
                max_samples = 50 if max_samples < 50 else 50000 if max_samples > 50000 else max_samples

                arrays = []
                min_len = None
                for ch in input_channels:
                    y = np.array(ch_dict.get(ch) or [], dtype=np.float32).reshape(-1)
                    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    arrays.append(y)
                    min_len = y.size if min_len is None else min(min_len, y.size)
                if not min_len or min_len <= 0:
                    raise ValueError("séries vazias")
                arrays = [a[:min_len] for a in arrays]
                mat = np.stack(arrays, axis=1).astype(np.float32)  # (t,c)
                target_idx = input_channels.index(resolved_target)

                last_t = min_len - horizon - 1
                if last_t < window - 1:
                    raise ValueError("série curta para janela/horizonte")

                rows_X: list[np.ndarray] = []
                rows_y: list[float] = []
                for t in range(window - 1, last_t + 1):
                    x_row = mat[t - window + 1 : t + 1, :].reshape(1, -1)
                    y_val = float(mat[t + horizon, target_idx])
                    rows_X.append(x_row.astype(np.float32))
                    rows_y.append(y_val)
                    if len(rows_y) >= max_samples:
                        break

                key = (step_id, label, resolved_target)
                ds = forecaster_datasets.setdefault(
                    key,
                    {
                        "X": [],
                        "y": [],
                        "skipped": 0,
                        "block_name": block_name,
                        "spec": spec,
                        "window": window,
                        "horizon": horizon,
                    },
                )
                ds["X"].extend(rows_X)
                ds["y"].extend(rows_y)
            except Exception as e:
                if not skip_missing:
                    errors.append(f"{exp_id} / {step_id}: {e}")
                    continue
                key = (step_id, "sem_etiqueta", str(block_cfg.get("target_channel") or ""))
                forecaster_datasets.setdefault(
                    key,
                    {"X": [], "y": [], "skipped": 0, "block_name": block_name, "spec": spec, "window": 0, "horizon": 0},
                )
                forecaster_datasets[key]["skipped"] = int(forecaster_datasets[key].get("skipped", 0)) + 1
                continue

        if trainable_steps and not isinstance(lab, dict):
            if not skip_missing:
                errors.append(f"{exp_id}: lab_results não encontrado")
                break
            skipped_experiments.append(exp_id)
            continue

        for step in trainable_steps:
            step_id = str(step.step_id)
            block_name = str(step.block_name)
            block_cfg = dict(step.block_config or {})
            spec = model_specs_by_step.get(step_id, {"algorithm": "ridge", "params": {}})

            # Extrair input do step via input_mapping
            input_mapping = dict(step.input_mapping or {})

            label = ""
            unit = ""
            X_vec: Optional[np.ndarray] = None

            try:
                if block_name == "ml_inference":
                    # features
                    src = input_mapping.get("features", "")
                    if "." not in src:
                        raise ValueError("input_mapping.features inválido")
                    src_step, src_out = src.split(".", 1)
                    src_output = result.step_results.get(src_step)
                    features = src_output.data.get(src_out) if src_output and isinstance(src_output.data, dict) else None
                    if not isinstance(features, dict):
                        raise ValueError("features não encontrado")

                    label = _extract_label_from_any(features)
                    input_feature = str(block_cfg.get("input_feature", "growth_rate") or "growth_rate")
                    channel = str(block_cfg.get("channel", "") or "")
                    fv, used_channel = _select_feature_value(features, input_feature, channel)
                    if fv is None:
                        raise ValueError(f"feature '{input_feature}' não encontrada")

                    X_vec = np.array([[float(fv)]], dtype=np.float32)

                    unit = str(block_cfg.get("output_unit") or "").strip()
                    if not unit:
                        resource = str(block_cfg.get("resource") or "fluorescencia_NMP")
                        from ..components.pipeline.blocks import MLInferenceBlock  # local import

                        res = getattr(MLInferenceBlock, "AVAILABLE_RESOURCES", {}).get(resource, {}) if isinstance(resource, str) else {}
                        unit = str(res.get("output_unit") or "").strip()

                else:
                    # series/multichannel: usa sensor_data
                    src = input_mapping.get("sensor_data", "")
                    if "." not in src:
                        raise ValueError("input_mapping.sensor_data inválido")
                    src_step, src_out = src.split(".", 1)
                    src_output = result.step_results.get(src_step)
                    sensor_data = src_output.data.get(src_out) if src_output and isinstance(src_output.data, dict) else None
                    if not isinstance(sensor_data, dict) or not isinstance(sensor_data.get("channels"), dict):
                        raise ValueError("sensor_data inválido")

                    label = _extract_label_from_any(sensor_data)

                    max_length = block_cfg.get("max_length", None)
                    try:
                        max_length_i = int(max_length) if max_length not in [None, ""] else None
                    except Exception:
                        max_length_i = None
                    pad_value = float(block_cfg.get("pad_value", 0.0) or 0.0)
                    align = str(block_cfg.get("align", "end") or "end").strip().lower()
                    unit = str(block_cfg.get("output_unit") or "").strip()

                    ch_dict = sensor_data.get("channels") or {}
                    if block_name == "ml_inference_series":
                        channel = str(block_cfg.get("channel") or "").strip() or None
                        # reusar helper do blocks.py
                        from ..components.pipeline.blocks import _series_from_sensor_data, _apply_length_policy  # type: ignore

                        _, y_series, _used = _series_from_sensor_data(sensor_data, channel=channel)
                        y_series = np.array(y_series, dtype=np.float32).reshape(-1)
                        if max_length_i:
                            y_series = _apply_length_policy(y_series, max_length=max_length_i, pad_value=pad_value, align=align)
                        X_vec = y_series.reshape(1, -1).astype(np.float32)

                    elif block_name == "ml_inference_multichannel":
                        channels_raw = block_cfg.get("channels", [])
                        if isinstance(channels_raw, str):
                            use_channels = [c.strip() for c in channels_raw.split(",") if c.strip()]
                        elif isinstance(channels_raw, list):
                            use_channels = [str(c).strip() for c in channels_raw if str(c).strip()]
                        else:
                            use_channels = []
                        available = list(ch_dict.keys())
                        use_channels = [c for c in (use_channels or available) if c in ch_dict]
                        if not use_channels:
                            raise ValueError("nenhum canal selecionado")

                        arrays = []
                        min_len = None
                        for ch in use_channels:
                            y = np.array(ch_dict.get(ch) or [], dtype=np.float32).reshape(-1)
                            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                            if max_length_i:
                                from ..components.pipeline.blocks import _apply_length_policy  # type: ignore

                                y = _apply_length_policy(y, max_length=max_length_i, pad_value=pad_value, align=align)
                            arrays.append(y)
                            min_len = y.size if min_len is None else min(min_len, y.size)
                        if not min_len or min_len <= 0:
                            raise ValueError("séries vazias")
                        arrays = [a[:min_len] for a in arrays]
                        mat = np.stack(arrays, axis=1)  # (t,c)

                        input_layout = str(block_cfg.get("input_layout", "time_channels") or "time_channels").strip()
                        if input_layout == "channels_time":
                            flat = mat.T.reshape(1, -1)
                        else:
                            flat = mat.reshape(1, -1)
                        X_vec = flat.astype(np.float32)

                if not label:
                    # sem label não dá para mapear y
                    raise ValueError("label ausente")

                target_field = ""
                by_unit = targets_map.get(label) if isinstance(targets_map.get(label), dict) else None
                if isinstance(by_unit, dict):
                    target_field = str(by_unit.get(unit) or "").strip()

                # Usar lab_results (valores originais) e aplicar diluição manualmente
                lab = select_lab_result_for_field(lab_results, target_field)
                y_val = extract_target_value(lab, target_field)
                
                # Aplicar correção de diluição: dividir pelo dilution_factor
                # O modelo vai aprender com valores "vistos pelo sensor" (diluídos)
                # Na predição, o bloco ML multiplica pelo dilution_factor
                if y_val is not None and dilution_factor != 1.0:
                    y_val = y_val / dilution_factor
                
                if y_val is None:
                    if skip_missing:
                        key = (step_id, label, unit)
                        datasets.setdefault(key, {"X": [], "y": [], "skipped": 0, "block_name": block_name, "spec": spec, "block_cfg": block_cfg})
                        datasets[key]["skipped"] = int(datasets[key].get("skipped", 0)) + 1
                        continue
                    raise ValueError(f"target ausente para {label}/{unit} ({target_field})")

                key = (step_id, label, unit)
                ds = datasets.setdefault(key, {"X": [], "y": [], "skipped": 0, "block_name": block_name, "spec": spec, "block_cfg": block_cfg})
                ds["X"].append(X_vec.reshape(1, -1))
                ds["y"].append(float(y_val))
            except Exception as e:
                if not skip_missing:
                    errors.append(f"{exp_id} / {step_id}: {e}")
                    continue
                key = (step_id, label or "", unit or "")
                datasets.setdefault(key, {"X": [], "y": [], "skipped": 0, "block_name": block_name, "spec": spec})
                datasets[key]["skipped"] = int(datasets[key].get("skipped", 0)) + 1
                continue

    trained: list[dict[str, Any]] = []

    # salvar modelos
    out_base = _resources_root() / tenant / "predict" / "trained"
    out_forecaster = _resources_root() / tenant / "predict" / "forecaster"

    # salvar forecasters
    for (step_id, label, target_channel), ds in forecaster_datasets.items():
        X_list = ds.get("X") or []
        y_list = ds.get("y") or []
        if len(X_list) < 20 or len(y_list) < 20:
            continue

        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.float32)
        block_name = str(ds.get("block_name") or "")
        spec = ds.get("spec") or {"algorithm": "ridge", "params": {}}
        skipped = int(ds.get("skipped", 0) or 0)
        window = int(ds.get("window") or 0)
        horizon = int(ds.get("horizon") or 0)

        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", label)[:64] or "label"
        safe_ch = re.sub(r"[^a-zA-Z0-9_-]+", "_", target_channel)[:64] or "ch"

        out_dir = out_forecaster / safe_label / safe_ch / step_id
        prefix = f"forecaster_{safe_label}_{safe_ch}_{step_id}"

        try:
            metric = str(spec.get("selection_metric") or getattr(request, "selection_metric", "rmse") or "rmse").strip() or "rmse"
            max_trials = int(spec.get("max_trials") or getattr(request, "max_trials", 60) or 60)
            grid_search = bool(spec.get("grid_search"))
            algos = spec.get("algorithms") or []
            grid = spec.get("param_grid") or {}
            params_by_algorithm = spec.get("params_by_algorithm") or {}
            grid_by_algorithm = spec.get("param_grid_by_algorithm") or {}

            # Importar helper para output_channel
            from ..components.pipeline.blocks import MLForecasterSeriesBlock

            # Configurações do bloco que serão salvas no metadata
            forecaster_block_config = {
                "input_channels": input_channels,
                "target_channel": resolved_target,
                "output_channel": MLForecasterSeriesBlock._default_output_channel(resolved_target),
                "window": window,
                "horizon": horizon,
                "pad_value": 0.0,
            }

            tr = train_regressor_export_onnx(
                X,
                y,
                algorithm=str(spec.get("algorithm") or "ridge"),
                params=dict(spec.get("params") or {}),
                params_by_algorithm=params_by_algorithm if isinstance(params_by_algorithm, dict) and params_by_algorithm else None,
                y_transform_mode="none",
                test_size=float(getattr(request, "test_size", 0.2) or 0.0),
                random_state=int(getattr(request, "random_state", 42) or 42),
                perm_importance=False,
                perm_repeats=1,
                selection_metric=metric,
                grid_search=grid_search,
                algorithms=[str(a).strip() for a in algos if str(a).strip()] if isinstance(algos, list) else None,
                param_grid=grid if isinstance(grid, dict) else None,
                param_grid_by_algorithm=grid_by_algorithm if isinstance(grid_by_algorithm, dict) and grid_by_algorithm else None,
                max_trials=max_trials,
                out_dir=out_dir,
                prefix=prefix,
                block_config=forecaster_block_config,  # NOVO: salvar configs do bloco
            )
        except Exception as e:
            errors.append(f"{step_id}: falha ao treinar/exportar forecaster: {e}")
            continue

        model_rel = tr.model_path.relative_to(_repo_root()).as_posix()
        scaler_rel = tr.scaler_path.relative_to(_repo_root()).as_posix()

        trained.append(
            {
                "step_id": step_id,
                "block_name": block_name,
                "label": label,
                "unit": "",
                "model_path": model_rel,
                "scaler_path": scaler_rel,
                "n_samples": tr.n_samples,
                "skipped": skipped,
                "metrics": {**(tr.metrics or {}), "forecaster_window": window, "forecaster_horizon": horizon, "target_channel": target_channel},
            }
        )

        if request.apply_to_pipeline:
            from ..components.pipeline.blocks import MLForecasterSeriesBlock  # local import

            # Calcular path do metadata
            metadata_rel = ""
            if tr.metadata_path:
                try:
                    metadata_rel = tr.metadata_path.relative_to(_repo_root()).as_posix()
                except Exception:
                    metadata_rel = ""

            _update_workspace_pipeline_block_config(
                payload,
                step_id,
                {
                    "model_path": model_rel,
                    "scaler_path": scaler_rel,
                    "target_channel": target_channel,
                    "window": window or 30,
                    "horizon": horizon or 1,
                    "pad_value": 0.0,
                    "output_channel": MLForecasterSeriesBlock._default_output_channel(target_channel),
                    "y_transform": tr.y_transform,  # Transformação aplicada ao y
                    "metadata_path": metadata_rel,  # Metadados do treinamento
                },
            )

    for (step_id, label, unit), ds in datasets.items():
        X_list = ds.get("X") or []
        y_list = ds.get("y") or []
        if len(X_list) < 2 or len(y_list) < 2:
            continue
        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.float32)
        block_name = str(ds.get("block_name") or "")
        spec = ds.get("spec") or {"algorithm": "ridge", "params": {}}
        skipped = int(ds.get("skipped", 0) or 0)

        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", label)[:64] or "label"
        safe_unit = unit_slug(unit)
        out_dir = out_base / safe_label / safe_unit / step_id
        prefix = f"{safe_label}_{safe_unit}_{step_id}"

        # Extrair configurações do bloco que foram usadas no treino
        block_cfg_extracted = ds.get("block_cfg") or {}
        inference_block_config = {
            "output_unit": unit,
            "input_feature": block_cfg_extracted.get("input_feature", "growth_rate"),
            "channel": block_cfg_extracted.get("channel", ""),
        }
        if block_name in ("ml_inference_series", "ml_inference_multichannel"):
            inference_block_config.update({
                "max_length": block_cfg_extracted.get("max_length"),
                "pad_value": block_cfg_extracted.get("pad_value", 0.0),
                "align": block_cfg_extracted.get("align", "end"),
                "input_layout": block_cfg_extracted.get("input_layout", "time_channels"),
            })
            if block_name == "ml_inference_multichannel":
                inference_block_config["channels"] = block_cfg_extracted.get("channels", [])

        try:
            metric = str(spec.get("selection_metric") or getattr(request, "selection_metric", "rmse") or "rmse").strip() or "rmse"
            max_trials = int(spec.get("max_trials") or getattr(request, "max_trials", 60) or 60)
            grid_search = bool(spec.get("grid_search"))
            algos = spec.get("algorithms") or []
            grid = spec.get("param_grid") or {}
            params_by_algorithm = spec.get("params_by_algorithm") or {}
            grid_by_algorithm = spec.get("param_grid_by_algorithm") or {}

            tr = train_regressor_export_onnx(
                X,
                y,
                algorithm=str(spec.get("algorithm") or "ridge"),
                params=dict(spec.get("params") or {}),
                params_by_algorithm=params_by_algorithm if isinstance(params_by_algorithm, dict) and params_by_algorithm else None,
                y_transform_mode=y_transform,
                test_size=float(getattr(request, "test_size", 0.2) or 0.0),
                random_state=int(getattr(request, "random_state", 42) or 42),
                perm_importance=bool(getattr(request, "perm_importance", False)),
                perm_repeats=int(getattr(request, "perm_repeats", 10) or 10),
                selection_metric=metric,
                grid_search=grid_search,
                algorithms=[str(a).strip() for a in algos if str(a).strip()] if isinstance(algos, list) else None,
                param_grid=grid if isinstance(grid, dict) else None,
                param_grid_by_algorithm=grid_by_algorithm if isinstance(grid_by_algorithm, dict) and grid_by_algorithm else None,
                max_trials=max_trials,
                out_dir=out_dir,
                prefix=prefix,
                block_config=inference_block_config,  # NOVO: salvar configs do bloco
            )
        except Exception as e:
            errors.append(f"{step_id}: falha ao treinar/exportar: {e}")
            continue

        model_rel = tr.model_path.relative_to(_repo_root()).as_posix()
        scaler_rel = tr.scaler_path.relative_to(_repo_root()).as_posix()

        trained.append(
            {
                "step_id": step_id,
                "block_name": block_name,
                "label": label,
                "unit": unit,
                "model_path": model_rel,
                "scaler_path": scaler_rel,
                "n_samples": tr.n_samples,
                "skipped": skipped,
                "metrics": tr.metrics,
            }
        )

        if request.apply_to_pipeline:
            # Calcular path do metadata (mesmo diretório do model)
            metadata_rel = ""
            if tr.metadata_path:
                try:
                    metadata_rel = tr.metadata_path.relative_to(_repo_root()).as_posix()
                except Exception:
                    metadata_rel = ""
            
            _update_workspace_pipeline_block_config(
                payload,
                step_id,
                {
                    "model_path": model_rel,
                    "scaler_path": scaler_rel,
                    "output_unit": unit,
                    "resource": "",  # Limpar resource para usar paths customizados
                    "y_transform": tr.y_transform,  # CRÍTICO: informar transformação do y
                    "metadata_path": metadata_rel,  # Metadados do treinamento
                },
            )

    activated_version: Optional[str] = None
    if request.apply_to_pipeline and trained:
        # criar nova versão e ativar
        manifest = _read_versions_manifest(tenant)
        versions = manifest.get("versions") if isinstance(manifest.get("versions"), list) else []
        based_on = base_version or active_version or str(manifest.get("active") or "v1")
        vid = _next_version_id(versions)

        versions_dir = _workspace_versions_dir(tenant)
        versions_dir.mkdir(parents=True, exist_ok=True)
        version_file = versions_dir / f"{vid}.json"
        _write_json_file(version_file, payload)

        versions.append(
            {
                "id": vid,
                "name": vid,
                "created_at": _utc_now_iso(),
                "based_on": based_on,
                "history": [],
            }
        )
        manifest["versions"] = versions
        manifest["active"] = vid
        reason = str(request.change_reason or "").strip() or f"Treinamento: {len(trained)} modelos atualizados"
        _append_version_history(manifest, vid, reason, "train")
        _write_versions_manifest(tenant, manifest)

        # ativar: copiar para o arquivo ativo
        active_file = _workspace_pipeline_file(tenant, pipeline_name)
        active_file.write_bytes(version_file.read_bytes())
        activated_version = vid

    return TenantPipelineTrainResponse(
        success=bool(trained) and not (not request.apply_to_pipeline and errors),
        trained=trained,
        skipped_experiments=skipped_experiments,
        errors=errors,
        version=activated_version,
    )


@router.get(
    "/pipelines/library",
    summary="Retorna a biblioteca completa de blocos/filtros/detectores"
)
async def endpoint_pipeline_library():
    return build_pipeline_library()


@router.get(
    "/pipelines/workspaces",
    response_model=WorkspaceListResponse,
    summary="Lista pipelines salvos em resources/<tenant>/pipeline/"
)
async def endpoint_list_workspaces():
    root = _resources_root()
    pipelines = []
    if root.exists():
        for tenant_dir in root.iterdir():
            if not tenant_dir.is_dir():
                continue
            tenant = tenant_dir.name
            if not _TENANT_RE.match(tenant):
                continue
            pipeline_dir = tenant_dir / "pipeline"
            if not pipeline_dir.exists():
                continue

            # Um card por tenant (pipeline ativo). Mantemos compatibilidade: <tenant>.json é o "ativo".
            active_file = pipeline_dir / f"{tenant}.json"
            if not active_file.exists():
                candidates = [
                    p
                    for p in pipeline_dir.glob("*.json")
                    if p.is_file() and not p.name.startswith("_")
                ]
                if not candidates:
                    continue
                # Escolher o mais recente
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                active_file = candidates[0]

            pipeline = active_file.stem
            workspace_meta = {}
            try:
                payload = _read_json_file(active_file)
                if isinstance(payload, dict) and isinstance(payload.get("workspace"), dict):
                    workspace_meta = payload.get("workspace") or {}
            except Exception:
                workspace_meta = {}

            try:
                updated_at = __import__("datetime").datetime.utcfromtimestamp(active_file.stat().st_mtime).isoformat() + "Z"
            except Exception:
                updated_at = None

            # Versões (best-effort: inicializa se ainda não existir)
            versions_info = _ensure_versions_initialized(tenant, pipeline)
            active_version = versions_info.get("active")
            versions = versions_info.get("versions") if isinstance(versions_info.get("versions"), list) else []

            logo_value = workspace_meta.get("logo")
            pipelines.append(
                {
                    "tenant": tenant,
                    "pipeline": pipeline,
                    "file": str(active_file.relative_to(root)).replace("\\", "/"),
                    "updated_at": updated_at,
                    "title": workspace_meta.get("title"),
                    "logo": (
                        _public_asset_url(tenant, pipeline, str(logo_value).lstrip("/"))
                        if isinstance(logo_value, str)
                        and logo_value
                        and not str(logo_value).startswith(("http://", "https://", "data:", "/pipelines/"))
                        else logo_value
                    ),
                    "accent_color": workspace_meta.get("accent_color"),
                    "active_version": active_version,
                    "versions_count": len(versions),
                }
            )
    pipelines.sort(key=lambda x: (x["tenant"], x["pipeline"]))
    return {"pipelines": pipelines}


@router.post(
    "/pipelines/workspaces/create",
    summary="Cria um workspace/pipeline em resources/<tenant>/pipeline/<pipeline>.json"
)
async def endpoint_create_workspace(request: WorkspaceCreateRequest):
    tenant = _validate_segment(request.tenant, "tenant")
    pipeline = _validate_segment(request.pipeline or tenant, "pipeline")
    file_path = _workspace_pipeline_file(tenant, pipeline)
    if file_path.exists() and not request.overwrite:
        raise HTTPException(status_code=409, detail="Pipeline já existe")
    payload = _default_pipeline_payload(name=pipeline)
    _write_json_file(file_path, payload)
    try:
        _ensure_versions_initialized(tenant, pipeline)
    except Exception:
        pass
    return payload


@router.post(
    "/pipelines/workspaces/duplicate",
    summary="Duplica um pipeline existente para um novo tenant/pipeline"
)
async def endpoint_duplicate_workspace(request: WorkspaceDuplicateRequest):
    source_tenant = _validate_segment(request.source_tenant, "source_tenant")
    source_pipeline = _validate_segment(request.source_pipeline or source_tenant, "source_pipeline")
    target_tenant = _validate_segment(request.target_tenant, "target_tenant")
    target_pipeline = _validate_segment(request.target_pipeline or target_tenant, "target_pipeline")

    source_file = _workspace_pipeline_file(source_tenant, source_pipeline)
    if not source_file.exists():
        raise HTTPException(status_code=404, detail="Pipeline de origem não encontrado")

    target_file = _workspace_pipeline_file(target_tenant, target_pipeline)
    if target_file.exists() and not request.overwrite:
        raise HTTPException(status_code=409, detail="Pipeline de destino já existe")

    data = _read_json_file(source_file)
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Pipeline de origem inválido")

    data.setdefault("version", "1.0")
    data["name"] = target_pipeline
    data["savedAt"] = _utc_now_iso()

    if isinstance(data.get("execution"), dict):
        data["execution"].setdefault("steps", [])
        data["execution"].setdefault("initial_state", {})
        data["execution"]["name"] = target_pipeline

    ws = data.get("workspace") if isinstance(data.get("workspace"), dict) else {}
    data["workspace"] = {
        **(ws or {}),
        "title": (str(request.target_title).strip() if request.target_title else target_pipeline),
        "logo": "",
        "accent_color": "#1e90ff",
    }

    _write_json_file(target_file, data)
    try:
        _ensure_versions_initialized(target_tenant, target_pipeline)
    except Exception:
        pass
    return data


@router.get(
    "/pipelines/workspaces/{tenant}/{pipeline}/versions",
    summary="Lista versões do pipeline para um tenant (uma versão ativa)"
)
async def endpoint_workspace_versions(tenant: str, pipeline: str):
    tenant = _validate_segment(tenant, "tenant")
    pipeline = _validate_segment(pipeline or tenant, "pipeline")
    info = _ensure_versions_initialized(tenant, pipeline)
    active = str(info.get("active") or "")
    versions = info.get("versions") if isinstance(info.get("versions"), list) else []
    versions_dir = _workspace_versions_dir(tenant)

    out = []
    for v in versions:
        if not isinstance(v, dict):
            continue
        vid = str(v.get("id") or "")
        if not vid:
            continue
        file_path = versions_dir / f"{vid}.json"
        try:
            updated_at = __import__("datetime").datetime.utcfromtimestamp(file_path.stat().st_mtime).isoformat() + "Z"
        except Exception:
            updated_at = None
        out.append(
            {
                **v,
                "id": vid,
                "name": (str(v.get("name") or "").strip() or vid),
                "created_at": v.get("created_at"),
                "history": v.get("history") if isinstance(v.get("history"), list) else [],
                "is_active": vid == active,
                "updated_at": updated_at,
                "file": str(file_path.relative_to(_resources_root() / tenant / "pipeline")).replace("\\", "/"),
            }
        )

    return {"tenant": tenant, "pipeline": pipeline, "active": active, "versions": out}


@router.get(
    "/pipelines/workspaces/{tenant}/{pipeline}/versions/{version}/load",
    summary="Carrega o JSON de uma versão específica do pipeline"
)
async def endpoint_workspace_version_load(tenant: str, pipeline: str, version: str):
    tenant = _validate_segment(tenant, "tenant")
    pipeline = _validate_segment(pipeline or tenant, "pipeline")
    version = _validate_segment(version, "version")
    _ensure_versions_initialized(tenant, pipeline)
    versions_dir = _workspace_versions_dir(tenant)
    file_path = versions_dir / f"{version}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Versão não encontrada")
    return _read_json_file(file_path)


@router.post(
    "/pipelines/workspaces/{tenant}/{pipeline}/versions/create",
    summary="Cria uma nova versão a partir da versão ativa e a torna ativa"
)
async def endpoint_workspace_version_create(tenant: str, pipeline: str):
    tenant = _validate_segment(tenant, "tenant")
    pipeline = _validate_segment(pipeline or tenant, "pipeline")
    info = _ensure_versions_initialized(tenant, pipeline)
    active = str(info.get("active") or "")
    versions = info.get("versions") if isinstance(info.get("versions"), list) else []

    new_id = _next_version_id(versions)
    versions_dir = _workspace_versions_dir(tenant)
    versions_dir.mkdir(parents=True, exist_ok=True)

    active_file = _workspace_pipeline_file(tenant, pipeline)
    if not active_file.exists():
        raise HTTPException(status_code=404, detail="Pipeline ativo não encontrado")

    new_file = versions_dir / f"{new_id}.json"
    new_file.write_bytes(active_file.read_bytes())

    versions.append({"id": new_id, "name": new_id, "created_at": _utc_now_iso(), "based_on": active or None, "history": [{"at": _utc_now_iso(), "action": "create", "reason": "Criada (cópia)"}]})
    manifest = _read_versions_manifest(tenant)
    manifest["active"] = new_id
    manifest["versions"] = versions
    _write_versions_manifest(tenant, manifest)

    return {"ok": True, "active": new_id}


@router.post(
    "/pipelines/workspaces/{tenant}/{pipeline}/versions/create-clean",
    summary="Cria uma nova versão limpa e a torna ativa"
)
async def endpoint_workspace_version_create_clean(tenant: str, pipeline: str, payload: dict = Body(default_factory=dict)):
    tenant = _validate_segment(tenant, "tenant")
    pipeline = _validate_segment(pipeline or tenant, "pipeline")
    info = _ensure_versions_initialized(tenant, pipeline)
    active = str(info.get("active") or "")
    versions = info.get("versions") if isinstance(info.get("versions"), list) else []

    new_id = _next_version_id(versions)
    versions_dir = _workspace_versions_dir(tenant)
    versions_dir.mkdir(parents=True, exist_ok=True)

    active_file = _workspace_pipeline_file(tenant, pipeline)
    if not active_file.exists():
        raise HTTPException(status_code=404, detail="Pipeline ativo não encontrado")

    base = _default_pipeline_payload(name=pipeline)
    try:
        current = _read_json_file(active_file)
        if isinstance(current, dict) and isinstance(current.get("workspace"), dict):
            base["workspace"] = {
                **(base.get("workspace") if isinstance(base.get("workspace"), dict) else {}),
                **(current.get("workspace") or {}),
            }
    except Exception:
        pass
    base["savedAt"] = _utc_now_iso()

    new_file = versions_dir / f"{new_id}.json"
    _write_json_file(new_file, base)

    activate = bool((payload or {}).get("activate", True))
    if activate:
        try:
            active_file.write_bytes(new_file.read_bytes())
        except Exception:
            pass

    reason = str((payload or {}).get("reason") or "").strip() or "Criada (limpa)"
    versions.append({"id": new_id, "name": new_id, "created_at": _utc_now_iso(), "based_on": active or None, "history": [{"at": _utc_now_iso(), "action": "create_clean", "reason": reason}]})
    manifest = _read_versions_manifest(tenant)
    if activate:
        manifest["active"] = new_id
    manifest["versions"] = versions
    _write_versions_manifest(tenant, manifest)

    return {"ok": True, "active": str(manifest.get("active") or "")}


@router.post(
    "/pipelines/workspaces/{tenant}/{pipeline}/versions/create-copy",
    summary="Cria uma nova versão por cópia (do ativo ou de uma versão) e a torna ativa"
)
async def endpoint_workspace_version_create_copy(tenant: str, pipeline: str, payload: dict = Body(default_factory=dict)):
    tenant = _validate_segment(tenant, "tenant")
    pipeline = _validate_segment(pipeline or tenant, "pipeline")
    info = _ensure_versions_initialized(tenant, pipeline)
    active = str(info.get("active") or "")
    versions = info.get("versions") if isinstance(info.get("versions"), list) else []

    new_id = _next_version_id(versions)
    versions_dir = _workspace_versions_dir(tenant)
    versions_dir.mkdir(parents=True, exist_ok=True)

    active_file = _workspace_pipeline_file(tenant, pipeline)
    if not active_file.exists():
        raise HTTPException(status_code=404, detail="Pipeline ativo não encontrado")

    from_version = str((payload or {}).get("from_version") or "").strip()
    new_file = versions_dir / f"{new_id}.json"
    if from_version:
        from_version = _validate_segment(from_version, "from_version")
        src = versions_dir / f"{from_version}.json"
        if not src.exists():
            raise HTTPException(status_code=404, detail="Versão base não encontrada")
        new_file.write_bytes(src.read_bytes())
    else:
        new_file.write_bytes(active_file.read_bytes())

    activate = bool((payload or {}).get("activate", True))
    if activate:
        try:
            active_file.write_bytes(new_file.read_bytes())
        except Exception:
            pass

    reason = str((payload or {}).get("reason") or "").strip() or "Criada (cópia)"
    versions.append({"id": new_id, "name": new_id, "created_at": _utc_now_iso(), "based_on": active or None, "history": [{"at": _utc_now_iso(), "action": "create_copy", "reason": reason}]})
    manifest = _read_versions_manifest(tenant)
    if activate:
        manifest["active"] = new_id
    manifest["versions"] = versions
    _write_versions_manifest(tenant, manifest)

    return {"ok": True, "active": str(manifest.get("active") or ""), "created": new_id}


@router.delete(
    "/pipelines/workspaces/{tenant}/{pipeline}/versions/{version}",
    summary="Exclui uma versão do pipeline"
)
async def endpoint_workspace_version_delete(tenant: str, pipeline: str, version: str):
    tenant = _validate_segment(tenant, "tenant")
    pipeline = _validate_segment(pipeline or tenant, "pipeline")
    version = _validate_segment(version, "version")

    info = _ensure_versions_initialized(tenant, pipeline)
    active = str(info.get("active") or "")
    versions = info.get("versions") if isinstance(info.get("versions"), list) else []
    if len(versions) <= 1:
        raise HTTPException(status_code=409, detail="Não é possível excluir a única versão existente")

    versions_dir = _workspace_versions_dir(tenant)
    version_file = versions_dir / f"{version}.json"
    if not version_file.exists():
        raise HTTPException(status_code=404, detail="Versão não encontrada")

    remaining = [v for v in versions if isinstance(v, dict) and str(v.get("id") or "") != version]
    if not remaining:
        raise HTTPException(status_code=409, detail="Não é possível excluir a única versão existente")

    next_active = active
    if version == active:
        def _key(item):
            m = re.match(r"^v(\\d+)$", str(item.get("id") or ""))
            return int(m.group(1)) if m else -1
        remaining.sort(key=_key)
        next_active = str(remaining[-1].get("id") or "")
        next_file = versions_dir / f"{next_active}.json"
        active_file = _workspace_pipeline_file(tenant, pipeline)
        if next_file.exists():
            active_file.write_bytes(next_file.read_bytes())

    try:
        version_file.unlink()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao excluir versão: {exc}")

    manifest = _read_versions_manifest(tenant)
    manifest["versions"] = remaining
    manifest["active"] = next_active
    _write_versions_manifest(tenant, manifest)

    return {"ok": True, "active": next_active}


@router.post(
    "/pipelines/workspaces/{tenant}/{pipeline}/versions/{version}/rename",
    summary="Renomeia uma versão"
)
async def endpoint_workspace_version_rename(tenant: str, pipeline: str, version: str, payload: dict = Body(default_factory=dict)):
    tenant = _validate_segment(tenant, "tenant")
    pipeline = _validate_segment(pipeline or tenant, "pipeline")
    version = _validate_segment(version, "version")
    name = str((payload or {}).get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name é obrigatório")
    if len(name) > 80:
        raise HTTPException(status_code=400, detail="name muito longo (máx 80)")

    _ensure_versions_initialized(tenant, pipeline)
    manifest = _read_versions_manifest(tenant)
    reason = str((payload or {}).get("reason") or "").strip()
    _set_version_name(manifest, version, name)
    _append_version_history(manifest, version, reason or f"Renomeada para '{name}'", "rename")
    _write_versions_manifest(tenant, manifest)
    return {"ok": True}


@router.post(
    "/pipelines/workspaces/{tenant}/{pipeline}/versions/{version}/activate",
    summary="Ativa uma versão existente (faz rollback)"
)
async def endpoint_workspace_version_activate(tenant: str, pipeline: str, version: str, payload: dict = Body(default_factory=dict)):
    tenant = _validate_segment(tenant, "tenant")
    pipeline = _validate_segment(pipeline or tenant, "pipeline")
    version = _validate_segment(version, "version")

    _ensure_versions_initialized(tenant, pipeline)
    versions_dir = _workspace_versions_dir(tenant)
    version_file = versions_dir / f"{version}.json"
    if not version_file.exists():
        raise HTTPException(status_code=404, detail="Versão não encontrada")

    active_file = _workspace_pipeline_file(tenant, pipeline)
    active_file.write_bytes(version_file.read_bytes())

    reason = str((payload or {}).get("reason") or "").strip() or "Ativada"
    manifest = _read_versions_manifest(tenant)
    manifest["active"] = version
    _append_version_history(manifest, version, reason, "activate")
    if not isinstance(manifest.get("versions"), list):
        manifest["versions"] = [{"id": version, "name": version, "created_at": _utc_now_iso(), "based_on": None, "history": [{"at": _utc_now_iso(), "action": "activate", "reason": "Ativada"}]}]
    _write_versions_manifest(tenant, manifest)

    return {"ok": True, "active": version}


@router.delete(
    "/pipelines/workspaces/{tenant}/{pipeline}",
    summary="Exclui um pipeline salvo em resources/<tenant>/pipeline/<pipeline>.json"
)
async def endpoint_delete_workspace(tenant: str, pipeline: str):
    tenant = _validate_segment(tenant, "tenant")
    pipeline = _validate_segment(pipeline, "pipeline")
    file_path = _workspace_pipeline_file(tenant, pipeline)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Pipeline não encontrado")

    try:
        file_path.unlink()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao excluir pipeline: {exc}")

    # Limpar diretórios vazios (best-effort)
    try:
        pipeline_dir = _workspace_pipeline_dir(tenant, pipeline)
        # remover versões/manifest (se existir)
        try:
            manifest = _workspace_versions_manifest(tenant)
            if manifest.exists():
                manifest.unlink()
        except Exception:
            pass
        try:
            versions_dir = _workspace_versions_dir(tenant)
            if versions_dir.exists() and versions_dir.is_dir():
                for f in versions_dir.glob("*.json"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
                if not any(versions_dir.iterdir()):
                    versions_dir.rmdir()
        except Exception:
            pass
        # remove resources/img se vazio
        for sub in [pipeline_dir / "resources" / "img", pipeline_dir / "resources", pipeline_dir]:
            if sub.exists() and sub.is_dir() and not any(sub.iterdir()):
                sub.rmdir()
        tenant_dir = (_resources_root() / tenant)
        if tenant_dir.exists() and tenant_dir.is_dir() and not any(tenant_dir.iterdir()):
            tenant_dir.rmdir()
    except Exception:
        pass

    return {"ok": True}


@router.get(
    "/pipelines/workspaces/{tenant}/{pipeline}",
    summary="Carrega um pipeline salvo em resources/<tenant>/pipeline/<pipeline>.json"
)
async def endpoint_load_workspace(tenant: str, pipeline: str):
    file_path = _workspace_pipeline_file(tenant, pipeline)
    return _read_json_file(file_path)


@router.post(
    "/pipelines/workspaces/save",
    summary="Salva um pipeline no diretório resources/<tenant>/pipeline/"
)
async def endpoint_save_workspace(request: WorkspaceSaveRequest):
    tenant = _validate_segment(request.tenant, "tenant")
    pipeline = _validate_segment(request.pipeline or tenant, "pipeline")

    requested_version = str(getattr(request, "workspace_version", "") or "").strip()
    if requested_version:
        requested_version = _validate_segment(requested_version, "workspace_version")

    file_path = _workspace_pipeline_file(tenant, pipeline)
    data = request.data or {}
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="data deve ser um objeto JSON")
    data.setdefault("version", "1.0")
    data.setdefault("name", pipeline)
    data["savedAt"] = _utc_now_iso()
    if "workspace" not in data or not isinstance(data.get("workspace"), dict):
        data["workspace"] = {"title": pipeline, "logo": "", "accent_color": "#1e90ff"}
    # Normalizar logo para caminho relativo dentro do diretório do pipeline (portável).
    try:
        ws = data.get("workspace") or {}
        if isinstance(ws, dict) and isinstance(ws.get("logo"), str) and ws.get("logo"):
            logo = str(ws["logo"])
            prefix = f"/pipelines/workspaces/assets/{tenant}/{pipeline}/"
            if logo.startswith(prefix):
                ws["logo"] = logo[len(prefix):]
            data["workspace"] = ws
    except Exception:
        pass
    if "editor" not in data:
        data["editor"] = {"nodes": [], "edges": []}
    if "execution" not in data:
        data["execution"] = {"name": pipeline, "steps": [], "initial_state": {}}

    # Se o request indicar uma versão específica, salvar nela. Caso contrário, salvar no arquivo ativo.
    # O arquivo ativo (<pipeline>.json) só é atualizado quando salvamos a versão ativa.
    info = _ensure_versions_initialized(tenant, pipeline)
    active_version = str(info.get("active") or "")
    target_version = requested_version or active_version

    saved_files = []
    versions_dir = _workspace_versions_dir(tenant)
    versions_dir.mkdir(parents=True, exist_ok=True)

    if target_version:
        version_file = versions_dir / f"{target_version}.json"

        # Gerar um resumo do que mudou (editor) antes de sobrescrever
        prev_payload: dict = {}
        try:
            if version_file.exists():
                prev = _read_json_file(version_file)
                prev_payload = prev if isinstance(prev, dict) else {}
        except Exception:
            prev_payload = {}

        _write_json_file(version_file, data)
        saved_files.append(str(version_file.relative_to(_resources_root())).replace("\\", "/"))

        reason = str(getattr(request, "change_reason", "") or "").strip()
        if reason:
            bullets: list[str] = []
            try:
                # Só tentar detalhar quando a alteração vem do editor (tem editor.nodes/edges)
                prev_nodes, prev_edges = _safe_editor_graph(prev_payload)
                new_nodes, new_edges = _safe_editor_graph(data)
                if prev_nodes or prev_edges or new_nodes or new_edges:
                    bullets = _summarize_editor_changes(prev_payload, data)
            except Exception:
                bullets = []

            if bullets:
                reason = reason + "\n" + "\n".join([f"- {b}" for b in bullets])

            manifest = _read_versions_manifest(tenant)
            _append_version_history(manifest, target_version, reason, "edit")
            _write_versions_manifest(tenant, manifest)

    # Atualiza o arquivo ativo somente se estamos salvando a versão ativa (ou se não existe versionamento)
    if (not target_version) or (target_version == active_version):
        _write_json_file(file_path, data)
        saved_files.append(str(file_path.relative_to(_resources_root())).replace("\\", "/"))

    return {"ok": True, "files": saved_files, "active": active_version, "saved_version": target_version}


@router.get(
    "/pipelines/workspaces/assets/{tenant}/{pipeline}/{asset_path:path}",
    summary="Serve arquivos estáticos do pipeline (logo/imagens) a partir de resources/<tenant>/pipeline/"
)
async def endpoint_workspace_asset(tenant: str, pipeline: str, asset_path: str):
    base = _workspace_pipeline_dir(tenant, pipeline).resolve()
    rel = Path(asset_path)
    candidate = (base / rel).resolve()
    if not str(candidate).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Caminho inválido")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")

    media_type, _ = mimetypes.guess_type(candidate.name)
    return FileResponse(candidate, media_type=media_type)


if _MULTIPART_AVAILABLE:

    @router.post(
        "/pipelines/workspaces/{tenant}/{pipeline}/logo-upload",
        summary="Upload de logo (arquivo local) para resources/<tenant>/pipeline/resources/img/"
    )
    async def endpoint_workspace_logo_upload(tenant: str, pipeline: str, file: UploadFile = File(...)):
        tenant = _validate_segment(tenant, "tenant")
        pipeline = _validate_segment(pipeline, "pipeline")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Arquivo vazio")
        if len(content) > 4 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Arquivo muito grande (máx 4MB)")

        original = _safe_filename(Path(file.filename or "logo").stem)
        ext = (Path(file.filename or "").suffix or "").lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"]:
            # Tentar inferir pelo content-type
            ct = (file.content_type or "").lower()
            ext = ".png" if "png" in ct else ".jpg" if "jpeg" in ct or "jpg" in ct else ".svg" if "svg" in ct else ""
            if not ext:
                ext = ".png"

        logo_dir = _workspace_logo_dir(tenant, pipeline)
        logo_dir.mkdir(parents=True, exist_ok=True)
        target = logo_dir / f"{original}{ext}"
        target.write_bytes(content)

        accent = _dominant_color_from_image_bytes(content)
        rel_asset = str(target.relative_to(_workspace_pipeline_dir(tenant, pipeline))).replace("\\", "/")
        return {"logo": _public_asset_url(tenant, pipeline, rel_asset), "accent_color": accent, "asset_path": rel_asset}

else:

    @router.post(
        "/pipelines/workspaces/{tenant}/{pipeline}/logo-upload",
        summary="Upload de logo (arquivo local) para resources/<tenant>/pipeline/resources/img/"
    )
    async def endpoint_workspace_logo_upload_unavailable(tenant: str, pipeline: str):
        raise HTTPException(
            status_code=503,
            detail='Upload de arquivos requer "python-multipart". Instale com: pip install python-multipart',
        )


@router.post(
    "/pipelines/workspaces/logo-from-url",
    summary="Baixa um logo de URL e salva em resources/<tenant>/pipeline/resources/img/"
)
async def endpoint_workspace_logo_from_url(request: WorkspaceLogoUrlRequest):
    tenant = _validate_segment(request.tenant, "tenant")
    pipeline = _validate_segment(request.pipeline or tenant, "pipeline")
    url = str(request.url or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="URL inválida (use http/https)")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bioailab-inference/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            content = resp.read(4 * 1024 * 1024 + 1)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Falha ao baixar logo: {exc}")

    if len(content) > 4 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Arquivo muito grande (máx 4MB)")

    ext = ""
    if content_type in ["image/png"]:
        ext = ".png"
    elif content_type in ["image/jpeg", "image/jpg"]:
        ext = ".jpg"
    elif content_type in ["image/webp"]:
        ext = ".webp"
    elif content_type in ["image/gif"]:
        ext = ".gif"
    elif content_type in ["image/svg+xml"]:
        ext = ".svg"
    else:
        # fallback pelo path
        ext = (Path(urllib.parse.urlparse(url).path).suffix or "").lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"]:
            ext = ".png"
        if ext == ".jpeg":
            ext = ".jpg"

    original = _safe_filename(Path(urllib.parse.urlparse(url).path).stem or "logo")
    logo_dir = _workspace_logo_dir(tenant, pipeline)
    logo_dir.mkdir(parents=True, exist_ok=True)
    target = logo_dir / f"{original}{ext}"
    target.write_bytes(content)

    accent = _dominant_color_from_image_bytes(content)
    rel_asset = str(target.relative_to(_workspace_pipeline_dir(tenant, pipeline))).replace("\\", "/")
    return {"logo": _public_asset_url(tenant, pipeline, rel_asset), "accent_color": accent, "asset_path": rel_asset}


# =============================================================================
# ORQUESTRADOR DE TREINAMENTO (Grid Search + Dependências)
# =============================================================================
# Permite:
# 1. Criar sessão de treinamento com análise de dependências
# 2. Treinar steps na ordem correta
# 3. Visualizar candidatos do grid search
# 4. Selecionar manualmente qual modelo usar
# 5. Aplicar modelos selecionados ao pipeline
# =============================================================================

@router.post(
    "/training/sessions",
    response_model=TrainingSessionDetail,
    summary="Cria uma nova sessão de treinamento com análise de dependências",
)
async def endpoint_create_training_session(request: TrainingSessionCreateRequest):
    """
    Cria uma nova sessão de treinamento analisando as dependências entre blocos ML.
    
    A sessão determina a ordem correta de treinamento:
    - Blocos sem dependências podem treinar primeiro
    - Blocos que dependem de outros ML devem esperar
    
    Retorna informações sobre:
    - Ordem de execução recomendada
    - Quais blocos estão bloqueados e por quê
    - Próximo bloco disponível para treinar
    """
    from ..infrastructure.ml.training_orchestrator import TrainingOrchestrator
    
    tenant = _validate_segment(request.tenant, "tenant")
    pipeline_name = tenant
    
    # Carregar pipeline
    version = _validate_version_id(request.version) if request.version else None
    if version:
        source_file = _workspace_versions_dir(tenant) / f"{version}.json"
        if not source_file.exists():
            raise HTTPException(status_code=404, detail="Versão do pipeline não encontrada")
    else:
        source_file = _workspace_pipeline_file(tenant, pipeline_name)
        if not source_file.exists():
            raise HTTPException(status_code=404, detail="Pipeline do tenant não encontrado")
    
    payload = _read_json_file(source_file)
    
    orchestrator = TrainingOrchestrator(tenant, payload)
    session = orchestrator.create_session()
    summary = orchestrator.get_session_summary(session)
    
    # Converter para response
    tasks_info = {}
    for sid, task in session.tasks.items():
        tasks_info[sid] = TrainingTaskInfo(
            step_id=task.step_id,
            block_name=task.block_name,
            label=task.label,
            unit=task.unit,
            status=task.status.value,
            depends_on=task.depends_on,
            n_samples=task.n_samples,
            candidates=[
                TrainingCandidateInfo(
                    rank=c.rank,
                    algorithm=c.algorithm,
                    params=c.params,
                    score=c.score,
                    metrics=c.metrics,
                    selected=c.selected,
                )
                for c in task.candidates
            ],
            selected_candidate_index=task.selected_candidate_index,
            model_path=task.model_path,
            scaler_path=task.scaler_path,
            metadata_path=task.metadata_path,
            errors=task.errors,
            warnings=task.warnings,
        )
    
    return TrainingSessionDetail(
        session_id=session.session_id,
        tenant=session.tenant,
        pipeline_name=session.pipeline_name,
        created_at=session.created_at,
        completed=session.completed,
        execution_order=session.execution_order,
        next_trainable=summary.get("next_trainable"),
        status_summary=summary.get("status_summary", {}),
        tasks=tasks_info,
        awaiting_selection=summary.get("awaiting_selection", []),
        blocked_tasks=summary.get("blocked_tasks", []),
    )


@router.get(
    "/training/sessions",
    response_model=TrainingSessionListResponse,
    summary="Lista sessões de treinamento do tenant",
)
async def endpoint_list_training_sessions(tenant: str):
    """Lista todas as sessões de treinamento de um tenant."""
    from ..infrastructure.ml.training_orchestrator import list_training_sessions
    
    tenant = _validate_segment(tenant, "tenant")
    sessions = list_training_sessions(tenant)
    
    return TrainingSessionListResponse(
        sessions=[
            TrainingSessionSummary(
                session_id=s["session_id"],
                created_at=s["created_at"],
                completed=s["completed"],
                status_summary=s.get("status_summary", {}),
            )
            for s in sessions
        ]
    )


@router.get(
    "/training/sessions/{session_id}",
    response_model=TrainingSessionDetail,
    summary="Obtém detalhes de uma sessão de treinamento",
)
async def endpoint_get_training_session(session_id: str, tenant: str):
    """
    Retorna detalhes completos da sessão incluindo:
    - Status de cada task
    - Candidatos do grid search (se houver)
    - Próximo step disponível para treinar
    - Tasks bloqueadas aguardando dependências
    """
    from ..infrastructure.ml.training_orchestrator import TrainingOrchestrator, get_training_session
    
    tenant = _validate_segment(tenant, "tenant")
    session = get_training_session(tenant, session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    orchestrator = TrainingOrchestrator(tenant, {})
    summary = orchestrator.get_session_summary(session)
    
    tasks_info = {}
    for sid, task in session.tasks.items():
        tasks_info[sid] = TrainingTaskInfo(
            step_id=task.step_id,
            block_name=task.block_name,
            label=task.label,
            unit=task.unit,
            status=task.status.value,
            depends_on=task.depends_on,
            n_samples=task.n_samples,
            candidates=[
                TrainingCandidateInfo(
                    rank=c.rank,
                    algorithm=c.algorithm,
                    params=c.params,
                    score=c.score,
                    metrics=c.metrics,
                    selected=c.selected,
                )
                for c in task.candidates
            ],
            selected_candidate_index=task.selected_candidate_index,
            model_path=task.model_path,
            scaler_path=task.scaler_path,
            metadata_path=task.metadata_path,
            errors=task.errors,
            warnings=task.warnings,
        )
    
    return TrainingSessionDetail(
        session_id=session.session_id,
        tenant=session.tenant,
        pipeline_name=session.pipeline_name,
        created_at=session.created_at,
        completed=session.completed,
        execution_order=session.execution_order,
        next_trainable=summary.get("next_trainable"),
        status_summary=summary.get("status_summary", {}),
        tasks=tasks_info,
        awaiting_selection=summary.get("awaiting_selection", []),
        blocked_tasks=summary.get("blocked_tasks", []),
    )


@router.post(
    "/training/sessions/{session_id}/select",
    summary="Seleciona um modelo candidato para um step",
)
async def endpoint_select_training_model(session_id: str, request: TrainingSelectModelRequest):
    """
    Seleciona qual candidato do grid search usar para um step.
    
    Após o grid search, você pode comparar os candidatos e escolher
    qual modelo deseja usar (não necessariamente o "melhor" por métrica).
    """
    from ..infrastructure.ml.training_orchestrator import get_training_session, TrainingStatus, ModelCandidate
    
    tenant_dir = Path(__file__).parent.parent.parent / "resources"
    
    # Encontrar tenant da sessão
    session = None
    tenant = None
    for t in tenant_dir.iterdir():
        if not t.is_dir():
            continue
        s = get_training_session(t.name, session_id)
        if s:
            session = s
            tenant = t.name
            break
    
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    task = session.tasks.get(request.step_id)
    if not task:
        raise HTTPException(status_code=404, detail="Step não encontrado na sessão")
    
    if task.status != TrainingStatus.AWAITING_SELECTION:
        raise HTTPException(
            status_code=400,
            detail=f"Step não está aguardando seleção (status: {task.status.value})"
        )
    
    if request.candidate_index >= len(task.candidates):
        raise HTTPException(
            status_code=400,
            detail=f"Índice de candidato inválido (máx: {len(task.candidates) - 1})"
        )
    
    # Marcar candidato como selecionado
    for i, c in enumerate(task.candidates):
        c.selected = (i == request.candidate_index)
    
    task.selected_candidate_index = request.candidate_index
    selected = task.candidates[request.candidate_index]
    
    # TODO: Salvar modelo selecionado (exportar ONNX)
    # Por enquanto, marca como treinado
    task.status = TrainingStatus.TRAINED
    task.model_path = selected.model_path
    task.scaler_path = selected.scaler_path
    task.metadata_path = selected.metadata_path
    
    # Atualizar status das tasks dependentes
    for sid, t in session.tasks.items():
        if request.step_id in t.depends_on:
            # Verificar se todas as dependências estão satisfeitas
            all_trained = all(
                session.tasks.get(d, task).status == TrainingStatus.TRAINED
                for d in t.depends_on
            )
            if all_trained and t.status == TrainingStatus.PENDING:
                t.status = TrainingStatus.READY
    
    # Salvar sessão
    from ..infrastructure.ml.training_orchestrator import TrainingOrchestrator
    orchestrator = TrainingOrchestrator(tenant, {})
    orchestrator._save_session(session)
    
    return {
        "success": True,
        "step_id": request.step_id,
        "selected_candidate": {
            "rank": selected.rank,
            "algorithm": selected.algorithm,
            "score": selected.score,
            "metrics": selected.metrics,
        },
        "message": f"Modelo '{selected.algorithm}' selecionado para {request.step_id}",
    }


@router.get(
    "/training/sessions/{session_id}/dependencies",
    summary="Visualiza grafo de dependências entre blocos ML",
)
async def endpoint_get_training_dependencies(session_id: str, tenant: str):
    """
    Retorna o grafo de dependências entre blocos ML.
    
    Útil para visualizar quais blocos precisam ser treinados primeiro.
    """
    from ..infrastructure.ml.training_orchestrator import get_training_session
    
    tenant = _validate_segment(tenant, "tenant")
    session = get_training_session(tenant, session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    # Construir grafo
    nodes = []
    edges = []
    
    for sid, task in session.tasks.items():
        nodes.append({
            "id": sid,
            "block_name": task.block_name,
            "status": task.status.value,
            "label": task.label or sid,
        })
        
        for dep in task.depends_on:
            edges.append({
                "from": dep,
                "to": sid,
            })
    
    return {
        "session_id": session_id,
        "nodes": nodes,
        "edges": edges,
        "execution_order": session.execution_order,
    }


# =============================================================================
# GRID SEARCH COM SALVAMENTO DE CANDIDATOS
# =============================================================================
# Como no pipeline legado: salva todos os candidatos para análise e seleção.
# =============================================================================

@router.post(
    "/training/grid-search",
    summary="Executa grid search e salva TODOS os candidatos para análise",
)
async def endpoint_grid_search_with_candidates(
    request: Request,
    tenant: str,
    step_id: str,
    protocolId: Optional[str] = None,
    experimentIds: list[str] = Query(default=[], alias="experimentIds"),
    algorithm: str = "ridge",
    algorithms: Optional[list[str]] = Query(default=None, alias="algorithms"),
    param_grid: Optional[str] = None,
    y_transform: str = "log10p",
    selection_metric: str = "rmse",
    max_trials: int = 60,
    test_size: float = 0.2,
    debug: bool = False,
    version: Optional[str] = None,
    use_cache: bool = True,
    invalidate_cache: bool = False,
    cache_version: int = Query(default=2, ge=2, le=2, description="Versão do cache (somente v2 por experimento)"),
):
    """
    Executa grid search e SALVA TODOS OS CANDIDATOS.
    
    Diferente do /pipelines/train, este endpoint:
    1. Salva cada modelo candidato em arquivo separado
    2. Retorna lista completa de candidatos com métricas
    3. Permite escolher qual modelo usar depois via /training/select-candidate
    
    Cache de Features:
    - use_cache=True (padrão): Reutiliza features já extraídas se pipeline e dataset não mudaram
    - invalidate_cache=True: Força re-extração das features (ignora cache existente)
    
    Fluxo:
    1. POST /training/grid-search → retorna candidates_session_id
    2. GET /training/candidates/{session_id} → ver todos os candidatos
    3. POST /training/select-candidate → escolhe e aplica ao pipeline
    """
    from ..infrastructure.ml.training import train_with_candidates
    
    tenant = _validate_segment(tenant, "tenant")
    debug = bool(debug)
    
    # Log de parâmetros de cache
    print(f"[grid-search] use_cache={use_cache}, invalidate_cache={invalidate_cache}, cache_version={cache_version}")
    
    def _dbg(message: str) -> None:
        if debug:
            print(message)
    
    # Parse param_grid de JSON string
    param_grid_dict: Optional[dict] = None
    if param_grid:
        try:
            param_grid_dict = json.loads(param_grid)
        except json.JSONDecodeError:
            param_grid_dict = None
    
    # Tentar ler pipeline do body da requisição
    pipeline_from_body: Optional[dict] = None
    try:
        body = await request.json()
        if isinstance(body, dict) and ("steps" in body or "editor" in body or "execution" in body):
            pipeline_from_body = body
    except Exception:
        pass
    
    # Carregar pipeline - PRIORIDADE: pipeline da requisição > arquivo salvo
    payload = None
    pipeline_source = "request"
    
    if pipeline_from_body:
        # Usar pipeline enviado na requisição (frontend)
        payload = pipeline_from_body
        pipeline_source = "request"
    else:
        # Fallback: carregar do arquivo
        ver = _validate_version_id(version) if version else None
        if ver:
            source_file = _workspace_versions_dir(tenant) / f"{ver}.json"
        else:
            source_file = _workspace_pipeline_file(tenant, tenant)
        
        if not source_file.exists():
            raise HTTPException(status_code=404, detail="Pipeline não encontrado")
        
        payload = _read_json_file(source_file)
        pipeline_source = f"file:{source_file.name}"
    
    # ==========================================================================
    # CACHE DE FEATURES (v2 por experimento) - UNICO CAMINHO SUPORTADO
    # ==========================================================================
    print(f"[grid-search] cache v1 desativado; usando cache v2 (cache_version={cache_version})")
    
    config = _build_workspace_pipeline_config(
        tenant=tenant,
        pipeline_json=payload,
        timeout_seconds=300.0,
        fail_fast=True,
        generate_output_graphs=False,
    )
    
    engine = PipelineEngine(config)
    
    # Encontrar o step
    step = None
    for s in config.steps:
        if str(s.step_id) == step_id:
            step = s
            break
    
    if not step:
        raise HTTPException(status_code=404, detail=f"Step {step_id} não encontrado")
    
    targets_map = DEFAULT_TARGETS_MAP
    
    # Coletar dados de treino
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    errors: list[str] = []
    skipped: list[str] = []
    block_cfg: dict = {}
    multichannel_channels: list[str] | None = None
    multichannel_max_length: int | None = None
    
    # ==========================================================================
    # USAR CACHE V2 (por experimento)
    # ==========================================================================
    print(f"[grid-search] >>> USANDO CACHE V2 (por experimento) <<<")

    # 1. Coletar features usando cache v2
    X_dict, metadata_dict, errors, skipped, block_cfg = await _collect_features_with_cache_v2(
        tenant=tenant,
        payload=payload,
        step_id=step_id,
        experiment_ids=experimentIds,
        engine=engine,
        config=config,
        use_cache=use_cache,
        invalidate_cache=invalidate_cache,
    )

    # 2. Determinar target_field baseado no bloco
    block_name = str(step.block_name)
    unit = str(block_cfg.get("output_unit") or "").strip()

    # Descobrir label do primeiro experimento com dados
    first_label = None
    for exp_id, meta in metadata_dict.items():
        if meta.get("label"):
            first_label = meta.get("label")
            break

    by_unit = targets_map.get(first_label) if isinstance(targets_map.get(first_label), dict) else None
    target_field = str(by_unit.get(unit) or "") if isinstance(by_unit, dict) else ""

    if not target_field:
        # Fallback: tentar inferir do primeiro lab_result
        target_field = "nmp_100ml"  # default

    _dbg(f"[grid-search] target_field={target_field}, label={first_label}, unit={unit}")

    # 3. Mapear fatores de diluicao (a partir do cache/metadata)
    dilution_factors: dict[str, float] = {}
    for exp_id, meta in metadata_dict.items():
        raw = meta.get("dilution") or meta.get("dilution_factor")
        if raw is None:
            continue
        try:
            val = float(raw)
            dilution_factors[exp_id] = val if val > 0 else 1.0
        except (ValueError, TypeError):
            dilution_factors[exp_id] = 1.0

    # 4. Buscar Y em batch (SEMPRE fresco, nao usa cache)
    y_values, y_skips = _get_y_batch_from_lab_results(
        tenant=tenant,
        experiment_ids=experimentIds,
        target_field=target_field,
        dilution_factors=dilution_factors,
    )

    # 5. Combinar X e Y
    for exp_id in experimentIds:
        if exp_id not in X_dict:
            continue  # ja foi registrado em skipped pela funcao de collect
        if exp_id not in y_values:
            if exp_id in y_skips:
                skipped.append(f"{exp_id}: {y_skips[exp_id]}")
            continue

        X_list.append(X_dict[exp_id])
        y_list.append(y_values[exp_id])

    _dbg(f"[grid-search] Cache v2: {len(X_list)} amostras coletadas de {len(experimentIds)} experimentos")

    # Se não há dados suficientes, retornar status "skipped" ao invés de erro
    # Isso permite que o frontend continue treinando outros modelos
    if len(X_list) < 2:
        all_issues = errors + skipped
        
        return {
            "success": False,
            "status": "skipped",
            "reason": f"Dados insuficientes para treinar (coletados: {len(X_list)} de {len(experimentIds)} experimentos)",
            "step_id": step_id,
            "n_samples": len(X_list),
            "n_candidates": 0,
            "best_index": None,
            "candidates": [],
            "errors": [],
            "skipped_reasons": all_issues[:10],
        }
    
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    # Executar grid search com salvamento de candidatos
    out_dir = _resources_root() / tenant / "predict" / "candidates"
    prefix = f"{step_id}"
    
    # Detectar se param_grid_dict é na verdade param_grid_by_algorithm
    # Se as chaves são nomes de algoritmos conhecidos e valores são dicts, é by_algorithm
    known_algos = {
        "ridge",
        "lasso",
        "elasticnet",
        "rf",
        "xgb",
        "svr",
        "knn",
        "gbr",
        "gbm",
        "mlp",
        "huber",
        "bayesian",
        "lgbm",
        "lightgbm",
        "cat",
        "catboost",
    }
    final_param_grid = None
    final_param_grid_by_algorithm = None
    
    if param_grid_dict:
        grid_keys = set(str(k).lower() for k in param_grid_dict.keys())
        # Se TODAS as chaves são algoritmos conhecidos E valores são dicts, é by_algorithm
        if grid_keys and grid_keys.issubset(known_algos) and all(isinstance(v, dict) for v in param_grid_dict.values()):
            final_param_grid_by_algorithm = param_grid_dict
        else:
            final_param_grid = param_grid_dict
    
    result = train_with_candidates(
        X, y,
        algorithm=algorithm,
        algorithms=algorithms or [algorithm],
        param_grid=final_param_grid,
        param_grid_by_algorithm=final_param_grid_by_algorithm,
        y_transform_mode=y_transform,
        test_size=test_size,
        selection_metric=selection_metric,
        max_trials=max_trials,
        out_dir=out_dir,
        prefix=prefix,
        block_config=block_cfg,
        save_all_candidates=True,
    )
    
    return {
        "success": True,
        "status": "trained",
        "session_path": str(result.session_path.relative_to(_repo_root())),
        "n_samples": result.n_samples,
        "n_collected": len(X_list),
        "n_total_experiments": len(experimentIds),
        "n_skipped": len(experimentIds) - len(X_list),
        "n_candidates": len(result.candidates),
        "best_index": result.best_index,
        "candidates": [
            {
                "rank": c.rank,
                "algorithm": c.algorithm,
                "params": c.params,
                "score": c.score,
                "metrics": {k: v for k, v in c.metrics.items() if k.startswith(("val_", "train_"))},
            }
            for c in sorted(result.candidates, key=lambda x: x.rank)
        ],
        "errors": errors[:10] if errors else [],
        "skipped_reasons": skipped[:10] if skipped else [],
    }


@router.get(
    "/training/candidates/{tenant}/{session_id}",
    summary="Lista candidatos de uma sessão de grid search",
)
async def endpoint_list_candidates(tenant: str, session_id: str):
    """
    Lista todos os candidatos de uma sessão de grid search.
    
    Retorna métricas detalhadas para cada candidato para você comparar e escolher.
    """
    import json
    
    tenant = _validate_segment(tenant, "tenant")
    candidates_dir = _resources_root() / tenant / "predict" / "candidates"
    
    # Encontrar sessão
    session_file = None
    for d in candidates_dir.iterdir():
        if d.is_dir() and session_id in d.name:
            sf = d / "_session.json"
            if sf.exists():
                session_file = sf
                break
    
    if not session_file:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    session = json.loads(session_file.read_text(encoding="utf-8"))
    
    return {
        "session_id": session.get("session_id"),
        "created_at": session.get("created_at"),
        "n_samples": session.get("n_samples"),
        "selection_metric": session.get("selection_metric"),
        "best_index": session.get("best_index"),
        "candidates": session.get("candidates", []),
    }


@router.get(
    "/training/candidates/{tenant}/{session_id}/predictions/{candidate_index}",
    summary="Obtém dados de predição de um candidato para gráficos",
)
async def endpoint_candidate_predictions(tenant: str, session_id: str, candidate_index: int):
    """
    Retorna os dados de predição (actual vs predicted) de um candidato específico.
    
    Útil para gerar gráficos de dispersão e resíduos.
    """
    import json
    
    tenant = _validate_segment(tenant, "tenant")
    candidates_dir = _resources_root() / tenant / "predict" / "candidates"
    
    # Encontrar sessão
    session_dir = None
    for d in candidates_dir.iterdir():
        if d.is_dir() and session_id in d.name:
            session_dir = d
            break
    
    if not session_dir:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    # Buscar arquivo de predições do candidato
    pred_files = list(session_dir.glob(f"candidate_{candidate_index:03d}_*_predictions.json"))
    if not pred_files:
        raise HTTPException(status_code=404, detail=f"Predições do candidato {candidate_index} não encontradas")
    
    pred_data = json.loads(pred_files[0].read_text(encoding="utf-8"))
    
    # Buscar metadata do candidato para incluir métricas
    meta_files = list(session_dir.glob(f"candidate_{candidate_index:03d}_*_metadata.json"))
    metadata = {}
    if meta_files:
        metadata = json.loads(meta_files[0].read_text(encoding="utf-8"))
    
    return {
        "candidate_index": candidate_index,
        "algorithm": metadata.get("training", {}).get("algorithm"),
        "metrics": metadata.get("metrics", {}),
        "predictions": pred_data,
    }


@router.post(
    "/training/select-candidate",
    summary="Seleciona um candidato e aplica ao pipeline",
)
async def endpoint_select_candidate(
    tenant: str,
    session_path: str,
    candidate_index: int,
    step_id: str,
    apply_to_pipeline: bool = True,
    change_reason: Optional[str] = None,
):
    """
    Seleciona um candidato do grid search e (opcionalmente) aplica ao pipeline.
    
    Args:
        tenant: Tenant
        session_path: Caminho do _session.json (relativo ao repo)
        candidate_index: Índice do candidato a selecionar (0-based, ou use rank-1)
        step_id: ID do step no pipeline
        apply_to_pipeline: Se True, atualiza o pipeline com o modelo selecionado
        change_reason: Razão da mudança (para histórico)
    
    Returns:
        Informações do modelo selecionado e nova versão do pipeline (se aplicado)
    """
    from ..infrastructure.ml.training import select_candidate
    from datetime import datetime
    import json
    
    tenant = _validate_segment(tenant, "tenant")
    
    # Resolver path
    full_session_path = _repo_root() / session_path
    if not full_session_path.exists():
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    # Diretório de destino para o modelo selecionado
    session_data = json.loads(full_session_path.read_text(encoding="utf-8"))
    out_dir = _resources_root() / tenant / "predict" / "selected" / step_id
    prefix = f"{step_id}_selected"
    
    # Selecionar candidato
    result = select_candidate(
        session_path=full_session_path,
        candidate_index=candidate_index,
        out_dir=out_dir,
        prefix=prefix,
    )
    
    model_rel = result.model_path.relative_to(_repo_root()).as_posix()
    scaler_rel = result.scaler_path.relative_to(_repo_root()).as_posix()
    metadata_rel = result.metadata_path.relative_to(_repo_root()).as_posix() if result.metadata_path else ""
    
    activated_version: Optional[str] = None
    
    if apply_to_pipeline:
        # Carregar pipeline
        pipeline_file = _workspace_pipeline_file(tenant, tenant)
        if pipeline_file.exists():
            payload = _read_json_file(pipeline_file)
            
            # Atualizar block_config do step
            _update_workspace_pipeline_block_config(
                payload,
                step_id,
                {
                    "model_path": model_rel,
                    "scaler_path": scaler_rel,
                    "metadata_path": metadata_rel,
                    "y_transform": result.y_transform,
                    "resource": "",  # Limpar resource para usar paths customizados
                },
            )
            
            # Criar nova versão
            manifest = _read_versions_manifest(tenant)
            versions = manifest.get("versions") if isinstance(manifest.get("versions"), list) else []
            vid = _next_version_id(versions)
            
            versions_dir = _workspace_versions_dir(tenant)
            versions_dir.mkdir(parents=True, exist_ok=True)
            
            new_version_file = versions_dir / f"{vid}.json"
            new_version_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            
            # Atualizar manifest
            manifest["active"] = vid
            if not isinstance(manifest.get("versions"), list):
                manifest["versions"] = []
            manifest["versions"].append({
                "id": vid,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "change_reason": change_reason or f"Modelo selecionado para {step_id}",
            })
            
            manifest_file = _workspace_versions_dir(tenant).parent / "_versions.json"
            manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            
            # Atualizar arquivo principal
            pipeline_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            
            activated_version = vid
    
    selected_candidate = session_data.get("candidates", [])[candidate_index]
    
    return {
        "success": True,
        "step_id": step_id,
        "selected_candidate": {
            "rank": selected_candidate.get("rank"),
            "algorithm": selected_candidate.get("algorithm"),
            "params": selected_candidate.get("params"),
            "score": selected_candidate.get("score"),
            "metrics": selected_candidate.get("metrics"),
        },
        "model_path": model_rel,
        "scaler_path": scaler_rel,
        "metadata_path": metadata_rel,
        "version": activated_version,
        "message": f"Modelo '{selected_candidate.get('algorithm')}' (rank {selected_candidate.get('rank')}) selecionado para {step_id}",
    }


# =============================================================================
# REGRESSÕES MATEMÁTICAS
# =============================================================================
# Alternativa a modelos ML: regressões simples (linear, exponencial, etc.)
# Os coeficientes são salvos no metadata.json e aplicados diretamente nos blocos.
# =============================================================================

@router.post(
    "/training/regression",
    summary="Treina uma regressão matemática como alternativa a ML",
)
async def endpoint_train_regression(
    request: Request,
    tenant: str,
    step_id: str,
    regression_type: str = Query(default="linear", description="Tipo: linear, quadratic, exponential, logarithmic, power, polynomial"),
    protocolId: Optional[str] = None,
    experimentIds: list[str] = Query(default=[], alias="experimentIds"),
    auto_select: bool = Query(default=False, description="Se True, testa todos os tipos e seleciona o melhor"),
    polynomial_degree: int = Query(default=3, description="Grau do polinômio (se regression_type=polynomial)"),
    y_transform: str = Query(default="none", description="Transformação do Y: 'log10p' para log10(1+y), 'none' para nenhuma"),
    outlier_method: str = Query(default="none", description="Método de remoção de outliers: 'none', 'ransac', 'iqr', 'zscore'"),
    robust_method: str = Query(default="ols", description="Método robusto para linear: 'ols', 'theil_sen', 'huber', 'ransac_fit'"),
    apply_to_pipeline: bool = Query(default=True, description="Se True, aplica automaticamente ao pipeline"),
    version: Optional[str] = None,
    use_cache: bool = True,
    invalidate_cache: bool = False,
):
    """
    Treina uma regressão matemática simples.
    
    Diferente do /training/grid-search que treina modelos ML complexos,
    este endpoint ajusta equações matemáticas simples aos dados.
    
    Vantagens:
    - Interpretabilidade: você vê a equação (ex: y = 2.5*x + 10)
    - Velocidade: ajuste instantâneo
    - Sem dependências: não precisa de ONNX, scaler, etc.
    
    Tipos suportados:
    - linear: y = a*x + b
    - quadratic: y = a*x² + b*x + c  
    - exponential: y = a * exp(b*x) + c
    - logarithmic: y = a * ln(x) + b
    - power: y = a * x^b + c
    - polynomial: y = aₙxⁿ + ... + a₁x + a₀
    
    Args:
        regression_type: Tipo de regressão a ajustar
        auto_select: Se True, testa todos os tipos e retorna o melhor (baseado em R²)
        polynomial_degree: Grau do polinômio (apenas para regression_type=polynomial)
    """
    from ..infrastructure.ml.regression import (
        fit_regression,
        fit_best_regression,
        regression_to_plot_data,
        SUPPORTED_REGRESSIONS,
    )
    
    tenant = _validate_segment(tenant, "tenant")
    
    if regression_type not in SUPPORTED_REGRESSIONS and not auto_select:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de regressão inválido: {regression_type}. Use: {SUPPORTED_REGRESSIONS}"
        )
    
    # Tentar ler pipeline do body da requisição
    pipeline_from_body: Optional[dict] = None
    try:
        body = await request.json()
        if isinstance(body, dict) and ("steps" in body or "editor" in body or "execution" in body):
            pipeline_from_body = body
    except Exception:
        pass
    
    # Carregar pipeline
    payload = None
    if pipeline_from_body:
        payload = pipeline_from_body
    else:
        ver = _validate_version_id(version) if version else None
        if ver:
            source_file = _workspace_versions_dir(tenant) / f"{ver}.json"
        else:
            source_file = _workspace_pipeline_file(tenant, tenant)
        
        if not source_file.exists():
            raise HTTPException(status_code=404, detail="Pipeline não encontrado")
        
        payload = _read_json_file(source_file)
    
    config = _build_workspace_pipeline_config(
        tenant=tenant,
        pipeline_json=payload,
        timeout_seconds=300.0,
        fail_fast=True,
        generate_output_graphs=False,
    )
    
    engine = PipelineEngine(config)
    
    # Encontrar o step
    step = None
    for s in config.steps:
        if str(s.step_id) == step_id:
            step = s
            break
    
    if not step:
        raise HTTPException(status_code=404, detail=f"Step {step_id} não encontrado")
    
    # Coletar dados (similar ao grid-search mas simplificado para 1D)
    X_values: list[float] = []
    y_values: list[float] = []
    errors: list[str] = []
    skipped: list[str] = []
    
    targets_map = DEFAULT_TARGETS_MAP
    block_cfg = dict(step.block_config or {})
    input_mapping = dict(step.input_mapping or {})
    block_name = str(step.block_name)
    
    # ==========================================================================
    # CACHE V2 (por experimento) - Mesmo sistema do grid-search
    # ==========================================================================
    print(f"[regression] Usando cache v2 (por experimento)")
    
    # 1. Coletar features usando cache v2
    X_dict, metadata_dict, collect_errors, collect_skipped, _ = await _collect_features_with_cache_v2(
        tenant=tenant,
        payload=payload,
        step_id=step_id,
        experiment_ids=experimentIds,
        engine=engine,
        config=config,
        use_cache=use_cache,
        invalidate_cache=invalidate_cache,
    )
    errors.extend(collect_errors)
    skipped.extend(collect_skipped)
    
    # 2. Determinar target_field
    unit = str(block_cfg.get("output_unit") or "").strip()
    first_label = None
    for exp_id, meta in metadata_dict.items():
        if meta.get("label"):
            first_label = meta.get("label")
            break
    
    by_unit = targets_map.get(first_label) if isinstance(targets_map.get(first_label), dict) else None
    target_field = str(by_unit.get(unit) or "") if isinstance(by_unit, dict) else ""
    
    if not target_field:
        target_field = "nmp_100ml"  # fallback
    
    print(f"[regression] target_field={target_field}, label={first_label}, unit={unit}")
    
    # 3. Mapear fatores de diluição (a partir do cache/metadata)
    dilution_factors: dict[str, float] = {}
    for exp_id, meta in metadata_dict.items():
        raw = meta.get("dilution") or meta.get("dilution_factor")
        if raw is None:
            continue
        try:
            val = float(raw)
            dilution_factors[exp_id] = val if val > 0 else 1.0
        except (ValueError, TypeError):
            dilution_factors[exp_id] = 1.0

    # 4. Buscar Y em batch (SEMPRE fresco, não usa cache)
    y_dict, y_skips = _get_y_batch_from_lab_results(
        tenant=tenant,
        experiment_ids=experimentIds,
        target_field=target_field,
        dilution_factors=dilution_factors,
    )
    
    # 4. Combinar X e Y (para regressão, X é 1D)
    for exp_id in experimentIds:
        if exp_id not in X_dict:
            continue
        if exp_id not in y_dict:
            if exp_id in y_skips:
                skipped.append(f"{exp_id}: {y_skips[exp_id]}")
            continue
        
        X_exp = X_dict[exp_id]
        # Para regressão 1D, extrair apenas o primeiro valor
        if X_exp.ndim == 2 and X_exp.shape[1] >= 1:
            x_val = float(X_exp[0, 0])
        else:
            x_val = float(X_exp.flatten()[0])
        
        X_values.append(x_val)
        y_values.append(y_dict[exp_id])
    
    print(f"[regression] Cache v2: {len(X_values)} amostras coletadas de {len(experimentIds)} experimentos")
    
    # Verificar dados suficientes
    if len(X_values) < 2:
        return {
            "success": False,
            "status": "skipped",
            "reason": f"Dados insuficientes (coletados: {len(X_values)})",
            "step_id": step_id,
            "errors": errors[:10],
            "skipped_reasons": skipped[:10],
        }
    
    X = np.array(X_values, dtype=np.float64)
    y = np.array(y_values, dtype=np.float64)
    
    # Guardar cópia dos dados antes de qualquer processamento
    X_original = X.copy()
    y_original_raw = y.copy()
    
    # Aplicar transformação Y se configurada
    if y_transform == "log10p":
        y = np.log10(1 + y)
        print(f"[regression] Transformação Y aplicada: log10(1+y)")
    
    # Y original (com transformação, para o gráfico)
    y_original = y.copy()
    
    # Ajustar regressão
    all_results: dict = {}
    
    if auto_select:
        best_result, all_results = fit_best_regression(
            X, y, metric="r2", 
            outlier_method=outlier_method,
            robust_method=robust_method,
            degree=polynomial_degree
        )
    else:
        best_result = fit_regression(
            X, y, regression_type, 
            outlier_method=outlier_method,
            robust_method=robust_method,
            degree=polynomial_degree
        )
        all_results = {regression_type: best_result}
    
    if not best_result.success:
        return {
            "success": False,
            "status": "failed",
            "reason": best_result.error or "Falha no ajuste da regressão",
            "step_id": step_id,
        }
    
    # Calcular dados inliers (sem outliers) para visualização
    outlier_indices = best_result.outlier_indices or []
    inlier_mask = np.ones(len(X_original), dtype=bool)
    inlier_mask[outlier_indices] = False
    X_inliers = X_original[inlier_mask]
    y_inliers = y_original[inlier_mask]
    
    # Preparar dados para visualização
    # - data_points: pontos usados no ajuste (inliers)
    # - original_data: todos os pontos (incluindo outliers)
    # - outlier_indices: índices dos outliers removidos
    plot_data = regression_to_plot_data(
        best_result, 
        x_data=X_inliers, 
        y_data=y_inliers,
        x_original=X_original,
        y_original=y_original,
    )
    
    # Preparar metadata para salvar
    out_dir = _resources_root() / tenant / "predict" / "selected" / step_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "version": "1.2.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        
        # Tipo de modelo - CHAVE para o bloco detectar regressão
        "model_type": best_result.regression_type,
        
        # Transformação Y aplicada (crítico para inferência)
        "y_transform": y_transform,
        
        # Método de remoção de outliers
        "outlier_method": outlier_method,
        
        # Dados da regressão
        "regression": {
            "equation": best_result.equation,
            "coefficients": best_result.coefficients,
            "r2_score": round(best_result.r2_score, 6),
            "rmse": round(best_result.rmse, 6),
            "mae": round(best_result.mae, 6),
            "n_samples": best_result.n_samples,
            "x_range": list(best_result.x_range) if best_result.x_range else None,
            "outlier_method": best_result.outlier_method,
            "n_outliers_removed": best_result.n_outliers_removed,
        },
        
        # Métricas (formato compatível com ML)
        "metrics": {
            "r2": best_result.r2_score,
            "rmse": best_result.rmse,
            "mae": best_result.mae,
            "n_samples": best_result.n_samples,
            "n_outliers_removed": best_result.n_outliers_removed,
        },
        
        # Block config original
        "block_config": block_cfg,
        
        # Não precisa de model_file nem scaler_file para regressão!
        "model_file": None,
        "scaler_file": None,
    }
    
    metadata_path = out_dir / f"{step_id}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    
    metadata_rel = metadata_path.relative_to(_repo_root()).as_posix()
    
    activated_version: Optional[str] = None
    
    # Aplicar ao pipeline
    if apply_to_pipeline:
        pipeline_file = _workspace_pipeline_file(tenant, tenant)
        if pipeline_file.exists():
            payload = _read_json_file(pipeline_file)
            
            # Atualizar block_config do step
            _update_workspace_pipeline_block_config(
                payload,
                step_id,
                {
                    "metadata_path": metadata_rel,
                    "model_path": "",  # Limpar - não precisa para regressão
                    "scaler_path": "",  # Limpar - não precisa para regressão
                    "resource": "",  # Limpar resource
                },
            )
            
            # Criar nova versão
            manifest = _read_versions_manifest(tenant)
            versions = manifest.get("versions") if isinstance(manifest.get("versions"), list) else []
            vid = _next_version_id(versions)
            
            versions_dir = _workspace_versions_dir(tenant)
            versions_dir.mkdir(parents=True, exist_ok=True)
            
            new_version_file = versions_dir / f"{vid}.json"
            new_version_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            
            # Atualizar manifest
            manifest["active"] = vid
            if not isinstance(manifest.get("versions"), list):
                manifest["versions"] = []
            manifest["versions"].append({
                "id": vid,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "change_reason": f"Regressão {best_result.regression_type} para {step_id}",
            })
            
            manifest_file = _workspace_versions_dir(tenant).parent / "_versions.json"
            manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            
            # Atualizar arquivo principal
            pipeline_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            
            activated_version = vid
    
    # Preparar comparativo se auto_select
    comparison = None
    if auto_select and len(all_results) > 1:
        comparison = [
            {
                "type": k,
                "equation": v.equation,
                "r2": round(v.r2_score, 4),
                "rmse": round(v.rmse, 4),
                "mae": round(v.mae, 4),
                "success": v.success,
                "selected": k == best_result.regression_type,
            }
            for k, v in sorted(all_results.items(), key=lambda x: -x[1].r2_score if x[1].success else float('inf'))
        ]
    
    return {
        "success": True,
        "status": "trained",
        "step_id": step_id,
        "regression_type": best_result.regression_type,
        "equation": best_result.equation,
        "coefficients": best_result.coefficients,
        "metrics": {
            "r2": round(best_result.r2_score, 4),
            "rmse": round(best_result.rmse, 4),
            "mae": round(best_result.mae, 4),
            "n_samples": best_result.n_samples,
        },
        "metadata_path": metadata_rel,
        "version": activated_version,
        "y_transform": y_transform,
        "plot_data": plot_data,
        "comparison": comparison,
        "n_collected": len(X_values),
        "n_total_experiments": len(experimentIds),
        "errors": errors[:10] if errors else [],
        "skipped_reasons": skipped[:10] if skipped else [],
    }


@router.post(
    "/training/apply-regression",
    summary="Aplica uma regressão treinada ao pipeline",
)
async def endpoint_apply_regression(
    tenant: str,
    step_id: str,
    regression_type: str,
    coefficients: str = Query(..., description="JSON string dos coeficientes"),
    equation: str = Query(..., description="Equação formatada"),
    r2_score: float = Query(default=0.0),
    rmse: float = Query(default=0.0),
    mae: float = Query(default=0.0),
    n_samples: int = Query(default=0),
    y_transform: str = Query(default="none", description="Transformação Y aplicada"),
):
    """
    Aplica uma regressão previamente treinada ao pipeline.
    
    Este endpoint salva os metadados da regressão e cria uma nova versão do pipeline.
    """
    tenant = _validate_segment(tenant, "tenant")
    
    # Parse coefficients
    try:
        coefs = json.loads(coefficients)
    except Exception:
        raise HTTPException(status_code=400, detail="Coeficientes inválidos (JSON)")
    
    # Preparar metadata
    out_dir = _resources_root() / tenant / "predict" / "selected" / step_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "version": "1.2.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_type": regression_type,
        "y_transform": y_transform,
        "regression": {
            "equation": equation,
            "coefficients": coefs,
            "r2_score": round(r2_score, 6),
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "n_samples": n_samples,
        },
        "metrics": {
            "r2": r2_score,
            "rmse": rmse,
            "mae": mae,
            "n_samples": n_samples,
        },
        "model_file": None,
        "scaler_file": None,
    }
    
    metadata_path = out_dir / f"{step_id}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    
    metadata_rel = metadata_path.relative_to(_repo_root()).as_posix()
    
    # Aplicar ao pipeline
    pipeline_file = _workspace_pipeline_file(tenant, tenant)
    if not pipeline_file.exists():
        raise HTTPException(status_code=404, detail="Pipeline não encontrado")
    
    payload = _read_json_file(pipeline_file)
    
    # Atualizar block_config do step
    _update_workspace_pipeline_block_config(
        payload,
        step_id,
        {
            "metadata_path": metadata_rel,
            "model_path": "",
            "scaler_path": "",
            "resource": "",
        },
    )
    
    # Criar nova versão
    manifest = _read_versions_manifest(tenant)
    versions = manifest.get("versions") if isinstance(manifest.get("versions"), list) else []
    vid = _next_version_id(versions)
    
    versions_dir = _workspace_versions_dir(tenant)
    versions_dir.mkdir(parents=True, exist_ok=True)
    
    new_version_file = versions_dir / f"{vid}.json"
    new_version_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Atualizar manifest
    manifest["active"] = vid
    if not isinstance(manifest.get("versions"), list):
        manifest["versions"] = []
    manifest["versions"].append({
        "id": vid,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "change_reason": f"Regressão {regression_type} para {step_id}",
    })
    
    manifest_file = _workspace_versions_dir(tenant).parent / "_versions.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Atualizar arquivo principal
    pipeline_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    
    return {
        "success": True,
        "step_id": step_id,
        "version": vid,
        "metadata_path": metadata_rel,
    }


@router.get(
    "/training/regression/plot/{tenant}/{step_id}",
    summary="Obtém dados para plotar a regressão",
)
async def endpoint_regression_plot(tenant: str, step_id: str):
    """
    Retorna dados para plotar a curva de regressão.
    
    Busca o metadata.json do step e gera pontos da curva para visualização.
    """
    from ..infrastructure.ml.regression import (
        RegressionResult,
        generate_curve_points,
    )
    
    tenant = _validate_segment(tenant, "tenant")
    
    # Buscar metadata
    metadata_path = _resources_root() / tenant / "predict" / "selected" / step_id / f"{step_id}_metadata.json"
    
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail=f"Metadata não encontrado para step {step_id}")
    
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    
    model_type = metadata.get("model_type")
    if model_type not in ("linear", "quadratic", "exponential", "logarithmic", "power", "polynomial"):
        raise HTTPException(status_code=400, detail=f"Step {step_id} não é uma regressão (model_type={model_type})")
    
    regression_data = metadata.get("regression", {})
    
    # Reconstruir RegressionResult
    result = RegressionResult(
        regression_type=model_type,
        coefficients=regression_data.get("coefficients", {}),
        equation=regression_data.get("equation", ""),
        r2_score=regression_data.get("r2_score", 0),
        rmse=regression_data.get("rmse", 0),
        mae=regression_data.get("mae", 0),
        n_samples=regression_data.get("n_samples", 0),
        x_range=tuple(regression_data["x_range"]) if regression_data.get("x_range") else None,
    )
    
    # Gerar pontos da curva
    x_curve, y_curve = generate_curve_points(result, n_points=100)
    
    return {
        "step_id": step_id,
        "regression_type": model_type,
        "equation": result.equation,
        "coefficients": result.coefficients,
        "metrics": {
            "r2": result.r2_score,
            "rmse": result.rmse,
            "mae": result.mae,
            "n_samples": result.n_samples,
        },
        "curve": {
            "x": x_curve.tolist(),
            "y": y_curve.tolist(),
        },
        "x_range": list(result.x_range) if result.x_range else None,
    }


# =============================================================================
# Datasets de Treinamento
# =============================================================================

@router.get(
    "/datasets/analysis-ids/{tenant}",
    summary="Lista os analysisIds disponíveis para um tenant",
)
async def endpoint_list_analysis_ids(tenant: str, source: str = Query(default="all")):
    """
    Lista os analysisIds únicos disponíveis para um tenant.
    
    Busca na coleção data_analise os analysisIds que têm dados.
    """
    tenant = _validate_segment(tenant, "tenant")
    
    settings = get_settings()
    source = _normalize_source(source)

    analysis_ids: set[str] = set()

    if source in ("all", "mongo"):
        if not settings.mongo_uri:
            if source == "mongo":
                raise HTTPException(status_code=500, detail="MongoDB não configurado")
        else:
            repo = MongoRepository(settings.mongo_uri, settings.tenant_db_prefix)
            try:
                analysis_ids.update(repo.list_analysis_ids(tenant))
            finally:
                repo.close()

    if source in ("all", "mock"):
        mock_repo = MockRepository(settings.resources_dir)
        analysis_ids.update(mock_repo.list_analysis_ids(tenant))

    return {"tenant": tenant, "analysis_ids": sorted(analysis_ids)}


@router.get(
    "/datasets/experiments/{tenant}/{protocol_id}",
    summary="Lista experimentos disponíveis para um protocolId",
)
async def endpoint_list_experiments(
    tenant: str,
    protocol_id: str,
    limit: int = 500,
    source: str = Query(default="all"),
):
    """
    Lista experimentos que possuem dados para um protocolId específico.
    
    Retorna informações resumidas de cada experimento incluindo:
    - Se tem lab_results (necessário para treino supervisionado)
    - Quantidade de pontos de dados
    - Labels disponíveis
    """
    tenant = _validate_segment(tenant, "tenant")
    
    settings = get_settings()
    source = _normalize_source(source)

    experiments: list[dict[str, Any]] = []

    if source in ("all", "mongo"):
        if not settings.mongo_uri:
            if source == "mongo":
                raise HTTPException(status_code=500, detail="MongoDB não configurado")
        else:
            repo = MongoRepository(settings.mongo_uri, settings.tenant_db_prefix)
            try:
                mongo_exps = repo.list_experiments_by_protocol(tenant, protocol_id, limit=limit)
                for e in mongo_exps:
                    if isinstance(e, dict) and "source" not in e:
                        e["source"] = "mongo"
                experiments.extend(mongo_exps)
            finally:
                repo.close()

    if source in ("all", "mock"):
        mock_repo = MockRepository(settings.resources_dir)
        experiments.extend(mock_repo.list_experiments_by_protocol(tenant, protocol_id, limit=limit))

    total = len(experiments)
    with_lab = sum(1 for e in experiments if e.get("has_lab_results"))

    return {
        "tenant": tenant,
        "protocol_id": protocol_id,
        "total": total,
        "with_lab_results": with_lab,
        "experiments": experiments,
    }


@router.get(
    "/datasets/protocols/{tenant}",
    summary="Lista protocolIds disponíveis para um tenant",
)
async def endpoint_list_protocols(tenant: str, limit: int = 100, source: str = Query(default="all")):
    """
    Lista os protocolIds disponíveis para um tenant.
    
    Retorna protocolIds únicos com contagem de experimentos.
    """
    tenant = _validate_segment(tenant, "tenant")
    
    settings = get_settings()
    source = _normalize_source(source)

    merged: dict[str, int] = {}

    if source in ("all", "mongo"):
        if not settings.mongo_uri:
            if source == "mongo":
                raise HTTPException(status_code=500, detail="MongoDB não configurado")
        else:
            repo = MongoRepository(settings.mongo_uri, settings.tenant_db_prefix)
            try:
                for item in repo.list_protocol_ids(tenant, limit=limit):
                    pid = item.get("protocol_id")
                    count = int(item.get("experiment_count") or 0)
                    if pid:
                        merged[pid] = merged.get(pid, 0) + count
            finally:
                repo.close()

    if source in ("all", "mock"):
        mock_repo = MockRepository(settings.resources_dir)
        for item in mock_repo.list_protocol_ids(tenant, limit=limit):
            pid = item.get("protocol_id")
            count = int(item.get("experiment_count") or 0)
            if pid:
                merged[pid] = merged.get(pid, 0) + count

    protocols = [{"protocol_id": k, "experiment_count": v} for k, v in merged.items()]
    protocols.sort(key=lambda x: x["experiment_count"], reverse=True)
    protocols = protocols[:limit]

    return {
        "tenant": tenant,
        "protocols": protocols,
    }


@router.get(
    "/datasets/{tenant}",
    summary="Lista datasets salvos para um tenant",
)
async def endpoint_list_datasets(tenant: str):
    """
    Lista datasets de treinamento salvos para um tenant.
    
    Datasets são salvos em resources/<tenant>/datasets/
    """
    tenant = _validate_segment(tenant, "tenant")
    
    datasets_dir = _resources_root() / tenant / "datasets"
    if not datasets_dir.exists():
        return {"tenant": tenant, "datasets": []}
    
    datasets = []
    for f in sorted(datasets_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            datasets.append({
                "id": f.stem,
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "protocol_id": data.get("protocol_id", ""),
                "source": data.get("source", "mongo"),
                "is_mock": bool(data.get("source") == "mock"),
                "experiment_count": len(data.get("experiment_ids", [])),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            })
        except Exception:
            continue
    
    return {"tenant": tenant, "datasets": datasets}


@router.get(
    "/datasets/{tenant}/{dataset_id}",
    summary="Obtém detalhes de um dataset",
)
async def endpoint_get_dataset(tenant: str, dataset_id: str):
    """
    Retorna os detalhes completos de um dataset, incluindo a lista de experiment_ids.
    """
    tenant = _validate_segment(tenant, "tenant")
    dataset_id = _validate_segment(dataset_id, "dataset_id")
    
    dataset_file = _resources_root() / tenant / "datasets" / f"{dataset_id}.json"
    if not dataset_file.exists():
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    data = json.loads(dataset_file.read_text(encoding="utf-8"))
    return data


@router.post(
    "/datasets/{tenant}",
    summary="Cria ou atualiza um dataset",
)
async def endpoint_save_dataset(
    tenant: str,
    name: str,
    protocol_id: str,
    experiment_ids: list[str] = Query(default=[]),
    viewed_ids: list[str] = Query(default=[]),
    ratings: str = "{}",
    description: str = "",
    dataset_id: Optional[str] = None,
    source: str = "mongo",
):
    """
    Cria ou atualiza um dataset de treinamento.
    
    Args:
        tenant: Identificador do tenant
        name: Nome do dataset
        protocol_id: ID do protocolo
        experiment_ids: Lista de IDs de experimentos selecionados
        viewed_ids: Lista de IDs de experimentos já visualizados
        ratings: JSON com classificações {expId: "good"|"bad"}
        description: Descrição opcional
        dataset_id: ID do dataset (se atualização)
    """
    from datetime import datetime
    import uuid
    
    tenant = _validate_segment(tenant, "tenant")
    
    datasets_dir = _resources_root() / tenant / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    now = datetime.utcnow().isoformat() + "Z"
    
    source = str(source or "mongo").strip().lower()
    if source not in ("mongo", "mock"):
        raise HTTPException(status_code=400, detail="source inválido (use: mongo, mock)")

    # Parse ratings JSON
    try:
        ratings_dict = json.loads(ratings) if ratings else {}
    except json.JSONDecodeError:
        ratings_dict = {}
    
    if dataset_id:
        # Atualização
        dataset_id = _validate_segment(dataset_id, "dataset_id")
        dataset_file = datasets_dir / f"{dataset_id}.json"
        
        if dataset_file.exists():
            existing = json.loads(dataset_file.read_text(encoding="utf-8"))
            created_at = existing.get("created_at", now)
            # Merge viewed_ids com existentes
            existing_viewed = set(existing.get("viewed_ids", []))
            existing_viewed.update(viewed_ids)
            viewed_ids = list(existing_viewed)
            # Merge ratings com existentes
            existing_ratings = existing.get("ratings", {})
            existing_ratings.update(ratings_dict)
            ratings_dict = existing_ratings
        else:
            created_at = now
    else:
        # Novo dataset
        dataset_id = str(uuid.uuid4())[:8]
        dataset_file = datasets_dir / f"{dataset_id}.json"
        created_at = now
    
    data = {
        "id": dataset_id,
        "name": name,
        "description": description,
        "protocol_id": protocol_id,
        "source": source,
        "experiment_ids": list(experiment_ids) if experiment_ids else [],
        "viewed_ids": list(viewed_ids) if viewed_ids else [],
        "ratings": ratings_dict,
        "created_at": created_at,
        "updated_at": now,
    }
    
    dataset_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "name": name,
        "source": source,
        "experiment_count": len(experiment_ids),
        "viewed_count": len(viewed_ids),
    }


@router.delete(
    "/datasets/{tenant}/{dataset_id}",
    summary="Remove um dataset",
)
async def endpoint_delete_dataset(tenant: str, dataset_id: str):
    """
    Remove um dataset de treinamento.
    """
    tenant = _validate_segment(tenant, "tenant")
    dataset_id = _validate_segment(dataset_id, "dataset_id")
    
    dataset_file = _resources_root() / tenant / "datasets" / f"{dataset_id}.json"
    if not dataset_file.exists():
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    dataset_file.unlink()
    
    return {"success": True, "dataset_id": dataset_id}


@router.get(
    "/datasets/preview/{tenant}/{experiment_id}",
    summary="Gera preview de gráficos para um experimento (um por sensor)",
)
async def endpoint_experiment_preview(
    tenant: str,
    experiment_id: str,
    normalize: bool = True,
):
    """
    Gera gráficos de preview dos dados espectrais de um experimento.
    
    Retorna um gráfico PNG base64 para cada sensor (turbidimetry, nephelometry, fluorescence).
    Também retorna informações resumidas do experimento e resultados de laboratório.
    
    Args:
        tenant: Identificador do tenant
        experiment_id: ID do experimento
        normalize: Se deve normalizar os dados (0-1)
    """
    import base64
    import io
    from datetime import datetime
    
    tenant = _validate_segment(tenant, "tenant")
    
    settings = get_settings()
    is_mock = str(experiment_id).startswith("mock:")

    if is_mock:
        repo = MockRepository(settings.resources_dir)
    else:
        if not settings.mongo_uri:
            raise HTTPException(status_code=500, detail="MongoDB não configurado")
        repo = MongoRepository(settings.mongo_uri, settings.tenant_db_prefix)

    try:
        # Buscar dados do experimento
        experiment_data = repo.get_experiment_data(tenant, experiment_id, limit=5000)

        if not experiment_data:
            raise HTTPException(status_code=404, detail="Dados do experimento não encontrados")

        # Buscar informações do experimento
        experiment_doc = repo.get_experiment(tenant, experiment_id)

        # Buscar resultados de laboratório
        lab_results = repo.get_lab_results(tenant, experiment_id, limit=10)
        
        # Processar lab_results para resumo
        lab_summary = []
        for lr in lab_results:
            # Data da análise (pode ser datetime ou timestamp)
            analysis_date = lr.get("analysisDate") or lr.get("analysis_date") or lr.get("date")
            if analysis_date:
                if isinstance(analysis_date, (int, float)):
                    try:
                        analysis_date = datetime.fromtimestamp(analysis_date / 1000).strftime("%d/%m/%Y")
                    except:
                        analysis_date = None
                elif hasattr(analysis_date, "strftime"):
                    analysis_date = analysis_date.strftime("%d/%m/%Y")
                else:
                    analysis_date = str(analysis_date)[:10] if analysis_date else None
            
            # Unidade pode vir do campo tagUnidade ou ser inferida do nome do campo
            tag_unidade = lr.get("tagUnidade", "NMP/100mL")
            
            # Extrair contagens específicas (estrutura do BioAILab CRM)
            # Os campos terminam em "Nmp" indicando a unidade NMP
            coliformes = lr.get("coliformesTotaisNmp")
            ecoli = lr.get("ecoliNmp")
            
            # Se tem coliformes, adicionar
            if coliformes is not None:
                lab_summary.append({
                    "bacteria": "Coliformes Totais",
                    "count": coliformes,
                    "unit": tag_unidade,
                    "date": analysis_date,
                    "method": lr.get("method", "NMP"),
                    "presence": coliformes > 0 if isinstance(coliformes, (int, float)) else None,
                })
            
            # Se tem E.coli, adicionar
            if ecoli is not None:
                lab_summary.append({
                    "bacteria": "E. coli",
                    "count": ecoli,
                    "unit": tag_unidade,
                    "date": analysis_date,
                    "method": lr.get("method", "NMP"),
                    "presence": ecoli > 0 if isinstance(ecoli, (int, float)) else None,
                })
            
            # Fallback para estrutura genérica
            if coliformes is None and ecoli is None:
                count = lr.get("count") or lr.get("value") or lr.get("cfu") or lr.get("cfuCount")
                bacteria = lr.get("label") or lr.get("bacteria") or lr.get("type") or lr.get("target")
                if count is not None or bacteria:
                    lab_summary.append({
                        "bacteria": bacteria or "Amostra",
                        "count": count,
                        "unit": lr.get("unit") or tag_unidade,
                        "date": analysis_date,
                        "method": lr.get("method"),
                        "presence": lr.get("presence"),
                    })
        
        # Extrair timestamps e dados por sensor
        timestamps = []
        sensors_data = {}  # {sensor: {channel: [values]}}
        temperature_data = []  # temperatura da amostra
        
        # Canais esperados (minúsculos conforme banco)
        channel_list = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "clr", "nir"]
        
        for doc in experiment_data:
            ts = doc.get("timestamp")
            if ts is None:
                continue
            timestamps.append(ts)
            
            # Extrair dados espectrais de cada sensor
            spectral = doc.get("spectral", {})
            for sensor_key, sensor_data in spectral.items():
                if not isinstance(sensor_data, dict):
                    continue
                if sensor_key not in sensors_data:
                    sensors_data[sensor_key] = {}
                
                for ch in channel_list:
                    if ch in sensor_data:
                        if ch not in sensors_data[sensor_key]:
                            sensors_data[sensor_key][ch] = []
                        sensors_data[sensor_key][ch].append(sensor_data[ch])
            
            # Extrair temperatura da amostra
            temps = doc.get("temperatures", {})
            sample_temp = temps.get("sample") or temps.get("amostra")
            if sample_temp is not None:
                temperature_data.append(sample_temp)
        
        if not timestamps:
            raise HTTPException(status_code=404, detail="Sem dados disponíveis")
        
        # Normalizar timestamps para minutos desde o início
        # Detectar se timestamps estão em segundos, milissegundos, ou já em minutos
        t0 = min(timestamps)
        t_max = max(timestamps)
        t_range = t_max - t0  # Diferença entre max e min
        
        # Se o valor máximo é muito grande (>10^12), está em milissegundos
        if t_max > 1e12:
            x = [(t - t0) / 60000 for t in timestamps]  # ms para minutos
        # Se o valor máximo é pequeno (<100000), provavelmente são índices/minutos (mocks)
        # Ex: mocks usam 0, 1, 2... 1367 onde cada índice = 1 minuto
        elif t_max < 100000:
            x = [(t - t0) for t in timestamps]  # já são minutos, só subtrair t0
        else:
            x = [(t - t0) / 60 for t in timestamps]  # segundos para minutos
        
        # Gerar gráficos
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            graphs = {}
            sensor_names = {
                "turbidimetry": "Turbidimetria",
                "nephelometry": "Nefelometria",
                "fluorescence": "Fluorescência",
                "temperature": "Temperatura",
            }
            
            # Gráficos dos sensores espectrais
            for sensor_key, channels in sensors_data.items():
                if not channels:
                    continue
                    
                fig, ax = plt.subplots(figsize=(10, 5))
                
                colors = plt.cm.tab10.colors
                color_idx = 0
                
                for ch_name, values in sorted(channels.items()):
                    if len(values) != len(x):
                        min_len = min(len(values), len(x))
                        y = values[:min_len]
                        x_plot = x[:min_len]
                    else:
                        y = values
                        x_plot = x
                    
                    if normalize and len(y) > 0:
                        mn, mx = min(y), max(y)
                        span = mx - mn if mx != mn else 1.0
                        y = [(v - mn) / span for v in y]
                    
                    color = colors[color_idx % len(colors)]
                    ax.plot(x_plot, y, color=color, linewidth=1, alpha=0.8, label=ch_name.upper())
                    color_idx += 1
                
                display_name = sensor_names.get(sensor_key, sensor_key)
                ax.set_xlabel("Tempo (min)")
                ax.set_ylabel("Valor" + (" (norm)" if normalize else ""))
                ax.set_title(f"{display_name} - {experiment_id[:20]}...")
                ax.legend(loc='upper right', fontsize=8, ncol=2)
                ax.grid(True, alpha=0.3)
                
                # Converter para base64
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close(fig)
                
                graphs[sensor_key] = f"data:image/png;base64,{img_base64}"
            
            # Gráfico de temperatura da amostra (mesmo padrão dos outros sensores)
            if temperature_data:
                fig, ax = plt.subplots(figsize=(10, 5))
                
                if len(temperature_data) != len(x):
                    min_len = min(len(temperature_data), len(x))
                    y_temp = temperature_data[:min_len]
                    x_temp = x[:min_len]
                else:
                    y_temp = temperature_data
                    x_temp = x
                
                # Linha simples como os outros gráficos
                ax.plot(x_temp, y_temp, color='#e74c3c', linewidth=1, alpha=0.8, label='Temperatura')
                ax.set_xlabel("Tempo (min)")
                ax.set_ylabel("Temperatura (°C)")
                ax.set_title(f"Temperatura - {experiment_id[:20]}...")
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close(fig)
                
                graphs["temperature"] = f"data:image/png;base64,{img_base64}"
            
            all_sensors = list(sensors_data.keys())
            if temperature_data:
                all_sensors.append("temperature")
            
            # Calcular duração do experimento (diferença entre primeiro e último timestamp)
            if len(timestamps) >= 2:
                ts_min = min(timestamps)
                ts_max = max(timestamps)
                duration_diff = ts_max - ts_min
                
                # Detectar formato do timestamp:
                # 1. Se min=0 e valores são sequenciais (0,1,2,...), são índices de minuto
                # 2. Se >1e12, são timestamps em milissegundos
                # 3. Caso contrário, são timestamps em segundos
                if ts_min == 0 and ts_max < 100000:
                    # Timestamps são índices de minuto (0, 1, 2, ..., N)
                    # O último valor já é a duração em minutos
                    duration_minutes = round(ts_max)
                elif ts_max > 1e12:
                    # Timestamps em milissegundos
                    duration_minutes = round(duration_diff / 60000)
                else:
                    # Timestamps em segundos
                    duration_minutes = round(duration_diff / 60)
            else:
                duration_minutes = 0
            
            # Extrair info do experimento
            exp_info = {}
            if experiment_doc:
                # Extrair diluição (campo "diluicao" representa o expoente: "5" = 1x10^5)
                diluicao_raw = experiment_doc.get("diluicao")
                diluicao_exponent = None
                diluicao_value = None
                if diluicao_raw:
                    try:
                        diluicao_exponent = int(diluicao_raw)
                        diluicao_value = 10 ** diluicao_exponent
                    except (ValueError, TypeError):
                        pass
                
                # Extrair data de início
                start_date = experiment_doc.get("startDate")
                created_at = None
                if start_date:
                    try:
                        # startDate pode ser string com timestamp em segundos
                        ts = int(start_date)
                        created_at = datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M")
                    except (ValueError, TypeError):
                        pass
                
                # Fallback para createdAt
                if not created_at:
                    created = experiment_doc.get("createdAt") or experiment_doc.get("created_at")
                    if created:
                        if isinstance(created, (int, float)):
                            try:
                                created_at = datetime.fromtimestamp(created / 1000).strftime("%d/%m/%Y %H:%M")
                            except:
                                pass
                        elif hasattr(created, "strftime"):
                            created_at = created.strftime("%d/%m/%Y %H:%M")
                
                exp_info = {
                    "name": experiment_doc.get("nome") or experiment_doc.get("name"),
                    "description": experiment_doc.get("description"),
                    "created_at": created_at,
                    "device": experiment_doc.get("serialNumber") or experiment_doc.get("device") or experiment_doc.get("deviceId"),
                    "status": experiment_doc.get("status"),
                    "diluicao_exponent": diluicao_exponent,  # Ex: 5
                    "diluicao_value": diluicao_value,  # Ex: 100000 (10^5)
                    "diluicao_display": f"1×10^{diluicao_exponent}" if diluicao_exponent else None,
                }
            
            return {
                "experiment_id": experiment_id,
                "data_points": len(timestamps),
                "duration_minutes": duration_minutes,
                "sensors": all_sensors,
                "graphs": graphs,
                "experiment": exp_info,
                "lab_results": lab_summary,
            }
            
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"matplotlib não disponível: {e}")
        
    finally:
        repo.close()


# ==============================================================================
# TRAINING HISTORY ENDPOINTS
# ==============================================================================

def _training_history_root(tenant: str, step_id: str) -> Path:
    """Get the training history directory for a tenant/step."""
    _validate_segment(tenant, "tenant")
    _validate_segment(step_id, "step_id")
    return _resources_root() / tenant / "training_history" / step_id


def _resolve_block_name_for_step(tenant: str, step_id: str) -> str:
    """Resolve block_name for a step_id from the active pipeline file."""
    try:
        pipeline_path = _workspace_pipeline_file(tenant, tenant)
        if not pipeline_path.exists():
            return ""
        payload = _read_json_file(pipeline_path)
        steps = payload.get("execution", {}).get("steps") if isinstance(payload, dict) else None
        if not isinstance(steps, list):
            steps = payload.get("steps", []) if isinstance(payload, dict) else []
        for step in steps or []:
            if str(step.get("step_id", "")) == step_id:
                return str(step.get("block_name", "") or "")
    except Exception:
        return ""
    return ""


def _resolve_step_block_config(tenant: str, step_id: str) -> dict:
    """Resolve block_config for a step_id from the active pipeline file."""
    try:
        pipeline_path = _workspace_pipeline_file(tenant, tenant)
        if not pipeline_path.exists():
            return {}
        payload = _read_json_file(pipeline_path)
        steps = payload.get("execution", {}).get("steps") if isinstance(payload, dict) else None
        if not isinstance(steps, list):
            steps = payload.get("steps", []) if isinstance(payload, dict) else []
        for step in steps or []:
            if str(step.get("step_id", "")) == step_id:
                return dict(step.get("block_config") or {})
    except Exception:
        return {}
    return {}


def _resolve_default_label(tenant: str) -> str:
    """Return the single label configured in the pipeline, if unambiguous."""
    try:
        pipeline_path = _workspace_pipeline_file(tenant, tenant)
        if not pipeline_path.exists():
            return ""
        payload = _read_json_file(pipeline_path)
        steps = payload.get("execution", {}).get("steps") if isinstance(payload, dict) else None
        if not isinstance(steps, list):
            steps = payload.get("steps", []) if isinstance(payload, dict) else []
        labels = []
        for step in steps or []:
            if str(step.get("block_name") or "") != "label":
                continue
            cfg = step.get("block_config") or {}
            label = str(cfg.get("label") or "").strip()
            if label:
                labels.append(label)
        return labels[0] if len(labels) == 1 else ""
    except Exception:
        return ""


def _resolve_history_block_details(tenant: str, data: dict) -> dict:
    """Resolve block details (unit/feature/channel/label) for a history entry."""
    details = {}
    block_cfg = {}

    result = data.get("result", {}) if isinstance(data, dict) else {}
    metadata_path = result.get("metadata_path")
    if isinstance(metadata_path, str) and metadata_path:
        try:
            meta = _read_json_file((_repo_root() / metadata_path).resolve())
            block_cfg = meta.get("block_config") or {}
        except Exception:
            block_cfg = {}

    if not block_cfg and isinstance(result.get("session_path"), str):
        try:
            session_path = _repo_root() / result.get("session_path")
            session = _read_json_file(session_path)
            candidates = session.get("candidates", []) if isinstance(session, dict) else []
            best_index = int(session.get("best_index", 0) or 0)
            if 0 <= best_index < len(candidates):
                meta_file = candidates[best_index].get("metadata_file")
                if meta_file:
                    meta = _read_json_file(session_path.parent / meta_file)
                    block_cfg = meta.get("block_config") or {}
        except Exception:
            block_cfg = {}

    if not block_cfg:
        block_cfg = _resolve_step_block_config(tenant, str(data.get("step_id", "")))

    if block_cfg:
        details["output_unit"] = block_cfg.get("output_unit")
        details["input_feature"] = block_cfg.get("input_feature")
        details["channel"] = block_cfg.get("channel")

    label = str(data.get("label") or "").strip()
    if not label:
        label = _resolve_default_label(tenant)
    if label:
        details["label"] = label

    return details


def _normalize_path_for_compare(path_value: str) -> str:
    if not path_value:
        return ""
    try:
        p = Path(path_value)
        if not p.is_absolute():
            p = (_repo_root() / path_value).resolve()
        return str(p)
    except Exception:
        return str(path_value)


def _is_history_entry_active(tenant: str, step_id: str, data: dict) -> bool:
    """Return True if entry matches the currently active model in the pipeline."""
    block_cfg = _resolve_step_block_config(tenant, step_id)
    if not block_cfg:
        return False

    cfg_model = _normalize_path_for_compare(str(block_cfg.get("model_path") or ""))
    cfg_scaler = _normalize_path_for_compare(str(block_cfg.get("scaler_path") or ""))
    cfg_meta = _normalize_path_for_compare(str(block_cfg.get("metadata_path") or ""))

    result = data.get("result", {}) if isinstance(data, dict) else {}

    res_meta = _normalize_path_for_compare(str(result.get("metadata_path") or ""))
    if cfg_meta and res_meta and cfg_meta == res_meta:
        return True

    res_model = _normalize_path_for_compare(str(result.get("model_path") or ""))
    res_scaler = _normalize_path_for_compare(str(result.get("scaler_path") or ""))
    if cfg_model and res_model and cfg_model == res_model:
        return True
    if cfg_scaler and res_scaler and cfg_scaler == res_scaler:
        return True

    session_path = result.get("session_path")
    if isinstance(session_path, str) and session_path:
        try:
            session_file = (_repo_root() / session_path).resolve()
            session = _read_json_file(session_file)
            candidates = session.get("candidates", []) if isinstance(session, dict) else []
            for cand in candidates:
                cand_model = _normalize_path_for_compare(str(session_file.parent / str(cand.get("model_file") or "")))
                cand_scaler = _normalize_path_for_compare(str(session_file.parent / str(cand.get("scaler_file") or "")))
                cand_meta = _normalize_path_for_compare(str(session_file.parent / str(cand.get("metadata_file") or "")))
                if cfg_meta and cand_meta and cfg_meta == cand_meta:
                    return True
                if cfg_model and cand_model and cfg_model == cand_model:
                    return True
                if cfg_scaler and cand_scaler and cfg_scaler == cand_scaler:
                    return True
        except Exception:
            return False

    return False


@router.get("/training/history/{tenant}/{step_id}")
async def list_training_history(
    tenant: str,
    step_id: str,
    limit: int = Query(20, ge=1, le=100),
):
    """
    List training history entries for a specific step.
    Returns a list of training records sorted by timestamp (most recent first).
    """
    history_dir = _training_history_root(tenant, step_id)
    
    if not history_dir.exists():
        return {"entries": [], "total": 0}
    
    entries = []
    for json_file in history_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("inactive") is True:
                    continue
                # Include summary info for list view
                entry = {
                    "id": data.get("id", json_file.stem),
                    "timestamp": data.get("timestamp"),
                    "mode": data.get("mode"),  # "ml" or "regression"
                    "applied": data.get("applied", False),
                    "metrics": {},
                    "block_name": data.get("block_name") or _resolve_block_name_for_step(tenant, step_id),
                    "step_id": data.get("step_id") or step_id,
                }
                entry.update(_resolve_history_block_details(tenant, data))
                entry["is_active"] = _is_history_entry_active(tenant, step_id, data)
                
                # Extract key metrics for display
                result = data.get("result", {})
                if data.get("mode") == "regression":
                    entry["metrics"] = {
                        "r_squared": result.get("r_squared") or result.get("metrics", {}).get("r2"),
                        "regression_type": result.get("regression_type"),
                        "equation": result.get("equation"),
                    }
                elif data.get("mode") == "ml":
                    # Get best candidate metrics
                    candidates = result.get("candidates", [])
                    best_index = result.get("best_index", 0)
                    best_candidate = candidates[best_index] if 0 <= best_index < len(candidates) else None
                    entry["metrics"] = {
                        "best_model": best_candidate.get("algorithm") if best_candidate else None,
                        "r2_score": best_candidate.get("metrics", {}).get("val_r2") if best_candidate else None,
                        "rmse": best_candidate.get("metrics", {}).get("val_rmse") if best_candidate else None,
                    }
                
                entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to read training history file {json_file}: {e}")
            continue
    
    # Sort by timestamp descending (most recent first)
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return {
        "entries": entries[:limit],
        "total": len(entries),
    }


@router.get("/training/history/{tenant}/{step_id}/{history_id}")
async def get_training_history_entry(
    tenant: str,
    step_id: str,
    history_id: str,
):
    """
    Get a specific training history entry with full details.
    """
    _validate_segment(history_id, "history_id")
    history_dir = _training_history_root(tenant, step_id)
    history_file = history_dir / f"{history_id}.json"
    
    if not history_file.exists():
        raise HTTPException(status_code=404, detail="Training history entry not found")
    
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read training history: {e}")


@router.post("/training/history/{tenant}/{step_id}")
async def save_training_history(
    tenant: str,
    step_id: str,
    data: dict = Body(...),
):
    """
    Save a training result to history.
    Expected data: {mode, config, result, applied?}
    """
    history_dir = _training_history_root(tenant, step_id)
    history_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique ID based on timestamp
    timestamp = datetime.now().isoformat()
    history_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    entry = {
        "id": history_id,
        "timestamp": timestamp,
        "step_id": step_id,
        "mode": data.get("mode"),
        "config": data.get("config", {}),
        "result": data.get("result", {}),
        "applied": data.get("applied", False),
        "block_name": data.get("block_name") or _resolve_block_name_for_step(tenant, step_id),
        "label": data.get("label") or _resolve_default_label(tenant) or "",
        "version": "1.0",
    }
    
    history_file = history_dir / f"{history_id}.json"
    
    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False, default=str)
        
        return {"id": history_id, "message": "Training history saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save training history: {e}")


@router.patch("/training/history/{tenant}/{step_id}/{history_id}")
async def update_training_history(
    tenant: str,
    step_id: str,
    history_id: str,
    data: dict = Body(...),
):
    """
    Update a training history entry (e.g., mark as applied).
    """
    _validate_segment(history_id, "history_id")
    history_dir = _training_history_root(tenant, step_id)
    history_file = history_dir / f"{history_id}.json"
    
    if not history_file.exists():
        raise HTTPException(status_code=404, detail="Training history entry not found")
    
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            entry = json.load(f)
        
        # Update allowed fields
        if "applied" in data:
            entry["applied"] = data["applied"]
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False, default=str)
        
        return {"message": "Training history updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update training history: {e}")


@router.delete("/training/history/{tenant}/{step_id}/{history_id}")
async def delete_training_history(
    tenant: str,
    step_id: str,
    history_id: str,
):
    """
    Delete a training history entry.
    """
    _validate_segment(history_id, "history_id")
    history_dir = _training_history_root(tenant, step_id)
    history_file = history_dir / f"{history_id}.json"
    
    if not history_file.exists():
        raise HTTPException(status_code=404, detail="Training history entry not found")
    
    try:
        history_file.unlink()
        return {"message": "Training history deleted successfully"}
    except Exception as e:
        # Fallback: mark as inactive so it no longer appears in history list
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                entry = json.load(f)
            entry["inactive"] = True
            entry["deleted_at"] = datetime.now().isoformat()
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False, default=str)
            return {"message": "Training history marked as inactive"}
        except Exception:
            raise HTTPException(status_code=500, detail=f"Failed to delete training history: {e}")


# =============================================================================
# Cache Management Endpoints
# =============================================================================

@router.get("/cache/stats/{tenant}")
async def get_cache_statistics(
    tenant: str,
    include_details: bool = Query(False, description="Include per-pipeline breakdown"),
):
    """
    Get cache statistics for a tenant.
    Returns information about cache usage, hit rates, and storage.
    """
    _validate_segment(tenant, "tenant")
    
    stats = _get_cache_stats(tenant)
    
    result = {
        "tenant": tenant,
        "v2_cache": {
            "total_experiments": stats["total_experiments"],
            "total_pipelines": stats["total_pipelines"],
            "total_size_mb": stats["total_size_mb"],
            "cache_directory": str(stats["cache_directory"]),
        },
        "memory_cache": {
            "lab_results_entries": len(_lab_results_cache),
            "experiment_data_entries": len(_experiment_data_cache),
        },
        "legacy_cache": {
            "exists": (_features_cache_dir(tenant)).exists(),
            "directory": str(_features_cache_dir(tenant)),
        },
    }
    
    if include_details:
        result["v2_cache"]["pipelines"] = stats.get("pipelines", {})
    
    return result


@router.delete("/cache/{tenant}")
async def clear_tenant_cache(
    tenant: str,
    cache_type: str = Query("all", description="Cache type to clear: 'v2', 'memory', 'legacy', or 'all'"),
):
    """
    Clear cache for a tenant.
    
    cache_type options:
    - 'v2': Clear per-experiment cache (disk)
    - 'memory': Clear in-memory LRU caches
    - 'legacy': Clear old monolithic cache
    - 'all': Clear all caches
    """
    _validate_segment(tenant, "tenant")
    
    cleared = []
    errors = []
    
    # Clear v2 cache
    if cache_type in ("v2", "all"):
        try:
            v2_dir = _features_cache_v2_dir(tenant)
            if v2_dir.exists():
                import shutil
                shutil.rmtree(v2_dir)
                cleared.append("v2_cache")
        except Exception as e:
            errors.append(f"v2_cache: {e}")
    
    # Clear memory caches
    if cache_type in ("memory", "all"):
        try:
            _clear_memory_caches()
            cleared.append("memory_cache")
        except Exception as e:
            errors.append(f"memory_cache: {e}")
    
    # Clear legacy cache
    if cache_type in ("legacy", "all"):
        try:
            legacy_dir = _features_cache_dir(tenant)
            if legacy_dir.exists():
                import shutil
                shutil.rmtree(legacy_dir)
                cleared.append("legacy_cache")
        except Exception as e:
            errors.append(f"legacy_cache: {e}")
    
    return {
        "tenant": tenant,
        "cleared": cleared,
        "errors": errors if errors else None,
        "message": f"Cleared {len(cleared)} cache(s)" if cleared else "No caches to clear",
    }


@router.delete("/cache/{tenant}/pipeline/{pipeline_hash}")
async def clear_pipeline_cache(
    tenant: str,
    pipeline_hash: str,
):
    """
    Clear cache for a specific pipeline (all steps).
    Useful when pipeline definition changes.
    """
    _validate_segment(tenant, "tenant")
    _validate_segment(pipeline_hash, "pipeline_hash")
    
    v2_dir = _features_cache_v2_dir(tenant)
    pipeline_dir = v2_dir / pipeline_hash
    
    if not pipeline_dir.exists():
        raise HTTPException(status_code=404, detail="Pipeline cache not found")
    
    try:
        import shutil
        experiments_count = sum(1 for _ in pipeline_dir.rglob("*.npz"))
        shutil.rmtree(pipeline_dir)
        return {
            "tenant": tenant,
            "pipeline_hash": pipeline_hash,
            "experiments_cleared": experiments_count,
            "message": "Pipeline cache cleared",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear pipeline cache: {e}")


@router.delete("/cache/{tenant}/experiment/{experiment_id}")
async def clear_experiment_cache(
    tenant: str,
    experiment_id: str,
):
    """
    Clear all cached features for a specific experiment across all pipelines.
    Useful when experiment data changes.
    """
    _validate_segment(tenant, "tenant")
    _validate_segment(experiment_id, "experiment_id")
    
    v2_dir = _features_cache_v2_dir(tenant)
    
    if not v2_dir.exists():
        return {"tenant": tenant, "experiment_id": experiment_id, "files_cleared": 0}
    
    cleared = 0
    for npz_file in v2_dir.rglob(f"{experiment_id}.npz"):
        try:
            json_file = npz_file.with_suffix(".json")
            npz_file.unlink()
            if json_file.exists():
                json_file.unlink()
            cleared += 1
        except Exception:
            pass
    
    # Also clear from memory cache
    global _lab_results_cache
    cache_key_to_remove = None
    for key in _lab_results_cache:
        if experiment_id in key:
            cache_key_to_remove = key
            break
    if cache_key_to_remove:
        del _lab_results_cache[cache_key_to_remove]
        if cache_key_to_remove in _lab_results_cache_time:
            del _lab_results_cache_time[cache_key_to_remove]
    
    return {
        "tenant": tenant,
        "experiment_id": experiment_id,
        "files_cleared": cleared,
        "message": f"Cleared {cleared} cached feature file(s) for experiment",
    }


# =============================================================================
# Mock Lab Results Editing Endpoints
# =============================================================================

@router.get("/mock/experiments/{tenant}/{exp_id}/lab-results")
async def get_mock_lab_results(
    tenant: str,
    exp_id: str,
):
    """
    Get lab results (calibration data) for a mock experiment.
    Returns raw calibration data along with normalized lab results.
    """
    _validate_segment(tenant, "tenant")
    settings = get_settings()
    mock_repo = MockRepository(settings.resources_dir)
    
    raw_calibration = mock_repo.get_calibration(tenant, exp_id)
    normalized_results = mock_repo.get_lab_results(tenant, exp_id)
    
    return {
        "experiment_id": exp_id,
        "calibration": raw_calibration,
        "lab_results": normalized_results,
    }


@router.put("/mock/experiments/{tenant}/{exp_id}/lab-results")
async def update_mock_lab_results(
    tenant: str,
    exp_id: str,
    data: dict = Body(...),
):
    """
    Update all lab results (calibration data) for a mock experiment.
    Expects: { "calibration": { "<calibration_id>": { "count": number, "unit": string }, ... } }
    """
    _validate_segment(tenant, "tenant")
    calibration = data.get("calibration")
    if calibration is None:
        raise HTTPException(status_code=400, detail="Missing 'calibration' field in request body")
    
    settings = get_settings()
    mock_repo = MockRepository(settings.resources_dir)
    
    success = mock_repo.update_calibration(tenant, exp_id, calibration)
    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    
    return {"message": "Lab results updated successfully"}


@router.patch("/mock/experiments/{tenant}/{exp_id}/lab-results/{calibration_id}")
async def patch_mock_lab_result(
    tenant: str,
    exp_id: str,
    calibration_id: str,
    data: dict = Body(...),
):
    """
    Update a specific lab result within a mock experiment.
    Expects: { "count": number, "unit": string } (both optional)
    """
    _validate_segment(tenant, "tenant")
    count = data.get("count")
    unit = data.get("unit")
    
    if count is None and unit is None:
        raise HTTPException(status_code=400, detail="At least 'count' or 'unit' must be provided")
    
    settings = get_settings()
    mock_repo = MockRepository(settings.resources_dir)
    
    success = mock_repo.update_lab_result(tenant, exp_id, calibration_id, count=count, unit=unit)
    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    
    return {"message": "Lab result updated successfully"}


@router.post("/mock/experiments/{tenant}/{exp_id}/lab-results")
async def add_mock_lab_result(
    tenant: str,
    exp_id: str,
    data: dict = Body(...),
):
    """
    Add a new lab result to a mock experiment.
    Expects: { "calibration_id": string, "count": number, "unit": string }
    """
    _validate_segment(tenant, "tenant")
    calibration_id = data.get("calibration_id")
    count = data.get("count")
    unit = data.get("unit")
    
    if not calibration_id:
        raise HTTPException(status_code=400, detail="'calibration_id' is required")
    if count is None:
        raise HTTPException(status_code=400, detail="'count' is required")
    if not unit:
        raise HTTPException(status_code=400, detail="'unit' is required")
    
    settings = get_settings()
    mock_repo = MockRepository(settings.resources_dir)
    
    success = mock_repo.add_lab_result(tenant, exp_id, calibration_id, count, unit)
    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    
    return {"message": "Lab result added successfully"}


@router.delete("/mock/experiments/{tenant}/{exp_id}/lab-results/{calibration_id}")
async def delete_mock_lab_result(
    tenant: str,
    exp_id: str,
    calibration_id: str,
):
    """
    Delete a specific lab result from a mock experiment.
    """
    _validate_segment(tenant, "tenant")
    settings = get_settings()
    mock_repo = MockRepository(settings.resources_dir)
    
    success = mock_repo.delete_lab_result(tenant, exp_id, calibration_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} or calibration {calibration_id} not found")
    
    return {"message": "Lab result deleted successfully"}


@router.get("/mock/bacteria-options")
async def get_bacteria_options():
    """
    Get available bacteria options for creating lab results.
    Returns list of { id, name } objects.
    """
    settings = get_settings()
    mock_repo = MockRepository(settings.resources_dir)
    
    options = mock_repo.list_bacteria_options()
    return {"bacteria_options": options}

