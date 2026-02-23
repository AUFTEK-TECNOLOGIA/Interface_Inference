# BioAI Lab Inference - Visao detalhada dos 4 blocos

Este documento descreve, de forma detalhada, como os 4 blocos principais do sistema sao gerenciados no backend atual. Ele foi escrito para alinhamento de time e para servir de guia de navegacao no codigo.

## 0) Mapa rapido de onde tudo fica

- API FastAPI: `main.py` e `src/interface/router.py`
- Pipeline (engine, config e blocos): `src/components/pipeline/engine.py`, `src/components/pipeline/base.py`, `src/components/pipeline/blocks.py`
- Configuracao de pipeline por tenant (fallback): `src/infrastructure/config/tenant_loader.py`
- Persistencia de pipeline do Pipeline Studio: `resources/<tenant>/pipeline/<pipeline>.json` e `resources/<tenant>/pipeline/versions/*.json`
- Datasets de treino: `resources/<tenant>/datasets/*.json`
- Artefatos treinados: `resources/<tenant>/predict/trained/...`
- Treinamento ML e export ONNX: `src/infrastructure/ml/training.py`
- Simulacao GUI: endpoints `POST /pipelines/simulate` e `GET /pipelines/blocks`

## 1) Criacao e manutencao de datasets

### 1.1 Objetivo do bloco

Este bloco e responsavel por:
- listar dados disponiveis (protocolos, experimentos, analysisIds)
- gerar previews graficos dos experimentos
- permitir curadoria (good/bad) e selecao de experimentos
- persistir datasets de treino por tenant

### 1.2 Endpoints envolvidos

Local: `src/interface/router.py` (secao "Datasets de Treinamento")

- `GET /datasets/analysis-ids/{tenant}`
  - Lista analysisIds com dados (mongo e/ou mock)
  - Parametro `source=all|mongo|mock`

- `GET /datasets/protocols/{tenant}`
  - Lista protocolIds e quantidade de experimentos
  - Parametros: `limit`, `source`

- `GET /datasets/experiments/{tenant}/{protocol_id}`
  - Lista experimentos com resumo (tem lab_results, qtd de pontos, labels)
  - Parametros: `limit`, `source`

- `GET /datasets/{tenant}`
  - Lista datasets salvos em `resources/<tenant>/datasets/`

- `GET /datasets/{tenant}/{dataset_id}`
  - Retorna dataset completo (inclui lista de experiment_ids)

- `POST /datasets/{tenant}`
  - Cria/atualiza dataset
  - Campos principais: `name`, `protocol_id`, `experiment_ids`, `viewed_ids`, `ratings`, `description`, `dataset_id`, `source`

- `DELETE /datasets/{tenant}/{dataset_id}`
  - Remove dataset salvo

- `GET /datasets/preview/{tenant}/{experiment_id}`
  - Gera previews de graficos por sensor (turbidimetry, nephelometry, fluorescence)
  - Usa Matplotlib e retorna PNG base64 + resumo do experimento/lab_results

### 1.3 Persistencia e formato do dataset

Local: `resources/<tenant>/datasets/<dataset_id>.json`

Campos salvos:
- `id`, `name`, `description`
- `protocol_id`
- `source` (mongo|mock)
- `experiment_ids` (lista de experimentos selecionados)
- `viewed_ids` (experimentos ja vistos pela UI)
- `ratings` (objeto `{expId: "good"|"bad"}`)
- `created_at`, `updated_at`

### 1.4 Origem dos dados (Mongo vs Mock)

- Repositorios: `MongoRepository` e `MockRepository`
- Configuracao via `.env` (`MONGO_URI`, `TENANT_DB_PREFIX`)
- Quando `source=mock`, dados vem de `resources/` (modo local)

## 2) Criacao e execucao do fluxo (pipeline) via blocos

### 2.1 Objetivo do bloco

Este bloco define e executa pipelines modulares:
- um pipeline e uma sequencia de steps (blocos)
- cada bloco processa um tipo de dado e entrega outro
- o pipeline pode vir de duas fontes:
  - JSON salvo no Pipeline Studio (workspace)
  - fallback: configuracao interna do tenant

### 2.2 Onde o pipeline e definido

- Pipeline Studio salva em:
  - `resources/<tenant>/pipeline/<pipeline>.json` (arquivo ativo)
  - `resources/<tenant>/pipeline/versions/*.json` (versoes)
  - manifest: `resources/<tenant>/pipeline/_versions.json`

- Configuracao fallback (legado):
  - `src/infrastructure/config/tenant_loader.py`

### 2.3 Endpoints do pipeline

Local: `src/interface/router.py`

- `GET /pipelines/blocks`
  - Lista blocos disponiveis (via `BlockRegistry`)

- `GET /pipelines/library`
  - Biblioteca completa de blocos, filtros e detectores (catalogo para UI)

- `POST /pipelines/execute`
  - Executa o pipeline ativo do tenant
  - Preferencia: pipeline salvo no workspace
  - Fallback: pipeline gerado do tenant.json

### 2.4 Como o pipeline e montado e executado

- Builder do pipeline a partir de JSON do workspace:
  - `_build_workspace_pipeline_config()` em `src/interface/router.py`
  - Converte `execution.steps` em `PipelineStep`

- Engine principal:
  - `PipelineEngine` em `src/components/pipeline/engine.py`
  - Executa steps, respeita `depends_on`, paralelismo e timeout

- Blocos:
  - Registrados em `BlockRegistry` (`src/components/pipeline/base.py`)
  - Implementacoes em `src/components/pipeline/blocks.py`

### 2.5 Estrutura basica de um step

Cada step no JSON do Pipeline Studio tem:
- `step_id`: identificador unico
- `block_name`: nome do bloco registrado
- `block_config`: configuracoes do bloco
- `depends_on`: lista de steps anteriores
- `input_mapping`: mapeia entradas do bloco para saidas anteriores

### 2.6 Saidas e resposta final

- O resultado do pipeline tem `step_results` por step
- Normalmente a resposta da API vem do bloco `response_builder`
- Se nao existir, o endpoint tenta o primeiro step com campo `response`

## 3) Simulacao do fluxo

### 3.1 Objetivo do bloco

Permite simular um pipeline sem salva-lo, diretamente pela UI.
Isso e usado para depuracao visual no Pipeline Studio.

### 3.2 Endpoint de simulacao

- `POST /pipelines/simulate` em `src/interface/router.py`

Entrada (schemas em `src/interface/schemas.py`):
- `name`, `description`
- `steps` (lista de `PipelineStep` declarativos)
- `initial_state` (estado inicial do pipeline)
- configuracoes como `max_parallel`, `timeout_seconds`, `fail_fast`

Saida:
- `pipeline_id`, `success`, `duration_ms`, `errors`
- `step_results` (dados serializados por step)
- `steps` (metadata por step: status, duracao, erro)

### 3.3 Simulacao local de tenants (modo dev)

Existe suporte para simular pipelines gerados por tenant:
- `src/components/pipeline/tenant_pipeline_builder.py`
- Metodo `simulate_prediction` executa o pipeline localmente

## 4) Treinamento de blocos de ML

### 4.1 Objetivo do bloco

Treina modelos supervisionados usando:
- features extraidas pelo pipeline
- lab_results como target (y)

Ao final, o sistema gera artefatos e opcionalmente atualiza o pipeline.

### 4.2 Endpoint principal de treino

- `POST /pipelines/train` em `src/interface/router.py`

Fluxo resumido:
1. Carrega o pipeline (versao ativa ou versao passada no request)
2. Monta `PipelineConfig`
3. Identifica steps treinaveis (`ml_inference`, `ml_inference_series`, `ml_inference_multichannel`)
4. Opcionalmente inclui forecasters (`ml_forecaster_series`)
5. Prefiltra experimentos (metadados apenas)
6. Executa pipeline com cache por experimento
7. Monta datasets por step e treina
8. Salva modelos e metadata
9. Atualiza pipeline (se `apply_to_pipeline=true`)
10. Cria nova versao e ativa (se aplicavel)

### 4.3 Otimizacao de treinamento (cache e prefiltragem)

Detalhado em `docs/OTIMIZACAO_TREINAMENTO.md`:
- `_analyze_ml_block_requirements()`: analisa requisitos do bloco
- `_prefilter_experiments_for_training()`: filtra por lab_results/diluicao/etc
- `_execute_experiments_with_cache()`: executa pipeline uma unica vez por experimento

### 4.4 Artefatos gerados

- Treino/export e feito por `train_regressor_export_onnx()` em `src/infrastructure/ml/training.py`
- Artefatos salvos em:
  - `resources/<tenant>/predict/trained/<label>/<unit>/<step_id>/`
  - `*_model.onnx`
  - `*_scaler.joblib`
  - `*_metadata.json`

O `metadata.json` guarda informacoes criticas:
- algoritmo, params e metricas
- transformacao do y (ex: log10p)
- configuracoes do bloco no momento do treino

### 4.5 Atualizacao do pipeline com os modelos

Quando `apply_to_pipeline=true`, o pipeline e atualizado com:
- `model_path`, `scaler_path`
- `metadata_path`
- `y_transform`
- configs especificas (ex: input_channel, output_unit, window/horizon para forecaster)

Isso e feito em `_update_workspace_pipeline_block_config()`.

### 4.6 Sessao de treino e orquestracao

Endpoints adicionais:
- `POST /training/sessions` (cria sessao)
- `GET /training/sessions` (lista)
- `GET /training/sessions/{session_id}` (detalhes)
- `GET /training/sessions/{session_id}/dependencies` (grafo de dependencia)

Esses endpoints usam `src/infrastructure/ml/training_orchestrator.py`.

### 4.7 Grid-search com candidatos

- `POST /training/grid-search`
  - Treina varios candidatos e salva todos
  - Usa cache v2 por experimento

- `GET /training/candidates/{tenant}/{session_id}`
- `POST /training/select-candidate`

A selecao atualiza o pipeline com o candidato escolhido.

## 5) Relacao entre os 4 blocos (visao end-to-end)

1. Usuario cria/curadoria dataset
   - endpoints `/datasets/*`
   - salva lista de experimentos e ratings

2. Pipeline e montado no Pipeline Studio
   - UI consome `/pipelines/blocks` e `/pipelines/simulate`
   - JSON salvo em `resources/<tenant>/pipeline/<pipeline>.json`

3. Simulacao valida o fluxo
   - `/pipelines/simulate` roda o pipeline em memoria
   - retorna outputs por step e feedback visual

4. Treinamento usa o dataset
   - `/pipelines/train` recebe experimentIds (dataset)
   - executa pipeline, coleta features e lab_results
   - gera modelos, salva artefatos e atualiza pipeline

## 6) Pontos de atencao operacionais

- Variaveis de ambiente obrigatorias:
  - `MONGO_URI`, `TENANT_DB_PREFIX`, `SPECTRAL_API_URL`, `TOKEN`
- O pipeline pode falhar se:
  - blocos esperados nao existem no JSON
  - dados de experimento/lab_results estao incompletos
- A simulacao e segura (nao escreve nada), mas o treino grava artefatos em `resources/`

## 7) Referencias diretas no codigo

- Datasets: `src/interface/router.py`
- Pipeline engine: `src/components/pipeline/engine.py`
- Blocos: `src/components/pipeline/blocks.py`
- Config de pipeline do workspace: `src/interface/router.py`
- Treinamento ML: `src/infrastructure/ml/training.py`
- Otimizacao do treino: `docs/OTIMIZACAO_TREINAMENTO.md`

---

Se quiser, eu tambem posso incluir um diagrama (ASCII ou Mermaid) mostrando as dependencias entre os blocos e os arquivos de persistencia.
