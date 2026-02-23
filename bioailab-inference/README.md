# bioailab-inference

Sistema de inferencia para analise microbiologica usando dados espectrais.

## Execucao da API

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

A API fica disponivel em http://localhost:8001 e exposta pelo FastAPI definido em `main.py`.

## Configuracao

1. Copie o arquivo de exemplo: `cp .env.example .env`
2. Ajuste as variaveis conforme o ambiente (URL espectral, MongoDB e tenant).
3. Carregue as variaveis antes de iniciar a API (`set VAR=valor` no Windows ou `export VAR=valor` no Linux/macOS).

### Variaveis de ambiente principais

- `SPECTRAL_API_URL`: endpoint usado para conversao espectral (padrao `https://spectral.bioailab.com.br/convert`).
- `MONGO_URI`: cadeia de conexao com o MongoDB usado para buscar os dados experimentais.
- `TENANT_DB_PREFIX`: prefixo aplicado para montar o banco de cada tenant.
- `TOKEN`: valor utilizado para autenticacao basica interna (padrao `1337` se nao definido).

## Configuracao da API espectral

O pipeline usa o servico espectral para converter as leituras. Para alterar o destino basta definir:

```bash
export SPECTRAL_API_URL="https://sua-api-espectral.com/convert"
```

## Carregamento de dados

O pipeline consulta automaticamente o MongoDB:

1. Busca dados reais da colecao `data_analise` filtrando por `experiment_id`.
2. Quando necessario, gera dados simulados para manter o fluxo de teste.

### Configuracao do MongoDB

```bash
# URI padrao
mongodb+srv://golang:s0meyLEQavWmUBmx@iot.2ypazq2.mongodb.net/?retryWrites=true&w=majority&appName=iot

# Prefixo padrao
bioailab_
```

Para usar outra instancia:

```bash
export MONGO_URI="sua_uri_mongodb"
export TENANT_DB_PREFIX="seu_prefixo_"
```

## Arquitetura do pipeline

1. **data_extraction**: coleta dados do experimento.
2. **preprocessing**: normaliza e prepara os sinais.
3. **spectral_conversion**: executa a conversao espectral.
4. **signal_filters**: aplica filtros de sinal.
5. **growth_detection**: identifica crescimento microbiano.
6. **curve_fitting**: ajusta curvas (Baranyi, Gompertz, etc.).
7. **feature_extraction**: extrai caracteristicas microbiologicas.
8. **ml_inference**: roda os modelos de machine learning.

## Desenvolvimento

1. Instale as dependencias com `pip install -r requirements.txt`.
2. Execute os testes automatizados (`pytest` ou scripts em `test_*.py`).
3. Aplique suas alteracoes e rode o servidor localmente com `uvicorn main:app --reload`.

## Simulador e gerador de tenants

- O módulo `src/components/pipeline/tenant_pipeline_builder.py` monta pipelines em blocos (n8n-like) a partir de cada prediction configurada no `tenants.json`.
- Para gerar o JSON declarativo use `python generate_tenant_pipeline.py --tenant <nome> --output pipeline_configs/<nome>_pipeline.json`.
- O builder expõe o método `simulate_prediction` para executar o pipeline completo localmente (útil para mockar tenants antes de subir a API).
- Existe também a GUI React em `apps/pipeline-studio`, baseada em React Flow. Para rodar:
  ```bash
  cd apps/pipeline-studio
  cp .env.example .env   # Ajuste REACT_APP_API_URL se necessário
  npm install
  npm run dev
  ```
  Ela consome `/pipelines/blocks` e `/pipelines/simulate` da API, permitindo montar blocos visualmente, simular e exportar o JSON para alimentar os tenants.
