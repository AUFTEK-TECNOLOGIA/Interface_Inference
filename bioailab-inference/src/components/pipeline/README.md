# Pipeline Framework

Framework modular para criação de pipelines de processamento baseados em blocos conectáveis.

## Visão Geral

O Pipeline Framework permite construir fluxos de processamento complexos conectando blocos independentes. Cada bloco representa uma unidade de processamento específica que pode ser testada, reutilizada e combinada com outros blocos.

## Conceitos Principais

### Blocos (Blocks)
Unidades básicas de processamento que:
- Recebem dados de entrada padronizados
- Executam processamento específico
- Produzem dados de saída padronizados
- São independentes e reutilizáveis

### Pipeline Engine
Orquestrador que:
- Executa blocos em sequência ou paralelo
- Gerencia dependências entre blocos
- Trata erros e recupera de falhas
- Fornece métricas de performance

### Contexto de Execução
Informações compartilhadas entre blocos:
- ID do pipeline e bloco
- Tempos de execução
- Status de sucesso/falha
- Metadados customizados

## Blocos Disponíveis

### GrowthDetectionBlock
**Propósito**: Detecta presença de crescimento microbiano
- **Entrada**: Dados do sensor (timestamps, valores)
- **Saída**: Boolean has_growth + detalhes da detecção

### CurveFittingBlock
**Propósito**: Ajusta modelos matemáticos aos dados
- **Entrada**: Dados do sensor + resultado da detecção
- **Saída**: Resultado do ajuste + dados fitted

### FeatureExtractionBlock
**Propósito**: Extrai features dos dados ajustados
- **Entrada**: Dados fitted + nome do extrator
- **Saída**: Features extraídas + metadados

### MLInferenceBlock
**Propósito**: Executa inferência com modelo de ML
- **Entrada**: Features + caminho do modelo
- **Saída**: Predição + confiança

## Exemplo de Uso

```python
from components.pipeline import (
    PipelineEngine, PipelineConfig, PipelineStep,
    GrowthDetectionBlock, CurveFittingBlock, FeatureExtractionBlock
)

# Configurar pipeline
config = PipelineConfig(
    name="bio_inference_pipeline",
    description="Pipeline completo de inferência microbiológica",
    steps=[
        PipelineStep(
            block_name="growth_detection",
            step_id="detect_growth"
        ),
        PipelineStep(
            block_name="curve_fitting",
            step_id="fit_curve",
            depends_on=["detect_growth"]
        ),
        PipelineStep(
            block_name="feature_extraction",
            step_id="extract_features",
            depends_on=["fit_curve"],
            block_config={"extractor_name": "microbial"}
        )
    ]
)

# Criar engine
engine = PipelineEngine(config)

# Dados de entrada
input_data = {
    "sensor_data": {
        "timestamps": [0, 60, 120, 180, 240, 300],
        "values": [0.1, 0.15, 0.3, 0.8, 1.2, 1.3]
    }
}

# Executar pipeline
result = engine.execute(input_data)

if result.success:
    print("Pipeline executado com sucesso!")
    print(f"Duração: {result.duration_ms:.2f}ms")

    # Acessar resultados
    features = result.step_results["extract_features"].data["features"]
    print(f"Features extraídas: {features}")
else:
    print("Pipeline falhou:")
    for error in result.errors:
        print(f"  - {error}")
```

## Configuração Declarativa

Pipelines podem ser definidos via configuração:

```python
config = PipelineConfig(
    name="my_pipeline",
    steps=[
        {
            "block_name": "growth_detection",
            "step_id": "step1"
        },
        {
            "block_name": "curve_fitting",
            "step_id": "step2",
            "depends_on": ["step1"],
            "block_config": {"model_preference": ["baranyi"]}
        }
    ],
    max_parallel=2,
    timeout_seconds=60.0,
    fail_fast=True
)
```

## Tratamento de Erros

O framework inclui robusto tratamento de erros:

- **Fail Fast**: Parar execução no primeiro erro
- **Timeout**: Limite de tempo para execução total
- **Recuperação**: Continuar execução mesmo com falhas parciais
- **Logging**: Rastreamento completo de execução

## Extensibilidade

### Criando Novos Blocos

```python
from components.pipeline import Block, BlockInput, BlockOutput, BlockRegistry

@BlockRegistry.register
class MyCustomBlock(Block):
    name = "my_custom"
    description = "Meu bloco personalizado"

    def execute(self, input_data: BlockInput) -> BlockOutput:
        # Processamento customizado
        result = input_data.get_required("my_input") * 2

        return BlockOutput(
            data={"my_output": result},
            context=input_data.context
        )
```

### Registrando Blocos

Blocos são automaticamente registrados com o decorator `@BlockRegistry.register`.

## Monitoramento e Observabilidade

Cada execução gera métricas detalhadas:
- Tempo de execução por bloco
- Status de sucesso/falha
- Dependências executadas
- Metadados customizados

## Benefícios

- **Modularidade**: Componentes independentes e testáveis
- **Reutilização**: Blocos podem ser usados em múltiplos pipelines
- **Manutenibilidade**: Mudanças isoladas não quebram outros blocos
- **Escalabilidade**: Execução paralela e distribuída
- **Observabilidade**: Rastreamento completo de execução