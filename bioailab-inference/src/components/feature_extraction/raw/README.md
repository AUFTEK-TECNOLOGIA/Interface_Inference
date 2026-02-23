# Raw Data Feature Extractor

Extrator simples e direto para dados brutos, sem pré-processamento ou ajuste de curva. Serve como fallback quando outros métodos falham.

## Visão Geral

Este extrator trabalha diretamente com os dados originais, aplicando apenas suavização leve e cálculo de derivadas numéricas básicas. Ideal para situações onde:

- Curve fitting falhou
- Análise rápida é necessária
- Dados têm muito ruído para modelos complexos
- Comparação direta entre sinais é desejada

## Features Extraídas

### Características Básicas
- **Amplitude**: Diferença simples entre valor final e inicial
- **Ponto de Inflexão**: Local do máximo da primeira derivada
- **Valor no Ponto de Inflexão**: Valor da curva na inflexão aproximada

### Análise de Derivadas
- **Pico da Primeira Derivada**: Máxima taxa de variação
  - Tempo e valor do ponto de máxima velocidade
- **Pico da Segunda Derivada**: Máxima aceleração
  - Tempo e valor do ponto de máxima aceleração

### Estatísticas Básicas
- **Área sob a Curva (AUC)**: Integral numérica simples
- **Valor Inicial/Final**: Pontos extremos da série

## Quando Usar

Ideal para:
- **Fallback**: Quando outros extratores falham
- **Análise Preliminar**: Verificação rápida de dados
- **Dados Ruidosos**: Onde ajuste de curva não é confiável
- **Comparação Direta**: Entre sinais sem normalização
- **Debugging**: Verificar se dados têm crescimento básico

## Vantagens

- **Sempre funciona**: Não depende de condições específicas
- **Rápido**: Cálculos mínimos e diretos
- **Robusto**: Tolera dados de baixa qualidade
- **Simples**: Fácil de entender e debugar

## Limitações

- **Aproximado**: Features são estimativas grosseiras
- **Menos informativo**: Não extrai parâmetros microbiológicos
- **Sensível a ruído**: Derivadas numéricas amplificam ruído
- **Baixa confiança**: Score de confiança reduzido (0.8)

## Parâmetros de Controle

- **Sigma de suavização**: Reduz ruído nas derivadas (padrão: adaptativo)
- **Suavização gaussiana**: Aplica filtro leve antes das derivadas

## Exemplo de Uso

```python
from components.feature_extraction.raw import RawDataExtractor

extractor = RawDataExtractor()
features = extractor.extract(time_data, raw_od_data)

print(f"Amplitude: {features.amplitude}")
print(f"Inflection time: {features.inflection_time} min")
print(f"AUC: {features.auc}")
print(f"Confidence: {features.confidence_score}")  # ~0.8
```