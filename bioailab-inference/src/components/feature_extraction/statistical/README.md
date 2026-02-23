# Statistical Feature Extractor

Extrator que usa análise numérica estatística para identificar características de crescimento sem depender de modelos matemáticos pré-definidos.

## Visão Geral

Este extrator aplica técnicas estatísticas puras aos dados brutos:
- Suavização gaussiana para reduzir ruído
- Cálculo numérico de derivadas
- Análise de pontos críticos baseada em estatística

Ideal quando não se quer assumir nenhum modelo específico de crescimento.

## Features Extraídas

### Características Geométricas
- **Amplitude**: Diferença total entre máximo e mínimo
- **Ponto de Inflexão**: Local de mudança de concavidade
- **Tempo do Ponto de Inflexão**: Quando ocorre a inflexão
- **Valor no Ponto de Inflexão**: Valor da curva na inflexão

### Análise de Derivadas
- **Pico da Primeira Derivada**: Máxima taxa de variação
  - Tempo e valor do pico de velocidade
- **Pico da Segunda Derivada**: Máxima aceleração
  - Tempo e valor do pico de aceleração

### Estatísticas Básicas
- **Área sob a Curva (AUC)**: Integral numérica da curva
- **Valor Inicial/Final**: Pontos extremos da série

## Quando Usar

Ideal para:
- Dados brutos sem ajuste de curva prévio
- Análise exploratória de dados
- Quando modelos específicos não se aplicam
- Comparação estatística entre curvas
- Dados com características não-sigmoidais

## Vantagens

- **Model-free**: Não assume forma específica da curva
- **Robusto**: Funciona com dados ruidosos
- **Flexível**: Adequado para qualquer tipo de crescimento
- **Estatístico**: Baseado em propriedades numéricas puras

## Limitações

- Requer dados com derivadas bem definidas
- Sensível a parâmetros de suavização
- Menos preciso que extratores baseados em modelos
- Não extrai parâmetros microbiológicos específicos

## Parâmetros de Controle

- **Sigma de suavização**: Controla o nível de suavização gaussiana
- **Thresholds**: Critérios para detecção de pontos críticos

## Exemplo de Uso

```python
from components.feature_extraction.statistical import StatisticalFeatureExtractor

extractor = StatisticalFeatureExtractor()
features = extractor.extract(time_data, raw_data)

print(f"Amplitude: {features.amplitude}")
print(f"Inflection time: {features.inflection_time} min")
print(f"First derivative peak: {features.first_derivative_peak_value}")
print(f"AUC: {features.auc}")
```