# Fitted Curve Feature Extractor

Extrator especializado para curvas que passaram por ajuste matemático usando modelos como Richards, Gompertz, Logistic ou Baranyi.

## Visão Geral

Este extrator trabalha com dados que já foram suavizados e ajustados por modelos matemáticos. Foca em características geométricas puras da curva resultante, sem fazer suposições sobre o significado biológico dos parâmetros.

## Features Extraídas

### Características Geométricas
- **Amplitude**: Diferença total entre valor final e inicial
- **Ponto de Inflexão**: Local onde a curvatura da curva muda
- **Tempo do Ponto de Inflexão**: Quando ocorre a mudança de concavidade
- **Valor no Ponto de Inflexão**: Valor da curva no ponto de inflexão

### Derivadas
- **Pico da Primeira Derivada**: Máxima taxa de variação (tempo e valor)
- **Pico da Segunda Derivada**: Máxima aceleração/desaceleração (tempo e valor)

### Estatísticas Básicas
- **Área sob a Curva (AUC)**: Integral da curva completa
- **Valor Inicial/Final**: Pontos extremos da série temporal

## Quando Usar

Ideal para:
- Dados que passaram por curve fitting (Richards, Gompertz, Logistic, Baranyi)
- Curvas suavizadas com modelos matemáticos
- Análise geométrica pura sem interpretação biológica
- Comparação entre diferentes ajustes de curva

## Vantagens

- **Robusto**: Funciona bem com dados ruidosos após ajuste
- **Geométrico**: Foca em propriedades matemáticas da curva
- **Flexível**: Adequado para qualquer forma de curva sigmoidal
- **Rápido**: Não requer cálculos microbiológicos complexos

## Limitações

- Não extrai parâmetros microbiológicos específicos (lag time, μmax, etc.)
- Depende da qualidade do ajuste prévio da curva
- Menos informativo biologicamente que extratores especializados

## Exemplo de Uso

```python
from components.feature_extraction.fitted import FittedCurveExtractor

extractor = FittedCurveExtractor()
features = extractor.extract(time_data, fitted_curve_data)

print(f"Amplitude: {features.amplitude}")
print(f"Inflection time: {features.inflection_time} min")
print(f"Inflection value: {features.inflection_value}")
print(f"AUC: {features.auc}")
```