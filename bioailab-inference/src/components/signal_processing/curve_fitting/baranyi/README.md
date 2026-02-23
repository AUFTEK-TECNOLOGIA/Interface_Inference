# Modelo de Baranyi-Roberts

## O que é

O modelo de Baranyi-Roberts é um modelo avançado para curvas de crescimento microbiano que modela explicitamente a fase lag (período de adaptação inicial). É especialmente útil em microbiologia preditiva.

## Características

- **Fase lag explícita**: Modela o período de adaptação inicial
- **Quatro parâmetros**: Permite controle preciso do crescimento
- **Transições suaves**: Entre todas as fases de crescimento
- **Microbiologia**: Padrão em modelagem de crescimento bacteriano

## Parâmetros

- `y0`: Log da população inicial
- `ymax`: Log da população máxima (assíntota superior)
- `mu_max`: Taxa específica de crescimento máximo
- `h0`: Parâmetro relacionado ao estado fisiológico inicial (controla a duração da fase lag)

## Equação

```
y = y0 + μmax * F(t) - ln(1 + (exp(μmax * F(t)) - 1) / exp(ymax - y0))
```

Onde a função de ajuste F(t) é:
```
F(t) = t + (1/μmax) * ln(exp(-μmax*t) + exp(-h0) - exp(-μmax*t - h0))
```

## Quando usar

- Para curvas de crescimento microbiano com fase lag significativa
- Em microbiologia preditiva e controle de qualidade
- Quando o modelo logístico simples não captura a fase lag
- Para estudos detalhados de crescimento populacional

## Como usar

```python
import numpy as np
from src.components.signal_processing.curve_fitting.baranyi import BaranyiModel
from src.components.signal_processing.curve_fitting import BlockInput

# Dados de crescimento bacteriano (escala logarítmica)
data = np.column_stack([tempo_horas, log_populacao])
input_block = BlockInput(data=data)

# Ajustar modelo
model = BaranyiModel()
result = model.fit(input_block)

print("Parâmetros ajustados:")
print(f"y0 (população inicial): {result.parameters['y0']}")
print(f"ymax (população máxima): {result.parameters['ymax']}")
print(f"mu_max (taxa máxima): {result.parameters['mu_max']}")
print(f"h0 (parâmetro lag): {result.parameters['h0']}")
print(f"R²: {result.r_squared}")
```

## Interpretação dos Parâmetros

- **y0**: Ponto de partida do crescimento (normalmente baixo)
- **ymax**: Capacidade de suporte máxima atingida
- **mu_max**: Velocidade máxima de crescimento (fase exponencial)
- **h0**: Controla duração da fase lag (valores maiores = lag mais longo)

## Entradas e Saídas

- **Entradas**: `BlockInput` com dados 2D (tempo, log_população)
- **Saídas**: `BlockOutput` com parâmetros ajustados e métricas

## Vantagens

- **Fase lag realista**: Melhor representação do crescimento microbiano
- **Parâmetros interpretáveis**: Significado biológico claro
- **Flexibilidade**: Adequado para diferentes condições
- **Precisão**: Melhor ajuste para dados microbiológicos

## Comparação com outros modelos

- **vs Logístico**: Inclui fase lag explícita
- **vs Gompertz**: Transições mais suaves, fase lag modelada
- **vs Richards**: Focado especificamente em crescimento microbiano

## Aplicações

- Crescimento microbiano em alimentos
- Microbiologia preditiva
- Controle de qualidade de alimentos
- Estudos de crescimento bacteriano
- Modelagem de fermentação