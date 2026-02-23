# Modelo de Richards

## O que é

O modelo de Richards é uma generalização do modelo logístico que permite maior flexibilidade na forma da curva sigmoide. É especialmente útil para descrever processos de crescimento com diferentes graus de assimetria.

## Características

- **Flexibilidade**: Generalização do modelo logístico
- **Assimetria ajustável**: Dependendo dos parâmetros
- **Quatro parâmetros**: Permite mais controle sobre a forma
- **Aplicações diversas**: Crescimento biológico, adoção tecnológica

## Quando usar

- Para curvas sigmoides com assimetria variável
- Quando o logístico simples não se ajusta bem
- Em estudos de crescimento complexos
- Para modelar processos com saturação não-simétrica

## Como usar

```python
import numpy as np
from src.components.signal_processing.curve_fitting.richards import RichardsModel
from src.components.signal_processing.curve_fitting import BlockInput

# Dados de crescimento
data = np.column_stack([tempo, valor])
input_block = BlockInput(data=data)

# Ajustar modelo completo
model = RichardsModel()
result = model.fit(input_block)

print("Parâmetros ajustados:")
print(f"A (amplitude): {result.parameters['A']}")
print(f"K (taxa): {result.parameters['K']}")
print(f"T (inflexão): {result.parameters['T']}")
print(f"M (forma): {result.parameters['M']}")
print(f"R²: {result.r_squared}")
```

## Parâmetros

- `A`: Amplitude (assíntota superior)
- `K`: Taxa de crescimento
- `T`: Tempo do ponto de inflexão
- `M`: Parâmetro de forma (controla assimetria)

## Fórmula Geral

```
y = A / (1 + exp(K * (T - x)))^(1/M)
```

**Casos especiais:**
- M = 1: Equivalente ao modelo logístico
- M → ∞: Aproxima o modelo de Gompertz
- M < 1: Crescimento mais rápido no início
- M > 1: Crescimento mais lento no início

## Interpretação do Parâmetro M

- **M = 1**: Curva simétrica (equivalente ao logístico)
- **M < 1**: Crescimento mais rápido no início, mais lento no final
- **M > 1**: Crescimento mais lento no início, mais rápido no final
- **M → 0**: Aproxima função degrau
- **M → ∞**: Aproxima modelo de Gompertz (assimétrico)

## Derivadas

### Primeira derivada (taxa de crescimento)
```
dy/dx = A * (K/M) * exp(K*(T-x)) / (1 + exp(K*(T-x)))^(1/M + 1)
```

### Segunda derivada
Calculada numericamente devido à complexidade analítica.

## Entradas e Saídas

- **Entradas**: `BlockInput` com dados 2D (tempo, valor)
- **Saídas**: `BlockOutput` com parâmetros e métricas de ajuste

## Vantagens

- **Flexibilidade**: Pode modelar diferentes formas sigmoides
- **Generalização**: Inclui logístico e Gompertz como casos especiais
- **Aplicações amplas**: Biologia, economia, tecnologia
- **Parâmetros interpretáveis**: Cada parâmetro tem significado claro

## Comparação com outros modelos

- **vs Logístico**: Richards permite assimetria (quando M ≠ 1)
- **vs Gompertz**: Richards é mais geral
- **vs Baranyi**: Richards foca na forma da curva, não no lag

## Aplicações

- Crescimento populacional complexo
- Difusão de inovações
- Crescimento tumoral
- Desenvolvimento econômico
- Adoção de tecnologias