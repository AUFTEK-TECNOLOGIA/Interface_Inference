# Filtro Exponencial

## O que é

Os filtros exponenciais aplicam suavização com resposta exponencial, dando mais peso a pontos recentes. São filtros IIR eficientes para tempo real.

## Tipos Disponíveis

### ExponentialMovingAverageFilter (EMA)
- **Média móvel exponencial**: Peso exponencial decrescente
- **Parâmetros**: `alpha` (fator de suavização)

### DoubleExponentialFilter
- **Suavização dupla**: Para tendências lineares
- **Parâmetros**: `alpha`, `beta`

### TripleExponentialFilter (Holt-Winters)
- **Suavização tripla**: Para tendências e sazonalidade
- **Parâmetros**: `alpha`, `beta`, `gamma`

## Quando usar

- Para processamento em tempo real
- Quando pontos recentes são mais importantes
- Para sinais com tendências
- Quando memória é limitada (fácil de implementar)

## Como usar

```python
import numpy as np
from src.components.signal_processing.filters.exponential import ExponentialMovingAverageFilter
from src.components.signal_processing.filters import BlockInput

# Dados de série temporal
data = np.column_stack([timestamps, valores])
input_block = BlockInput(data=data)

# Aplicar EMA
filter = ExponentialMovingAverageFilter(alpha=0.1)
output = filter.process(input_block)

print("Suavização aplicada:", output.success)
print("Alpha:", 0.1)
```

## Parâmetros

### ExponentialMovingAverageFilter
- `alpha`: Fator de suavização (0-1, default=0.1)
  - Alto alpha: menos suavização, mais responsivo
  - Baixo alpha: mais suavização, mais lento

### DoubleExponentialFilter
- `alpha`: Para nível (default=0.1)
- `beta`: Para tendência (default=0.1)

### TripleExponentialFilter
- `alpha`: Para nível
- `beta`: Para tendência
- `gamma`: Para sazonalidade

## Fórmula EMA

```
EMA[0] = data[0]
EMA[t] = alpha * data[t] + (1 - alpha) * EMA[t-1]
```

## Entradas e Saídas

- **Entradas**: `BlockInput` com dados 2D
- **Saídas**: `BlockOutput` com sinal suavizado

## Vantagens

- **Leve**: Pouca memória e computação
- **Causal**: Usa apenas dados passados
- **Adaptável**: Alpha controla responsividade
- **Tempo real**: Pode processar ponto a ponto

## Exemplo

```
Sinal ruidoso: [1.0, 1.1, 0.9, 1.2, 0.8, 1.3]
Alpha: 0.2
EMA: [1.0, 1.02, 0.988, 1.03, 0.964, 1.05]
```