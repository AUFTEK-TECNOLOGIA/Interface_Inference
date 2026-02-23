# Filtro Mediano

## O que é

O filtro mediano substitui cada ponto pela mediana dos valores na janela ao redor, sendo excelente para remover ruído impulsivo (spikes) enquanto preserva bordas e transições abruptas.

## Tipos Disponíveis

### MedianFilter
- **Filtro mediano básico**: Janela deslizante com mediana
- **Parâmetros**: `window` (tamanho da janela)

### AdaptiveMedianFilter
- **Filtro adaptativo**: Ajusta tamanho da janela baseado no ruído local
- **Parâmetros**: `max_window`, `threshold`

## Quando usar

- Para remover outliers e spikes
- Quando preservar bordas é crítico
- Para sinais com ruído impulsivo
- Não recomendado para sinais suaves (pode introduzir saltos)

## Como usar

```python
import numpy as np
from src.components.signal_processing.filters.median import MedianFilter
from src.components.signal_processing.filters import BlockInput

# Dados com ruído impulsivo
data = np.column_stack([timestamps, sinal_ruidoso])
input_block = BlockInput(data=data)

# Aplicar filtro mediano
filter = MedianFilter(window=5)
output = filter.process(input_block)

print("Filtro aplicado:", output.success)
print("Metadados:", output.metadata)
```

## Parâmetros

### MedianFilter
- `window`: Tamanho da janela (ímpar, default=5)

### AdaptiveMedianFilter
- `max_window`: Tamanho máximo da janela
- `threshold`: Limite para adaptação

## Entradas e Saídas

- **Entradas**: `BlockInput` com dados 2D [x, y]
- **Saídas**: `BlockOutput` com sinal filtrado

## Exemplo

```
Janela: 5 pontos
Sinal ruidoso: [1, 2, 100, 4, 5, 6]  # spike em 100
Sinal filtrado: [1, 2, 4, 4, 5, 6]   # spike removido
```