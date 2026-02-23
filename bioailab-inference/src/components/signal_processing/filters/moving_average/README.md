# Filtro de Média Móvel

## O que é

O filtro de média móvel calcula a média de uma janela deslizante de pontos do sinal, suavizando variações rápidas e preservando tendências gerais. É um dos filtros mais simples e eficazes para remover ruído de alta frequência.

## Tipos Disponíveis

### MovingAverageFilter (SMA)
- **Média Móvel Simples**: Cada ponto é substituído pela média aritmética dos pontos na janela
- **Alinhamento**: `center` (padrão), `left` (causal), `right` (anti-causal)

### WeightedMovingAverageFilter (WMA)
- **Média Móvel Ponderada**: Pontos mais recentes têm maior peso
- **Pesos**: Linear decrescente por padrão, customizável

## Quando usar

- Para suavizar sinais ruidosos
- Quando velocidade de processamento é prioridade
- Para dados com tendências lineares
- Não recomendado para sinais com bordas ou mudanças abruptas

## Como usar

```python
import numpy as np
from src.components.signal_processing.filters.moving_average import MovingAverageFilter
from src.components.signal_processing.filters import BlockInput

# Preparar dados
data = np.column_stack([timestamps, valores_sinal])  # Shape (n, 2)
input_block = BlockInput(data=data, metadata={"source": "sensor"})

# Aplicar filtro de média móvel
filter = MovingAverageFilter(window=5, alignment="center")
output = filter.process(input_block)

print("Dados originais:", input_block.data.shape)
print("Dados filtrados:", output.data.shape)
print("Sucesso:", output.success)
```

## Parâmetros

### MovingAverageFilter
- `window`: Tamanho da janela (int, default=5)
- `alignment`: Alinhamento da janela ("center", "left", "right")

### WeightedMovingAverageFilter
- `window`: Tamanho da janela (int)
- `weights`: Lista de pesos (opcional, default=linear crescente)

## Entradas e Saídas

- **Entradas**: `BlockInput` com `data` (array 2D shape (n, 2) para [x, y])
- **Saídas**: `BlockOutput` com `data` filtrado, `metadata` com params do filtro

## Exemplo de Saída

```
Dados originais: (100, 2)
Dados filtrados: (100, 2)
Sucesso: True
Metadados: {'filter_applied': 'moving_average', 'filter_params': {'window': 5, 'alignment': 'center'}}
```