# Normalizador MinMax

## O que é

O Normalizador MinMax é usado para **escalar dados para o intervalo [0, 1]**. Ele subtrai o valor mínimo e divide pela amplitude (máximo - mínimo), garantindo que todos os valores fiquem entre 0 e 1. É útil quando você quer preservar a distribuição relativa dos dados e garantir valores positivos.

## Quando usar

- Para algoritmos que esperam valores no intervalo [0, 1]
- Quando a distribuição relativa dos dados é importante
- Para dados sem outliers extremos

## Como usar

```python
import numpy as np
from src.components.signal_processing.normalizers.minmax import MinMaxNormalizer
from src.components.signal_processing.normalizers import BlockInput

# Preparar dados
data = np.column_stack([timestamps, valores_sinal])  # Shape (n, 2)
input_block = BlockInput(data=data, metadata={"source": "sensor"})

# Processar
normalizer = MinMaxNormalizer()
output = normalizer.process(input_block)

print("Dados originais:", input_block.data)
print("Dados normalizados:", output.data)
print("Sucesso:", output.success)
print("Metadados:", output.metadata)
```

## Entradas e Saídas

- **Entradas**: `BlockInput` com `data` (array 2D shape (n, 2) para [x, y]) e `metadata` (opcional)
- **Saídas**: `BlockOutput` com `data` (array 2D normalizado), `metadata` (com params), `success`, `error`

### Exemplo de Saída (Blocos)
```
Dados originais: [[ 1 10]
                  [ 2 20]
                  [ 3 30]
                  [ 4 40]
                  [ 5 50]]
Dados normalizados: [[0.   0.  ]
                     [0.25 0.25]
                     [0.5  0.5 ]
                     [0.75 0.75]
                     [1.   1.  ]]
Sucesso: True
Metadados: {'normalization_params': {'method': 'minmax', 'x_params': {'min': 1.0, 'max': 5.0, 'range': 4.0}, 'y_params': {'min': 10.0, 'max': 50.0, 'range': 40.0}}, 'method': 'minmax', 'source': 'sensor'}
```