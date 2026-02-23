# Normalizador Z-Score

## O que é

O Normalizador Z-Score (ou Standardization) **padroniza os dados para média 0 e desvio padrão 1**. Ele subtrai a média e divide pelo desvio padrão, resultando em uma distribuição normal padrão. É ideal para algoritmos que assumem dados centrados em zero.

## Quando usar

- Para algoritmos de ML que esperam média zero (ex: regressão linear, redes neurais)
- Quando os dados seguem distribuição normal
- Para comparar features com escalas diferentes

## Como usar

```python
import numpy as np
from src.components.signal_processing.normalizers.zscore import ZScoreNormalizer
from src.components.signal_processing.normalizers import BlockInput

# Dados de exemplo
data = np.column_stack([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])  # Shape (5, 2)
input_block = BlockInput(data=data, metadata={"standardized": True})

# Criar normalizador
normalizer = ZScoreNormalizer()

# Processar
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
Dados normalizados: [[-1.41421356 -1.41421356]
                     [-0.70710678 -0.70710678]
                     [ 0.          0.        ]
                     [ 0.70710678  0.70710678]
                     [ 1.41421356  1.41421356]]
Sucesso: True
Metadados: {'normalization_params': {'method': 'zscore', 'x_params': {'mean': 3.0, 'std': 1.4142135623730951}, 'y_params': {'mean': 30.0, 'std': 14.142135623730951}}, 'method': 'zscore', 'standardized': True}
```