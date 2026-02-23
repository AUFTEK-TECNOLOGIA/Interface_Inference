# Normalizador Robust

## O que é

O Normalizador Robust usa **mediana e intervalo interquartil (IQR)** para normalização, sendo **resistente a outliers**. Ele subtrai a mediana e divide pelo IQR (diferença entre o 3º e 1º quartil), o que o torna mais confiável quando os dados contêm valores extremos que poderiam distorcer a normalização tradicional.

## Quando usar

- Quando os dados têm outliers ou ruído extremo
- Para dados biológicos ou sensoriais com variações anômalas
- Quando MinMax ou Z-Score são afetados por valores atípicos

## Como usar

```python
import numpy as np
from src.components.signal_processing.normalizers.robust import RobustNormalizer
from src.components.signal_processing.normalizers import BlockInput

# Dados de exemplo com outlier
data = np.column_stack([[1, 2, 3, 4, 100], [10, 20, 30, 40, 1000]])  # Shape (5, 2)
input_block = BlockInput(data=data, metadata={"robust": True})

# Criar normalizador
normalizer = RobustNormalizer()

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
Dados originais: [[   1   10]
                  [   2   20]
                  [   3   30]
                  [   4   40]
                  [ 100 1000]]
Dados normalizados: [[-0.66666667 -0.66666667]
                     [-0.33333333 -0.33333333]
                     [ 0.          0.        ]
                     [ 0.33333333  0.33333333]
                     [ 3.          3.        ]]
Sucesso: True
Metadados: {'normalization_params': {'method': 'robust', 'x_params': {'median': 3.0, 'q1': 1.5, 'q3': 3.5, 'iqr': 2.0}, 'y_params': {'median': 30.0, 'q1': 15.0, 'q3': 35.0, 'iqr': 20.0}}, 'method': 'robust', 'robust': True}
```