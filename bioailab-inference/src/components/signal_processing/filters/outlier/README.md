# Filtro de Outliers

## O que é

Os filtros de outliers identificam e removem valores extremos do sinal que podem distorcer análises posteriores, usando métodos estatísticos robustos.

## Tipos Disponíveis

### OutlierRemovalFilter
- **Filtro básico**: Remove pontos além de threshold fixo
- **Parâmetros**: `threshold`, `method`

### IQROutlierFilter
- **Baseado em IQR**: Usa intervalo interquartil para detectar outliers
- **Parâmetros**: `multiplier` (fator do IQR)

### MADOutlierFilter
- **Baseado em MAD**: Usa desvio mediano absoluto
- **Parâmetros**: `threshold`

## Quando usar

- Para remover valores extremos antes de modelagem
- Quando outliers podem afetar curve fitting
- Para dados de sensores com falhas intermitentes
- Antes de normalização ou outros filtros

## Como usar

```python
import numpy as np
from src.components.signal_processing.filters.outlier import IQROutlierFilter
from src.components.signal_processing.filters import BlockInput

# Dados com outliers
data = np.column_stack([timestamps, sinal_com_outliers])
input_block = BlockInput(data=data)

# Remover outliers baseado em IQR
filter = IQROutlierFilter(multiplier=1.5)
output = filter.process(input_block)

print("Outliers removidos:", output.success)
print("Pontos restantes:", len(output.data))
```

## Parâmetros

### IQROutlierFilter
- `multiplier`: Fator do IQR (default=1.5, ~2.7σ)

### MADOutlierFilter
- `threshold`: Limite em unidades de MAD (default=3.0)

## Método de Detecção

### IQR Method
```
Q1 = percentil 25
Q3 = percentil 75
IQR = Q3 - Q1
Limite inferior = Q1 - multiplier * IQR
Limite superior = Q3 + multiplier * IQR
```

### MAD Method
```
Mediana = median(dados)
MAD = median(|dados - mediana|)
Limite = mediana ± threshold * MAD
```

## Entradas e Saídas

- **Entradas**: `BlockInput` com dados 2D
- **Saídas**: `BlockOutput` com outliers removidos, metadados com contagem

## Exemplo

```
Dados originais: 1000 pontos
Outliers detectados: 23
Dados filtrados: 977 pontos
Método: IQR (multiplier=1.5)
```