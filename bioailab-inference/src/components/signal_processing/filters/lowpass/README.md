# Filtros Passa-Baixa

## O que é

Os filtros passa-baixa atenuam componentes de alta frequência do sinal, preservando tendências e variações lentas. São essenciais para remover ruído e suavizar dados de sensores.

## Tipos Disponíveis

### LowPassFilter
- **Filtro exponencial simples**: IIR de primeira ordem
- **Parâmetros**: `cutoff_ratio` (0-1), `order` (número de passadas)

### ButterworthFilter
- **Filtro IIR com resposta plana**: Máxima planicidade na banda de passagem
- **Parâmetros**: `cutoff_freq`, `sample_rate`, `order`
- **Requer**: scipy

### ChebyshevFilter
- **Filtro com ripple controlado**: Permite ripple na banda de passagem
- **Parâmetros**: `cutoff_freq`, `sample_rate`, `order`, `ripple`

## Quando usar

- Para remover ruído de alta frequência
- Quando tendências de longo prazo são importantes
- Para preparar dados para análise de crescimento
- Butterworth: resposta plana desejada
- Chebyshev: maior atenuação na banda rejeitada

## Como usar

```python
import numpy as np
from src.components.signal_processing.filters.lowpass import LowPassFilter, ButterworthFilter
from src.components.signal_processing.filters import BlockInput

# Preparar dados
data = np.column_stack([timestamps, valores_sinal])
input_block = BlockInput(data=data)

# Filtro exponencial simples
lp_filter = LowPassFilter(cutoff_ratio=0.1, order=2)
output = lp_filter.process(input_block)

# Filtro Butterworth (requer scipy)
bw_filter = ButterworthFilter(cutoff_freq=0.1, sample_rate=1.0, order=4)
output2 = bw_filter.process(input_block)

print("LowPass aplicado:", output.success)
print("Butterworth aplicado:", output2.success)
```

## Parâmetros

### LowPassFilter
- `cutoff_ratio`: Razão de corte (0-1, default=0.1)
- `order`: Número de passadas (int, default=1)

### ButterworthFilter
- `cutoff_freq`: Frequência de corte (Hz)
- `sample_rate`: Taxa de amostragem (Hz, default=None para normalizado)
- `order`: Ordem do filtro (default=4)
- `filtfilt`: Aplicar bidirecional (default=False)

## Entradas e Saídas

- **Entradas**: `BlockInput` com dados 2D [x, y]
- **Saídas**: `BlockOutput` com sinal filtrado, metadados do filtro

## Exemplo

```
Dados originais: (1000, 2)
Filtro: LowPass (cutoff_ratio=0.1, order=2)
Sucesso: True
Metadados: {'filter_applied': 'lowpass', 'filter_params': {'cutoff_ratio': 0.1, 'order': 2}}
```