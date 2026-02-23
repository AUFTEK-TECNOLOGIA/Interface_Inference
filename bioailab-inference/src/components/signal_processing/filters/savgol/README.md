# Filtro Savitzky-Golay

## O que é

O filtro Savitzky-Golay ajusta um polinômio de baixa ordem aos pontos na janela e usa o valor central do polinômio ajustado. Preserva características do sinal como picos e vales melhor que outros filtros.

## Características

- **Preservação de forma**: Mantém largura e altura de picos
- **Suavização controlada**: Ordem do polinômio controla suavização
- **Derivadas**: Pode calcular derivadas suavizadas
- **Requer**: scipy

## Quando usar

- Para preservar forma de picos em espectros
- Quando derivadas são necessárias
- Para dados analíticos com características importantes
- Melhor que média móvel para sinais complexos

## Como usar

```python
import numpy as np
from src.components.signal_processing.filters.savgol import SavGolFilter
from src.components.signal_processing.filters import BlockInput

# Dados espectrais
data = np.column_stack([wavelengths, intensities])
input_block = BlockInput(data=data)

# Aplicar filtro Savitzky-Golay
filter = SavGolFilter(window_length=15, polyorder=3)
output = filter.process(input_block)

print("Filtro aplicado:", output.success)
print("Janela:", 15, "Polinômio ordem:", 3)
```

## Parâmetros

- `window_length`: Tamanho da janela (ímpar, ≥ polyorder+1)
- `polyorder`: Ordem do polinômio (default=2)
- `deriv`: Ordem da derivada (default=0)
- `delta`: Espaçamento entre pontos (default=1.0)
- `mode`: Como tratar bordas ('mirror', 'constant', 'nearest', 'wrap')

## Entradas e Saídas

- **Entradas**: `BlockInput` com dados 2D [x, y]
- **Saídas**: `BlockOutput` com sinal suavizado

## Comparação com Média Móvel

| Aspecto | Savitzky-Golay | Média Móvel |
|---------|----------------|-------------|
| Preservação de picos | Excelente | Ruim |
| Velocidade | Mais lento | Mais rápido |
| Derivadas | Sim | Não |
| Complexidade | Alta | Baixa |

## Exemplo

```
Dados: espectro com picos largos
Janela: 15 pontos, polinômio ordem 3
Resultado: picos preservados, ruído removido
```