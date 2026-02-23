# Modelo de Gompertz

## O que é

O modelo de Gompertz é um modelo assimétrico usado para descrever curvas de crescimento sigmoidal, especialmente em microbiologia. Caracteriza-se por um crescimento inicial mais lento comparado ao logístico.

## Características

- **Assimetria**: Ponto de inflexão em aproximadamente 37% da amplitude máxima
- **Crescimento inicial lento**: Mais realista para muitos processos biológicos
- **Três parâmetros**: Amplitude, taxa de crescimento, tempo de inflexão

## Quando usar

- Para curvas de crescimento microbiano assimétricas
- Quando o crescimento inicial é lento
- Em microbiologia e ecologia populacional
- Para modelar processos com saturação assimétrica

## Como usar

```python
import numpy as np
from src.components.signal_processing.curve_fitting.gompertz import GompertzModel
from src.components.signal_processing.curve_fitting import BlockInput

# Dados de crescimento
data = np.column_stack([tempo, populacao])
input_block = BlockInput(data=data)

# Ajustar modelo
model = GompertzModel()
result = model.fit(input_block)

print("Parâmetros:", result.parameters)
print("R²:", result.r_squared)
```

## Parâmetros

- `A`: Amplitude (assíntota superior, população máxima)
- `K`: Taxa de crescimento (parâmetro de forma)
- `T`: Tempo de inflexão (onde a taxa de crescimento é máxima)

## Fórmula

```
y = A * exp(-exp(K * (T - x)))
```

## Derivadas

### Primeira derivada
```
dy/dx = A * K * exp(-exp(K*(T-x))) * exp(K*(T-x))
```

### Segunda derivada
```
d²y/dx² = A * K² * exp(-exp(K*(T-x))) * exp(K*(T-x)) * (exp(K*(T-x)) - 1)
```

## Entradas e Saídas

- **Entradas**: `BlockInput` com dados 2D (tempo, valor)
- **Saídas**: `BlockOutput` com parâmetros ajustados e métricas de qualidade

## Vantagens

- **Assimetria natural**: Melhor para muitos processos biológicos
- **Parâmetros interpretáveis**: Tempo de inflexão claro
- **Microbiologia**: Padrão em crescimento bacteriano
- **Flexibilidade**: Adequado para diferentes tipos de saturação

## Comparação com outros modelos

- **vs Logístico**: Gompertz tem crescimento inicial mais lento
- **vs Baranyi**: Gompertz não modela fase lag explicitamente
- **vs Richards**: Gompertz é caso especial do Richards (forma=1)

## Aplicações

- Crescimento microbiano
- Crescimento tumoral
- Desenvolvimento populacional
- Curvas de adoção tecnológica