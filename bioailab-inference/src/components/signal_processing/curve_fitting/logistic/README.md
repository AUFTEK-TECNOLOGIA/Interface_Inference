# Modelo Logístico

## O que é

O modelo logístico é o modelo sigmoide clássico mais simples e amplamente usado. Descreve crescimento com saturação simétrica, onde a taxa de crescimento máxima ocorre no ponto médio da curva.

## Características

- **Simetria**: Ponto de inflexão exatamente em 50% da amplitude máxima
- **Três parâmetros**: Amplitude, taxa de crescimento, tempo de inflexão
- **Interpretável**: Parâmetros têm significado biológico claro
- **Simples**: Menos parâmetros que modelos mais complexos

## Quando usar

- Para curvas de crescimento simétricas
- Quando o processo tem saturação clara
- Em ecologia populacional
- Para modelar adoção de tecnologias ou difusão

## Como usar

```python
import numpy as np
from src.components.signal_processing.curve_fitting.logistic import LogisticModel
from src.components.signal_processing.curve_fitting import BlockInput

# Dados de crescimento populacional
data = np.column_stack([tempo, populacao])
input_block = BlockInput(data=data)

# Ajustar modelo
model = LogisticModel()
result = model.fit(input_block)

print("Capacidade de suporte (A):", result.parameters['A'])
print("Taxa de crescimento (K):", result.parameters['K'])
print("Tempo de inflexão (T):", result.parameters['T'])
```

## Parâmetros

- `A`: Amplitude (capacidade de suporte, população máxima)
- `K`: Taxa de crescimento (parâmetro de forma)
- `T`: Tempo do ponto de inflexão (onde crescimento é máximo)

## Fórmula

```
y = A / (1 + exp(-K * (x - T)))
```

## Derivadas

### Primeira derivada (taxa de crescimento)
```
dy/dx = (A * K * exp(-K*(x-T))) / (1 + exp(-K*(x-T)))^2
```

### Segunda derivada
```
d²y/dx² = (A * K² * exp(-K*(x-T)) * (1 - exp(-K*(x-T)))) / (1 + exp(-K*(x-T)))^3
```

## Entradas e Saídas

- **Entradas**: `BlockInput` com dados 2D (tempo, valor)
- **Saídas**: `BlockOutput` com parâmetros ajustados e métricas

## Vantagens

- **Simplicidade**: Poucos parâmetros, fácil interpretação
- **Simetria**: Adequado para muitos processos biológicos
- **Estabilidade**: Convergência geralmente boa
- **Fundamentação**: Base teórica sólida em ecologia

## Comparação com outros modelos

- **vs Gompertz**: Logístico é simétrico, Gompertz assimétrico
- **vs Baranyi**: Logístico não modela fase lag
- **vs Richards**: Logístico é caso especial (forma=1)

## Aplicações

- Crescimento populacional
- Epidemias e difusão de doenças
- Adoção de inovações
- Crescimento tumoral
- Curvas de aprendizado