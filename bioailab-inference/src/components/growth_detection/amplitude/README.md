# Detector de Amplitude

## O que é

O detector de amplitude identifica crescimento bacteriano baseado na amplitude relativa do sinal. Ele verifica se a diferença entre os valores máximo e mínimo é significativa em relação a uma referência (normalmente o maior valor absoluto).

## Como funciona

- Calcula a amplitude: `amplitude = max(y) - min(y)`
- Calcula a referência: `reference = max(|max(y)|, |min(y)|, ε)`
- Amplitude percentual: `amplitude_percent = (amplitude / reference) * 100`
- Crescimento detectado se `amplitude_percent >= min_amplitude_percent`

## Quando usar

- Para sinais com mudança clara de amplitude
- Quando há variação significativa entre mínimo e máximo
- Em dados onde a amplitude total indica crescimento
- Cenários com crescimento bem definido

## Como usar

```python
import numpy as np
from src.components.growth_detection.amplitude import AmplitudeDetector
from src.components.growth_detection import GrowthDetectionConfig

# Dados de absorbância ao longo do tempo
tempo = np.array([0, 1, 2, 3, 4, 5])
absorbancia = np.array([0.1, 0.15, 0.25, 0.45, 0.65, 0.85])

# Configuração
config = GrowthDetectionConfig(
    min_amplitude_percent=20.0,  # Requer amplitude >= 20%
    expected_direction="increasing"
)

# Detecção
detector = AmplitudeDetector(config=config)
resultado = detector.detect(tempo, absorbancia)

print(f"Crescimento detectado: {resultado.has_growth}")
print(f"Amplitude percentual: {resultado.amplitude_percent:.1f}%")
print(f"Confiança: {resultado.confidence:.2f}")
```

## Parâmetros de Configuração

- `min_amplitude_percent`: Amplitude mínima requerida (padrão: 10%)
- `expected_direction`: Direção esperada ("auto", "increasing", "decreasing")
- `smooth_sigma`: Suavização Gaussiana (padrão: 1.0)

## Interpretação dos Resultados

- **Amplitude percentual**: Indica a magnitude da mudança relativa
- **Confiança**: Baseada na proximidade do threshold mínimo
- **Direção**: "increasing", "decreasing" ou "unknown"

## Vantagens

- **Simples e direto**: Fácil de entender e interpretar
- **Robusto**: Funciona bem com dados ruidosos
- **Rápido**: Cálculo eficiente
- **Intuitivo**: Amplitude clara indica crescimento

## Limitações

- Pode falhar em sinais com amplitude pequena mas crescimento consistente
- Sensível a outliers extremos
- Não considera a forma da curva de crescimento

## Exemplos

### Cenário 1: Crescimento claro
```
Dados: [0.1, 0.2, 0.4, 0.7, 1.0, 1.5]
Amplitude: 1.4
Referência: 1.5
Percentual: 93.3% → CRESCIMENTO DETECTADO
```

### Cenário 2: Amplitude insuficiente
```
Dados: [1.0, 1.05, 0.95, 1.02, 0.98, 1.01]
Amplitude: 0.1
Referência: 1.05
Percentual: 9.5% → NENHUM CRESCIMENTO (abaixo de 10%)
```

## Comparação com outros detectores

- **vs Ratio**: Amplitude olha para variação total, Ratio compara início/fim
- **vs Derivative**: Amplitude detecta mudança geral, Derivative detecta taxas
- **vs Combined**: Amplitude é componente básico, Combined combina detectores