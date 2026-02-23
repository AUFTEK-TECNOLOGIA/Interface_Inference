# Detector de Razão

## O que é

O detector de razão identifica crescimento comparando os valores médios no início e no fim da série temporal. Ele calcula a razão entre a média final e a média inicial para determinar se houve uma mudança significativa.

## Como funciona

- Define janela: `window = max(3, n//10)` pontos
- Calcula médias:
  - `start_mean = média(y[:window])`
  - `end_mean = média(y[-window:])`
- Calcula razão: `ratio = end_mean / start_mean`
- Detecta crescimento baseado na razão e direção

## Quando usar

- Para sinais com tendência consistente ao longo do tempo
- Quando a proporção entre início e fim é mais importante que amplitude absoluta
- Em dados onde o crescimento é gradual e uniforme
- Cenários com crescimento linear ou exponencial suave

## Como usar

```python
import numpy as np
from src.components.growth_detection.ratio import RatioDetector
from src.components.growth_detection import GrowthDetectionConfig

# Dados de densidade óptica
tempo = np.array([0, 2, 4, 6, 8, 10])
do = np.array([0.05, 0.08, 0.15, 0.35, 0.75, 1.20])

# Configuração
config = GrowthDetectionConfig(
    min_growth_ratio=1.5,  # Requer razão >= 1.5
    expected_direction="increasing"
)

# Detecção
detector = RatioDetector(config=config)
resultado = detector.detect(tempo, do)

print(f"Crescimento detectado: {resultado.has_growth}")
print(f"Razão início/fim: {resultado.ratio:.2f}")
print(f"Confiança: {resultado.confidence:.2f}")
```

## Parâmetros de Configuração

- `min_growth_ratio`: Razão mínima requerida (padrão: 1.2)
- `expected_direction`: Direção esperada ("auto", "increasing", "decreasing")
- `smooth_sigma`: Suavização Gaussiana (padrão: 1.0)

## Interpretação dos Resultados

- **Razão > 1**: Valores finais maiores (crescimento)
- **Razão < 1**: Valores finais menores (decréscimo)
- **Razão ≈ 1**: Sem mudança significativa
- **Confiança**: Baseada na distância do threshold

## Vantagens

- **Robusto ao ruído**: Usa médias de janelas, não valores individuais
- **Detecta tendências**: Bom para crescimento gradual
- **Interpretável**: Razão direta de entender
- **Flexível**: Funciona com diferentes direções

## Limitações

- Pode falhar em sinais com amplitude alta mas sem tendência clara
- Sensível ao tamanho da janela
- Não detecta crescimento não-monotônico

## Exemplos

### Cenário 1: Crescimento gradual
```
Dados: [0.1, 0.15, 0.25, 0.45, 0.70, 1.0]
Janela: 2 pontos
start_mean: 0.125, end_mean: 0.85
Razão: 6.8 → CRESCIMENTO DETECTADO
```

### Cenário 2: Sem tendência clara
```
Dados: [1.0, 1.2, 0.8, 1.1, 0.9, 1.0]
Janela: 2 pontos
start_mean: 1.1, end_mean: 0.95
Razão: 0.86 → NENHUM CRESCIMENTO
```

## Comparação com outros detectores

- **vs Amplitude**: Ratio detecta tendência, Amplitude detecta variação total
- **vs Derivative**: Ratio compara médias, Derivative analisa taxas instantâneas
- **vs Combined**: Ratio é componente básico, Combined combina detectores