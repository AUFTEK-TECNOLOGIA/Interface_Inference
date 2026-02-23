# Detector de Derivada

## O que é

O detector de derivada identifica crescimento analisando a taxa de mudança do sinal ao longo do tempo. Ele calcula a derivada da série temporal e procura por picos significativos que indiquem fases de crescimento ativo.

## Como funciona

- Suaviza o sinal para reduzir ruído
- Calcula derivada: `dy/dt = d(y_smooth)/dt`
- Analisa estatísticas da derivada:
  - Média, desvio padrão, máximo, mínimo
- Detecta direção predominante (crescente/decrescente)
- Verifica se o pico da derivada supera o threshold de ruído

## Quando usar

- Para detectar crescimento mesmo com amplitude total pequena
- Em sinais com fase de crescimento exponencial clara
- Quando há crescimento rápido em meio a dados estáveis
- Cenários onde a taxa de mudança é mais importante que amplitude absoluta

## Como usar

```python
import numpy as np
from src.components.growth_detection.derivative import DerivativeDetector
from src.components.growth_detection import GrowthDetectionConfig

# Dados com crescimento exponencial
tempo = np.linspace(0, 8, 30)
concentracao = 0.1 * np.exp(0.4 * tempo)  # Crescimento exponencial

# Configuração
config = GrowthDetectionConfig(
    noise_threshold_percent=5.0,  # Threshold de ruído 5%
    expected_direction="increasing",
    smooth_sigma=1.5
)

# Detecção
detector = DerivativeDetector(config=config)
resultado = detector.detect(tempo, concentracao)

print(f"Crescimento detectado: {resultado.has_growth}")
print(f"Direção: {resultado.direction}")
print(f"Confiança: {resultado.confidence:.2f}")
```

## Parâmetros de Configuração

- `noise_threshold_percent`: Threshold de ruído (% do sinal médio)
- `expected_direction`: Direção esperada ("auto", "increasing", "decreasing")
- `smooth_sigma`: Suavização Gaussiana (padrão: 1.0)

## Interpretação dos Resultados

- **Pico da derivada**: Maior taxa de mudança detectada
- **Direção**: Baseada na integral positiva vs negativa
- **Confiança**: Baseada na magnitude do pico vs ruído
- **Detalhes**: Inclui estatísticas completas da derivada

## Vantagens

- **Sensível**: Detecta crescimento sutil com amplitude pequena
- **Robusto**: Usa análise estatística da derivada
- **Direto**: Mede taxa de crescimento diretamente
- **Flexível**: Funciona com diferentes formas de crescimento

## Limitações

- Requer muitos pontos (≥10) para cálculo confiável
- Sensível ao nível de suavização
- Pode falhar em sinais muito ruidosos
- Mais complexo de interpretar

## Exemplos

### Cenário 1: Crescimento exponencial
```
Dados: 0.1 * exp(0.4*t) de t=0 a 8
Derivada máxima: ~0.15 (alta taxa de crescimento)
Threshold de ruído: ~0.005
Resultado: CRESCIMENTO DETECTADO (taxa >> ruído)
```

### Cenário 2: Sinal ruidoso sem tendência
```
Dados: ruído aleatório ±0.01 ao redor de 1.0
Derivada máxima: ~0.002
Threshold de ruído: ~0.005
Resultado: NENHUM CRESCIMENTO (abaixo do threshold)
```

## Comparação com outros detectores

- **vs Amplitude**: Derivative detecta taxas, Amplitude detecta mudança total
- **vs Ratio**: Derivative analisa mudança instantânea, Ratio compara médias
- **vs Combined**: Derivative é especializado, Combined combina abordagens