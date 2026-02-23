# Outlier Removal - Remoção de Valores Anômalos

## O que é

O OutlierRemoval detecta e remove (ou interpola) valores anômalos nos dados de sensores que podem distorcer análises posteriores, usando métodos estatísticos robustos.

## Métodos Disponíveis

### Z-Score (`method="zscore"`)
- Detecta pontos que estão a mais de N desvios padrão da média
- **Threshold padrão**: 3.0 (99.7% dos dados em distribuição normal)
- Melhor para dados com distribuição aproximadamente normal

### IQR - Intervalo Interquartil (`method="iqr"`)
- Usa quartis para definir limites: [Q1 - k×IQR, Q3 + k×IQR]
- **Threshold padrão**: 1.5 (outliers moderados)
- Mais robusto que z-score para dados não-normais

### MAD - Desvio Mediano Absoluto (`method="mad"`)
- Usa mediana e MAD para detecção
- **Threshold padrão**: 3.0
- Mais robusto a outliers extremos

## Estratégias de Substituição

### Remover (`replace_strategy="remove"`)
- Remove completamente os pontos detectados como outliers
- Reduz o tamanho do dataset
- Mantém timestamps e canais sincronizados

### Interpolar (`replace_strategy="interpolate"`)
- Substitui outliers por valores interpolados dos vizinhos
- Mantém o tamanho original do dataset
- Preserva a estrutura temporal

## Quando usar

- Antes de curve fitting para evitar distorções
- Para limpar dados de sensores com falhas intermitentes
- Antes de normalização ou outros filtros
- Quando spikes podem afetar análise estatística

## Como usar

```python
from src.components.signal_processing.preprocessing.outlier_removal import OutlierRemovalProcessor

# Dados do sensor
sensor_data = {
    "timestamps": [0, 1, 2, 3, 4, 5],
    "channels": {
        "f1": [100, 110, 500, 120, 130, 140]  # 500 é outlier
    }
}

# Remover outliers usando z-score
processor = OutlierRemovalProcessor(
    method="zscore",
    threshold=2.0,
    replace_strategy="interpolate"
)

result = processor.process(sensor_data)
print(f"Outliers detectados: {result.outlier_info['total_outliers_detected']}")
```

## Parâmetros

- `method`: Método de detecção ('zscore', 'iqr', 'mad')
- `threshold`: Limite para classificar como outlier
  - zscore: número de desvios padrão (default: 3.0)
  - iqr: multiplicador do IQR (default: 1.5)
  - mad: número de MADs (default: 3.0)
- `replace_strategy`: O que fazer com outliers ('remove', 'interpolate')

## Saídas

- `sensor_data`: Dados processados (sem outliers ou com interpolação)
- `outlier_info`: Estatísticas (total detectado, por canal, índices)

## Integração no Pipeline

Bloco disponível como `outlier_removal` no Pipeline Studio, na etapa de Preparação.

## Referência

Para filtros de outliers mais específicos (aplicados a sinais individuais), veja também:
- `src/components/signal_processing/filters/outlier/` - Filtros de outlier para sinais
