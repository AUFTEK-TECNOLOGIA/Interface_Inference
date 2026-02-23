# Microbial Growth Feature Extractor

Extrator especializado em parâmetros de crescimento microbiano, seguindo os modelos clássicos de microbiologia preditiva.

## Visão Geral

Este extrator identifica e quantifica as quatro fases clássicas do crescimento microbiano:
1. **Fase Lag (λ)**: Período de adaptação onde não há crescimento significativo
2. **Fase Exponencial**: Crescimento rápido e constante
3. **Fase Estacionária**: Equilíbrio entre crescimento e morte
4. **Fase de Declínio**: Redução da população devido à exaustão de nutrientes

## Parâmetros Extraídos

### Parâmetros Primários
- **Lag Time (λ)**: Tempo até o início do crescimento exponencial (minutos)
- **Taxa Máxima de Crescimento (μmax)**: Velocidade máxima específica (OD/minuto)
- **Capacidade de Suporte (K)**: População máxima atingida
- **Amplitude**: Diferença total entre valor final e inicial

### Parâmetros Derivados
- **Tempo de Geração**: Tempo necessário para duplicação (minutos)
- **Tempo para Fase Estacionária**: Quando o crescimento para (minutos)
- **Área sob a Curva (AUC)**: Integral da curva de crescimento
- **Valor Inicial/Final**: Pontos extremos da curva

## Quando Usar

Ideal para:
- Curvas de crescimento bacteriano em meios líquidos
- Dados turbidimétricos (OD600) ou espectrofotométricos
- Experimentos microbiológicos com fases de crescimento distintas
- Análise de parâmetros cinéticos de crescimento

## Limitações

- Requer dados com pelo menos uma fase exponencial clara
- Menos adequado para dados com muito ruído
- Assume crescimento sigmoidal clássico (não linear)

## Referências

- Zwietering, M. H. et al. (1990). Modeling bacterial growth curves. *Applied and Environmental Microbiology*
- Baranyi, J. & Roberts, T. A. (1994). A dynamic approach to predicting bacterial growth. *International Journal of Food Microbiology*

## Exemplo de Uso

```python
from components.feature_extraction.microbial import MicrobialGrowthExtractor

extractor = MicrobialGrowthExtractor()
features = extractor.extract(time_data, od_data)

print(f"Lag time: {features.lag_time} min")
print(f"Max growth rate: {features.max_growth_rate} OD/min")
print(f"Carrying capacity: {features.carrying_capacity}")
```