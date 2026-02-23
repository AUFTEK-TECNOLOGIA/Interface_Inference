# Pré-processamento de Sinais

Módulo contendo operações de preparação de dados antes da análise principal.

## Componentes

### TimeSlice
Corte temporal de dados do experimento. Permite remover dados do início e/ou fim baseado em tempo ou índice.

- [time_slice/](./time_slice/) - Corte por intervalo de tempo ou índice

### OutlierRemoval  
Remoção de valores anômalos que podem distorcer análises.

- [outlier_removal/](./outlier_removal/) - Detecção e remoção de outliers

## Quando usar

1. **TimeSlice**: 
   - Remover período de estabilização inicial
   - Focar em janela temporal específica
   - Excluir dados após fim do experimento

2. **OutlierRemoval**:
   - Limpar dados antes de curve fitting
   - Remover spikes de sensores
   - Preparar dados para análise estatística

## Integração com Pipeline

Estes processadores estão disponíveis como blocos no Pipeline Studio:
- `time_slice` - Bloco de corte temporal
- `outlier_removal` - Bloco de remoção de outliers
