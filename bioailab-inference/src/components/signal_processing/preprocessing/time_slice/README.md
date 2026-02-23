# Time Slice - Corte Temporal de Dados

## O que é

O TimeSlice permite cortar dados do experimento por intervalo de tempo (em minutos) ou por índices. Útil para remover períodos de estabilização, focar em janelas específicas ou excluir dados inválidos.

## Tipos Disponíveis

### TimeSliceProcessor
- **Corte por tempo**: Define início e fim em minutos
- **Corte por índice**: Define início e fim por posição
- **Normalização**: Opção de resetar timestamps para zero

## Quando usar

- Remover período de estabilização inicial do sensor
- Focar em janela temporal específica de interesse
- Excluir dados após término do experimento
- Sincronizar dados de múltiplos sensores

## Como usar

```python
import numpy as np
from src.components.signal_processing.preprocessing.time_slice import TimeSliceProcessor

# Dados do sensor
sensor_data = {
    "timestamps": [0, 60, 120, 180, 240, 300],  # segundos
    "channels": {
        "f1": [100, 110, 120, 130, 140, 150]
    }
}

# Cortar primeiros 2 minutos (0-120s)
processor = TimeSliceProcessor(
    slice_mode="time",
    start_time_min=0.0,
    end_time_min=2.0,
    normalize_time=True
)

result = processor.process(sensor_data)
print(f"Pontos originais: 6")
print(f"Pontos após corte: {len(result['timestamps'])}")
```

## Parâmetros

### Modo Tempo (`slice_mode="time"`)
- `start_time_min`: Tempo inicial em minutos (default: 0.0)
- `end_time_min`: Tempo final em minutos (default: None = até o fim)

### Modo Índice (`slice_mode="index"`)
- `start_index`: Índice inicial (default: 0)
- `end_index`: Índice final (default: None = até o fim)

### Opções Gerais
- `normalize_time`: Se True, subtrai o primeiro timestamp de todos (default: True)

## Saídas

- `sensor_data`: Dados processados com timestamps e canais cortados
- `slice_info`: Informações do corte (pontos removidos, offset aplicado, etc.)

## Integração no Pipeline

Bloco disponível como `time_slice` no Pipeline Studio, na etapa de Preparação.
