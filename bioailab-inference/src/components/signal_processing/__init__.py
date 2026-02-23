"""
Módulo de Processamento de Sinais.

Contém componentes para processamento, filtragem e preparação
de sinais de sensores:

- filters/: Filtros de sinal (mediana, savgol, etc.)
- normalizers/: Normalizadores de sinal
- curve_fitting/: Ajuste de curvas
- preprocessing/: Pré-processamento (time slice, outlier removal)
"""

# Importar submódulos principais
from . import filters
from . import normalizers
from . import curve_fitting
from . import preprocessing
