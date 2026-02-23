# Blocos de pipeline (lista atual)

Lista extraida de `src/components/pipeline/blocks.py` (decorator `@BlockRegistry.register`).

| Bloco | Classe | Categoria | O que faz | Como faz |
| --- | --- | --- | --- | --- |
| `amplitude_detector` | AmplitudeDetectorBlock | detection | Detecta crescimento pela amplitude relativa do sinal (max-min) | Calcula a metrica descrita e compara com thresholds/config para gerar o sinal. |
| `and_gate` | AndGateBlock | control_flow | Porta AND: true se AMBAS condicoes forem true | Avalia booleanos e controla o fluxo/ramificacao entre steps. |
| `boolean_extractor` | BooleanExtractorBlock | extraction | Extrai um booleano de source_data e passa sensor_data adiante | Seleciona e organiza os canais do sensor/experimento a partir do input do pipeline. |
| `cmyk_conversion` | CMYKConversionBlock | conversion | Converte para CMYK (ciano, magenta, amarelo, preto) | Aplica transformacao matematica dos canais para o espaco de cor indicado. |
| `condition_branch` | ConditionBranchBlock | control_flow | Ramifica o fluxo: se condicao=true vai para if_true, senao para if_false | Avalia booleanos e controla o fluxo/ramificacao entre steps. |
| `condition_gate` | ConditionGateBlock | control_flow | Portao condicional: passa dados somente se condicao == valor esperado | Avalia booleanos e controla o fluxo/ramificacao entre steps. |
| `control_state_extraction` | ControlStateExtractionBlock | extraction | Extrai dados do estado de controle (erros e sinais de controle) | Seleciona e organiza os canais do sensor/experimento a partir do input do pipeline. |
| `curve_fit` | CurveFitBlock | curve_fitting | Ajusta modelo matematico aos dados (Richards, Gompertz, etc.) | Ajusta modelos matematicos e retorna parametros/curva ajustada. |
| `curve_fit_best` | CurveFitBestBlock | curve_fitting | Encontra o melhor modelo de ajuste automaticamente | Ajusta modelos matematicos e retorna parametros/curva ajustada. |
| `curve_fitting_deprecated` | CurveFittingBlock_DEPRECATED | curve_fitting | [DEPRECATED] Use curve_fit ou curve_fit_best | Ajusta modelos matematicos e retorna parametros/curva ajustada. |
| `derivative` | DerivativeBlock | math | Calcula derivada (taxa de variacao) dos canais | Calcula a derivada numerica por canal. |
| `derivative_detector` | DerivativeDetectorBlock | detection | Detecta crescimento pela analise da taxa de variacao (derivada) | Calcula a metrica descrita e compara com thresholds/config para gerar o sinal. |
| `experiment_fetch` | ExperimentFetchBlock | extraction | Busca documento de experimento e seus dados brutos no repositorio | Le o repositorio configurado e carrega dados do experimento e lab_results. |
| `exponential_filter` | ExponentialFilterBlock | filtering | Filtro de media movel exponencial (EMA) | Aplica o filtro configurado canal a canal (ex.: media movel, savgol, lowpass). |
| `features_merge` | FeaturesMergeBlock | feature_extraction | Combina multiplos blocos de features em um unico output | Junta vetores de features e normaliza o formato de saida. |
| `fluorescence_extraction` | FluorescenceExtractionBlock | extraction | Extrai dados do sensor de fluorescencia (10 canais espectrais) | Seleciona e organiza os canais do sensor/experimento a partir do input do pipeline. |
| `growth_features` | GrowthFeaturesBlock | feature_extraction | Extrai features de crescimento numericamente das curvas ajustadas | Calcula atributos de crescimento a partir das curvas ajustadas. |
| `hsb_conversion` | HSBConversionBlock | conversion | Converte para HSB (matiz, saturacao, brilho) | Aplica transformacao matematica dos canais para o espaco de cor indicado. |
| `hsv_conversion` | HSVConversionBlock | conversion | Converte para HSV (matiz, saturacao, valor) | Aplica transformacao matematica dos canais para o espaco de cor indicado. |
| `integral` | IntegralBlock | math | Calcula integral acumulada (area sob a curva) | Calcula a integral acumulada por canal. |
| `lab_conversion` | LABConversionBlock | conversion | Converte para CIE LAB (luminosidade, a*, b*) | Aplica transformacao matematica dos canais para o espaco de cor indicado. |
| `label` | LabelBlock | label | Adiciona uma label/tag para agrupar resultados | Anexa label ao contexto e propaga no pipeline. |
| `lowpass_filter` | LowpassFilterBlock | filtering | Filtro passa-baixa Butterworth | Aplica o filtro configurado canal a canal (ex.: media movel, savgol, lowpass). |
| `median_filter` | MedianFilterBlock | filtering | Filtro de mediana - remove ruido impulsivo | Aplica o filtro configurado canal a canal (ex.: media movel, savgol, lowpass). |
| `merge` | MergeBlock | control_flow | Junta dois fluxos condicionais - passa adiante o que estiver ativo | Avalia booleanos e controla o fluxo/ramificacao entre steps. |
| `ml_detector` | MLDetectorBlock | ml | Detector ML (serie  booleano) com score (auto-configurado apos treino) | Carrega modelo/scaler/metadata e gera booleano + score a partir da serie. |
| `ml_forecaster_series` | MLForecasterSeriesBlock | ml | Preve uma serie por janela temporal (auto-configurado apos treino) | Carrega modelo/scaler/metadata e executa inferencia sobre as features/series. |
| `ml_inference` | MLInferenceBlock | ml | Executa inferencia ML com modelos ONNX | Carrega modelo/scaler/metadata e executa inferencia sobre as features/series. |
| `ml_inference_multichannel` | MLInferenceMultichannelBlock | ml | Executa inferencia ML a partir de multiplos canais (auto-configurado apos treino) | Carrega modelo/scaler/metadata e executa inferencia sobre as features/series. |
| `ml_inference_series` | MLInferenceSeriesBlock | ml | Executa inferencia ML a partir de uma serie temporal (um canal) | Carrega modelo/scaler/metadata e executa inferencia sobre as features/series. |
| `ml_transform_series` | MLTransformSeriesBlock | ml | Aplica um modelo ML para transformar uma serie (auto-configurado apos treino) | Carrega modelo/scaler/metadata e executa inferencia sobre as features/series. |
| `moving_average_filter` | MovingAverageFilterBlock | filtering | Filtro de media movel - suavizacao basica | Aplica o filtro configurado canal a canal (ex.: media movel, savgol, lowpass). |
| `nema_currents_extraction` | NemaCurrentsExtractionBlock | extraction | Extrai dados das correntes NEMA (bobinas A e B) | Seleciona e organiza os canais do sensor/experimento a partir do input do pipeline. |
| `nephelometry_extraction` | NephelometryExtractionBlock | extraction | Extrai dados do sensor de nefelometria (10 canais espectrais) | Seleciona e organiza os canais do sensor/experimento a partir do input do pipeline. |
| `normalize` | NormalizeBlock | normalization | Normaliza os dados (minmax, zscore, robust) | Aplica normalizacao (minmax, zscore, robust) conforme config. |
| `not_gate` | NotGateBlock | control_flow | Porta NOT: inverte o valor booleano | Avalia booleanos e controla o fluxo/ramificacao entre steps. |
| `numeric_compare` | NumericCompareBlock | control_flow | Compara valor numerico com threshold (gera condition) | Compara o valor com o threshold e gera condition booleana. |
| `or_gate` | OrGateBlock | control_flow | Porta OR: true se PELO MENOS UMA condicao for true | Avalia booleanos e controla o fluxo/ramificacao entre steps. |
| `outlier_removal` | OutlierRemovalBlock | outlier | Remove outliers dos dados usando diferentes metodos | Remove outliers por metodo configurado. |
| `peltier_currents_extraction` | PeltierCurrentsExtractionBlock | extraction | Extrai dados das correntes Peltier | Seleciona e organiza os canais do sensor/experimento a partir do input do pipeline. |
| `power_supply_extraction` | PowerSupplyExtractionBlock | extraction | Extrai dados da fonte de alimentacao (tensao e corrente) | Seleciona e organiza os canais do sensor/experimento a partir do input do pipeline. |
| `ratio_detector` | RatioDetectorBlock | detection | Detecta crescimento pela razao entre valores iniciais e finais | Calcula a metrica descrita e compara com thresholds/config para gerar o sinal. |
| `resonant_frequencies_extraction` | ResonantFrequenciesExtractionBlock | extraction | Extrai dados das frequencias ressonantes | Seleciona e organiza os canais do sensor/experimento a partir do input do pipeline. |
| `response_builder` | ResponseBuilderBlock | response | Monta o JSON de resposta final (suporta agrupamento por label) | Agrega outputs do pipeline e monta o payload final. |
| `response_merge` | ResponseMergeBlock | response | Seleciona a resposta ativa entre multiplas respostas parciais | Seleciona/mescla respostas parciais e retorna a ativa. |
| `response_pack` | ResponsePackBlock | response | Empacota uma resposta parcial (para merge e saida unica) | Agrega outputs do pipeline e monta o payload final. |
| `rgb_conversion` | RGBConversionBlock | conversion | Converte para RGB (vermelho, verde, azul) | Aplica transformacao matematica dos canais para o espaco de cor indicado. |
| `savgol_filter` | SavgolFilterBlock | filtering | Filtro Savitzky-Golay - preserva picos e caracteristicas | Aplica o filtro configurado canal a canal (ex.: media movel, savgol, lowpass). |
| `sensor_fusion` | SensorFusionBlock | other | Combinar sensores (multissensor  sensor_data) | Mescla canais de sensores em um unico `sensor_data`. |
| `shape_features` | ShapeFeaturesBlock | feature_extraction | Extrai features de forma (inflexao, picos, area sob curva, etc.) | Calcula atributos de forma a partir das series/curvas. |
| `signal_filters` | SignalFiltersBlock | filtering | Aplica filtros de sinal pos-conversao espectral | Aplica o filtro configurado canal a canal (ex.: media movel, savgol, lowpass). |
| `spectral_conversion` | SpectralConversionBlock | conversion | Converte canais espectrais usando API externa | Chama a API externa configurada e substitui os canais convertidos. |
| `statistical_features` | StatisticalFeaturesBlock | feature_extraction | Extrai features estatisticas (max, min, mean, std, etc.) | Calcula estatisticas basicas por canal e retorna o vetor. |
| `temperatures_extraction` | TemperaturesExtractionBlock | extraction | Extrai dados de temperatura (8 pontos de medicao) | Seleciona e organiza os canais do sensor/experimento a partir do input do pipeline. |
| `temporal_features` | TemporalFeaturesBlock | feature_extraction | Extrai features temporais (time_to_max, time_to_threshold, etc.) | Calcula tempos e marcos temporais a partir das series. |
| `time_slice` | TimeSliceBlock | slicing | Corta dados por intervalo de tempo (inicio e fim) | Corta a serie temporal pelo intervalo configurado. |
| `turbidimetry_extraction` | TurbidimetryExtractionBlock | extraction | Extrai dados do sensor de turbidimetria (10 canais espectrais) | Seleciona e organiza os canais do sensor/experimento a partir do input do pipeline. |
| `value_in_list` | ValueInListBlock | control_flow | Verifica se um valor esta em uma lista (gera condition) | Verifica inclusao em lista e gera condition booleana. |
| `xyy_conversion` | xyYConversionBlock | conversion | Converte para CIE xyY (cromaticidade + luminancia) | Aplica transformacao matematica dos canais para o espaco de cor indicado. |
| `xyz_conversion` | XYZConversionBlock | conversion | Converte para CIE XYZ (espaco perceptual base) | Aplica transformacao matematica dos canais para o espaco de cor indicado. |
