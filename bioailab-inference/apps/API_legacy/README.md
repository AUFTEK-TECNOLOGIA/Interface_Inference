# UFC Flex Pipeline 🚀

## Visão-geral

Pipeline para previsão e visualização hora a hora do crescimento microbiano utilizando leituras dos sensores VIS1, VIS2 e UV. A versão atual da linha de comando é **v2025-06**.

## Instalação

```bash
git clone <repo>
cd bioailab-ml
pip install -r requirements.txt
```
Requer Python >= 3.10. O pipeline usa GPU com CUDA se disponível, mas também funciona apenas com CPU. Todos os modelos de treinamento, forecaster e predição selecionam automaticamente a GPU quando presente e caem para CPU caso contrário.

Se você tem uma GPU, pode rodar o script que detecta a versão do CUDA e instala o PyTorch adequado:

```powershell
# No Windows PowerShell (modo administrador):
.\scripts\setup_torch_cuda.ps1

## Uso rápido

### CLI

```bash
python pipeline_ufc_flex.py \
  -s UV:R,G,B -u "NPM/mL" \
  --models mlp,cnn,rf \
  --mlp-layers 3 --mlp-hidden 128 \
  --perm-imp --simulate --auto-start --thr 0.05 \
  --grid-search --param-grid '{"mlp":{"mlp_layers":[2,3]}}' \
  --metrics-out metrics.csv --save-plots figs
```
Use `rf` ou `random_forest` para ativar o modelo Random Forest.

### GUI

```bash
streamlit run src/gui.py
```

## Tabela de parâmetros

| Flag | Descrição | Default |
|------|-----------|---------|
|`-s, --sensor`|sensor:canal1,canal2,… (repetível)|-|
|`-u, --unit`|unidade alvo|`UFC/mL`|
|`--models`|modelos a testar (use `rf` ou `random_forest` para Random Forest)|`mlp`|
|`--data-dir`|diretório de dados|`data`|
|`--fct-hidden`|neurônios do forecaster|`64`|
|`--fct-layers`|camadas do forecaster|`2`|
|`--mlp-layers`|camadas do MLP|`2`|
|`--mlp-hidden`|neurônios por camada MLP|`64`|
|`--lstm-layers`|camadas LSTM do regressor|`2`|
|`--lstm-units`|unidades por camada LSTM|`64`|
|`--lstm-dropout`|dropout LSTM|`0.2`|
|`--gbm-n-estimators`|estimadores GBM|`800`|
|`--gbm-lr`|learning rate GBM|`0.03`|
|`--cnn-kernel`|tamanho do kernel CNN|`5`|
|`--cnn-depth`|profundidade CNN|`2`|
|`--rf-n-estimators`|árvores RandomForest|`100`|
|`--rf-max-depth`|profundidade máx RF|`-`|
|`--svr-kernel`|kernel do SVR|`rbf`|
|`--svr-c`|parâmetro C do SVR|`1.0`|
|`--svr-epsilon`|épsilon do SVR|`0.1`|
|`--xgb-n-estimators`|estimadores XGBoost|`200`|
|`--xgb-lr`|learning rate XGB|`0.05`|
|`--xgb-max-depth`|profundidade XGB|`3`|
|`--simulate, -sim`|simulação hora a hora|`False`|
|`--auto-start`|detecção automática|`False`|
|`--thr, -t`|limiar para auto-start|`0.05`|
|`--sim-size`|ensaios na simulação|`3`|
|`--save-plots`|diretório para figuras|-|
|`--export-preds`|CSV de previsões|-|
|`--grid-search`|ativa grid-search|`False`|
|`--param-grid`|grades JSON/YAML|-|
|`--metrics-out`|CSV de métricas|-|
|`--plots-out`|figuras comparativas|-|
|`--verbose`|logging detalhado|`False`|
|`--perm-imp`|permutation-importance|`False`|
|`--slice-start`|corta pontos iniciais de cada série|`0`|
|`--slice-end`|corta pontos finais de cada série|`0`|
