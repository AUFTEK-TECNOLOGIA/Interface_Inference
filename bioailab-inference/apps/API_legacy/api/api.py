# api/api.py

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any, List, Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from src.training import train_forecaster, train_regressor
from src.utils import parse_sensors
from src.config import (
    TEST_SIZE,
    MODEL_DIR,
)
from src.inference import inference
from fastapi.responses import JSONResponse


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def _jsonable(x: Any) -> Any:
    """Transforma numpy / objetos em tipos JSON-serializáveis."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


# ──────────────────────────────────────────────────────────────────────────────
# App e CORS
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="BioAiLab Pipeline API")

# Permitir chamadas do seu front (e.g. localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # use ["*"] em desenvolvimento se preferir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Schemas (Novos: forecaster/regressor separados)
# ──────────────────────────────────────────────────────────────────────────────


class ExperimentSpec(BaseModel):
    sensors: list[str]
    units: list[str]


class DataSlice(BaseModel):
    start: int = 0
    end: int = 0


class DataSpec(BaseModel):
    jsonFile: str | None = None
    directory: str = "data"
    slice: DataSlice = DataSlice()


class ForecasterModelSpec(BaseModel):
    architecture: str = "lstm"
    hiddenUnits: int = Field(..., gt=0)
    layers: int = Field(..., ge=1)
    dropout: float = Field(..., ge=0.0, le=0.9)
    bidirectional: bool = True
    window: int = Field(..., ge=1)
    horizon: int = Field(..., ge=1)
    targetLen: int = Field(..., ge=1)


class TrainingSpec(BaseModel):
    epochs: int = Field(..., ge=1)
    batchSize: int = Field(..., ge=1)
    learningRate: float = Field(..., gt=0)
    testSize: float = Field(0.2)


class ParamSearchSpec(BaseModel):
    enabled: bool = False
    grid: dict[str, list[Any]] | None = None


class ForecasterTrainRequest(BaseModel):
    experiment: ExperimentSpec
    data: DataSpec
    model: ForecasterModelSpec
    training: TrainingSpec
    paramSearch: ParamSearchSpec = Field(default_factory=ParamSearchSpec)

class ForecasterTrainResponse(BaseModel):
    tag: str
    channels: Dict[str, list[str]]
    trials: list[dict[str, Any]] | None = None


class ModelSpec(BaseModel):
    name: str
    params: Dict[str, Any] | None = None
    grid: Dict[str, List[Any]] | None = None

    @model_validator(mode="after")
    def _params_or_grid(self):
        if not self.params and not self.grid:
            raise ValueError("Modelo precisa de 'params' ou 'grid'.")
        return self


class RegressorTrainRequest(BaseModel):
    featureFile: str
    units: list[str]
    models: List[ModelSpec] = Field(..., min_length=1)

    testSize: float = TEST_SIZE
    augment: str | None = None
    nAugs: int = 1
    permImp: bool = False


class RegressorModelResponse(BaseModel):
    name: str
    modelFile: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    featureImportances: List[float] | None = None


class RegressorTrainResponse(BaseModel):
    tag: str
    models: List[RegressorModelResponse]


class ForecasterInferenceRequest(BaseModel):
    ensaio: Dict[str, Any]
    sensors: List[str]
    units: List[str]
    fctFile: str | None
    sliceStart: int = 0




class ForecasterInferenceResponse(BaseModel):
    tFull: List[int]
    yForecast: List[List[float]]      # n_chan × T
    tPartial: List[int] | None = None
    yTruePartial: List[List[float]] | None = None
    growthStart: int


class RegressorInferenceRequest(BaseModel):
    ensaio: Dict[str, Any]
    sensors: List[str]
    units: List[str]
    model: str
    modelFile: str | None = None
    forecasterFile: str | None = None
    sliceStart: int = 0


class RegressorInferenceResponse(BaseModel):
    yRegression: float


# ──────────────────────────────────────────────────────────────────────────────
# NOVOS ENDPOINTS – Fluxo separado
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/train_forecaster")  # <- sem response_model
def train_forecaster_endpoint(req: ForecasterTrainRequest):
    """Treina APENAS o forecaster e gera features por trial."""
    try:
        mixes = parse_sensors(req.experiment.sensors)
        data_json = (
            Path(req.data.jsonFile)
            if req.data.jsonFile
            else Path(req.data.directory) / "experiments_filtered.json"
        )
        grid = req.paramSearch.grid if req.paramSearch.grid else None

        params = SimpleNamespace(
            hiddenUnits=req.model.hiddenUnits,
            layers=req.model.layers,
            dropout=req.model.dropout,
            bidirectional=req.model.bidirectional,
            window=req.model.window,
            horizon=req.model.horizon,
            targetLen=req.model.targetLen,
            epochs=req.training.epochs,
            batchSize=req.training.batchSize,
            learningRate=req.training.learningRate,
            testSize=req.training.testSize,
            sliceStart=req.data.slice.start,
            sliceEnd=req.data.slice.end,
            paramSearch=req.paramSearch.enabled,
            paramGrid=grid,
        )

        training_data = train_forecaster(
            mixes, req.experiment.units, params, data_json
        )

        trials = training_data.get("trials")
        payload: Dict[str, Any] = {
            "tag": training_data["tag"],
            "channels": training_data["channels"],
        }
        if trials is not None:
            payload["trials"] = trials

        # Remove quaisquer chaves com valor None (garantia extra)
        payload = {k: v for k, v in payload.items() if v is not None}

        return JSONResponse(content=payload)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train_regressor", response_model=RegressorTrainResponse)
def train_regressor_endpoint(req: RegressorTrainRequest):
    """Treina regressores a partir de features previamente extraídas."""
    try:
        feat_path = (
            Path(MODEL_DIR) / req.featureFile
            if not Path(req.featureFile).is_absolute()
            else Path(req.featureFile)
        )
        if not feat_path.exists():
            raise HTTPException(400, f"featureFile não encontrado: {feat_path}")

        data = np.load(feat_path, allow_pickle=True)
        arr_full = data["full_curves"]
        arr_y = data["y"]
        arr_ids = data["uuids"]
        channels = (
            data["channels"].item() if hasattr(data["channels"], "item") else data["channels"]
        )
        tag = str(data["tag"])

        training_data = {
            "full_curves": [arr_full[i] for i in range(arr_full.shape[0])],
            "y": arr_y,
            "uuids": arr_ids.tolist(),
            "channels": channels,
            "tag": tag,
        }

        model_specs = [m.dict() for m in req.models]
        results = train_regressor(
            model_specs,
            training_data,
            req.units,
            test_size=req.testSize,
            augment=req.augment,
            n_augs=req.nAugs,
            perm_imp=req.permImp,
        )

        results_json: list[dict[str, Any]] = []
        for r in results:
            r_json = _jsonable(r)
            r_json["modelFile"] = Path(r_json.pop("model_file")).name
            if "feature_importances_" in r_json:
                r_json["featureImportances"] = r_json.pop("feature_importances_")
            results_json.append(r_json)

        return RegressorTrainResponse(tag=tag, models=results_json)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/infer_forecaster", response_model=ForecasterInferenceResponse)
def infer_forecaster_endpoint(req: ForecasterInferenceRequest):
    """
    Inferência APENAS do forecaster. Internamente utiliza a mesma função 'inference'
    e simplesmente ignora o resultado de regressão.
    """
    try:
        mixes = parse_sensors(req.sensors)
        y_full, _y_reg, y_true_partial, growth_idx = inference(
            ensaio            = req.ensaio,
            sensors           = mixes,
            units             = req.units,
            model_name        = "noop",            # rótulo qualquer; será ignorado do lado do regressor
            model_file        = None,              # não usado
            forecaster_file   = req.fctFile,
            slice_start       = req.sliceStart,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ForecasterInferenceResponse(
        tFull=list(range(req.sliceStart, req.sliceStart + y_full.shape[0])),
        tPartial=(list(range(req.sliceStart, req.sliceStart + y_true_partial.shape[0]))
                  if y_true_partial.size else None),
        yForecast=y_full.T.tolist(),
        yTruePartial=(y_true_partial.T.tolist() if y_true_partial.size else None),
        growthStart=int(growth_idx) if growth_idx is not None else -1
    )


@app.post("/infer_regressor", response_model=RegressorInferenceResponse)
def infer_regressor_endpoint(req: RegressorInferenceRequest):
    """
    Inferência APENAS do regressor. Usa 'inference' para compor a curva com o forecaster
    e depois aplica o regressor, retornando apenas yRegression.
    """
    try:
        mixes = parse_sensors(req.sensors)
        _y_full, y_reg, _y_true_partial, _growth_idx = inference(
            ensaio            = req.ensaio,
            sensors           = mixes,
            units             = req.units,
            model_name        = req.model,
            model_file        = req.modelFile,
            forecaster_file   = req.forecasterFile,
            slice_start       = req.sliceStart,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return RegressorInferenceResponse(
        yRegression=float(y_reg)
    )
