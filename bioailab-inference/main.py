"""
BioAI Lab - API Unificada

API para processamento de dados de experimentos e predição de bactérias.
Arquitetura Clean com responsabilidades separadas.
"""

import os

# Force a non-GUI backend for Matplotlib to avoid thread/GUI locks in workers.
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.interface.router import router


app = FastAPI(
    title="BioAI Lab - API Unificada",
    description="API para processamento de dados de experimentos e predição de bactérias.",
    version="2.0.0"
)

# CORS
origins = [
    "http://localhost",
    "http://localhost:3001",
    "http://localhost:3000",
    "http://localhost:3002",
    "http://localhost:3003",
    "https://www.bioailab.icu",
    "https://bioailab.com.br"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router
app.include_router(router)


@app.get("/", summary="Health check")
def read_root():
    return {"status": "API Unificada da BioAI Lab está online", "version": "2.0.0"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True, timeout_keep_alive=600000000)
