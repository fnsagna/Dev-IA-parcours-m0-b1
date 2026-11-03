"""API de prédiction d'annulation — Escale Atlantique (mission FastIA)."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from app.schemas import Booking, PredictionResponse

logger.remove()
logger.add(
    "logs/api.log",
    level="INFO",
    rotation="5 MB",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "model" / "cancellation_model.joblib"))
MODEL_INFO_PATH = ROOT_DIR / "model" / "model_info.json"

# Objets partagés par toutes les requêtes (ici : le modèle)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Exécuté UNE fois au démarrage : on charge le modèle en mémoire
    state["model"] = joblib.load(MODEL_PATH)
    yield
    # Exécuté à l'arrêt du serveur
    state.clear()

app = FastAPI(
    title="Escale Atlantique — Prédiction d'annulation",
    description="Estime la probabilité qu'une réservation soit annulée.",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health() -> dict:
    """Le service est-il en vie, et le modèle chargé ?"""
    return {"status": "ok", "model_loaded": "model" in state}

@app.get("/info")
def info() -> dict:
    """Nom du modèle, date d'entraînement et colonnes attendues."""
    import json
    data = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    return {
        "model_name": data["model_name"],
        "trained_on": data["trained_on"],
        "features": data["features"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(booking: Booking) -> PredictionResponse:
    """Prédit l'annulation d'une réservation.
    """
    start = time.perf_counter()

    try:
        model = state["model"]
        df = pd.DataFrame([booking.model_dump()]) 
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    duree_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "predict | entrée={} | annulation={} | proba={:.3f} | durée={:.1f}ms",
        booking.model_dump(), bool(pred), float(proba), duree_ms,
    )

    return PredictionResponse(annulation=bool(pred), probabilite_annulation=float(proba))

@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(bookings: list[Booking]) -> list[PredictionResponse]:
    df = pd.DataFrame([b.model_dump() for b in bookings])
    model = state["model"]
    preds = model.predict(df)
    probas = model.predict_proba(df)[:, 1]
    return [
        PredictionResponse(annulation=bool(p), probabilite_annulation=float(pr))
        for p, pr in zip(preds, probas)
    ]
