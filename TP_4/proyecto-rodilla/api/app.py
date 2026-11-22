"""FastAPI endpoint for the academic knee-grade demonstration."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference import MODEL_PATH, load_model, predict_image

app = FastAPI(
    title="Clasificador académico de artrosis de rodilla",
    version="1.0.0",
    description=(
        "Demostración académica de clasificación de radiografías en cinco grados. "
        "No es un dispositivo médico y no debe usarse para diagnóstico."
    ),
)


@app.get("/")
def root():
    return {
        "name": app.title,
        "docs": "/docs",
        "notice": "Uso académico únicamente; no usar para decisiones clínicas.",
    }


@app.get("/health")
def health():
    try:
        model = load_model()
        return {"status": "ok", "model_ready": True, "classes": model.classes_.tolist()}
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Modelo no disponible; ejecutá el pipeline de entrenamiento.") from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg", "image/bmp", "image/tiff"}:
        raise HTTPException(status_code=415, detail="Subí una imagen PNG, JPG, BMP o TIFF.")
    content = await file.read(15 * 1024 * 1024 + 1)
    if not content or len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="La imagen está vacía o supera 15 MB.")
    encoded = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(status_code=400, detail="No se pudo leer el archivo de imagen.")
    try:
        return predict_image(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

