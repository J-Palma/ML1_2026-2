"""Load the selected model and predict the Kellgren-Lawrence grade."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.features import extract_image_features

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "xgboost_forest.joblib"
CLASS_NAMES = ("Normal", "Dudosa", "Leve", "Moderada", "Severa")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict_image(image_gray: np.ndarray) -> dict:
    """Predict one grayscale knee radiograph with the saved XGBoost model."""
    if image_gray is None or image_gray.ndim != 2 or image_gray.size == 0:
        raise ValueError("La imagen debe ser una radiografía en escala de grises válida.")
    feature_dict, _, _ = extract_image_features(image_gray)
    features = np.fromiter(feature_dict.values(), dtype=np.float32).reshape(1, -1)
    model = load_model()
    probabilities = model.predict_proba(features)[0]
    predicted_class = int(np.argmax(probabilities))
    return {
        "class_index": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": float(probabilities[predicted_class]),
        "probabilities": {
            name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)
        },
        "notice": "Demostración académica; no usar para diagnóstico ni decisiones clínicas.",
    }
