"""Load the selected model and predict the Kellgren-Lawrence grade."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import json

from src.features import extract_image_features, PREPROCESSING_SPEC

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "best_model.joblib"
CLASS_NAMES = ("Normal", "Dudosa", "Leve", "Moderada", "Severa")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict_image(image_gray: np.ndarray) -> dict:
    """Predict one grayscale knee radiograph with the versioned final model."""
    if image_gray is None or image_gray.ndim != 2 or image_gray.size == 0:
        raise ValueError("La imagen debe ser una radiografía en escala de grises válida.")
    metadata = json.loads((PROJECT_DIR / 'models' / 'hyperparameters.json').read_text(encoding='utf-8'))
    if metadata.get('preprocessing') != PREPROCESSING_SPEC:
        raise ValueError('El modelo y el preprocesamiento tienen versiones distintas; reentrenar o restaurar ambos.')
    feature_dict, _, flat = extract_image_features(image_gray)
    features = np.fromiter(feature_dict.values(), dtype=np.float32).reshape(1, -1)
    if metadata.get('input_kind') == 'flat': features = flat.reshape(1, -1)
    model = load_model()
    probabilities = model.predict_proba(features)[0]
    # Use the classifier's decision, as in the reported evaluation. For SVC,
    # predict() and argmax(predict_proba()) can legitimately differ.
    predicted_class = int(model.predict(features)[0])
    return {
        "model": metadata['name'],
        "protocol": metadata['protocol'],
        "class_index": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": float(probabilities[list(model.classes_).index(predicted_class)]),
        "probabilities": {
            name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)
        },
        "notice": "Demostración académica; no usar para diagnóstico ni decisiones clínicas.",
        "probability_note": "Puntuaciones del modelo sin calibración clínica; no expresan certeza diagnóstica.",
    }
