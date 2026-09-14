"""FastAPI service for the breast-ultrasound external-testing research demo.

The API wraps the same frozen inference implementation used by the research
pipeline. It is a research prototype only and is not intended for clinical
diagnosis.
"""

import os
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference.model import (
    BreastUltrasoundClassifier,
    EXPECTED_ID2LABEL,
    MODEL_NAME,
    MODEL_REVISION,
)


RESEARCH_NOTICE = "Research use only. Not for clinical diagnosis."

MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")

app = FastAPI(
    title="Breast Ultrasound AI Validation API",
    version="0.2.0",
    description=(
        "Research prototype for external testing of a frozen breast-ultrasound "
        "Vision Transformer. Not intended for clinical diagnosis."
    ),
)

_classifier: BreastUltrasoundClassifier | None = None


def get_classifier() -> BreastUltrasoundClassifier:
    """Load the frozen classifier once and reuse it across requests."""

    global _classifier

    if _classifier is None:
        _classifier = BreastUltrasoundClassifier(device=MODEL_DEVICE)

    return _classifier


@app.get("/health")
def health() -> dict:
    """Return basic service status without forcing model initialization."""

    return {
        "status": "ok",
        "research_only": True,
        "model_loaded": _classifier is not None,
    }


@app.get("/model-info")
def model_info() -> dict:
    """Return the exact frozen model identity used by the research pipeline."""

    return {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "classes": [
            EXPECTED_ID2LABEL[index]
            for index in sorted(EXPECTED_ID2LABEL)
        ],
        "research_only": True,
        "disclaimer": RESEARCH_NOTICE,
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
) -> dict:
    """Run the frozen classifier on one uploaded ultrasound image."""

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    try:
        with Image.open(BytesIO(contents)) as image:
            image.load()
            prediction = get_classifier().predict(image)

    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file could not be decoded as an image.",
        ) from exc

    probabilities = prediction["probabilities"]

    sorted_probabilities = sorted(
        probabilities.values(),
        reverse=True,
    )

    confidence = sorted_probabilities[0]
    prediction_margin = (
        sorted_probabilities[0] - sorted_probabilities[1]
    )

    return {
        "predicted_class": prediction["predicted_label"],
        "probabilities": probabilities,
        "confidence": confidence,
        "prediction_margin": prediction_margin,
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
        },
        "device": prediction["device"],
        "research_only": True,
        "disclaimer": RESEARCH_NOTICE,
    }
