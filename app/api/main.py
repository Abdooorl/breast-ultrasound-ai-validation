"""FastAPI service for the breast-ultrasound external-testing research demo.

The API wraps the same frozen inference implementation used by the research
pipeline. It is a research prototype only and is not intended for clinical
diagnosis.
"""

from io import BytesIO
import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.inference.model import (
    BreastUltrasoundClassifier,
    EXPECTED_ID2LABEL,
    MODEL_NAME,
    MODEL_REVISION,
)


logger = logging.getLogger(__name__)

RESEARCH_NOTICE = "Research use only. Not for clinical diagnosis."

MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")

MAX_UPLOAD_BYTES = int(
    os.getenv(
        "MAX_UPLOAD_BYTES",
        str(10 * 1024 * 1024),
    )
)

MAX_IMAGE_PIXELS = int(
    os.getenv(
        "MAX_IMAGE_PIXELS",
        str(25_000_000),
    )
)

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
}


class ModelIdentity(BaseModel):
    name: str
    revision: str


class HealthResponse(BaseModel):
    status: str
    research_only: bool
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_revision: str
    classes: list[str]
    research_only: bool
    disclaimer: str


class AnalyzeResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    confidence: float
    prediction_margin: float
    model: ModelIdentity
    device: str
    research_only: bool
    disclaimer: str


app = FastAPI(
    title="Breast Ultrasound AI Validation API",
    version="0.3.0",
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
        _classifier = BreastUltrasoundClassifier(
            device=MODEL_DEVICE,
        )

    return _classifier


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Return service status without forcing model initialization."""

    return HealthResponse(
        status="ok",
        research_only=True,
        model_loaded=_classifier is not None,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info() -> ModelInfoResponse:
    """Return the exact frozen model identity used by the research pipeline."""

    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        classes=[
            EXPECTED_ID2LABEL[index]
            for index in sorted(EXPECTED_ID2LABEL)
        ],
        research_only=True,
        disclaimer=RESEARCH_NOTICE,
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
)
async def analyze(
    file: UploadFile = File(...),
) -> AnalyzeResponse:
    """Run the frozen classifier on one uploaded ultrasound image."""

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image type. "
                "Only PNG and JPEG images are accepted."
            ),
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Uploaded image exceeds the maximum allowed file size.",
        )

    try:
        with Image.open(BytesIO(contents)) as image:
            width, height = image.size

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded image has invalid dimensions.",
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Uploaded image exceeds the maximum allowed dimensions.",
                )

            image.load()

            try:
                prediction = get_classifier().predict(image)

            except Exception as exc:
                logger.exception(
                    "Model inference failed."
                )

                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Model inference is temporarily unavailable."
                    ),
                ) from exc

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file could not be decoded as a valid image.",
        ) from exc

    probabilities = prediction["probabilities"]

    sorted_probabilities = sorted(
        probabilities.values(),
        reverse=True,
    )

    confidence = sorted_probabilities[0]

    prediction_margin = (
        sorted_probabilities[0]
        - sorted_probabilities[1]
    )

    return AnalyzeResponse(
        predicted_class=prediction["predicted_label"],
        probabilities=probabilities,
        confidence=confidence,
        prediction_margin=prediction_margin,
        model=ModelIdentity(
            name=MODEL_NAME,
            revision=MODEL_REVISION,
        ),
        device=prediction["device"],
        research_only=True,
        disclaimer=RESEARCH_NOTICE,
    )
