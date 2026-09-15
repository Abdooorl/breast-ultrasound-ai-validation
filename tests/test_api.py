from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import app.api.main as api_main


client = TestClient(api_main.app)


class FakeClassifier:
    """Deterministic stand-in used only for API tests."""

    def predict(self, image):
        return {
            "predicted_index": 0,
            "predicted_label": "benign",
            "probabilities": {
                "benign": 0.70,
                "malignant": 0.20,
                "normal": 0.10,
            },
            "logits": {
                "benign": 2.0,
                "malignant": 1.0,
                "normal": 0.0,
            },
            "device": "cpu",
        }


class BrokenClassifier:
    """Stand-in that simulates an inference failure."""

    def predict(self, image):
        raise RuntimeError("simulated model failure")


def make_png_bytes(
    width: int = 16,
    height: int = 16,
) -> bytes:
    buffer = BytesIO()

    Image.new(
        "RGB",
        (width, height),
        "white",
    ).save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["research_only"] is True
    assert "model_loaded" in body


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()

    assert body["model_name"] == api_main.MODEL_NAME
    assert body["model_revision"] == api_main.MODEL_REVISION

    assert body["classes"] == [
        "benign",
        "malignant",
        "normal",
    ]

    assert body["research_only"] is True


def test_analyze(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "get_classifier",
        lambda: FakeClassifier(),
    )

    response = client.post(
        "/analyze",
        files={
            "file": (
                "sample.png",
                make_png_bytes(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_class"] == "benign"

    assert body["probabilities"] == {
        "benign": 0.70,
        "malignant": 0.20,
        "normal": 0.10,
    }

    assert body["confidence"] == pytest.approx(
        0.70
    )

    assert body["prediction_margin"] == pytest.approx(
        0.50
    )

    assert body["research_only"] is True

    assert body["model"]["name"] == api_main.MODEL_NAME

    assert (
        body["model"]["revision"]
        == api_main.MODEL_REVISION
    )


def test_analyze_rejects_empty_file():
    response = client.post(
        "/analyze",
        files={
            "file": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_analyze_rejects_non_image():
    response = client.post(
        "/analyze",
        files={
            "file": (
                "not-image.png",
                b"this is not an image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_analyze_rejects_unsupported_media_type():
    response = client.post(
        "/analyze",
        files={
            "file": (
                "sample.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415


def test_analyze_rejects_oversized_upload(
    monkeypatch,
):
    monkeypatch.setattr(
        api_main,
        "MAX_UPLOAD_BYTES",
        8,
    )

    response = client.post(
        "/analyze",
        files={
            "file": (
                "large.png",
                b"0123456789",
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def test_analyze_rejects_excessive_dimensions(
    monkeypatch,
):
    monkeypatch.setattr(
        api_main,
        "MAX_IMAGE_PIXELS",
        100,
    )

    response = client.post(
        "/analyze",
        files={
            "file": (
                "large-dimensions.png",
                make_png_bytes(
                    width=16,
                    height=16,
                ),
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def test_analyze_handles_model_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        api_main,
        "get_classifier",
        lambda: BrokenClassifier(),
    )

    response = client.post(
        "/analyze",
        files={
            "file": (
                "sample.png",
                make_png_bytes(),
                "image/png",
            )
        },
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "Model inference is temporarily unavailable."
        )
    }
