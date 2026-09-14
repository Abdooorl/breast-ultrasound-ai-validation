from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.api.main as api_main


client = TestClient(api_main.app)


class FakeClassifier:
    """Small deterministic stand-in used only for API tests."""

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


def make_png_bytes() -> bytes:
    buffer = BytesIO()

    Image.new(
        "RGB",
        (16, 16),
        "white",
    ).save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["research_only"] is True


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
    assert body["confidence"] == 0.70
    assert body["prediction_margin"] == pytest.approx(0.50)
    assert body["probabilities"]["malignant"] == 0.20
    assert body["research_only"] is True


def test_analyze_rejects_non_image():
    response = client.post(
        "/analyze",
        files={
            "file": (
                "not-image.txt",
                b"this is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
