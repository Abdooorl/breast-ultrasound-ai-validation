from pathlib import Path
from typing import Union

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAME = "Parveshiiii/breast-cancer-detector"
MODEL_REVISION = "7c4c1ac11f5f80d382cc641ec6f2d3989e7fa73c"

EXPECTED_ID2LABEL = {
    0: "benign",
    1: "malignant",
    2: "normal",
}


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve the requested inference device."""

    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class BreastUltrasoundClassifier:
    """Frozen pretrained breast-ultrasound image classifier."""

    def __init__(self, device: str = "auto") -> None:
        self.device = resolve_device(device)

        self.processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
        )

        self.model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
        )

        self.model.to(self.device)
        self.model.eval()

        actual_mapping = {
            int(index): label.lower()
            for index, label in self.model.config.id2label.items()
        }

        if actual_mapping != EXPECTED_ID2LABEL:
            raise RuntimeError(
                "Unexpected model class mapping. "
                f"Expected {EXPECTED_ID2LABEL}, got {actual_mapping}"
            )

    def predict(
        self,
        image: Union[str, Path, Image.Image],
    ) -> dict:
        """Return logits, probabilities and predicted class for one image."""

        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                pil_image = img.convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(
                "image must be a filesystem path or PIL.Image.Image"
            )

        inputs = self.processor(
            images=pil_image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probabilities = torch.softmax(logits, dim=-1)

        logits = logits.cpu()
        probabilities = probabilities.cpu()

        predicted_index = int(torch.argmax(probabilities).item())
        predicted_label = EXPECTED_ID2LABEL[predicted_index]

        return {
            "predicted_index": predicted_index,
            "predicted_label": predicted_label,
            "probabilities": {
                EXPECTED_ID2LABEL[index]: float(probabilities[index].item())
                for index in EXPECTED_ID2LABEL
            },
            "logits": {
                EXPECTED_ID2LABEL[index]: float(logits[index].item())
                for index in EXPECTED_ID2LABEL
            },
            "device": str(self.device),
        }