import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.model import (
    BreastUltrasoundClassifier,
    MODEL_NAME,
    MODEL_REVISION,
)


def normalize_yes_no(value: str) -> str:
    value = str(value).strip().lower()

    if value in {"yes", "y"}:
        return "yes"

    if value in {"no", "n"}:
        return "no"

    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen-model inference on BUS-UCLM."
    )

    parser.add_argument(
        "--info",
        type=Path,
        default=Path("data/raw/BUS-UCLM/INFO.csv"),
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/raw/BUS-UCLM/images"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/predictions/bus_uclm_predictions.csv"
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda", "auto"],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of images for smoke testing.",
    )

    args = parser.parse_args()

    metadata = pd.read_csv(args.info, sep=";")

    metadata.columns = [
        column.strip()
        for column in metadata.columns
    ]

    for column in [
        "Image",
        "Label",
        "Doppler",
        "Marks",
        "Combined",
    ]:
        metadata[column] = (
            metadata[column]
            .astype(str)
            .str.strip()
        )

    metadata = metadata.sort_values("Image").reset_index(drop=True)

    if args.limit is not None:
        metadata = metadata.head(args.limit)

    print(f"Images scheduled: {len(metadata)}")
    print(f"Device: {args.device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Revision: {MODEL_REVISION}")

    print("\nLoading classifier...")

    classifier = BreastUltrasoundClassifier(
        device=args.device
    )

    records = []

    for _, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Running inference",
    ):
        image_name = row["Image"]
        image_path = args.image_dir / image_name

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        result = classifier.predict(image_path)

        probabilities = result["probabilities"]
        logits = result["logits"]

        sorted_probabilities = sorted(
            probabilities.values(),
            reverse=True,
        )

        confidence = sorted_probabilities[0]
        probability_margin = (
            sorted_probabilities[0]
            - sorted_probabilities[1]
        )

        doppler = normalize_yes_no(row["Doppler"])
        marks = normalize_yes_no(row["Marks"])
        combined = normalize_yes_no(row["Combined"])

        clean_input = (
            doppler == "no"
            and marks == "no"
            and combined == "no"
        )

        records.append(
            {
                "image_id": image_name,
                "patient_id": image_name.split("_")[0],
                "ground_truth": row["Label"].lower(),
                "predicted_label": result["predicted_label"],
                "predicted_index": result["predicted_index"],

                "prob_benign": probabilities["benign"],
                "prob_malignant": probabilities["malignant"],
                "prob_normal": probabilities["normal"],

                "logit_benign": logits["benign"],
                "logit_malignant": logits["malignant"],
                "logit_normal": logits["normal"],

                "confidence": confidence,
                "probability_margin": probability_margin,

                "doppler": doppler,
                "marks": marks,
                "combined": combined,
                "clean_input": clean_input,

                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "device": result["device"],
            }
        )

    predictions = pd.DataFrame(records)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        args.output,
        index=False,
    )

    print("\n=== BATCH INFERENCE COMPLETE ===")
    print(f"Rows written: {len(predictions)}")
    print(f"Output: {args.output}")

    probability_sums = (
        predictions["prob_benign"]
        + predictions["prob_malignant"]
        + predictions["prob_normal"]
    )

    max_probability_error = (
        probability_sums - 1.0
    ).abs().max()

    print(
        "Maximum probability-sum error:",
        f"{max_probability_error:.12g}",
    )


if __name__ == "__main__":
    main()