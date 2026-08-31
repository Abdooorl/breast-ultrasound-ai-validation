import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.model import BreastUltrasoundClassifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen breast-ultrasound classifier on one image."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to an ultrasound image.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device.",
    )

    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image does not exist: {args.image}")

    classifier = BreastUltrasoundClassifier(device=args.device)
    result = classifier.predict(args.image)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()