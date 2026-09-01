import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    evaluate_malignant_binary,
    evaluate_multiclass,
)


def evaluate_subset(
    df: pd.DataFrame,
    name: str,
) -> dict:
    return {
        "subset": name,
        "multiclass": evaluate_multiclass(df),
        "malignant_vs_non_malignant": (
            evaluate_malignant_binary(df)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen BUS-UCLM predictions."
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "results/predictions/bus_uclm_predictions.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/metrics/baseline_metrics.json"
        ),
    )

    args = parser.parse_args()

    df = pd.read_csv(args.predictions)

    required_columns = {
        "ground_truth",
        "predicted_label",
        "prob_malignant",
        "clean_input",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    clean_mask = (
        df["clean_input"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    clean_df = df.loc[clean_mask].copy()

    results = {
        "primary_full_dataset": evaluate_subset(
            df,
            "full_bus_uclm",
        ),
        "secondary_clean_input": evaluate_subset(
            clean_df,
            "clean_input",
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(args.output, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        json.dumps(
            results,
            indent=2,
        )
    )

    print(
        f"\nSaved metrics to: {args.output}"
    )


if __name__ == "__main__":
    main()