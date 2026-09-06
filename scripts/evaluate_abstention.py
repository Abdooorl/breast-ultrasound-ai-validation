from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.abstention import (
    sweep_confidence_thresholds,
)


PREDICTIONS_PATH = Path(
    "results/predictions/"
    "bus_uclm_predictions.csv"
)

OUTPUT_DIR = Path(
    "results/uncertainty"
)

EXPECTED_SHA256 = (
    "634e5a38c5657d5c9801b873d53e504a"
    "b2595b280f5d4269c38d1b80ca543d76"
)

THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


def load_predictions() -> pd.DataFrame:
    actual_hash = hashlib.sha256(
        PREDICTIONS_PATH.read_bytes()
    ).hexdigest()

    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(
            "Frozen prediction checksum mismatch.\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Actual:   {actual_hash}"
        )

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    if len(df) != 683:
        raise RuntimeError(
            f"Expected 683 rows, found {len(df)}."
        )

    return df


def verify_sweep(
    results: pd.DataFrame,
) -> None:
    if len(results) != len(
        THRESHOLDS
    ):
        raise RuntimeError(
            "Unexpected number of thresholds."
        )

    if not np.all(
        np.diff(
            results["coverage"]
        ) <= 1e-12
    ):
        raise RuntimeError(
            "Coverage must not increase "
            "as threshold increases."
        )

    if not np.all(
        np.diff(
            results["abstention_rate"]
        ) >= -1e-12
    ):
        raise RuntimeError(
            "Abstention rate must not decrease "
            "as threshold increases."
        )

    if not (
        results["n_accepted"]
        + results["n_abstained"]
        == results["n_total"]
    ).all():
        raise RuntimeError(
            "Accepted + abstained count mismatch."
        )

    if not (
        results["accepted_malignant"]
        + results["abstained_malignant"]
        == results["total_malignant"]
    ).all():
        raise RuntimeError(
            "Malignant count mismatch."
        )

    print(
        "Threshold sweep validation: PASSED"
    )


def build_display_table(
    results: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "threshold",
        "coverage",
        "abstention_rate",
        "accepted_accuracy",
        "accepted_risk",
        "accepted_sensitivity",
        "accepted_specificity",
        "fn",
        "fp",
        "abstained_malignant",
        "malignant_coverage",
        "malignant_accepted_false_negative_rate",
    ]

    table = results[
        columns
    ].copy()

    numeric_columns = [
        "coverage",
        "abstention_rate",
        "accepted_accuracy",
        "accepted_risk",
        "accepted_sensitivity",
        "accepted_specificity",
        "malignant_coverage",
        "malignant_accepted_false_negative_rate",
    ]

    table[numeric_columns] = (
        table[numeric_columns]
        .round(4)
    )

    return table


def main() -> None:
    print(
        "=== CONFIDENCE-BASED "
        "ABSTENTION SWEEP ==="
    )

    df = load_predictions()

    print(
        "Frozen prediction checksum: verified"
    )
    print(
        "Images:",
        len(df),
    )
    print(
        "Patients:",
        df["patient_id"].nunique(),
    )
    print(
        "Thresholds:",
        THRESHOLDS,
    )
    print()

    results = (
        sweep_confidence_thresholds(
            df,
            THRESHOLDS,
        )
    )

    verify_sweep(results)

    display = build_display_table(
        results
    )

    print()
    print(
        display.to_string(
            index=False
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIR
        / "abstention_threshold_sweep.csv"
    )

    md_path = (
        OUTPUT_DIR
        / "abstention_threshold_sweep.md"
    )

    json_path = (
        OUTPUT_DIR
        / "abstention_threshold_sweep.json"
    )

    results.to_csv(
        csv_path,
        index=False,
    )

    md_path.write_text(
        display.to_markdown(
            index=False
        )
        + "\n"
    )

    json_path.write_text(
        json.dumps(
            results.to_dict(
                orient="records"
            ),
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        f"Saved: {csv_path}"
    )
    print(
        f"Saved: {md_path}"
    )
    print(
        f"Saved: {json_path}"
    )

    print()
    print(
        "=== ABSTENTION SWEEP COMPLETE ==="
    )


if __name__ == "__main__":
    main()