from typing import Any

import numpy as np
import pandas as pd


CLASS_LABELS = [
    "benign",
    "malignant",
    "normal",
]

PROBABILITY_COLUMNS = [
    "prob_benign",
    "prob_malignant",
    "prob_normal",
]


def validate_calibration_input(
    df: pd.DataFrame,
) -> None:
    required = {
        "ground_truth",
        "predicted_label",
        "confidence",
        *PROBABILITY_COLUMNS,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    unknown_labels = set(
        df["ground_truth"].unique()
    ).difference(CLASS_LABELS)

    if unknown_labels:
        raise ValueError(
            "Unknown ground-truth labels: "
            + ", ".join(
                sorted(unknown_labels)
            )
        )


def confidence_reliability(
    df: pd.DataFrame,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Calibration of maximum softmax confidence
    against prediction correctness.

    ECE uses equal-width confidence bins.
    """

    validate_calibration_input(df)

    confidence = df[
        "confidence"
    ].to_numpy(dtype=float)

    correct = (
        df["ground_truth"]
        == df["predicted_label"]
    ).to_numpy(dtype=bool)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_indices = np.digitize(
        confidence,
        edges[1:-1],
        right=False,
    )

    rows = []

    total = len(df)
    ece = 0.0

    for bin_index in range(n_bins):
        mask = (
            bin_indices == bin_index
        )

        count = int(mask.sum())

        lower = float(
            edges[bin_index]
        )

        upper = float(
            edges[bin_index + 1]
        )

        if count:
            mean_confidence = float(
                confidence[mask].mean()
            )

            empirical_accuracy = float(
                correct[mask].mean()
            )

            calibration_gap = float(
                abs(
                    empirical_accuracy
                    - mean_confidence
                )
            )

            ece += (
                count
                / total
                * calibration_gap
            )
        else:
            mean_confidence = np.nan
            empirical_accuracy = np.nan
            calibration_gap = np.nan

        rows.append(
            {
                "bin": bin_index + 1,
                "lower_bound": lower,
                "upper_bound": upper,
                "count": count,
                "mean_confidence": (
                    mean_confidence
                ),
                "empirical_accuracy": (
                    empirical_accuracy
                ),
                "calibration_gap": (
                    calibration_gap
                ),
            }
        )

    bins = pd.DataFrame(rows)

    overall_accuracy = float(
        correct.mean()
    )

    mean_confidence = float(
        confidence.mean()
    )

    summary = {
        "n_images": int(total),
        "n_bins": int(n_bins),
        "binning": "equal-width",
        "overall_accuracy": (
            overall_accuracy
        ),
        "mean_confidence": (
            mean_confidence
        ),
        "confidence_minus_accuracy": float(
            mean_confidence
            - overall_accuracy
        ),
        "ece": float(ece),
    }

    return bins, summary


def multiclass_brier_score(
    df: pd.DataFrame,
) -> float:
    """
    Multiclass Brier score:

    mean over cases of the sum across classes
    of (predicted probability - one-hot truth)^2.

    Lower is better. For three classes this
    ranges from 0 to 2.
    """

    validate_calibration_input(df)

    probabilities = df[
        PROBABILITY_COLUMNS
    ].to_numpy(dtype=float)

    label_to_index = {
        label: index
        for index, label
        in enumerate(CLASS_LABELS)
    }

    truth_indices = (
        df["ground_truth"]
        .map(label_to_index)
        .to_numpy(dtype=int)
    )

    targets = np.zeros_like(
        probabilities
    )

    targets[
        np.arange(len(df)),
        truth_indices,
    ] = 1.0

    squared_error = (
        probabilities
        - targets
    ) ** 2

    return float(
        squared_error.sum(
            axis=1
        ).mean()
    )