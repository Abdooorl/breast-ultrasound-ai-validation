from typing import Any

import numpy as np
import pandas as pd


def evaluate_confidence_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """
    Evaluate selective classification at one
    maximum-softmax confidence threshold.

    A case is accepted if confidence >= threshold.
    Otherwise the model abstains.
    """

    required = {
        "ground_truth",
        "predicted_label",
        "confidence",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    accepted_mask = (
        df["confidence"] >= threshold
    )

    accepted = df.loc[
        accepted_mask
    ].copy()

    abstained = df.loc[
        ~accepted_mask
    ].copy()

    n_total = len(df)
    n_accepted = len(accepted)
    n_abstained = len(abstained)

    coverage = (
        n_accepted / n_total
        if n_total
        else np.nan
    )

    abstention_rate = (
        n_abstained / n_total
        if n_total
        else np.nan
    )

    if n_accepted:
        accepted_accuracy = (
            accepted["ground_truth"]
            == accepted["predicted_label"]
        ).mean()
    else:
        accepted_accuracy = np.nan

    true_malignant = (
        df["ground_truth"] == "malignant"
    )

    true_non_malignant = ~true_malignant

    accepted_malignant = (
        accepted["ground_truth"]
        == "malignant"
    )

    accepted_non_malignant = (
        accepted["ground_truth"]
        != "malignant"
    )

    predicted_malignant = (
        accepted["predicted_label"]
        == "malignant"
    )

    tp = int(
        (
            accepted_malignant
            & predicted_malignant
        ).sum()
    )

    fn = int(
        (
            accepted_malignant
            & ~predicted_malignant
        ).sum()
    )

    tn = int(
        (
            accepted_non_malignant
            & ~predicted_malignant
        ).sum()
    )

    fp = int(
        (
            accepted_non_malignant
            & predicted_malignant
        ).sum()
    )

    accepted_sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else np.nan
    )

    accepted_specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else np.nan
    )

    total_malignant = int(
        true_malignant.sum()
    )

    total_non_malignant = int(
        true_non_malignant.sum()
    )

    accepted_malignant_count = int(
        accepted_malignant.sum()
    )

    abstained_malignant_count = int(
        (
            abstained["ground_truth"]
            == "malignant"
        ).sum()
    )

    accepted_non_malignant_count = int(
        accepted_non_malignant.sum()
    )

    abstained_non_malignant_count = int(
        (
            abstained["ground_truth"]
            != "malignant"
        ).sum()
    )

    malignant_coverage = (
        accepted_malignant_count
        / total_malignant
        if total_malignant
        else np.nan
    )

    non_malignant_coverage = (
        accepted_non_malignant_count
        / total_non_malignant
        if total_non_malignant
        else np.nan
    )

    malignant_accepted_false_negative_rate = (
        fn / total_malignant
        if total_malignant
        else np.nan
    )

    malignant_detected_or_abstained_rate = (
        (
            tp
            + abstained_malignant_count
        )
        / total_malignant
        if total_malignant
        else np.nan
    )

    accepted_error_count = int(
        (
            accepted["ground_truth"]
            != accepted["predicted_label"]
        ).sum()
    )

    accepted_risk = (
        1.0 - accepted_accuracy
        if np.isfinite(
            accepted_accuracy
        )
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "n_total": int(n_total),
        "n_accepted": int(n_accepted),
        "n_abstained": int(n_abstained),
        "coverage": float(coverage),
        "abstention_rate": float(
            abstention_rate
        ),
        "accepted_accuracy": float(
            accepted_accuracy
        ),
        "accepted_risk": float(
            accepted_risk
        ),
        "accepted_errors": (
            accepted_error_count
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accepted_sensitivity": float(
            accepted_sensitivity
        ),
        "accepted_specificity": float(
            accepted_specificity
        ),
        "total_malignant": (
            total_malignant
        ),
        "accepted_malignant": (
            accepted_malignant_count
        ),
        "abstained_malignant": (
            abstained_malignant_count
        ),
        "malignant_coverage": float(
            malignant_coverage
        ),
        "total_non_malignant": (
            total_non_malignant
        ),
        "accepted_non_malignant": (
            accepted_non_malignant_count
        ),
        "abstained_non_malignant": (
            abstained_non_malignant_count
        ),
        "non_malignant_coverage": float(
            non_malignant_coverage
        ),
        "malignant_accepted_false_negative_rate": float(
            malignant_accepted_false_negative_rate
        ),
        "malignant_detected_or_abstained_rate": float(
            malignant_detected_or_abstained_rate
        ),
    }


def sweep_confidence_thresholds(
    df: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    results = [
        evaluate_confidence_threshold(
            df,
            threshold,
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(results)