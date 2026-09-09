from pathlib import Path
import hashlib
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.calibration import (
    PROBABILITY_COLUMNS,
    confidence_reliability,
    multiclass_brier_score,
)


PREDICTIONS_PATH = Path(
    "results/predictions/"
    "bus_uclm_predictions.csv"
)

OUTPUT_DIR = Path(
    "results/calibration"
)

FIGURES_DIR = Path(
    "figures"
)

EXPECTED_SHA256 = (
    "634e5a38c5657d5c9801b873d53e504a"
    "b2595b280f5d4269c38d1b80ca543d76"
)

N_BINS = 10


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

    if (
        df["image_id"].nunique()
        != 683
    ):
        raise RuntimeError(
            "Expected 683 unique images."
        )

    return df


def validate_probabilities(
    df: pd.DataFrame,
) -> None:
    probabilities = df[
        PROBABILITY_COLUMNS
    ].to_numpy(dtype=float)

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Non-finite probabilities found."
        )

    if (
        probabilities.min() < 0
        or probabilities.max() > 1
    ):
        raise RuntimeError(
            "Probabilities outside [0, 1]."
        )

    sums = probabilities.sum(
        axis=1
    )

    max_sum_error = float(
        np.abs(
            sums - 1.0
        ).max()
    )

    if max_sum_error > 1e-6:
        raise RuntimeError(
            "Probability-sum validation failed."
        )

    recomputed_confidence = (
        probabilities.max(
            axis=1
        )
    )

    confidence_error = float(
        np.abs(
            recomputed_confidence
            - df[
                "confidence"
            ].to_numpy(dtype=float)
        ).max()
    )

    if confidence_error > 1e-6:
        raise RuntimeError(
            "Confidence validation failed."
        )

    print(
        "Probability validation: PASSED"
    )

    print(
        "Maximum probability-sum error:",
        f"{max_sum_error:.12g}",
    )


def save_reliability_figure(
    bins: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    nonempty = bins[
        bins["count"] > 0
    ]

    plt.figure(
        figsize=(6.5, 6)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )

    plt.plot(
        nonempty[
            "mean_confidence"
        ],
        nonempty[
            "empirical_accuracy"
        ],
        marker="o",
        linewidth=2,
        label="Model",
    )

    plt.xlabel(
        "Mean predicted confidence"
    )

    plt.ylabel(
        "Empirical accuracy"
    )

    plt.title(
        "Reliability Diagram"
    )

    plt.xlim(
        0,
        1,
    )

    plt.ylim(
        0,
        1,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    png_path = (
        FIGURES_DIR
        / "figure_7_reliability_diagram.png"
    )

    pdf_path = (
        FIGURES_DIR
        / "figure_7_reliability_diagram.pdf"
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {png_path}"
    )

    print(
        f"Saved: {pdf_path}"
    )


def main() -> None:
    print(
        "=== CALIBRATION EVALUATION ==="
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
        df[
            "patient_id"
        ].nunique(),
    )

    print()

    validate_probabilities(df)

    bins, summary = (
        confidence_reliability(
            df,
            n_bins=N_BINS,
        )
    )

    brier = (
        multiclass_brier_score(
            df
        )
    )

    summary[
        "multiclass_brier_score"
    ] = brier

    if not (
        0.0
        <= summary["ece"]
        <= 1.0
    ):
        raise RuntimeError(
            "ECE outside [0, 1]."
        )

    if not (
        0.0
        <= brier
        <= 2.0
    ):
        raise RuntimeError(
            "Brier score outside [0, 2]."
        )

    if (
        bins["count"].sum()
        != len(df)
    ):
        raise RuntimeError(
            "Calibration-bin count mismatch."
        )

    print()
    print(
        "Calibration validation: PASSED"
    )

    print()
    print(
        "Overall accuracy:",
        f"{summary['overall_accuracy']:.4f}",
    )

    print(
        "Mean confidence:",
        f"{summary['mean_confidence']:.4f}",
    )

    print(
        "Confidence - accuracy:",
        f"{summary['confidence_minus_accuracy']:.4f}",
    )

    print(
        "ECE:",
        f"{summary['ece']:.4f}",
    )

    print(
        "Multiclass Brier score:",
        f"{brier:.4f}",
    )

    print()
    print(
        bins.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bins_csv = (
        OUTPUT_DIR
        / "reliability_bins.csv"
    )

    bins_md = (
        OUTPUT_DIR
        / "reliability_bins.md"
    )

    summary_json = (
        OUTPUT_DIR
        / "calibration_summary.json"
    )

    bins.to_csv(
        bins_csv,
        index=False,
    )

    bins_md.write_text(
        bins.to_markdown(
            index=False,
            floatfmt=".4f",
        )
        + "\n"
    )

    summary_json.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        f"Saved: {bins_csv}"
    )

    print(
        f"Saved: {bins_md}"
    )

    print(
        f"Saved: {summary_json}"
    )

    save_reliability_figure(
        bins
    )

    print()
    print(
        "=== CALIBRATION EVALUATION COMPLETE ==="
    )


if __name__ == "__main__":
    main()