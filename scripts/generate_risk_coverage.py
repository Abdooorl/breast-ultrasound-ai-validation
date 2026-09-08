from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREDICTIONS_PATH = Path(
    "results/predictions/bus_uclm_predictions.csv"
)

OUTPUT_DIR = Path("results/uncertainty")
FIGURES_DIR = Path("figures")

EXPECTED_SHA256 = (
    "634e5a38c5657d5c9801b873d53e504a"
    "b2595b280f5d4269c38d1b80ca543d76"
)


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

    df = pd.read_csv(PREDICTIONS_PATH)

    required = {
        "image_id",
        "ground_truth",
        "predicted_label",
        "confidence",
    }

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if len(df) != 683:
        raise RuntimeError(
            f"Expected 683 rows, found {len(df)}."
        )

    return df


def compute_risk_coverage(
    df: pd.DataFrame,
) -> pd.DataFrame:
    ranked = df.sort_values(
        "confidence",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    ranked["correct"] = (
        ranked["ground_truth"]
        == ranked["predicted_label"]
    )

    ranked["cumulative_correct"] = (
        ranked["correct"].cumsum()
    )

    ranked["accepted_count"] = (
        np.arange(len(ranked)) + 1
    )

    ranked["coverage"] = (
        ranked["accepted_count"]
        / len(ranked)
    )

    ranked["accepted_accuracy"] = (
        ranked["cumulative_correct"]
        / ranked["accepted_count"]
    )

    ranked["risk"] = (
        1.0
        - ranked["accepted_accuracy"]
    )

    return ranked[
        [
            "accepted_count",
            "coverage",
            "accepted_accuracy",
            "risk",
            "confidence",
        ]
    ].copy()

def compute_summary(
    curve: pd.DataFrame,
) -> dict:
    full_coverage_row = curve.iloc[-1]

    aurc = float(
        np.trapezoid(
            curve["risk"],
            curve["coverage"],
        )
    )

    coverage_targets = [
        0.25,
        0.50,
        0.75,
        1.00,
    ]

    risk_at_coverage = {}

    for target in coverage_targets:
        index = (
            curve["coverage"]
            - target
        ).abs().idxmin()

        row = curve.loc[index]

        risk_at_coverage[
            f"{int(target * 100)}%"
        ] = {
            "coverage": float(
                row["coverage"]
            ),
            "accepted_count": int(
                row["accepted_count"]
            ),
            "accepted_accuracy": float(
                row["accepted_accuracy"]
            ),
            "risk": float(
                row["risk"]
            ),
            "confidence_cutoff": float(
                row["confidence"]
            ),
        }

    return {
        "n_images": int(
            full_coverage_row[
                "accepted_count"
            ]
        ),
        "full_coverage_accuracy": float(
            full_coverage_row[
                "accepted_accuracy"
            ]
        ),
        "full_coverage_risk": float(
            full_coverage_row["risk"]
        ),
        "aurc": aurc,
        "risk_at_coverage": (
            risk_at_coverage
        ),
        "note": (
            "AURC summarizes the empirical "
            "risk–coverage curve. Fixed coverage "
            "points are descriptive and are not "
            "used to select an operating threshold."
        ),
    }

def save_curve(
    curve: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    curve_path = (
        OUTPUT_DIR
        / "risk_coverage_curve.csv"
    )

    curve.to_csv(
        curve_path,
        index=False,
    )

    print(
        f"Saved: {curve_path}"
    )


def save_summary(
    summary: dict,
) -> None:
    path = (
        OUTPUT_DIR
        / "risk_coverage_summary.json"
    )

    path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print(
        f"Saved: {path}"
    )


def save_figure(
    curve: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        curve["coverage"],
        curve["risk"],
        linewidth=2,
    )

    plt.xlabel("Coverage")
    plt.ylabel(
        "Selective risk (1 − accepted accuracy)"
    )

    plt.title(
        "Risk–Coverage Curve"
    )

    plt.xlim(0, 1)
    plt.ylim(
        bottom=0
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    png_path = (
        FIGURES_DIR
        / "figure_6_risk_coverage_curve.png"
    )

    pdf_path = (
        FIGURES_DIR
        / "figure_6_risk_coverage_curve.pdf"
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


def verify_curve(
    curve: pd.DataFrame,
) -> None:
    if not np.all(
        np.diff(
            curve["coverage"]
        ) > 0
    ):
        raise RuntimeError(
            "Coverage must increase monotonically."
        )

    final_coverage = float(
        curve.iloc[-1]["coverage"]
    )

    if not np.isclose(
        final_coverage,
        1.0,
    ):
        raise RuntimeError(
            "Final coverage must equal 1."
        )

    if (
        curve["risk"].min() < 0
        or curve["risk"].max() > 1
    ):
        raise RuntimeError(
            "Risk outside [0, 1]."
        )

    print(
        "Risk–coverage validation: PASSED"
    )


def main() -> None:
    print(
        "=== RISK–COVERAGE ANALYSIS ==="
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
    print()

    curve = compute_risk_coverage(
        df
    )

    verify_curve(curve)

    summary = compute_summary(
        curve
    )

    print()
    print(
        "Full-coverage accuracy:",
        f"{summary['full_coverage_accuracy']:.4f}",
    )
    print(
        "Full-coverage risk:",
        f"{summary['full_coverage_risk']:.4f}",
    )

    print(
        "Risk at fixed coverage levels:"
    )

    for label, values in (
        summary["risk_at_coverage"].items()
    ):
        print(
            f"  {label}: "
            f"coverage={values['coverage']:.4f}, "
            f"accuracy={values['accepted_accuracy']:.4f}, "
            f"risk={values['risk']:.4f}, "
            f"n={values['accepted_count']}"
        )

    print(
        "AURC:",
        f"{summary['aurc']:.4f}",
    )

    print()

    save_curve(curve)
    save_summary(summary)
    save_figure(curve)

    print()
    print(
        "=== RISK–COVERAGE ANALYSIS COMPLETE ==="
    )


if __name__ == "__main__":
    main()
