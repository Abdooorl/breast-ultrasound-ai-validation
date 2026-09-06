from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


PREDICTIONS_PATH = Path(
    "results/predictions/bus_uclm_predictions.csv"
)

OUTPUT_DIR = Path("results/uncertainty")

EXPECTED_SHA256 = (
    "634e5a38c5657d5c9801b873d53e504a"
    "b2595b280f5d4269c38d1b80ca543d76"
)

PROBABILITY_COLUMNS = [
    "prob_benign",
    "prob_malignant",
    "prob_normal",
]


def verify_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_PATH}"
        )

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
        "patient_id",
        "ground_truth",
        "predicted_label",
        "confidence",
        "probability_margin",
        *PROBABILITY_COLUMNS,
    }

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    return df


def verify_uncertainty_fields(
    df: pd.DataFrame,
) -> tuple[float, float]:
    probabilities = df[
        PROBABILITY_COLUMNS
    ].to_numpy()

    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    recomputed_confidence = (
        sorted_probabilities[:, -1]
    )

    recomputed_margin = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )

    confidence_error = np.max(
        np.abs(
            df["confidence"].to_numpy()
            - recomputed_confidence
        )
    )

    margin_error = np.max(
        np.abs(
            df["probability_margin"].to_numpy()
            - recomputed_margin
        )
    )

    return (
        float(confidence_error),
        float(margin_error),
    )


def add_analysis_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["correct"] = (
        result["ground_truth"]
        == result["predicted_label"]
    )

    result["malignant_case"] = (
        result["ground_truth"]
        == "malignant"
    )

    result["malignant_false_negative"] = (
        result["malignant_case"]
        & (
            result["predicted_label"]
            != "malignant"
        )
    )

    result["malignant_true_positive"] = (
        result["malignant_case"]
        & (
            result["predicted_label"]
            == "malignant"
        )
    )

    return result


def describe_group(
    df: pd.DataFrame,
    group_name: str,
) -> dict:
    confidence = df["confidence"]
    margin = df["probability_margin"]

    return {
        "group": group_name,
        "n": int(len(df)),
        "confidence": {
            "mean": float(confidence.mean()),
            "median": float(confidence.median()),
            "q25": float(confidence.quantile(0.25)),
            "q75": float(confidence.quantile(0.75)),
            "min": float(confidence.min()),
            "max": float(confidence.max()),
        },
        "probability_margin": {
            "mean": float(margin.mean()),
            "median": float(margin.median()),
            "q25": float(margin.quantile(0.25)),
            "q75": float(margin.quantile(0.75)),
            "min": float(margin.min()),
            "max": float(margin.max()),
        },
    }


def build_summary(
    df: pd.DataFrame,
) -> dict:
    groups = [
        describe_group(
            df,
            "all_images",
        ),
        describe_group(
            df[df["correct"]],
            "correct_predictions",
        ),
        describe_group(
            df[~df["correct"]],
            "incorrect_predictions",
        ),
        describe_group(
            df[df["malignant_true_positive"]],
            "malignant_true_positives",
        ),
        describe_group(
            df[df["malignant_false_negative"]],
            "malignant_false_negatives",
        ),
    ]

    return {
        "dataset": "BUS-UCLM",
        "n_images": int(len(df)),
        "n_patients": int(
            df["patient_id"].nunique()
        ),
        "uncertainty_proxies": {
            "confidence": (
                "maximum softmax probability"
            ),
            "probability_margin": (
                "highest probability minus "
                "second-highest probability"
            ),
        },
        "groups": groups,
    }


def build_summary_table(
    summary: dict,
) -> pd.DataFrame:
    rows = []

    for group in summary["groups"]:
        rows.append(
            {
                "Group": group["group"],
                "N": group["n"],
                "Confidence mean": (
                    group["confidence"]["mean"]
                ),
                "Confidence median": (
                    group["confidence"]["median"]
                ),
                "Confidence Q1": (
                    group["confidence"]["q25"]
                ),
                "Confidence Q3": (
                    group["confidence"]["q75"]
                ),
                "Margin mean": (
                    group[
                        "probability_margin"
                    ]["mean"]
                ),
                "Margin median": (
                    group[
                        "probability_margin"
                    ]["median"]
                ),
                "Margin Q1": (
                    group[
                        "probability_margin"
                    ]["q25"]
                ),
                "Margin Q3": (
                    group[
                        "probability_margin"
                    ]["q75"]
                ),
            }
        )

    table = pd.DataFrame(rows)

    numeric_columns = [
        column
        for column in table.columns
        if column not in {"Group", "N"}
    ]

    table[numeric_columns] = (
        table[numeric_columns]
        .astype(float)
        .round(4)
    )

    return table


def main() -> None:
    print(
        "=== UNCERTAINTY PROXY ANALYSIS ==="
    )

    df = verify_predictions()

    print(
        "Frozen prediction checksum: verified"
    )
    print("Images:", len(df))
    print(
        "Patients:",
        df["patient_id"].nunique(),
    )

    confidence_error, margin_error = (
        verify_uncertainty_fields(df)
    )

    print()
    print(
        "Max confidence recomputation error:",
        f"{confidence_error:.12g}",
    )
    print(
        "Max margin recomputation error:",
        f"{margin_error:.12g}",
    )

    tolerance = 1e-6

    if confidence_error > tolerance:
        raise RuntimeError(
            "Confidence field validation failed."
        )

    if margin_error > tolerance:
        raise RuntimeError(
            "Probability-margin validation failed."
        )

    print(
        "Uncertainty field validation: PASSED"
    )

    analysis_df = add_analysis_columns(df)

    summary = build_summary(
        analysis_df
    )

    summary_table = build_summary_table(
        summary
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases_path = (
        OUTPUT_DIR
        / "uncertainty_cases.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "uncertainty_summary.json"
    )

    table_csv_path = (
        OUTPUT_DIR
        / "uncertainty_summary.csv"
    )

    table_md_path = (
        OUTPUT_DIR
        / "uncertainty_summary.md"
    )

    analysis_df.to_csv(
        cases_path,
        index=False,
    )

    with summary_path.open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    summary_table.to_csv(
        table_csv_path,
        index=False,
    )

    table_md_path.write_text(
        summary_table.to_markdown(
            index=False
        )
        + "\n"
    )

    print()
    print(summary_table.to_string(index=False))

    print()
    print(
        "Malignant cases:",
        int(
            analysis_df[
                "malignant_case"
            ].sum()
        ),
    )

    print(
        "Malignant true positives:",
        int(
            analysis_df[
                "malignant_true_positive"
            ].sum()
        ),
    )

    print(
        "Malignant false negatives:",
        int(
            analysis_df[
                "malignant_false_negative"
            ].sum()
        ),
    )

    print()
    print(f"Saved: {cases_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {table_csv_path}")
    print(f"Saved: {table_md_path}")

    print()
    print(
        "=== UNCERTAINTY ANALYSIS COMPLETE ==="
    )


if __name__ == "__main__":
    main()