from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


PREDICTIONS_PATH = Path(
    "results/predictions/bus_uclm_predictions.csv"
)

OUTPUT_DIR = Path(
    "results/failure_analysis"
)

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

    if len(df) != 683:
        raise RuntimeError(
            f"Expected 683 rows, found {len(df)}."
        )

    if df["image_id"].nunique() != 683:
        raise RuntimeError(
            "Expected 683 unique images."
        )

    required = {
        "image_id",
        "patient_id",
        "ground_truth",
        "predicted_label",
        "prob_benign",
        "prob_malignant",
        "prob_normal",
        "confidence",
        "probability_margin",
        "doppler",
        "marks",
        "combined",
        "clean_input",
    }

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    return df


def normalize_boolean_like(
    series: pd.Series,
) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        "Yes": True,
        "No": False,
        "Y": True,
        "N": False,
        "yes": True,
        "no": False,
        1: True,
        0: False,
    }

    normalized = series.map(mapping)

    if normalized.isna().any():
        bad = sorted(
            series[
                normalized.isna()
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Could not normalize boolean-like values: "
            + ", ".join(bad)
        )

    return normalized.astype(bool)


def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    for column in [
        "doppler",
        "marks",
        "combined",
        "clean_input",
    ]:
        result[column] = (
            normalize_boolean_like(
                result[column]
            )
        )

    result["is_malignant"] = (
        result["ground_truth"]
        == "malignant"
    )

    result["is_non_malignant"] = (
    ~result["is_malignant"]
)

    result["predicted_malignant"] = (
        result["predicted_label"]
        == "malignant"
    )

    result["malignant_true_positive"] = (
        result["is_malignant"]
        & result["predicted_malignant"]
    )

    result["malignant_false_negative"] = (
        result["is_malignant"]
        & ~result["predicted_malignant"]
    )

    result["malignant_false_positive"] = (
        ~result["is_malignant"]
        & result["predicted_malignant"]
    )

    result["malignant_true_negative"] = (
        ~result["is_malignant"]
        & ~result["predicted_malignant"]
    )

    return result


def numeric_summary(
    df: pd.DataFrame,
) -> dict:
    if len(df) == 0:
        return {
            "n": 0,
        }

    return {
        "n": int(len(df)),
        "confidence_mean": float(
            df["confidence"].mean()
        ),
        "confidence_median": float(
            df["confidence"].median()
        ),
        "confidence_q25": float(
            df["confidence"].quantile(
                0.25
            )
        ),
        "confidence_q75": float(
            df["confidence"].quantile(
                0.75
            )
        ),
        "margin_mean": float(
            df["probability_margin"].mean()
        ),
        "margin_median": float(
            df["probability_margin"].median()
        ),
        "prob_malignant_mean": float(
            df["prob_malignant"].mean()
        ),
        "prob_malignant_median": float(
            df["prob_malignant"].median()
        ),
    }


def metadata_summary(
    df: pd.DataFrame,
) -> dict:
    if len(df) == 0:
        return {
            "n": 0,
        }

    return {
        "n": int(len(df)),
        "doppler_count": int(
            df["doppler"].sum()
        ),
        "doppler_rate": float(
            df["doppler"].mean()
        ),
        "marks_count": int(
            df["marks"].sum()
        ),
        "marks_rate": float(
            df["marks"].mean()
        ),
        "combined_count": int(
            df["combined"].sum()
        ),
        "combined_rate": float(
            df["combined"].mean()
        ),
        "clean_input_count": int(
            df["clean_input"].sum()
        ),
        "clean_input_rate": float(
            df["clean_input"].mean()
        ),
    }


def patient_summary(
    df: pd.DataFrame,
    flag_column: str,
    eligible_column: str,
) -> pd.DataFrame:
    grouped = (
        df.groupby("patient_id")
        .agg(
            total_images=(
                "image_id",
                "size",
            ),
            eligible_images=(
                eligible_column,
                "sum",
            ),
            flagged_images=(
                flag_column,
                "sum",
            ),
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["flagged_images"] > 0
    ].copy()

    grouped[
        "flagged_rate_among_eligible"
    ] = (
        grouped["flagged_images"]
        / grouped["eligible_images"]
    )

    grouped = grouped.sort_values(
        [
            "flagged_images",
            "flagged_rate_among_eligible",
            "patient_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    return grouped

def false_negative_direction_table(
    fn: pd.DataFrame,
) -> pd.DataFrame:
    table = (
        fn["predicted_label"]
        .value_counts()
        .rename_axis(
            "predicted_as"
        )
        .reset_index(
            name="count"
        )
    )

    table["percentage"] = (
        table["count"]
        / len(fn)
    )

    return table


def false_positive_source_table(
    fp: pd.DataFrame,
) -> pd.DataFrame:
    table = (
        fp["ground_truth"]
        .value_counts()
        .rename_axis(
            "ground_truth"
        )
        .reset_index(
            name="count"
        )
    )

    table["percentage"] = (
        table["count"]
        / len(fp)
    )

    return table


def high_confidence_error_summary(
    fn: pd.DataFrame,
    fp: pd.DataFrame,
) -> pd.DataFrame:
    thresholds = [
        0.80,
        0.90,
        0.95,
    ]

    rows = []

    for threshold in thresholds:
        rows.append(
            {
                "threshold": threshold,
                "fn_at_or_above": int(
                    (
                        fn["confidence"]
                        >= threshold
                    ).sum()
                ),
                "fn_fraction": float(
                    (
                        fn["confidence"]
                        >= threshold
                    ).mean()
                ),
                "fp_at_or_above": int(
                    (
                        fp["confidence"]
                        >= threshold
                    ).sum()
                ),
                "fp_fraction": float(
                    (
                        fp["confidence"]
                        >= threshold
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_group_table(
    groups: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for name, group in groups.items():
        numeric = numeric_summary(group)
        metadata = metadata_summary(group)

        rows.append(
            {
                "group": name,
                "n": numeric["n"],
                "confidence_mean": (
                    numeric.get(
                        "confidence_mean",
                        np.nan,
                    )
                ),
                "confidence_median": (
                    numeric.get(
                        "confidence_median",
                        np.nan,
                    )
                ),
                "margin_mean": (
                    numeric.get(
                        "margin_mean",
                        np.nan,
                    )
                ),
                "margin_median": (
                    numeric.get(
                        "margin_median",
                        np.nan,
                    )
                ),
                "prob_malignant_mean": (
                    numeric.get(
                        "prob_malignant_mean",
                        np.nan,
                    )
                ),
                "doppler_rate": (
                    metadata.get(
                        "doppler_rate",
                        np.nan,
                    )
                ),
                "marks_rate": (
                    metadata.get(
                        "marks_rate",
                        np.nan,
                    )
                ),
                "combined_rate": (
                    metadata.get(
                        "combined_rate",
                        np.nan,
                    )
                ),
                "clean_input_rate": (
                    metadata.get(
                        "clean_input_rate",
                        np.nan,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def save_table(
    df: pd.DataFrame,
    filename: str,
) -> None:
    csv_path = (
        OUTPUT_DIR
        / f"{filename}.csv"
    )

    md_path = (
        OUTPUT_DIR
        / f"{filename}.md"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    md_path.write_text(
        df.to_markdown(
            index=False,
            floatfmt=".4f",
        )
        + "\n"
    )

    print(
        f"Saved: {csv_path}"
    )

    print(
        f"Saved: {md_path}"
    )


def main() -> None:
    print(
        "=== FAILURE ANALYSIS AUDIT ==="
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

    df = prepare_data(df)

    fn = df[
        df["malignant_false_negative"]
    ].copy()

    fp = df[
        df["malignant_false_positive"]
    ].copy()

    tp = df[
        df["malignant_true_positive"]
    ].copy()

    tn = df[
        df["malignant_true_negative"]
    ].copy()

    if len(fn) != 51:
        raise RuntimeError(
            f"Expected 51 malignant FNs, "
            f"found {len(fn)}."
        )

    if len(fp) != 46:
        raise RuntimeError(
            f"Expected 46 malignant FPs, "
            f"found {len(fp)}."
        )

    if len(tp) != 39:
        raise RuntimeError(
            f"Expected 39 malignant TPs, "
            f"found {len(tp)}."
        )

    if len(tn) != 547:
        raise RuntimeError(
            f"Expected 547 malignant TNs, "
            f"found {len(tn)}."
        )

    print()
    print(
        "Binary malignant audit counts:"
    )

    print(
        "TP:",
        len(tp),
    )

    print(
        "FN:",
        len(fn),
    )

    print(
        "FP:",
        len(fp),
    )

    print(
        "TN:",
        len(tn),
    )

    print()
    print(
        "Failure-count validation: PASSED"
    )

    groups = {
        "malignant_true_positive": tp,
        "malignant_false_negative": fn,
        "malignant_false_positive": fp,
        "malignant_true_negative": tn,
    }

    group_table = build_group_table(
        groups
    )

    fn_direction = (
        false_negative_direction_table(
            fn
        )
    )

    fp_source = (
        false_positive_source_table(
            fp
        )
    )

    high_confidence = (
        high_confidence_error_summary(
            fn,
            fp,
        )
    )

    fn_patients = patient_summary(
    df,
    "malignant_false_negative",
    "is_malignant",
    )

    fp_patients = patient_summary(
    df,
    "malignant_false_positive",
    "is_non_malignant",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    error_cases = pd.concat(
        [
            fn.assign(
                error_type=(
                    "malignant_false_negative"
                )
            ),
            fp.assign(
                error_type=(
                    "malignant_false_positive"
                )
            ),
        ],
        ignore_index=True,
    )

    error_case_columns = [
        "image_id",
        "patient_id",
        "error_type",
        "ground_truth",
        "predicted_label",
        "confidence",
        "probability_margin",
        "prob_benign",
        "prob_malignant",
        "prob_normal",
        "doppler",
        "marks",
        "combined",
        "clean_input",
    ]

    error_cases[
        error_case_columns
    ].to_csv(
        OUTPUT_DIR
        / "error_cases.csv",
        index=False,
    )

    save_table(
        group_table,
        "failure_group_summary",
    )

    save_table(
        fn_direction,
        "malignant_false_negative_directions",
    )

    save_table(
        fp_source,
        "malignant_false_positive_sources",
    )

    save_table(
        high_confidence,
        "high_confidence_errors",
    )

    save_table(
        fn_patients,
        "false_negative_patient_summary",
    )

    save_table(
        fp_patients,
        "false_positive_patient_summary",
    )

    summary = {
        "n_images": int(
            len(df)
        ),
        "n_patients": int(
            df["patient_id"].nunique()
        ),
        "binary_malignant_counts": {
            "tp": int(len(tp)),
            "fn": int(len(fn)),
            "fp": int(len(fp)),
            "tn": int(len(tn)),
        },
        "false_negative_numeric": (
            numeric_summary(fn)
        ),
        "false_negative_metadata": (
            metadata_summary(fn)
        ),
        "false_positive_numeric": (
            numeric_summary(fp)
        ),
        "false_positive_metadata": (
            metadata_summary(fp)
        ),
        "note": (
            "This audit is descriptive. "
            "Metadata associations are not "
            "interpreted as causal."
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "failure_audit_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "=== GROUP SUMMARY ==="
    )

    print(
        group_table.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print(
        "=== MALIGNANT FALSE NEGATIVE "
        "DIRECTIONS ==="
    )

    print(
        fn_direction.to_string(
            index=False
        )
    )

    print()
    print(
        "=== MALIGNANT FALSE POSITIVE "
        "SOURCES ==="
    )

    print(
        fp_source.to_string(
            index=False
        )
    )

    print()
    print(
        "=== HIGH-CONFIDENCE ERRORS ==="
    )

    print(
        high_confidence.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print(
        "FN patients:",
        len(fn_patients),
    )

    print(
        "FP patients:",
        len(fp_patients),
    )

    print()
    print(
        "Saved:",
        OUTPUT_DIR
        / "error_cases.csv",
    )

    print(
        "Saved:",
        summary_path,
    )

    print()
    print(
        "=== FAILURE ANALYSIS COMPLETE ==="
    )


if __name__ == "__main__":
    main()