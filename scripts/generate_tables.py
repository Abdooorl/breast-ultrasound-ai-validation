from pathlib import Path
import hashlib
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    CLASS_LABELS,
    evaluate_malignant_binary,
    evaluate_multiclass,
)


PREDICTIONS_PATH = Path(
    "results/predictions/bus_uclm_predictions.csv"
)

BOOTSTRAP_PATH = Path(
    "results/metrics/patient_bootstrap_ci.json"
)

TABLES_DIR = Path("results/tables")

EXPECTED_SHA256 = (
    "634e5a38c5657d5c9801b873d53e504ab2595b280f5d4269c38d1b80ca543d76"
)


def verify_frozen_predictions() -> pd.DataFrame:
    actual_sha256 = hashlib.sha256(
        PREDICTIONS_PATH.read_bytes()
    ).hexdigest()

    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            "Frozen prediction checksum mismatch.\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Actual:   {actual_sha256}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    if len(df) != 683:
        raise RuntimeError(
            f"Expected 683 rows, found {len(df)}."
        )

    if df["image_id"].nunique() != 683:
        raise RuntimeError(
            "Expected 683 unique image IDs."
        )

    return df


def load_bootstrap() -> dict:
    if not BOOTSTRAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing bootstrap file: {BOOTSTRAP_PATH}"
        )

    with BOOTSTRAP_PATH.open() as f:
        return json.load(f)


def ci_text(
    bootstrap: dict,
    metric_name: str,
) -> str:
    metric = bootstrap["metrics"][metric_name]
    ci = metric["ci_95"]

    return (
        f"{ci['lower']:.3f}–"
        f"{ci['upper']:.3f}"
    )


def save_table(
    df: pd.DataFrame,
    filename: str,
) -> None:
    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = TABLES_DIR / f"{filename}.csv"
    md_path = TABLES_DIR / f"{filename}.md"

    df.to_csv(
        csv_path,
        index=False,
    )

    md_path.write_text(
        df.to_markdown(index=False) + "\n"
    )

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


def table_three_class(
    df: pd.DataFrame,
    bootstrap: dict,
) -> None:
    metrics = evaluate_multiclass(df)

    rows = [
        {
            "Metric": "Accuracy",
            "Estimate": metrics["accuracy"],
            "95% patient-bootstrap CI": ci_text(
                bootstrap,
                "accuracy",
            ),
        },
        {
            "Metric": "Balanced accuracy",
            "Estimate": metrics["balanced_accuracy"],
            "95% patient-bootstrap CI": ci_text(
                bootstrap,
                "balanced_accuracy_3class",
            ),
        },
        {
            "Metric": "Macro precision",
            "Estimate": metrics["macro_precision"],
            "95% patient-bootstrap CI": "—",
        },
        {
            "Metric": "Macro recall",
            "Estimate": metrics["macro_recall"],
            "95% patient-bootstrap CI": "—",
        },
        {
            "Metric": "Macro F1",
            "Estimate": metrics["macro_f1"],
            "95% patient-bootstrap CI": ci_text(
                bootstrap,
                "macro_f1",
            ),
        },
        {
            "Metric": "Malignant sensitivity",
            "Estimate": metrics[
                "malignant_sensitivity"
            ],
            "95% patient-bootstrap CI": ci_text(
                bootstrap,
                "malignant_sensitivity",
            ),
        },
    ]

    table = pd.DataFrame(rows)

    table["Estimate"] = table[
        "Estimate"
    ].map(lambda x: f"{x:.3f}")

    save_table(
        table,
        "table_1_three_class_primary",
    )


def table_per_class(
    df: pd.DataFrame,
) -> None:
    metrics = evaluate_multiclass(df)

    rows = []

    for label in CLASS_LABELS:
        class_metrics = metrics[
            "per_class"
        ][label]

        rows.append(
            {
                "Class": label.capitalize(),
                "Precision": (
                    class_metrics["precision"]
                ),
                "Recall": (
                    class_metrics["recall"]
                ),
                "F1": (
                    class_metrics["f1"]
                ),
                "Support": int(
                    class_metrics["support"]
                ),
            }
        )

    table = pd.DataFrame(rows)

    for column in [
        "Precision",
        "Recall",
        "F1",
    ]:
        table[column] = table[column].map(
            lambda x: f"{x:.3f}"
        )

    save_table(
        table,
        "table_2_per_class_primary",
    )


def table_binary(
    df: pd.DataFrame,
    bootstrap: dict,
) -> None:
    metrics = evaluate_malignant_binary(df)

    metric_mapping = [
        (
            "Sensitivity",
            "sensitivity",
            "binary_sensitivity",
        ),
        (
            "Specificity",
            "specificity",
            "binary_specificity",
        ),
        (
            "PPV",
            "ppv",
            "binary_ppv",
        ),
        (
            "NPV",
            "npv",
            "binary_npv",
        ),
        (
            "F1",
            "f1",
            "binary_f1",
        ),
        (
            "Balanced accuracy",
            "balanced_accuracy",
            "binary_balanced_accuracy",
        ),
        (
            "ROC-AUC",
            "roc_auc",
            "roc_auc",
        ),
        (
            "PR-AUC",
            "pr_auc",
            "pr_auc",
        ),
    ]

    rows = []

    for display, metric_key, bootstrap_key in metric_mapping:
        rows.append(
            {
                "Metric": display,
                "Estimate": f"{metrics[metric_key]:.3f}",
                "95% patient-bootstrap CI": ci_text(
                    bootstrap,
                    bootstrap_key,
                ),
            }
        )

    table = pd.DataFrame(rows)

    save_table(
        table,
        "table_3_malignant_binary_primary",
    )


def table_binary_confusion(
    df: pd.DataFrame,
) -> None:
    metrics = evaluate_malignant_binary(df)

    table = pd.DataFrame(
        [
            {
                "Measure": "True positives",
                "Count": metrics["tp"],
            },
            {
                "Measure": "True negatives",
                "Count": metrics["tn"],
            },
            {
                "Measure": "False positives",
                "Count": metrics["fp"],
            },
            {
                "Measure": "False negatives",
                "Count": metrics["fn"],
            },
        ]
    )

    save_table(
        table,
        "table_4_malignant_confusion_counts",
    )


def table_clean_subset(
    df: pd.DataFrame,
) -> None:
    clean = df[
        df["clean_input"] == True
    ].copy()

    multiclass = evaluate_multiclass(clean)
    binary = evaluate_malignant_binary(clean)

    rows = [
        {
            "Metric": "Images",
            "Estimate": str(len(clean)),
        },
        {
            "Metric": "Patients",
            "Estimate": str(
                clean["patient_id"].nunique()
            ),
        },
        {
            "Metric": "3-class accuracy",
            "Estimate": (
                f"{multiclass['accuracy']:.3f}"
            ),
        },
        {
            "Metric": "3-class balanced accuracy",
            "Estimate": (
                f"{multiclass['balanced_accuracy']:.3f}"
            ),
        },
        {
            "Metric": "Macro F1",
            "Estimate": (
                f"{multiclass['macro_f1']:.3f}"
            ),
        },
        {
            "Metric": "Malignant sensitivity",
            "Estimate": (
                f"{binary['sensitivity']:.3f}"
            ),
        },
        {
            "Metric": "Specificity",
            "Estimate": (
                f"{binary['specificity']:.3f}"
            ),
        },
        {
            "Metric": "ROC-AUC",
            "Estimate": (
                f"{binary['roc_auc']:.3f}"
            ),
        },
        {
            "Metric": "PR-AUC",
            "Estimate": (
                f"{binary['pr_auc']:.3f}"
            ),
        },
    ]

    table = pd.DataFrame(rows)

    save_table(
        table,
        "table_5_clean_input_secondary",
    )


def verify_phase4_artifacts() -> None:
    required = [
        Path(
            "results/metrics/"
            "baseline_metrics.json"
        ),
        Path(
            "results/metrics/"
            "patient_bootstrap_ci.json"
        ),
        Path(
            "figures/"
            "figure_1_confusion_matrix_3class.pdf"
        ),
        Path(
            "figures/"
            "figure_2_malignant_roc_curve.pdf"
        ),
        Path(
            "figures/"
            "figure_3_malignant_precision_recall_curve.pdf"
        ),
        Path(
            "figures/"
            "figure_4_class_distribution.pdf"
        ),
        Path(
            "figures/"
            "figure_5_study_workflow.pdf"
        ),
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing Phase 4 artifacts:\n"
            + "\n".join(missing)
        )

    print(
        "Phase 4 artifact check: PASSED"
    )


def main() -> None:
    print(
        "=== BASELINE TABLE GENERATION ==="
    )

    df = verify_frozen_predictions()
    bootstrap = load_bootstrap()

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

    table_three_class(
        df,
        bootstrap,
    )

    table_per_class(df)

    table_binary(
        df,
        bootstrap,
    )

    table_binary_confusion(df)

    table_clean_subset(df)

    print()

    verify_phase4_artifacts()

    print()
    print(
        "=== BASELINE TABLES COMPLETE ==="
    )


if __name__ == "__main__":
    main()