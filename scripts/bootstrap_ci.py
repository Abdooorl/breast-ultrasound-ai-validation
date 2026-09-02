import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bootstrap import (
    patient_cluster_bootstrap,
    percentile_ci,
)
from src.evaluation.metrics import (
    evaluate_malignant_binary,
    evaluate_multiclass,
)


PREDICTIONS_PATH = Path(
    "results/predictions/bus_uclm_predictions.csv"
)

OUTPUT_PATH = Path(
    "results/metrics/patient_bootstrap_ci.json"
)

N_BOOTSTRAP = 5000
SEED = 42


def compute_metrics(df: pd.DataFrame) -> dict:
    multiclass = evaluate_multiclass(df)

    try:
        binary = evaluate_malignant_binary(df)
    except ValueError:
        binary = {
            "sensitivity": np.nan,
            "specificity": np.nan,
            "ppv": np.nan,
            "npv": np.nan,
            "f1": np.nan,
            "balanced_accuracy": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
        }

    return {
        "accuracy": multiclass["accuracy"],
        "balanced_accuracy_3class": (
            multiclass["balanced_accuracy"]
        ),
        "macro_f1": multiclass["macro_f1"],
        "malignant_sensitivity": (
            multiclass["malignant_sensitivity"]
        ),
        "binary_sensitivity": binary["sensitivity"],
        "binary_specificity": binary["specificity"],
        "binary_ppv": binary["ppv"],
        "binary_npv": binary["npv"],
        "binary_f1": binary["f1"],
        "binary_balanced_accuracy": (
            binary["balanced_accuracy"]
        ),
        "roc_auc": binary["roc_auc"],
        "pr_auc": binary["pr_auc"],
    }


def main() -> None:
    df = pd.read_csv(PREDICTIONS_PATH)

    print("=== PATIENT-CLUSTER BOOTSTRAP ===")
    print("Images:", len(df))
    print("Patients:", df["patient_id"].nunique())
    print("Bootstrap repetitions:", N_BOOTSTRAP)
    print("Seed:", SEED)
    print()

    point_estimates = compute_metrics(df)

    bootstrap = patient_cluster_bootstrap(
        df=df,
        metric_fn=compute_metrics,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )

    samples = bootstrap.pop("samples")

    confidence_intervals = {}

    for metric_name in point_estimates:
        values = [
            sample[metric_name]
            for sample in samples
        ]

        confidence_intervals[metric_name] = {
            "estimate": float(
                point_estimates[metric_name]
            ),
            "ci_95": percentile_ci(values),
        }

    result = {
        "method": (
            "patient-cluster percentile bootstrap"
        ),
        "bootstrap_unit": "patient_id",
        "confidence_level": 0.95,
        **bootstrap,
        "metrics": confidence_intervals,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    for name, result_metric in confidence_intervals.items():
        ci = result_metric["ci_95"]

        print(
            f"{name}: "
            f"{result_metric['estimate']:.4f} "
            f"[{ci['lower']:.4f}, "
            f"{ci['upper']:.4f}]"
        )

    print()
    print(
        "Valid bootstrap samples:",
        result["n_bootstrap_valid"],
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()