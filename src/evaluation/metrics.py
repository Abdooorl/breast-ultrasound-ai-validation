from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


CLASS_LABELS = ["benign", "malignant", "normal"]


def evaluate_multiclass(df: pd.DataFrame) -> dict[str, Any]:
    y_true = df["ground_truth"].to_numpy()
    y_pred = df["predicted_label"].to_numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        zero_division=0,
    )

    per_class = {}

    for i, label in enumerate(CLASS_LABELS):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "n_images": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "malignant_sensitivity": float(
            per_class["malignant"]["recall"]
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
        ).tolist(),
        "class_order": CLASS_LABELS,
        "per_class": per_class,
    }


def evaluate_malignant_binary(
    df: pd.DataFrame,
) -> dict[str, Any]:
    y_true = (
        df["ground_truth"].eq("malignant")
    ).astype(int).to_numpy()

    y_pred = (
        df["predicted_label"].eq("malignant")
    ).astype(int).to_numpy()

    y_score = df["prob_malignant"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    ppv = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_score,
    )

    return {
        "n_images": int(len(df)),
        "positive_class": "malignant",
        "negative_class": "non-malignant",
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_score,
            )
        ),
        "pr_auc": float(
            auc(
                recall,
                precision,
            )
        ),
        "binary_confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }