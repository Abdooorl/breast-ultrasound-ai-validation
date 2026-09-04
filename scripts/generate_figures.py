from pathlib import Path
import hashlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import CLASS_LABELS


PREDICTIONS_PATH = Path(
    "results/predictions/bus_uclm_predictions.csv"
)

FIGURES_DIR = Path("figures")

EXPECTED_SHA256 = (
    "634e5a38c5657d5c9801b873d53e504a"
    "b2595b280f5d4269c38d1b80ca543d76"
)


def verify_frozen_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_PATH}"
        )

    actual_sha256 = hashlib.sha256(
        PREDICTIONS_PATH.read_bytes()
    ).hexdigest()

    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            "Frozen prediction file checksum mismatch.\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Actual:   {actual_sha256}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    if len(df) != 683:
        raise RuntimeError(
            f"Expected 683 predictions, found {len(df)}."
        )

    if df["image_id"].nunique() != 683:
        raise RuntimeError(
            "Prediction file does not contain "
            "683 unique image IDs."
        )

    return df


def save_figure(fig, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    png_path = FIGURES_DIR / f"{filename}.png"
    pdf_path = FIGURES_DIR / f"{filename}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_confusion_matrix(df: pd.DataFrame) -> None:
    y_true = df["ground_truth"]
    y_pred = df["predicted_label"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
    )

    row_totals = cm.sum(axis=1, keepdims=True)
    row_percent = cm / row_totals

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(cm)

    ax.set_xticks(range(len(CLASS_LABELS)))
    ax.set_yticks(range(len(CLASS_LABELS)))

    ax.set_xticklabels(
        [label.capitalize() for label in CLASS_LABELS]
    )
    ax.set_yticklabels(
        [label.capitalize() for label in CLASS_LABELS]
    )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(
        "Three-Class Confusion Matrix on BUS-UCLM"
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n"
                f"({row_percent[i, j] * 100:.1f}%)",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Number of images",
    )

    save_figure(
        fig,
        "figure_1_confusion_matrix_3class",
    )


def plot_roc_curve(df: pd.DataFrame) -> None:
    y_true = (
        df["ground_truth"].eq("malignant")
    ).astype(int)

    y_score = df["prob_malignant"]

    fpr, tpr, _ = roc_curve(
        y_true,
        y_score,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_score,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"Model (AUC = {roc_auc:.3f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Chance",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title(
        "Malignant vs Non-Malignant ROC Curve"
    )

    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    save_figure(
        fig,
        "figure_2_malignant_roc_curve",
    )


def plot_precision_recall_curve(
    df: pd.DataFrame,
) -> None:
    y_true = (
        df["ground_truth"].eq("malignant")
    ).astype(int)

    y_score = df["prob_malignant"]

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            y_score,
        )
    )

    pr_auc = auc(
        recall,
        precision,
    )

    prevalence = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=f"Model (PR-AUC = {pr_auc:.3f})",
    )

    ax.axhline(
        prevalence,
        linestyle="--",
        linewidth=1,
        label=(
            "Malignant prevalence "
            f"({prevalence:.3f})"
        ),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")

    ax.set_title(
        "Malignant vs Non-Malignant "
        "Precision–Recall Curve"
    )

    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    save_figure(
        fig,
        "figure_3_malignant_precision_recall_curve",
    )


def plot_class_distribution(
    df: pd.DataFrame,
) -> None:
    counts = (
        df["ground_truth"]
        .value_counts()
        .reindex(CLASS_LABELS)
    )

    percentages = (
        counts / counts.sum() * 100
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    bars = ax.bar(
        [label.capitalize() for label in CLASS_LABELS],
        counts.values,
    )

    for bar, count, percentage in zip(
        bars,
        counts.values,
        percentages.values,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Number of images")

    ax.set_title(
        "BUS-UCLM Ground-Truth Class Distribution"
    )

    ax.set_ylim(
        0,
        counts.max() * 1.15,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    save_figure(
        fig,
        "figure_4_class_distribution",
    )


def plot_study_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.axis("off")

    boxes = [
        (
            0.03,
            "Public pretrained ViT\n"
            "Frozen model revision"
        ),
        (
            0.23,
            "BUS-UCLM\n"
            "683 images\n38 patients"
        ),
        (
            0.43,
            "External inference\n"
            "No retraining or\nfine-tuning"
        ),
        (
            0.63,
            "Baseline evaluation\n"
            "3-class + malignant vs\n"
            "non-malignant"
        ),
        (
            0.83,
            "Reliability analysis\n"
            "Bootstrap, uncertainty,\n"
            "calibration & abstention"
        ),
    ]

    box_width = 0.14
    box_height = 0.42
    y = 0.30

    for x, text in boxes:
        rectangle = plt.Rectangle(
            (x, y),
            box_width,
            box_height,
            fill=False,
            linewidth=1.5,
        )

        ax.add_patch(rectangle)

        ax.text(
            x + box_width / 2,
            y + box_height / 2,
            text,
            ha="center",
            va="center",
            fontsize=10,
        )

    for i in range(len(boxes) - 1):
        start_x = (
            boxes[i][0]
            + box_width
            + 0.005
        )

        end_x = (
            boxes[i + 1][0]
            - 0.005
        )

        ax.annotate(
            "",
            xy=(end_x, y + box_height / 2),
            xytext=(
                start_x,
                y + box_height / 2,
            ),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.5,
            },
        )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1)

    ax.set_title(
        "Study Workflow for External Validation "
        "of the Frozen Breast Ultrasound ViT",
        pad=15,
    )

    save_figure(
        fig,
        "figure_5_study_workflow",
    )


def write_captions() -> None:
    captions = """# Baseline Research Figures

## Figure 1 — Three-Class Confusion Matrix
Confusion matrix for the frozen pretrained Vision Transformer evaluated on all 683 BUS-UCLM images. Cells report image counts and row-wise percentages. No retraining or fine-tuning was performed.

## Figure 2 — Malignant vs Non-Malignant ROC Curve
Receiver operating characteristic curve using the model's continuous malignant-class probability to distinguish malignant from non-malignant BUS-UCLM images.

## Figure 3 — Malignant vs Non-Malignant Precision–Recall Curve
Precision–recall curve using malignant-class probability. The horizontal reference represents malignant prevalence in the full BUS-UCLM evaluation dataset.

## Figure 4 — BUS-UCLM Class Distribution
Ground-truth distribution of benign, malignant and normal images in the full 683-image BUS-UCLM external validation cohort.

## Figure 5 — Study Workflow
Overview of the external validation workflow: frozen public model reproduction, BUS-UCLM inference, baseline performance evaluation and subsequent reliability analysis. The model is evaluated without retraining or fine-tuning.
"""

    path = FIGURES_DIR / "README.md"
    path.write_text(captions)

    print(f"Saved: {path}")


def main() -> None:
    print("=== BASELINE FIGURE GENERATION ===")

    df = verify_frozen_predictions()

    print("Frozen prediction checksum: verified")
    print("Images:", len(df))
    print(
        "Patients:",
        df["patient_id"].nunique(),
    )
    print()

    plot_confusion_matrix(df)
    plot_roc_curve(df)
    plot_precision_recall_curve(df)
    plot_class_distribution(df)
    plot_study_workflow()
    write_captions()

    print()
    print(
        "=== ALL BASELINE FIGURES GENERATED ==="
    )


if __name__ == "__main__":
    main()