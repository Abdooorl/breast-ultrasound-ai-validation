# Baseline Research Figures

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

### Figure 6 — Risk–coverage curve

Confidence-ranked selective-classification performance on the frozen BUS-UCLM external test set. Coverage represents the proportion of cases retained for prediction, while selective risk is defined as one minus accepted-case accuracy. Lower coverage preferentially retains higher-confidence cases. The curve is descriptive and is not used to select a preferred operating threshold.
