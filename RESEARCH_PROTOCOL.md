# Research Protocol

## Working title
**Do Public Breast Ultrasound AI Models Generalize? External Validation and Uncertainty-Aware Evaluation of a Pretrained Vision Transformer**

## Primary objective
Evaluate the generalization performance of a publicly available pretrained breast-ultrasound Vision Transformer on the independently acquired BUS-UCLM dataset without retraining or fine-tuning.

## Research questions

### RQ1
How well does the pretrained classifier perform on BUS-UCLM under zero-shot external evaluation?

### RQ2
How does its independent external performance compare with the performance reported by the model author?

### RQ3
Can confidence-based selective classification / abstention reduce potentially unsafe errors while retaining useful coverage?

### RQ4
Which classes and image/case characteristics are associated with the highest error rates, particularly malignant false negatives?

### RQ5 — Optional
Do model attribution regions qualitatively correspond with radiologist-annotated lesion regions?

## Primary outcomes
- Malignant sensitivity
- Malignant specificity
- ROC-AUC
- PR-AUC
- Balanced accuracy
- Macro F1
- Patient-aware 95% confidence intervals

## Secondary outcomes
- Calibration metrics
- Coverage-risk trade-off under abstention
- False-negative / false-positive patterns

## Scope constraints
- No retraining or fine-tuning for primary external validation.
- Explainability is optional and must not delay the October 30 deadline.
- Results must be presented as research findings, not clinical diagnostic claims.
