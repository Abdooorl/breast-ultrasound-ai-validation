# Pretrained Model Audit

## 1. Model Identity

**Model:** Parveshiiii/breast-cancer-detector
**Revision:** 7c4c1ac11f5f80d382cc641ec6f2d3989e7fa73c
**Architecture:** ViTForImageClassification
**Model family:** Vision Transformer
**License:** Apache-2.0

## 2. Architecture

- Input size: 224 × 224
- Patch size: 16 × 16
- Hidden dimension: 768
- Transformer layers: 12
- Attention heads: 12

## 3. Output Classes

- 0 → benign
- 1 → malignant
- 2 → normal

## 4. Preprocessing

**Image processor:** ViTImageProcessor

- Resize: 224 × 224
- Resampling: bilinear
- Rescaling: 1 / 255
- Normalization mean: [0.5, 0.5, 0.5]
- Normalization std: [0.5, 0.5, 0.5]

The official Hugging Face image processor will be used during inference rather than manually recreating the preprocessing pipeline.

## 5. Verification Status

The architecture, class mapping, preprocessing configuration, model revision, and license above were independently verified from the Hugging Face model repository and configuration files.

## 6. Training Data Provenance

According to the model card, the model was fine-tuned using:

`gymprathap/Breast-Cancer-Ultrasound-Images-Dataset`

The repository tags identify the base model as:

`google/vit-base-patch16-224-in21k`

The model card reports approximately 1,578 breast ultrasound images across
normal, benign, and malignant classes, with approximately 1,500 samples used
for training.

These training details are treated as author-reported unless independently
reconstructed from the original training pipeline.

## 7. Author-Reported Training Procedure

According to the model card:

- Training duration: 12 epochs
- Approximate training size: 1,500 samples
- Data augmentation: 20% noise augmentation
- Final reported validation accuracy: 94.46%

The complete original training pipeline has not been independently reproduced
as part of this study.

## 8. Author-Reported Performance

The model developer reports:

- Internal validation accuracy: 94.46%

The model card also reports evaluation on:

`as-cle-bert/breastcanc-ultrasound-class`

Reported benchmark results:

- Total images: 647
- Images included in primary binary metrics: 644
- Predictions classified as `normal`: 3
- Accuracy: 96.12%
- Malignant precision: 94.26%
- Malignant sensitivity/recall: 93.81%
- Malignant F1-score: 94.03%

The three `normal` predictions were excluded from the reported primary binary
performance metrics because the benchmark dataset contains only benign and
malignant labels.

## 9. External Benchmark Provenance Concern

The reported external benchmark should be interpreted cautiously.

The benchmark dataset
`as-cle-bert/breastcanc-ultrasound-class`
states that its 647 images originate from the breast ultrasound dataset
published by Al-Dhabyani et al.

The training dataset used by the pretrained model also traces its underlying
breast ultrasound images to the Al-Dhabyani/BUSI dataset family.

Therefore, the reported benchmark does not appear to represent
institutionally independent external testing.

Exact image-level overlap between the model's training samples and the
reported benchmark has not been established in this audit. Consequently, no
claim of confirmed data leakage is made.

This limitation motivates the use of BUS-UCLM as the independent external test
dataset in the present study.

## 10. Intended Use and Limitations

According to the model card, the model is intended for breast ultrasound image
classification and should not be used as a standalone diagnostic system.

The model card identifies the following limitations or out-of-scope uses:

- Non-ultrasound imaging modalities such as mammography, MRI, or CT
- Images containing text overlays, annotations, calipers, or other artifacts
- Standalone clinical diagnosis without expert review
- Pediatric or non-breast ultrasound images
- Potential variation in performance across imaging equipment, patient
  populations, geography, and image quality

The present study treats the model strictly as a research model and does not
evaluate or claim clinical deployment readiness.

## 11. Audit Summary

The technical configuration of the model was independently verified from the
pinned Hugging Face revision.

The training procedure and previously reported performance metrics were
obtained from the developer's model card and are therefore identified as
author-reported.

A key provenance concern is that both the training source and the developer's
reported external benchmark appear to derive from the BUSI/Al-Dhabyani dataset
family. This prevents the reported benchmark from being treated as strong
evidence of institutionally independent generalization.

The present study will therefore perform external testing using BUS-UCLM
without retraining or fine-tuning the model.
