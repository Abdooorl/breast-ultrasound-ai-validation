# Tables

## Table 1. Model and external test cohort characteristics

| Characteristic | Value |
|---|---|
| Evaluated model | `Parveshiiii/breast-cancer-detector` |
| Model revision | `7c4c1ac11f5f80d382cc641ec6f2d3989e7fa73c` |
| Base architecture | Vision Transformer, ViT-B/16 |
| Base checkpoint | `google/vit-base-patch16-224-in21k` |
| Input size | 224 × 224 pixels |
| Patch size | 16 × 16 pixels |
| Output classes | Benign, malignant, normal |
| Model adaptation on external data | None |
| External dataset | BUS-UCLM |
| External test images | 683 |
| Patients | 38 |
| Normal images | 419 |
| Benign images | 174 |
| Malignant images | 90 |
| Ultrasound system | Siemens ACUSON S2000 |
| Acquisition period | 2022–2023 |
| Primary prediction unit | Image |
| Primary outcome | Malignant sensitivity |
| Bootstrap procedure | Patient-cluster bootstrap |
| Bootstrap replicates | 5,000 |
| Bootstrap random seed | 42 |
| Clean-input sensitivity subset | 521 images |

*Note:* Model weights, preprocessing, and class mapping were frozen before complete external inference. No BUS-UCLM image or label was used for retraining, fine-tuning, or recalibration.

## Table 2. External performance on BUS-UCLM with patient-clustered 95% confidence intervals

| Outcome | Estimate | 95% CI |
|---|---:|---:|
| **Three-class performance** |  |  |
| Accuracy | 0.4583 | 0.3756–0.5347 |
| Balanced accuracy | 0.5338 | 0.4564–0.6073 |
| Macro F1 | 0.4544 | 0.3608–0.5346 |
| **Malignant vs non-malignant performance** |  |  |
| Sensitivity | 0.4333 | 0.2417–0.6136 |
| Specificity | 0.9224 | 0.8788–0.9584 |
| Positive predictive value | 0.4588 | 0.2222–0.6863 |
| Negative predictive value | 0.9147 | 0.8574–0.9589 |
| F1 score | 0.4457 | 0.2446–0.6114 |
| Balanced accuracy | 0.6779 | 0.5776–0.7740 |
| ROC-AUC | 0.7855 | 0.7060–0.8645 |
| PR-AUC | 0.4619 | 0.2500–0.6487 |

*Note:* Confidence intervals were obtained using 5,000 patient-cluster bootstrap replicates. ROC-AUC and PR-AUC used the malignant softmax score as the continuous decision variable.

## Table 3. Prespecified confidence-based abstention threshold sweep

| Threshold | Overall coverage | Malignant coverage | Malignant abstained, n | Accepted accuracy | Accepted malignant sensitivity* | Accepted specificity* | Accepted malignant FN | Accepted FP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.9649 | 1.0000 | 0 | 0.4613 | 0.4333 | 0.9262 | 51 | 42 |
| 0.55 | 0.9151 | 0.9778 | 2 | 0.4592 | 0.4318 | 0.9311 | 50 | 37 |
| 0.60 | 0.8624 | 0.9444 | 5 | 0.4652 | 0.4471 | 0.9325 | 47 | 34 |
| 0.65 | 0.8053 | 0.9111 | 8 | 0.4691 | 0.4634 | 0.9316 | 44 | 32 |
| 0.70 | 0.7189 | 0.8556 | 13 | 0.4949 | 0.4935 | 0.9348 | 39 | 27 |
| 0.75 | 0.6486 | 0.8000 | 18 | 0.4989 | 0.4861 | 0.9326 | 37 | 25 |
| 0.80 | 0.5637 | 0.6667 | 30 | 0.5195 | 0.5500 | 0.9354 | 27 | 21 |
| 0.85 | 0.4553 | 0.6111 | 35 | 0.5434 | 0.5818 | 0.9297 | 23 | 18 |
| 0.90 | 0.3397 | 0.4778 | 47 | 0.6121 | 0.6279 | 0.9365 | 16 | 12 |
| 0.95 | 0.1742 | 0.1889 | 73 | 0.8403 | 0.7059 | 0.9608 | 5 | 4 |

*Note:* Overall coverage is the proportion of all 683 external images receiving an accepted prediction. Malignant coverage is the proportion of the 90 malignant images receiving an accepted prediction. Metrics marked with an asterisk were calculated only among accepted cases. FN = false negative; FP = false positive. Abstained observations were reported separately and were not counted as correct predictions. The complete prespecified threshold sweep is shown rather than selecting a single threshold after observing external-test labels.

## Table 4. Malignant and high-confidence error patterns during external testing

| Failure characteristic | Count | Percentage |
|---|---:|---:|
| **Malignant false negatives** | **51** | **100.0%** |
| Predicted benign | 48 | 94.1% |
| Predicted normal | 3 | 5.9% |
| FN confidence ≥ 0.80 | 27 | 52.9% |
| FN confidence ≥ 0.90 | 16 | 31.4% |
| FN confidence ≥ 0.95 | 5 | 9.8% |
| Malignant FN patients | 14 | — |
| FN from five highest-count patients | 29 | 56.9% |
| **False-positive malignant predictions** | **46** | **100.0%** |
| Reference normal | 30 | 65.2% |
| Reference benign | 16 | 34.8% |
| FP confidence ≥ 0.80 | 21 | 45.7% |
| FP confidence ≥ 0.90 | 12 | 26.1% |
| FP confidence ≥ 0.95 | 4 | 8.7% |
| False-positive patients | 16 | — |
| FP from five highest-count patients | 29 | 63.0% |

*Note:* Confidence refers to maximum-softmax confidence for the model's predicted class. Percentages for confidence thresholds and prediction directions are relative to the corresponding false-negative or false-positive group. Patient concentration is descriptive and does not establish a causal failure mechanism.
