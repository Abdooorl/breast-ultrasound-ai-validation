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
