## Abstract

### Objective
To evaluate the external performance and reliability of a publicly released pretrained breast-ultrasound Vision Transformer on BUS-UCLM without model adaptation, and to assess whether confidence-based abstention improves reliability.

### Methods
The frozen three-class classifier was evaluated on 683 BUS-UCLM images from 38 patients. The primary outcome was malignant sensitivity. Secondary analyses included malignant-versus-non-malignant discrimination, patient-cluster bootstrap confidence intervals, calibration, confidence and margin analysis, selective classification, risk-coverage analysis, and malignant false-negative auditing.

### Results
Malignant sensitivity was 0.4333 (95% CI 0.2417-0.6136), despite a malignant-score ROC-AUC of 0.7855 (95% CI 0.7060-0.8645). Of 51 malignant false negatives, 48 (94.1%) were classified as benign and 27 (52.9%) had maximum-softmax confidence of at least 0.80. Expected Calibration Error was 0.3386. Raising the acceptance threshold reduced accepted-case error but sharply reduced coverage; at a threshold of 0.90, coverage was 0.3397 and 16 malignant false negatives remained accepted.

### Conclusion
The frozen classifier showed limited external performance on BUS-UCLM, with low malignant sensitivity, substantial overconfidence, and high-confidence malignant-to-benign errors. Confidence-based abstention did not provide a reliable safety boundary on this external cohort because meaningful risk reduction required substantial loss of coverage and high-confidence malignant misses persisted.

**Keywords:** breast ultrasound; artificial intelligence; external testing; Vision Transformer; calibration; uncertainty; selective classification
