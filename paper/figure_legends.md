# Figure Legends

**Figure 1. Study workflow for frozen external evaluation of the pretrained breast-ultrasound Vision Transformer.** The publicly released classifier was applied to BUS-UCLM without retraining, fine-tuning, recalibration, or other adaptation. Predictions were evaluated for three-class performance, malignant-versus-non-malignant discrimination, patient-clustered uncertainty, confidence and calibration, selective classification, risk–coverage behavior, and failure patterns.

**Figure 2. Three-class confusion matrix for external BUS-UCLM evaluation.** Rows represent the BUS-UCLM reference class and columns represent model predictions. Among 90 malignant images, 48 were classified as benign, 39 as malignant, and 3 as normal. Malignant-to-benign classification was therefore the dominant malignant error direction.

**Figure 3. Receiver operating characteristic curve for malignant-versus-non-malignant discrimination.** The malignant softmax score was used as the continuous decision variable. ROC-AUC was 0.7855, with a patient-cluster bootstrap 95% confidence interval of 0.7060–0.8645. The curve represents ranking performance and should be interpreted separately from malignant sensitivity under the model's frozen hard-class decision rule.

**Figure 4. Precision–recall curve for malignant-versus-non-malignant discrimination.** The malignant softmax score was used as the continuous decision variable. PR-AUC was 0.4619, with a patient-cluster bootstrap 95% confidence interval of 0.2500–0.6487.

**Figure 5. Reliability of maximum-softmax confidence during external testing.** Mean maximum-softmax confidence was compared with empirical top-class accuracy across 10 equal-width confidence bins. Perfect calibration is represented by the identity line. Expected Calibration Error was 0.3386 and the multiclass Brier score was 0.8080. Maximum-softmax confidence substantially overestimated observed correctness on the external dataset.

**Figure 6. Risk–coverage relationship under confidence-based selective classification.** Predictions were ranked by maximum-softmax confidence and progressively lower-confidence predictions were excluded. Selective risk decreased as coverage decreased, indicating that confidence retained useful relative ranking information. However, substantial reductions in accepted-case risk required substantial reductions in coverage. The empirical area under the risk–coverage curve was 0.3831 and is reported descriptively.

**Supplementary Figure S1. BUS-UCLM class distribution in the full external test cohort.** The primary analysis included 683 images from 38 patients: 419 normal, 174 benign, and 90 malignant images.
