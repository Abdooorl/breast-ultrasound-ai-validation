# 3. Results

## 3.1 External test cohort

The complete eligible BUS-UCLM cohort comprised 683 breast-ultrasound images from 38 patients. The reference-standard class distribution was 419 normal images, 174 benign images, and 90 malignant images. All 683 images were retained for the primary external analysis. A prespecified clean-input sensitivity subset, excluding images flagged as containing visual marks/annotations, Doppler information, or combined/multi-panel content, contained 521 images.

The frozen inference output contained one prediction per image with no missing predictions, duplicate image rows, or class-probability inconsistencies. All analyses reported below were performed from the same frozen prediction table.

## 3.2 Three-class external performance

Across the full 683-image external test set, overall three-class accuracy was 0.4583 and balanced accuracy was 0.5338. Macro-averaged precision, recall, and F1 score were 0.5743, 0.5338, and 0.4544, respectively.

Class-specific performance differed substantially. For benign images, precision was 0.3262, recall was 0.8793, and F1 score was 0.4759. For malignant images, precision was 0.4588, recall was 0.4333, and F1 score was 0.4457. For normal images, precision was 0.9380, recall was 0.2888, and F1 score was 0.4416.

The three-class confusion matrix, ordered as benign, malignant, and normal, was: reference benign, 153 benign, 16 malignant, and 5 normal; reference malignant, 48 benign, 39 malignant, and 3 normal; reference normal, 268 benign, 30 malignant, and 121 normal. Thus, the dominant error for malignant images was assignment to the benign class, while many normal images were also assigned to benign.

## 3.3 Malignant versus non-malignant discrimination

For the prespecified binary analysis, benign and normal images were combined into a non-malignant class. The resulting confusion counts were 39 true positives, 547 true negatives, 46 false positives, and 51 false negatives.

Malignant sensitivity, the primary endpoint, was 0.4333. Specificity was 0.9224, positive predictive value was 0.4588, negative predictive value was 0.9147, F1 score was 0.4457, and balanced accuracy was 0.6779. Using the malignant softmax score as the continuous decision variable, ROC-AUC was 0.7855 and PR-AUC was 0.4619.

These results show that ranking performance was stronger than the sensitivity achieved by the model's default hard three-class decision rule. However, more than half of the malignant images were not assigned to the malignant class.

## 3.4 Patient-clustered confidence intervals

Patient-cluster bootstrap confidence intervals were calculated using 5,000 replicates with the patient as the resampling unit.

For the full external cohort, the 95% confidence interval for overall three-class accuracy was 0.3756 to 0.5347. The corresponding intervals were 0.4564 to 0.6073 for three-class balanced accuracy and 0.3608 to 0.5346 for macro F1 score.

For malignant detection, sensitivity was 0.4333 with a 95% confidence interval of 0.2417 to 0.6136. Specificity was 0.9224 (95% CI 0.8788–0.9584), positive predictive value was 0.4588 (95% CI 0.2222–0.6863), negative predictive value was 0.9147 (95% CI 0.8574–0.9589), and F1 score was 0.4457 (95% CI 0.2446–0.6114). Binary balanced accuracy was 0.6779 (95% CI 0.5776–0.7740), ROC-AUC was 0.7855 (95% CI 0.7060–0.8645), and PR-AUC was 0.4619 (95% CI 0.2500–0.6487).

## 3.5 Clean-input sensitivity analysis

The clean-input subset contained 521 images. Three-class accuracy was 0.3973, balanced accuracy was 0.5258, and macro F1 score was 0.3734.

Within this subset, the binary malignant-versus-non-malignant confusion counts were 14 true positives, 449 true negatives, 36 false positives, and 22 false negatives. Malignant sensitivity was 0.3889 and specificity was 0.9258. ROC-AUC was 0.8018 and PR-AUC was 0.2828.

Because the clean-input subset had a different case composition and contained only 36 malignant images, these values were treated as a sensitivity analysis rather than evidence that removal of flagged image types improved or worsened model performance.

## 3.6 Confidence and prediction-margin behavior

Across all 683 predictions, mean maximum-softmax confidence was 0.7961 and median confidence was 0.8313. Mean prediction margin was 0.6340 and median margin was 0.7009.

Correct predictions had higher confidence on average than incorrect predictions. Among 313 correct predictions, mean confidence was 0.8250 and median confidence was 0.8808, compared with 0.7717 and 0.7998 among 370 incorrect predictions. Mean prediction margin was 0.6802 for correct predictions and 0.5949 for incorrect predictions.

The difference was more pronounced within malignant cases. For the 39 malignant true positives, mean confidence was 0.8981 and mean prediction margin was 0.8127. For the 51 malignant false negatives, mean confidence remained relatively high at 0.8038, with a mean prediction margin of 0.6473. Lower confidence was therefore associated with error on average, but some malignant errors were still made with high confidence.

## 3.7 Confidence-based abstention

The prespecified threshold sweep demonstrated a trade-off between retained coverage and performance among accepted predictions.

At a confidence threshold of 0.50, coverage was 0.9649 and accepted-case accuracy was 0.4613. Malignant sensitivity among accepted cases was 0.4333, with 51 accepted malignant false negatives.

At a threshold of 0.70, coverage fell to 0.7189, accepted-case accuracy increased to 0.4949, and malignant sensitivity among accepted cases increased to 0.4935. The number of accepted malignant false negatives decreased to 39.

At a threshold of 0.80, coverage was 0.5637, accepted-case accuracy was 0.5195, and malignant sensitivity among accepted cases was 0.5500, with 27 accepted malignant false negatives.

At a threshold of 0.90, coverage decreased to 0.3397. Accepted-case accuracy was 0.6121, malignant sensitivity was 0.6279, and 16 malignant false negatives remained among accepted predictions.

At the most restrictive threshold of 0.95, only 0.1742 of all images were accepted. Accepted-case accuracy increased to 0.8403 and malignant sensitivity among accepted malignant cases was 0.7059. Five malignant false negatives remained accepted, while 73 of the 90 malignant images were abstained.

Increasingly strict confidence thresholds reduced the number of accepted malignant false negatives and lowered selective error, but this occurred at the cost of substantial loss of overall and malignant-case coverage.

## 3.8 Risk-coverage analysis

The continuous risk-coverage analysis showed that predictions ranked by maximum-softmax confidence contained useful ordering information.

At approximately 25% coverage, 171 images were accepted and accuracy was 0.7135, corresponding to a selective risk of 0.2865. At approximately 50% coverage, 342 images were accepted and accuracy was 0.5380, with risk 0.4620. At approximately 75% coverage, 512 images were accepted and accuracy was 0.4883, with risk 0.5117. At full coverage, accuracy was 0.4583 and risk was 0.5417.

The descriptive area under the empirical risk-coverage curve was 0.3831. The curve therefore showed that confidence ranking could identify a lower-risk subset, but meaningful reductions in accepted-case risk required substantial reductions in coverage.

## 3.9 Calibration

Maximum-softmax confidence was substantially higher than observed top-class correctness on the external dataset. Overall accuracy was 0.4583, while mean confidence was 0.7961, a difference of 0.3379.

Using 10 equal-width confidence bins, Expected Calibration Error was 0.3386. The multiclass Brier score was 0.8080.

The largest reliability gaps occurred in the high-confidence bins. For predictions with confidence between 0.80 and 0.90, mean confidence was 0.8531 while empirical accuracy was 0.3791. For predictions with confidence between 0.90 and 1.00, mean confidence was 0.9479 while empirical accuracy was 0.6121.

These results indicate substantial overconfidence under external testing. Confidence could still rank predictions by relative reliability while remaining poorly calibrated in absolute terms.

## 3.10 Failure analysis

The malignant error audit reproduced the binary confusion counts of 39 true positives, 547 true negatives, 46 false positives, and 51 false negatives.

Among the 51 malignant false negatives, 48 (94.1%) were predicted as benign and 3 (5.9%) were predicted as normal. The principal malignant failure mode was therefore malignant-to-benign misclassification.

Among the 46 false-positive malignant predictions, 30 (65.2%) originated from reference-normal images and 16 (34.8%) from reference-benign images.

High-confidence errors were common. Of the 51 malignant false negatives, 27 (52.9%) had maximum-softmax confidence of at least 0.80, 16 (31.4%) had confidence of at least 0.90, and 5 (9.8%) had confidence of at least 0.95. Among the 46 malignant false positives, 21 (45.7%) had confidence of at least 0.80, 12 (26.1%) at least 0.90, and 4 (8.7%) at least 0.95.

Malignant false negatives were distributed across 14 patients. The five patients with the largest false-negative counts accounted for 29 of 51 false negatives (56.9%). False positives were distributed across 16 patients, with the five patients contributing the largest false-positive counts accounting for 29 of 46 false positives (63.0%). These patient-level concentrations were described as clustering rather than as proof of a specific failure mechanism.

Metadata patterns were also examined descriptively. Among malignant false negatives, 56.9% were flagged as containing visual marks, 17.7% contained Doppler information, and 43.1% belonged to the prespecified clean-input subset. Because substantial numbers of errors also occurred among clean-input images, the observed failures could not be attributed solely to overlays or other flagged image characteristics.

Overall, the failure analysis showed that the model's most important error pattern was confident malignant-to-benign misclassification, and that confidence-based abstention reduced but did not eliminate this failure mode.
