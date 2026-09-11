# Failure Analysis Findings

## Results

### Malignant classification failures

In the malignant-versus-non-malignant evaluation, the frozen model produced 39 true positives, 51 false negatives, 46 false positives, and 547 true negatives. The dominant malignant failure mode was confusion with the benign class: 48 of 51 malignant false negatives (94.1%) were predicted as benign, whereas only 3 of 51 (5.9%) were predicted as normal.

Malignant false negatives were not restricted to low-confidence predictions. Their mean maximum-softmax confidence was 0.804 and their median confidence was 0.815. Of the 51 malignant false negatives, 27 (52.9%) had confidence greater than or equal to 0.80, 16 (31.4%) had confidence greater than or equal to 0.90, and 5 (9.8%) had confidence greater than or equal to 0.95. The mean top-two probability margin among malignant false negatives was 0.647, compared with 0.813 among malignant true positives.

False-positive malignant predictions arose more frequently from normal than benign cases. Of the 46 false positives, 30 (65.2%) had a normal ground-truth label and 16 (34.8%) had a benign ground-truth label. False positives had a mean confidence of 0.751 and a mean probability margin of 0.540.

### Patient-level error patterns

Malignant false negatives occurred across 14 patients, indicating that missed malignant cases were not confined to a single subject. However, errors were unevenly distributed. The five patients contributing the largest numbers of malignant false negatives accounted for 29 of 51 misses (56.9%). Several patients had all of their malignant images misclassified, including LOTI (6/6), ELCO (5/5), UNCU (5/5), CODE (4/4), FLKA (4/4), COVA (3/3), and FLBA (3/3). These patient-level rates should be interpreted cautiously because several denominators were small.

False-positive malignant predictions occurred across 16 patients. The five patients contributing the largest numbers of false positives accounted for 29 of 46 false alarms (63.0%). HESN had the largest concentration, with 9 of 13 eligible non-malignant images predicted as malignant.

### Image metadata patterns

Among malignant false negatives, 17.7% were Doppler images, 56.9% contained marks, and 43.1% met the predeclared clean-input definition. By comparison, 25.6% of malignant true positives were Doppler images, 64.1% contained marks, and 35.9% were clean inputs.

Among malignant false positives, 6.5% were Doppler images, 21.7% contained marks, and 78.3% were clean inputs. Corresponding values among malignant true negatives were 2.7%, 16.5%, and 82.1%, respectively. No combined-image cases occurred among malignant true positives, false negatives, or false positives.

These metadata comparisons are descriptive only and do not establish that Doppler content, annotations, or other acquisition characteristics caused classification errors.

### Calibration and confidence

The model was substantially overconfident on the external BUS-UCLM dataset. Overall three-class accuracy was 0.458, whereas mean maximum-softmax confidence was 0.796, giving a confidence-minus-accuracy difference of 0.338. Expected calibration error using 10 equal-width confidence bins was 0.339, and the multiclass Brier score was 0.808.

The largest calibration gap occurred in the 0.80-0.90 confidence interval, where 153 predictions had a mean confidence of 0.853 but an empirical accuracy of only 0.379. Even in the 0.90-1.00 interval, mean confidence was 0.948 while empirical accuracy was 0.612.

## Discussion

The failure analysis indicates that the principal safety-relevant weakness of the externally evaluated model was failure to distinguish malignant from benign-appearing cases. Nearly all malignant false negatives were assigned to the benign class rather than the normal class, suggesting that the principal external generalization problem was not simply failure to detect abnormality, but failure to correctly separate malignant and benign ultrasound appearances.

Confidence-based uncertainty provided useful ranking information but did not reliably identify all clinically important errors. Malignant true positives had higher average confidence and larger probability margins than malignant false negatives, supporting the use of confidence as a selective-classification signal. However, more than half of malignant false negatives still had confidence of at least 0.80, and approximately one third had confidence of at least 0.90. Therefore, high softmax confidence could not be interpreted as evidence that a prediction was correct.

This observation is consistent with the calibration analysis. The model's average confidence substantially exceeded its empirical accuracy, and large calibration gaps remained even in the highest-confidence bins. Accordingly, raw softmax output should be treated as a model-confidence score rather than as a calibrated estimate of clinical probability.

The patient-level analysis also showed that errors were not uniformly distributed. Although malignant false negatives occurred across 14 patients, more than half of all malignant misses arose from five patients, and several patients had all available malignant images misclassified. This pattern may reflect patient-specific image characteristics, lesion appearance, acquisition conditions, repeated correlated images, or other sources of distribution shift; however, the present analysis does not establish which factors were responsible.

Similarly, the descriptive metadata analysis did not support a simple explanation in which visible marks, Doppler imaging, or combined images alone account for model failure. A substantial proportion of malignant false negatives occurred in images classified as clean inputs, and false positives were also predominantly observed among clean-input images. These observations reinforce that failure cannot be attributed to a single recorded metadata feature without additional controlled analysis.

Taken together, the findings show that the frozen public model retained useful discriminatory signal on BUS-UCLM but did not generalize reliably at its default decision rule. Its malignant-score ranking performance was materially better than its hard classification sensitivity, while its softmax probabilities were poorly calibrated. Selective abstention reduced accepted-case error rates and malignant false negatives as confidence thresholds increased, but this improvement required progressively lower coverage and did not eliminate high-confidence malignant misses. These results support uncertainty-aware abstention as a risk-management mechanism for research evaluation, but not as evidence that the model is suitable for autonomous clinical diagnosis.
