# Research Protocol

**Protocol version:** 1.0  
**Protocol status:** Frozen before external inference  
**Freeze date:** 2026-08-28  
**Project:** `breast-ultrasound-ai-validation`

## Working title

**Does a Public Breast Ultrasound AI Model Generalize? External Testing and Uncertainty-Aware Abstention of a Pretrained Vision Transformer**

## 1. Study purpose

Public medical-imaging models can report strong performance on internal or closely related datasets while performing differently when exposed to images acquired at another institution, on another scanner, or from a different patient population. This study will independently test the generalization and reliability of a publicly available pretrained breast-ultrasound Vision Transformer on an independently acquired public dataset.

The study is an **external testing study**, not a model-development study. The pretrained model will be evaluated without retraining or fine-tuning on the external dataset. A prespecified uncertainty-aware selective-classification analysis will then examine whether allowing the model to abstain on low-confidence cases can reduce prediction errors while retaining useful coverage.

The study is for research purposes only and does not evaluate the model as a standalone clinical diagnostic device.

## 2. Primary objective

To evaluate the external generalization and reliability of a publicly available pretrained breast-ultrasound Vision Transformer on BUS-UCLM without retraining or fine-tuning, and to determine how confidence-based abstention changes the trade-off between prediction coverage and error.

## 3. Model under evaluation

- **Model:** `Parveshiiii/breast-cancer-detector`
- **Source:** Hugging Face
- **Task:** three-class breast-ultrasound image classification
- **Output classes:** `benign`, `malignant`, `normal`
- **Base architecture:** Vision Transformer (ViT)
- **Reported training dataset:** `gymprathap/Breast-Cancer-Ultrasound-Images-Dataset`
- **Weights:** frozen for all primary analyses
- **Model revision:** to be recorded and pinned before the first external inference run
- **Image processor / preprocessing configuration:** loaded from the released model repository where available and documented before inference

No BUS-UCLM image or label will be used to update model weights.

## 4. External testing dataset

The primary external testing dataset is **BUS-UCLM**, an independently acquired public breast-ultrasound dataset collected at Ciudad Real General University Hospital / University of Castilla-La Mancha, Spain.

Published dataset composition:

- 683 ultrasound images
- 38 patients
- 419 normal images
- 174 benign images
- 90 malignant images
- acquisition performed using a Siemens ACUSON S2000 system during 2022–2023
- lesion annotations supplied by expert breast radiologists
- malignancy confirmed by biopsy for malignant lesions
- segmentation masks supplied separately from the source images

### Unit of analysis

The model produces one prediction per ultrasound image. Therefore, the **prediction unit is the image**. Because multiple images can originate from the same patient, observations are not treated as statistically independent at the patient level. Patient identifiers will be retained during analysis and used for clustered uncertainty estimation.

## 5. Reference standard

The study will use the official BUS-UCLM class labels as the reference standard. The term **reference standard** is used instead of “ground truth” in line with CLAIM 2024 terminology.

For the binary malignancy analysis:

- `malignant` -> malignant
- `benign` -> non-malignant
- `normal` -> non-malignant

No official class label will be modified on the basis of the model prediction.

## 6. Research questions

### RQ1 — Three-class external generalization

**How well does the pretrained Vision Transformer classify normal, benign, and malignant breast-ultrasound images from an independently acquired external dataset without retraining or fine-tuning?**

Prespecified measures:

- overall accuracy
- balanced accuracy
- macro precision
- macro recall
- macro F1
- per-class precision, recall, and F1
- three-class confusion matrix

### RQ2 — Malignancy discrimination

**How well does the model distinguish malignant from non-malignant breast-ultrasound images during external testing?**

The binary analysis will collapse benign and normal images into the non-malignant class.

**Primary outcome:** malignant sensitivity.

Secondary binary measures:

- specificity
- positive predictive value (PPV / precision)
- negative predictive value (NPV)
- F1 score
- balanced accuracy
- ROC-AUC using the malignant model score
- PR-AUC using the malignant model score
- false-negative count and rate
- false-positive count and rate

### RQ3 — Confidence and calibration

**How well does model confidence correspond to actual correctness, and how well calibrated are its predictions on the external dataset?**

Prespecified analyses:

- maximum softmax probability as the primary confidence score
- top-two probability margin as a secondary uncertainty score
- confidence distribution for correct vs incorrect predictions
- reliability diagram
- Expected Calibration Error (ECE)
- Brier score

Raw softmax confidence will **not** be interpreted as a clinically calibrated probability of cancer.

### RQ4 — Uncertainty-aware abstention

**Can confidence-based selective classification reduce prediction errors, particularly malignant false negatives, while retaining useful prediction coverage?**

The model will be allowed to abstain when maximum softmax confidence falls below a prespecified threshold.

Threshold sweep:

`0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95`

At every threshold, the following will be reported:

- overall coverage
- overall abstention rate
- class-specific coverage
- malignant-case abstention rate
- accepted-case accuracy
- accepted-case balanced accuracy
- accepted-case malignant sensitivity
- accepted-case specificity
- accepted malignant false negatives
- accepted false positives
- risk-coverage relationship

Abstained cases will be reported separately and will not be silently counted as correct predictions. No single threshold will be selected after viewing BUS-UCLM labels and then presented as if it had been prespecified. The complete threshold curve will be reported.

Any threshold later used in a research demo will be clearly identified as a post-analysis implementation choice unless a separate prespecified rule is established before use.

### RQ5 — Failure patterns

**What systematic failure patterns occur during external testing, particularly among malignant false-negative predictions?**

Failure analysis will examine:

- actual class and predicted class
- confidence score
- second-highest predicted class
- top-two probability margin
- malignant -> benign errors
- malignant -> normal errors
- benign -> malignant errors
- normal -> malignant errors
- confidently incorrect predictions
- whether a prespecified abstention threshold would have deferred the error

Where BUS-UCLM metadata permit, prespecified subgroup descriptions will also examine whether errors differ for:

- images containing visual marks / annotations
- Doppler images
- combined or multi-panel images
- images without these flags

These subgroup analyses are descriptive and exploratory unless sample sizes support stable estimates.

## 7. Primary and secondary outcomes

### Primary outcome

**Image-level malignant sensitivity on the full eligible BUS-UCLM external testing set, with a patient-clustered 95% confidence interval.**

This endpoint is prioritized because malignant false negatives are a clinically important failure mode for a breast-imaging classifier.

### Secondary outcomes

- malignant specificity
- PPV
- NPV
- ROC-AUC
- PR-AUC
- binary F1
- three-class accuracy
- balanced accuracy
- macro F1
- calibration metrics
- false-negative / false-positive patterns
- risk-coverage behavior under abstention

No claim of clinical utility will be based on any single metric.

## 8. Inclusion and exclusion rules

### Primary analysis inclusion

All official BUS-UCLM source ultrasound images will be included if they:

1. can be successfully decoded,
2. have an official recognizable reference label, and
3. can be mapped to `normal`, `benign`, or `malignant`.

### Permitted exclusions

An image may be excluded only for a documented technical reason such as:

- corrupted or unreadable file
- missing official label
- duplicate file proven to be an accidental duplicate in the distributed dataset, if the duplicate policy is established during the dataset audit before final inference

### Prohibited outcome-based exclusions

Images will not be excluded because:

- the model is uncertain
- the model predicts the wrong class
- the image appears difficult
- inclusion lowers a metric

Every exclusion will be logged before final performance reporting.

## 9. Dataset audit before inference

Before the model is run over the complete dataset, the following will be audited and documented:

- total file count
- number of patients
- images per patient
- class distribution
- file types
- image dimensions
- unreadable files
- missing labels
- exact duplicate hashes
- suspicious filename or metadata inconsistencies
- available BUS-UCLM metadata flags
- visual-mark / annotation flags
- Doppler flags
- combined-image flags

The resulting manifest will preserve a stable image identifier, patient identifier, reference label, file path, and relevant metadata.

The full prediction table must not be generated until this audit is complete and the eligible-image manifest is frozen.

## 10. Preprocessing rules

The same preprocessing pipeline will be applied to every included image.

Primary preprocessing will follow the processor/configuration distributed with the pretrained model as closely as technically possible. Any deviation will be documented before full external inference.

The classifier will receive the ultrasound image only.

**BUS-UCLM segmentation masks will not be supplied as model inputs.** They are reserved for optional post-hoc explainability/localization analysis.

No image will receive hand-tuned preprocessing based on its prediction or label.

## 11. Primary external-testing rule

The external testing sequence is:

1. audit and freeze the eligible BUS-UCLM manifest,
2. pin the pretrained model revision,
3. pin the software environment,
4. freeze the preprocessing implementation,
5. run inference once over the eligible dataset,
6. save raw scores and predictions,
7. freeze the prediction file before performance analysis.

The model will not be retrained, fine-tuned, or optimized using BUS-UCLM.

BUS-UCLM labels may be used only for evaluation after predictions have been generated.

## 12. Prespecified sensitivity analysis for model-intended inputs

The model card states that the model is intended for clean breast-ultrasound images without certain overlays or artifacts. BUS-UCLM reports that some images contain visual marks, Doppler information, or combined scans.

Therefore, in addition to the primary all-eligible-image analysis, a **prespecified secondary sensitivity analysis** will evaluate the subset of BUS-UCLM images that best matches the released model's stated input conditions, where dataset metadata allow this subset to be defined objectively.

The all-eligible-image analysis remains primary. The clean-input subset will not replace it.

## 13. Statistical analysis plan

### Point estimates

Metrics will first be computed at the image level because the model produces one prediction per image.

### Patient-clustered confidence intervals

Because multiple images may belong to one patient, 95% confidence intervals will be estimated using a **patient-cluster bootstrap** rather than a naive image-level bootstrap.

Planned procedure:

- resampling unit: patient
- all images for a sampled patient remain together
- bootstrap replicates: 5,000
- random seed: 42
- confidence interval: percentile 95% CI unless a justified alternative is documented before final analysis

For a bootstrap replicate in which a metric is mathematically undefined because the required class is absent, that replicate will be recorded as invalid for that metric rather than assigned an artificial value. The number of valid bootstrap replicates will be reported for affected metrics.

### Comparisons with model-card performance

Performance reported by the model developer will be reproduced in the background/results tables for context where definitions are sufficiently clear.

These comparisons are **descriptive**, because the developer's datasets, label mappings, inclusion rules, and evaluation design are not identical to BUS-UCLM. The study will not treat a numerical difference between reported performance and BUS-UCLM performance as a formal paired statistical comparison.

## 14. Calibration analysis

Calibration will be assessed separately from discrimination.

Planned outputs:

- reliability plots
- ECE
- Brier score
- confidence distributions by correctness

Where appropriate, both three-class confidence behavior and malignant-vs-non-malignant confidence behavior will be described.

No post-hoc calibration method will be presented as part of the frozen baseline model. If temperature scaling or another calibration method is explored later, it will be labeled as a separate post-hoc experiment and will not overwrite baseline results.

## 15. Abstention and selective-risk analysis

The abstention experiment evaluates the existing model without altering its weights.

For threshold `t`:

- prediction accepted if `max_softmax >= t`
- prediction abstained if `max_softmax < t`

Coverage is the proportion of all eligible images receiving an accepted prediction.

Selective risk will be summarized primarily as the error rate among accepted predictions and displayed against coverage.

Because abstention can disproportionately affect a class, overall coverage alone is insufficient. Malignant coverage and malignant abstention counts will always be reported alongside accepted-case malignant sensitivity.

The analysis will emphasize the **trade-off**, not search for a threshold that maximizes a single metric on BUS-UCLM.

## 16. Failure-analysis rules

Primary attention will be given to malignant false negatives.

A frozen prediction table will be used for failure analysis. Images may be inspected qualitatively after the aggregate predictions are frozen, but inspection will not trigger retroactive exclusion from the primary analysis unless a genuine data-integrity error is discovered. Any such amendment must be documented.

Failure examples selected for figures will be described as illustrative cases and will not replace aggregate statistics.

## 17. Optional exploratory explainability analysis

If time permits after all core analyses are complete, an exploratory analysis may compare model attribution/attention regions with BUS-UCLM lesion segmentation masks.

Possible outputs include:

- qualitative attribution overlays
- lesion/attention overlap measures where methodologically defensible
- comparison of attribution patterns for correct vs incorrect classifications

This analysis is **optional** and may be removed without changing the primary study.

No attribution map will be described as proof that the model reasons like a radiologist.

## 18. Reproducibility

The project will record or preserve:

- Git commit hash for analysis code
- exact Hugging Face model identifier and revision
- Python version
- locked Python dependencies (`requirements-lock.txt`)
- random seed(s)
- dataset manifest
- patient identifiers in anonymized dataset form
- raw model scores
- predicted labels
- evaluation configuration
- generated metric files and figures

Raw patient images and large dataset files will not be committed to GitHub unless their license, privacy status, and repository-size implications have been reviewed. The public repository will instead provide acquisition/preparation instructions and code.

## 19. Reporting framework

The manuscript will follow the principles of the **Checklist for Artificial Intelligence in Medical Imaging (CLAIM) 2024 Update** where applicable.

In particular:

- use “external testing” rather than the ambiguous term “external validation” in the manuscript where appropriate,
- describe image acquisition and dataset provenance,
- describe the reference standard,
- state inclusion/exclusion criteria,
- document preprocessing and model configuration,
- clearly distinguish internal/model-card performance from independent external testing,
- report limitations and uncertainty,
- preserve reproducibility information.

## 20. Interpretation boundaries

The following claims are outside the scope of this study:

- that the model diagnoses breast cancer in patients
- that the model is safe for autonomous clinical use
- that BUS-UCLM performance represents all hospitals or populations
- that softmax confidence is an individual patient's probability of cancer
- that abstention makes the system clinically safe

The study evaluates model behavior on a public external dataset and should be interpreted as a research benchmark.

## 21. Definition of study success

Study success is **not** defined as achieving a particular accuracy, sensitivity, or AUC.

The study is successful if it delivers a transparent, reproducible, statistically appropriate external evaluation of the pretrained model, regardless of whether performance is strong, moderate, or poor.

## 22. Core scope vs optional scope

### Core — required

- model audit
- BUS-UCLM dataset audit
- frozen eligible-image manifest
- frozen external inference predictions
- three-class evaluation
- malignant vs non-malignant evaluation
- patient-clustered confidence intervals
- calibration analysis
- uncertainty analysis
- abstention threshold sweep
- risk-coverage analysis
- malignant false-negative analysis
- reproducible repository
- manuscript

### Optional — must not delay core study

- attribution / attention maps
- lesion-mask overlap analysis
- additional external datasets
- additional models
- advanced web interface
- deployment optimization

## 23. Protocol amendment policy

This protocol is frozen before complete BUS-UCLM external inference.

If a methodological change becomes necessary after freeze:

1. the original protocol text will remain traceable through Git history,
2. the change will be documented with a date and reason,
3. the amendment will state whether it was made before or after viewing relevant results,
4. baseline analyses defined in Protocol v1.0 will be retained whenever technically possible,
5. post-hoc analyses will be labeled as post-hoc rather than presented as prespecified.

This policy is intended to reduce outcome-driven analytical changes and preserve transparency.

## 24. Protocol freeze statement

**Protocol v1.0 is intended to be committed to the public project repository before complete external inference on BUS-UCLM. The predefined research questions, primary outcome, core inclusion rules, patient-cluster bootstrap, confidence threshold sweep, and core-vs-optional scope should not be silently changed after external results are observed.**

---

## Protocol references

1. Tejani AS, et al. _Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update._ Radiology: Artificial Intelligence. 2024;6(4):e240300. doi:10.1148/ryai.240300.
2. Vállez N, et al. _BUS-UCLM: Breast ultrasound lesion segmentation dataset._ Scientific Data. 2025;12:242. doi:10.1038/s41597-025-04562-3.
3. `Parveshiiii/breast-cancer-detector`, Hugging Face model card. Access details and exact model revision will be recorded at inference freeze.
