# Does a Public Breast Ultrasound AI Model Generalize? External Testing and Uncertainty-Aware Abstention of a Pretrained Vision Transformer

**Manuscript type:** Original Research / External Testing Study
**Authors:** [TO COMPLETE]
**Affiliations:** [TO COMPLETE]
**Corresponding author:** [TO COMPLETE]
## Abstract

### Objective
To evaluate the external performance and reliability of a publicly released pretrained breast-ultrasound Vision Transformer on BUS-UCLM without model adaptation, and to assess whether confidence-based abstention improves reliability.

### Methods
The frozen three-class classifier was evaluated on 683 BUS-UCLM images from 38 patients. The primary outcome was malignant sensitivity. Secondary analyses included malignant-versus-non-malignant discrimination, patient-cluster bootstrap confidence intervals, calibration, confidence and margin analysis, selective classification, risk-coverage analysis, and malignant false-negative auditing.

### Results
Malignant sensitivity was 0.4333 (95% CI 0.2417-0.6136), despite a malignant-score ROC-AUC of 0.7855 (95% CI 0.7060-0.8645). Of 51 malignant false negatives, 48 (94.1%) were classified as benign and 27 (52.9%) had maximum-softmax confidence of at least 0.80. Expected Calibration Error was 0.3386. Raising the acceptance threshold reduced accepted-case error but sharply reduced coverage; at a threshold of 0.90, coverage was 0.3397 and 16 malignant false negatives remained accepted.

### Conclusion
The frozen classifier showed limited external performance on BUS-UCLM, with low malignant sensitivity, substantial overconfidence, and high-confidence malignant-to-benign errors. Confidence-based abstention did not provide a reliable safety boundary because meaningful risk reduction required substantial loss of coverage and high-confidence malignant misses persisted.

**Keywords:** breast ultrasound; artificial intelligence; external testing; Vision Transformer; calibration; uncertainty; selective classification
# 1. Introduction and Related Work

## 1.1 Introduction

Breast ultrasound is widely used as an adjunct imaging modality for the evaluation of breast abnormalities, particularly in settings where sonography can provide complementary structural information beyond other imaging techniques. However, ultrasound interpretation is affected by operator dependence, acquisition variability, lesion heterogeneity, and differences in imaging equipment and clinical populations. These characteristics have made breast ultrasound an active area for computer-aided diagnosis and deep-learning research, with numerous studies reporting strong performance for lesion detection and classification tasks [@han2017breastus; @ma2026breastusreview].

High performance during model development does not, however, establish that an imaging model will retain the same behavior when applied to data acquired at a different institution, with a different scanner, patient population, acquisition protocol, or annotation process. External validation is therefore a critical step in assessing generalizability. A systematic review of deep-learning algorithms for radiologic diagnosis found that external validation remained comparatively uncommon and that most externally evaluated algorithms showed some reduction in performance outside their development data [@yu2022external]. This problem is particularly relevant for medical imaging systems trained on public datasets, where differences in acquisition conditions and data composition can produce substantial dataset shift.

Breast-ultrasound artificial intelligence is not exempt from this problem. Recent work has emphasized that dataset shift and patient-level data leakage can materially affect apparent model performance in breast-ultrasound benchmarks [@wang2026datasetshift]. At the same time, contemporary reviews of ultrasound-based artificial intelligence continue to identify robust external validation and translational evaluation as important gaps between promising research results and dependable clinical use [@ma2026breastusreview]. These concerns motivate evaluation strategies that go beyond reporting a single accuracy or area-under-the-curve value.

Reliability is especially important when a classifier is intended to distinguish malignant from non-malignant findings. A model may retain useful ranking ability while still producing an unsuitable hard-class decision at its default operating rule. In addition, modern neural networks may assign high softmax scores to incorrect predictions, meaning that apparent confidence should not automatically be interpreted as a calibrated probability of correctness [@guo2017calibration]. Predictive uncertainty can also become less reliable under dataset shift [@ovadia2019uncertainty]. Consequently, external evaluation should consider not only discrimination and classification performance, but also calibration, confidence behavior, and the characteristics of clinically important errors.

Selective classification provides one framework for examining whether model confidence can be used to defer uncertain predictions rather than forcing a decision for every input [@geifman2017selective]. In this setting, increasingly strict acceptance thresholds may reduce error among retained predictions, but only at the cost of reduced coverage. For medical-imaging research, this trade-off is important because an apparently improved metric among accepted cases may conceal a large proportion of abstained cases. Confidence-based abstention should therefore be evaluated as a risk-coverage trade-off rather than presented as a complete safety mechanism.

The present study independently evaluates a publicly released pretrained Vision Transformer breast-ultrasound classifier on BUS-UCLM, an independently acquired public dataset from the Hospital General Universitario de Ciudad Real in Spain [@vallez2025busuclm]. BUS-UCLM contains 683 images from 38 patients, including normal, benign, and malignant findings, and was acquired using a Siemens ACUSON S2000 system between 2022 and 2023. The dataset includes expert annotations, with malignant lesions reported as biopsy-confirmed [@vallez2025busuclm]. Importantly, the evaluated classifier was used without retraining, fine-tuning, recalibration, or adaptation to BUS-UCLM.

The study was designed as an external-testing investigation rather than a model-development exercise. Its primary endpoint was malignant sensitivity under the model's frozen three-class decision rule. Secondary analyses examined three-class and malignant-versus-non-malignant performance, patient-clustered confidence intervals, calibration, maximum-softmax confidence, prediction margin, confidence-based abstention, risk-coverage behavior, and failure patterns. Particular attention was given to malignant false negatives and to whether some of these errors remained highly confident. Reporting and reproducibility decisions were informed by the CLAIM 2024 recommendations for artificial-intelligence studies in medical imaging [@tejani2024claim].

The central question was therefore not whether a new breast-ultrasound model could be trained to perform well on BUS-UCLM, but whether an already released pretrained classifier would preserve useful and reliable behavior when transferred unchanged to an independent dataset. This distinction is important because external deployment exposes models to distributional conditions that are not represented by internal validation alone.

## 1.2 Related Work

Deep learning has been investigated for breast-ultrasound classification for several years. Earlier work demonstrated that convolutional neural networks could support differentiation of breast lesions in ultrasound images, including studies using large biopsy-confirmed cohorts [@han2017breastus]. Public datasets subsequently became important resources for reproducible breast-ultrasound research. One widely used example is the Breast Ultrasound Images (BUSI) dataset introduced by Al-Dhabyani and colleagues, which contains benign, malignant, and normal ultrasound images with associated lesion masks [@aldhabyani2020busi]. Such datasets have facilitated rapid development and comparison of classification and segmentation approaches, but strong results within a familiar public-data ecosystem do not necessarily establish performance under independent acquisition conditions.

Vision Transformers extended transformer architectures to image classification by representing an image as a sequence of patches processed through transformer layers [@dosovitskiy2021vit]. Their adoption in medical imaging has provided an alternative to conventional convolutional architectures, but the general problem of dataset shift remains architecture-independent: a high-capacity model can learn representations that perform strongly on its development distribution while degrading when image characteristics or population composition change.

BUS-UCLM provides a useful setting for external evaluation because it was independently collected and includes patient identifiers that permit patient-aware statistical analysis. The dataset contains 683 images from 38 patients, with 419 normal, 174 benign, and 90 malignant images [@vallez2025busuclm]. In addition to class labels, it contains metadata describing characteristics such as Doppler content and combined images, as well as separate segmentation masks. The associated dataset publication emphasizes its use for breast-ultrasound lesion analysis and provides expert-derived reference annotations [@vallez2025busuclm].

A closely related recent benchmark examined breast-ultrasound artificial intelligence under dataset shift while explicitly considering patient leakage [@wang2026datasetshift]. That work reinforces two issues relevant to the present study: first, that breast-ultrasound performance can depend strongly on how datasets and patient partitions are constructed; and second, that evaluation under distribution shift should be treated as a distinct problem from model development. The present study differs in emphasis by testing a fixed third-party public classifier without retraining and by coupling external classification performance with calibration, uncertainty proxies, selective classification, risk-coverage analysis, and a detailed audit of malignant classification failures.

Calibration and selective prediction form another important strand of related work. Guo and colleagues showed that modern neural networks can be poorly calibrated even when classification accuracy is high [@guo2017calibration]. Ovadia and colleagues further demonstrated that predictive uncertainty can deteriorate under dataset shift [@ovadia2019uncertainty]. Selective-classification methods seek to exploit uncertainty estimates by allowing a model to abstain on lower-confidence cases [@geifman2017selective]. In medical imaging, however, the practical value of abstention depends on both the errors removed and the proportion of cases deferred. For this reason, the present study evaluates confidence thresholds together with retained coverage, accepted-case risk, malignant-case coverage, and the number of malignant false negatives that remain accepted.

Taken together, previous work supports the need for external, patient-aware, reliability-focused evaluation of breast-ultrasound AI. The present investigation contributes a fully reproducible test of a frozen public Vision Transformer on an independent public cohort, with particular emphasis on malignant sensitivity, calibration, high-confidence failure, and the trade-off between abstention and coverage.
# 2. Materials and Methods

## 2.1 Study design and prespecified protocol

This study was designed as an independent external testing study of a publicly available breast-ultrasound image classifier. The objective was to evaluate whether a pretrained Vision Transformer (ViT), released through Hugging Face, generalized to an independently acquired breast-ultrasound dataset without retraining, fine-tuning, calibration fitting, or other model adaptation on the external data. A secondary objective was to characterize model uncertainty, calibration, selective-classification behavior, and clinically relevant error patterns, with particular emphasis on malignant false-negative predictions.

The analysis plan was prespecified in Research Protocol v1.0 and frozen before complete external inference on BUS-UCLM. The protocol defined the primary outcome, dataset inclusion rules, external-testing sequence, patient-cluster bootstrap procedure, uncertainty measures, abstention thresholds, calibration analysis, failure-analysis priorities, and the distinction between core and optional analyses. Methodological changes after protocol freeze were required to remain traceable and to be identified as post hoc where applicable. The study was conducted as a research benchmark and was not intended to evaluate the model as a standalone clinical diagnostic device. Reporting was guided by principles from the Checklist for Artificial Intelligence in Medical Imaging (CLAIM) 2024 Update. The overall external-testing workflow is summarized in Figure 1.

No new patient recruitment or prospective data collection was performed. The study used an openly released, previously collected breast-ultrasound dataset and a publicly released pretrained model.

## 2.2 Pretrained model

The model under evaluation was `Parveshiiii/breast-cancer-detector`, obtained from Hugging Face and pinned to revision `7c4c1ac11f5f80d382cc641ec6f2d3989e7fa73c` before complete external inference. The released model is a `ViTForImageClassification` classifier fine-tuned from `google/vit-base-patch16-224-in21k`. It performs three-class breast-ultrasound image classification with the label mapping defined in the released model configuration: class 0, benign; class 1, malignant; and class 2, normal.

The architecture uses 224 × 224 pixel inputs with 16 × 16 image patches, a hidden dimension of 768, 12 transformer encoder layers, and 12 attention heads. Model weights were kept frozen throughout all external analyses. No BUS-UCLM image, class label, patient identifier, or derived result was used to update model parameters, select alternative weights, or fine-tune the classifier.

The original model documentation reports training on a breast-ultrasound image dataset derived from the public BUSI/Al-Dhabyani dataset family. Developer-reported performance was treated only as contextual background because the original data, evaluation design, and inclusion rules were not identical to the present external test.

## 2.3 External test dataset

External testing was performed on BUS-UCLM, a publicly released breast-ultrasound dataset collected at Ciudad Real General University Hospital / University of Castilla-La Mancha, Spain. The dataset contains 683 ultrasound images from 38 patients, comprising 419 normal images, 174 benign images, and 90 malignant images. Images were acquired using a Siemens ACUSON S2000 ultrasound system during 2022–2023. The dataset includes expert annotations and separate lesion-segmentation masks; malignant lesions were reported by the dataset authors as biopsy-confirmed.

The image was the prediction unit because the model produces one classification per image. However, because multiple images may originate from the same patient, observations were not assumed to be statistically independent for uncertainty estimation. Patient identifiers supplied by the dataset were therefore retained in the analysis manifest and used for patient-clustered bootstrap confidence intervals.

The official BUS-UCLM class labels were used as the reference standard without modification based on model output. For the binary malignancy analysis, malignant images were coded as positive and normal plus benign images were combined into a non-malignant negative class. Key model and external-cohort characteristics are summarized in Table 1.

## 2.4 Dataset audit, eligibility, and sensitivity subset

Before full inference, the distributed BUS-UCLM files were audited for file count, patient count, class distribution, image dimensions, readability, label availability, exact duplicate hashes, mask availability, and available metadata flags. The audit also recorded whether images contained visual marks or annotations, Doppler information, or combined/multi-panel content. A frozen image manifest preserved stable image identifiers, patient identifiers, paths, reference labels, dimensions, and metadata flags.

The primary analysis included every official source image that could be decoded successfully and mapped to one of the three prespecified classes: normal, benign, or malignant. Outcome-based exclusions were prohibited. Images were not removed because they were difficult, uncertain, incorrectly classified, or reduced a performance metric. Segmentation masks were not used as classifier inputs.

A prespecified secondary sensitivity analysis evaluated a clean-input subset intended to more closely match the released model's stated input conditions. This subset excluded images flagged by BUS-UCLM metadata as containing visual marks/annotations, Doppler information, or combined/multi-panel content. The full eligible BUS-UCLM dataset remained the primary analysis population.

## 2.5 Image preprocessing and frozen inference

All images were processed using the image processor distributed with the pinned model revision. Source images were converted to RGB before preprocessing. The processor (`ViTImageProcessor`) resized images to 224 × 224 pixels using resampling mode 2 (bilinear interpolation), applied a rescaling factor of 0.00392156862745098 (1/255), and normalized image channels using mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)`. The same preprocessing pipeline was applied to every included image; no image-specific or label-informed preprocessing was performed.

Inference was implemented in PyTorch using the Hugging Face Transformers model and processor. The classifier was placed in evaluation mode, gradients were disabled, and inference was performed deterministically on the frozen model. For each image, the model produced three logits, which were transformed using the softmax function to obtain class scores. The class with the largest softmax score was recorded as the predicted class.

For reproducibility, the batch inference table stored the image identifier and path, patient identifier, reference class, predicted class, all three class scores, raw logits, maximum softmax confidence, top-two probability margin, model identifier, pinned model revision, execution device, and prespecified metadata flags. The completed prediction file was checked for row count, unique images, missing outputs, class-score sums, and model/revision consistency, and was then frozen before performance analyses. Its SHA-256 checksum was recorded so that all subsequent analyses used the same inference output.

## 2.6 Performance outcomes

The primary study outcome was image-level malignant sensitivity on the full external BUS-UCLM dataset, accompanied by a patient-clustered 95% confidence interval. Malignant sensitivity was prioritized because missed malignant cases represent the most consequential classification error considered in this study.

For three-class external performance, the following metrics were calculated: overall accuracy, balanced accuracy, macro precision, macro recall, macro F1 score, class-specific precision, recall and F1 score, and a three-class confusion matrix.

For binary malignant-versus-non-malignant evaluation, normal and benign images were combined as non-malignant. Binary outcomes included sensitivity, specificity, positive predictive value, negative predictive value, F1 score, balanced accuracy, false-negative and false-positive counts, receiver operating characteristic area under the curve (ROC-AUC), and precision-recall area under the curve (PR-AUC). ROC-AUC and PR-AUC were calculated using the model's malignant softmax score rather than the hard predicted label.

## 2.7 Patient-cluster bootstrap confidence intervals

Because repeated images from the same patient violate the independence assumption of a naive image-level bootstrap, uncertainty intervals were estimated using a patient-cluster bootstrap. The patient was the resampling unit, and all images belonging to each sampled patient were retained together within a bootstrap replicate.

A total of 5,000 bootstrap replicates were generated using random seed 42. For each replicate, the relevant image-level performance metrics were recomputed, and percentile 95% confidence intervals were obtained from the bootstrap distribution. When a metric was mathematically undefined in a replicate because the required class was absent, that replicate was treated as invalid for that metric rather than being assigned an artificial value.

## 2.8 Confidence and calibration analysis

Maximum softmax probability was used as the primary model-confidence measure. The prediction margin, defined as the difference between the largest and second-largest class probabilities, was used as a secondary uncertainty proxy. Confidence and margin distributions were summarized for correct and incorrect predictions and separately for malignant true-positive and malignant false-negative predictions.

Calibration was evaluated with respect to top-class correctness rather than interpreting the malignant score as an individualized clinical probability of cancer. A 10-bin equal-width reliability analysis over the interval [0, 1] compared mean maximum-softmax confidence with empirical accuracy within each confidence bin. Expected Calibration Error (ECE) was calculated as the sample-size-weighted mean absolute difference between bin accuracy and mean confidence.

A multiclass Brier score was also calculated as the mean, across images, of the summed squared difference between the three predicted class probabilities and the one-hot encoded reference class. No post-hoc recalibration method was applied to the frozen baseline model.

## 2.9 Selective classification and abstention analysis

Selective classification was evaluated without modifying model weights. At confidence threshold t, an image was accepted if its maximum softmax probability was greater than or equal to t and was abstained otherwise. The following prespecified thresholds were evaluated: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, and 0.95.

At each threshold, overall coverage, abstention rate, accepted-case accuracy, accepted-case malignant sensitivity, accepted-case specificity, accepted malignant false negatives, accepted false positives, malignant-case coverage, and malignant-case abstention were calculated. Abstained observations were reported separately and were not counted as correct predictions.

A continuous risk-coverage analysis was additionally generated by sorting observations in descending order of maximum softmax confidence and progressively expanding the accepted set. Coverage was defined as the proportion of all external images receiving an accepted prediction, and selective risk was defined as the classification error rate among accepted images. Accuracy and risk were summarized at representative coverage levels, and the area under the empirical risk-coverage curve (AURC) was calculated as a descriptive summary. No single confidence threshold was retrospectively presented as a clinically validated operating point.

## 2.10 Failure analysis

Failure analysis was performed using the frozen prediction table. For malignant-versus-non-malignant classification, each image was categorized as a true positive, true negative, false positive, or false negative. Primary attention was given to malignant false negatives.

For false-negative cases, the predicted class, malignant probability, maximum-softmax confidence, prediction margin, and metadata flags were summarized. Malignant false negatives were further separated according to whether the model predicted benign or normal. False-positive cases were summarized according to whether their reference class was normal or benign.

High-confidence errors were quantified descriptively at maximum-softmax confidence thresholds of 0.80, 0.90, and 0.95. Error concentration by patient was also summarized using patient-level counts and the proportion of eligible images for that patient that were misclassified. Available BUS-UCLM metadata, including visual marks, Doppler content, combined/multi-panel status, and membership in the clean-input subset, were compared descriptively across error groups. These analyses were not used to infer causal mechanisms, and repeated images from individual patients were not treated as independent evidence of an underlying failure mechanism.

## 2.11 Reproducibility and software

All analyses were implemented in Python 3.11.16. Core package versions in the frozen environment included PyTorch 2.13.0, Hugging Face Transformers 5.16.1, scikit-learn 1.9.0, NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1, Pillow 12.3.0, and Matplotlib 3.11.1. The software environment was frozen in the project repository using a locked dependency file. The exact Hugging Face model identifier and revision, preprocessing configuration, dataset manifest, raw model scores, prediction labels, random seed, generated result tables, figures, and prediction checksum were retained as reproducibility artifacts.

The public repository was structured so that model loading, inference, evaluation, bootstrap analysis, calibration analysis, selective-classification analysis, failure analysis, and figure/table generation were separated into reproducible scripts and modules. Raw BUS-UCLM patient images were not required to be committed to the repository; instead, the repository documents dataset acquisition and preparation and stores analysis outputs necessary to reproduce the reported results once the dataset is obtained from its official source.

Optional ViT attribution or lesion-mask overlap analysis was prespecified as exploratory and was not required for completion of the core external-testing study.
# 3. Results

## 3.1 External test cohort

The complete eligible BUS-UCLM cohort comprised 683 breast-ultrasound images from 38 patients. The reference-standard class distribution was 419 normal images, 174 benign images, and 90 malignant images (Supplementary Figure S1). All 683 images were retained for the primary external analysis. A prespecified clean-input sensitivity subset, excluding images flagged as containing visual marks/annotations, Doppler information, or combined/multi-panel content, contained 521 images.

The frozen inference output contained one prediction per image with no missing predictions, duplicate image rows, or class-probability inconsistencies. All analyses reported below were performed from the same frozen prediction table.

## 3.2 Three-class external performance

Across the full 683-image external test set, overall three-class accuracy was 0.4583 and balanced accuracy was 0.5338. Macro-averaged precision, recall, and F1 score were 0.5743, 0.5338, and 0.4544, respectively.

Class-specific performance differed substantially. For benign images, precision was 0.3262, recall was 0.8793, and F1 score was 0.4759. For malignant images, precision was 0.4588, recall was 0.4333, and F1 score was 0.4457. For normal images, precision was 0.9380, recall was 0.2888, and F1 score was 0.4416.

The three-class confusion matrix, ordered as benign, malignant, and normal, was: reference benign, 153 benign, 16 malignant, and 5 normal; reference malignant, 48 benign, 39 malignant, and 3 normal; reference normal, 268 benign, 30 malignant, and 121 normal. Thus, the dominant error for malignant images was assignment to the benign class, while many normal images were also assigned to benign (Figure 2).

## 3.3 Malignant versus non-malignant discrimination

For the prespecified binary analysis, benign and normal images were combined into a non-malignant class. The resulting confusion counts were 39 true positives, 547 true negatives, 46 false positives, and 51 false negatives.

Malignant sensitivity, the primary endpoint, was 0.4333. Specificity was 0.9224, positive predictive value was 0.4588, negative predictive value was 0.9147, F1 score was 0.4457, and balanced accuracy was 0.6779. Using the malignant softmax score as the continuous decision variable, ROC-AUC was 0.7855 and PR-AUC was 0.4619 (Figures 3 and 4).

These results show that ranking performance was stronger than the sensitivity achieved by the model's default hard three-class decision rule. However, more than half of the malignant images were not assigned to the malignant class.

## 3.4 Patient-clustered confidence intervals

Patient-cluster bootstrap confidence intervals were calculated using 5,000 replicates with the patient as the resampling unit.

For the full external cohort, the 95% confidence interval for overall three-class accuracy was 0.3756 to 0.5347. The corresponding intervals were 0.4564 to 0.6073 for three-class balanced accuracy and 0.3608 to 0.5346 for macro F1 score.

For malignant detection, sensitivity was 0.4333 with a 95% confidence interval of 0.2417 to 0.6136. Specificity was 0.9224 (95% CI 0.8788–0.9584), positive predictive value was 0.4588 (95% CI 0.2222–0.6863), negative predictive value was 0.9147 (95% CI 0.8574–0.9589), and F1 score was 0.4457 (95% CI 0.2446–0.6114). Binary balanced accuracy was 0.6779 (95% CI 0.5776–0.7740), ROC-AUC was 0.7855 (95% CI 0.7060–0.8645), and PR-AUC was 0.4619 (95% CI 0.2500–0.6487). Primary and secondary external performance estimates with patient-clustered 95% confidence intervals are summarized in Table 2.

## 3.5 Clean-input sensitivity analysis

The clean-input subset contained 521 images. Three-class accuracy was 0.3973, balanced accuracy was 0.5258, and macro F1 score was 0.3734.

Within this subset, the binary malignant-versus-non-malignant confusion counts were 14 true positives, 449 true negatives, 36 false positives, and 22 false negatives. Malignant sensitivity was 0.3889 and specificity was 0.9258. ROC-AUC was 0.8018 and PR-AUC was 0.2828.

Because the clean-input subset had a different case composition and contained only 36 malignant images, these values were treated as a sensitivity analysis rather than evidence that removal of flagged image types improved or worsened model performance.

## 3.6 Confidence and prediction-margin behavior

Across all 683 predictions, mean maximum-softmax confidence was 0.7961 and median confidence was 0.8313. Mean prediction margin was 0.6340 and median margin was 0.7009.

Correct predictions had higher confidence on average than incorrect predictions. Among 313 correct predictions, mean confidence was 0.8250 and median confidence was 0.8808, compared with 0.7717 and 0.7998 among 370 incorrect predictions. Mean prediction margin was 0.6802 for correct predictions and 0.5949 for incorrect predictions.

The difference was more pronounced within malignant cases. For the 39 malignant true positives, mean confidence was 0.8981 and mean prediction margin was 0.8127. For the 51 malignant false negatives, mean confidence remained relatively high at 0.8038, with a mean prediction margin of 0.6473. Lower confidence was therefore associated with error on average, but some malignant errors were still made with high confidence.

## 3.7 Confidence-based abstention

The prespecified threshold sweep demonstrated a trade-off between retained coverage and performance among accepted predictions. The complete prespecified threshold sweep is reported in Table 3.

At a confidence threshold of 0.50, coverage was 0.9649 and accepted-case accuracy was 0.4613. Malignant sensitivity among accepted cases was 0.4333, with 51 accepted malignant false negatives.

At a threshold of 0.70, coverage fell to 0.7189, accepted-case accuracy increased to 0.4949, and malignant sensitivity among accepted cases increased to 0.4935. The number of accepted malignant false negatives decreased to 39.

At a threshold of 0.80, coverage was 0.5637, accepted-case accuracy was 0.5195, and malignant sensitivity among accepted cases was 0.5500, with 27 accepted malignant false negatives.

At a threshold of 0.90, coverage decreased to 0.3397. Accepted-case accuracy was 0.6121, malignant sensitivity was 0.6279, and 16 malignant false negatives remained among accepted predictions.

At the most restrictive threshold of 0.95, only 0.1742 of all images were accepted. Accepted-case accuracy increased to 0.8403 and malignant sensitivity among accepted malignant cases was 0.7059. Five malignant false negatives remained accepted, while 73 of the 90 malignant images were abstained.

Increasingly strict confidence thresholds reduced the number of accepted malignant false negatives and lowered selective error, but this occurred at the cost of substantial loss of overall and malignant-case coverage.

## 3.8 Risk-coverage analysis

The continuous risk-coverage analysis showed that predictions ranked by maximum-softmax confidence contained useful ordering information.

At approximately 25% coverage, 171 images were accepted and accuracy was 0.7135, corresponding to a selective risk of 0.2865. At approximately 50% coverage, 342 images were accepted and accuracy was 0.5380, with risk 0.4620. At approximately 75% coverage, 512 images were accepted and accuracy was 0.4883, with risk 0.5117. At full coverage, accuracy was 0.4583 and risk was 0.5417.

The descriptive area under the empirical risk-coverage curve was 0.3831. The curve therefore showed that confidence ranking could identify a lower-risk subset, but meaningful reductions in accepted-case risk required substantial reductions in coverage (Figure 6).

## 3.9 Calibration

Maximum-softmax confidence was substantially higher than observed top-class correctness on the external dataset. Overall accuracy was 0.4583, while mean confidence was 0.7961, a difference of 0.3379.

Using 10 equal-width confidence bins, Expected Calibration Error was 0.3386. The multiclass Brier score was 0.8080.

The largest reliability gaps occurred in the high-confidence bins. For predictions with confidence between 0.80 and 0.90, mean confidence was 0.8531 while empirical accuracy was 0.3791. For predictions with confidence between 0.90 and 1.00, mean confidence was 0.9479 while empirical accuracy was 0.6121.

These results indicate substantial overconfidence under external testing. Confidence could still rank predictions by relative reliability while remaining poorly calibrated in absolute terms (Figure 5).

## 3.10 Failure analysis

The malignant error audit reproduced the binary confusion counts of 39 true positives, 547 true negatives, 46 false positives, and 51 false negatives. The principal malignant error patterns and high-confidence failures are summarized in Table 4.

Among the 51 malignant false negatives, 48 (94.1%) were predicted as benign and 3 (5.9%) were predicted as normal. The principal malignant failure mode was therefore malignant-to-benign misclassification.

Among the 46 false-positive malignant predictions, 30 (65.2%) originated from reference-normal images and 16 (34.8%) from reference-benign images.

High-confidence errors were common. Of the 51 malignant false negatives, 27 (52.9%) had maximum-softmax confidence of at least 0.80, 16 (31.4%) had confidence of at least 0.90, and 5 (9.8%) had confidence of at least 0.95. Among the 46 malignant false positives, 21 (45.7%) had confidence of at least 0.80, 12 (26.1%) at least 0.90, and 4 (8.7%) at least 0.95.

Malignant false negatives were distributed across 14 patients. The five patients with the largest false-negative counts accounted for 29 of 51 false negatives (56.9%). False positives were distributed across 16 patients, with the five patients contributing the largest false-positive counts accounting for 29 of 46 false positives (63.0%). These patient-level concentrations were described as clustering rather than as proof of a specific failure mechanism.

Metadata patterns were also examined descriptively. Among malignant false negatives, 56.9% were flagged as containing visual marks, 17.7% contained Doppler information, and 43.1% belonged to the prespecified clean-input subset. Because substantial numbers of errors also occurred among clean-input images, the observed failures could not be attributed solely to overlays or other flagged image characteristics.

Overall, the failure analysis showed that the model's most important error pattern was confident malignant-to-benign misclassification, and that confidence-based abstention reduced but did not eliminate this failure mode.
# 4. Discussion

This study examined how a publicly released breast-ultrasound Vision Transformer behaved when transferred unchanged to an independent external dataset. The principal finding was not simply that performance was lower on BUS-UCLM, but that different aspects of performance diverged under external testing. Malignant sensitivity under the frozen three-class decision rule was 0.4333, whereas the malignant softmax score retained a ROC-AUC of 0.7855. Discrimination and operating-point performance answer different questions: a model can preserve useful ranking information while still producing unsuitable categorical decisions at the decision rule inherited from its development setting. This distinction has also been demonstrated under acquisition shift in medical imaging, where sensitivity and specificity can drift even when ROC-AUC remains comparatively preserved [@roschewitz2023acquisition].

The most important failure pattern was the direction of the malignant errors. Forty-eight of the 51 malignant false negatives were assigned to the benign class rather than to normal. The present study cannot establish the mechanism responsible for this pattern, but several non-exclusive explanations are plausible. First, benign and malignant breast lesions can share overlapping sonographic appearances. Swoboda et al. examined histologically verified malignant tumors that mimicked fibroadenomas on ultrasound and showed that differentiating features may be subtle even in conventional sonographic assessment [@swoboda2025fibroadenoma]. Second, differences between the model-development distribution and BUS-UCLM may have altered the usefulness of features learned during training. Cross-institutional medical-imaging studies have shown that neural networks can exploit acquisition- or site-associated signals and may lose performance when those relationships change externally [@zech2018generalization; @ongly2024shortcut]. Third, the public BUSI dataset lineage associated with the released model is benign-majority [@aldhabyani2020busi]. If the model's effective training distribution retained a similar imbalance, learned class priors could have contributed to a tendency toward benign predictions under shift. However, the exact class composition of the released model's training subset is not documented in sufficient detail to test this explanation. These mechanisms should therefore be regarded as hypotheses rather than demonstrated causes because the present study did not perform causal attribution, controlled class-prior experiments, or representation-level analysis.

The external performance pattern is therefore best interpreted in the broader context of dataset shift. Differences in scanner hardware, acquisition settings, image processing, case mix, and institutional workflow can alter image distributions without changing the nominal diagnostic task. Zech et al. demonstrated substantial cross-hospital variability in a chest-radiograph classifier and showed that networks could identify hospital-associated information [@zech2018generalization]. Ong Ly et al. subsequently reported across 13 healthcare datasets that hidden data-acquisition biases could promote shortcut learning and lead to overestimation of generalization performance [@ongly2024shortcut]. More broadly, systematic evidence indicates that externally validated radiology models frequently show performance degradation outside their development data [@yu2022external]. Our data do not identify a particular shortcut or acquisition factor, but they reinforce the need to distinguish internal performance from behavior on independently acquired data.

The breast-ultrasound literature also illustrates the importance of development-data diversity and multicenter evaluation. Gu et al. developed and evaluated a breast-ultrasound deep-learning system using 14,043 images from 5,012 women collected across 32 hospitals [@gu2022multicenter]. Xiang et al. evaluated a multivendor, multicenter model using 45,909 images acquired with 42 ultrasound-machine types across four hospitals and reported external AUCs of 0.91-0.96 [@xiang2023multivendor]. These studies are not directly comparable with the present work because their models, datasets, training strategies, task definitions, and validation designs differ. Their relevance is instead methodological: breast-ultrasound AI can be developed and tested across broad acquisition conditions, whereas the present study asks a narrower question—what happens when an already released third-party model is transferred without adaptation to an independent public cohort.

Confidence behavior provided a second major finding. Correct predictions were more confident than incorrect predictions on average, and malignant true positives had higher confidence and larger prediction margins than malignant false negatives. However, confidence did not provide a reliable safety boundary. More than half of malignant false negatives had maximum-softmax confidence of at least 0.80, almost one third remained above 0.90, and some exceeded 0.95. This is consistent with prior work showing that neural-network softmax outputs can be poorly calibrated [@guo2017calibration] and that uncertainty quality may deteriorate under dataset shift [@ovadia2019uncertainty]. Ahmed et al. likewise emphasized that softmax values are not calibrated measures of model confidence and examined explicit failure-detection strategies in medical imaging [@ahmed2022failure]. The practical implication is that an externally misclassified case cannot be assumed to reveal itself through a low maximum-softmax score.

The calibration analysis made this limitation explicit. Mean maximum-softmax confidence was 0.7961 despite an overall three-class accuracy of 0.4583, and the Expected Calibration Error was 0.3386. In the 0.80-0.90 confidence bin, mean confidence was 0.8531 while empirical accuracy was only 0.3791. These findings indicate substantial overconfidence with respect to top-class correctness on BUS-UCLM. Importantly, poor absolute calibration does not mean that confidence contains no relative information. The model still ranked some predictions by reliability, which helps explain why selective classification improved accepted-case performance despite poor calibration. Zhang et al. showed in medical-image classification that the effectiveness of uncertainty-based referral is strongly related to calibration and that post-calibration can alter referral performance [@zhang2023calibration]. Our study did not recalibrate the model because doing so would have changed the frozen external-testing question.

The abstention results should therefore be interpreted as a coverage trade-off rather than as a clinical solution. Increasing the confidence threshold reduced accepted-case error and decreased the number of malignant false negatives remaining among accepted predictions. However, those improvements required progressively excluding more cases. At a threshold of 0.90, overall coverage fell to 0.3397. At 0.95, accepted-case accuracy reached 0.8403, but only 17.4% of all images remained accepted and 73 of the 90 malignant images were abstained. The risk-coverage curve showed the same pattern continuously: confidence ranking could identify a lower-risk subset, but substantial risk reduction required substantial loss of coverage. Selective classification is useful for characterizing this trade-off [@geifman2017selective], but these results do not establish a clinically safe operating threshold.

This distinction matters because confidence-based referral can appear attractive as a mechanism for withholding uncertain AI predictions. In the present study, some of the most important errors were not low-confidence predictions. High-confidence malignant false negatives therefore limit the protection that can be obtained from maximum-softmax abstention alone. More sophisticated uncertainty estimation, explicit out-of-distribution detection, or independently validated referral strategies may prove useful, but each would require separate prospective or external evaluation rather than being inferred from the present threshold sweep.

The clean-input sensitivity analysis also argues against attributing the observed degradation to one obvious image characteristic. Excluding images flagged for marks, Doppler content, or combined/multi-panel presentation did not restore malignant sensitivity; sensitivity in that subset was 0.3889. Because the clean subset had a different case composition and contained only 36 malignant images, this finding should not be interpreted as evidence that clean images were intrinsically harder. It instead shows that the observed failure pattern was not confined to visibly flagged inputs, consistent with the possibility that external shift reflects several interacting differences rather than a single artifact.

Patient-level concentration of errors further illustrates why image counts alone can overstate the amount of independent evidence. BUS-UCLM contains 683 images but only 38 patients. Malignant false negatives occurred across 14 patients, and the five patients with the largest false-negative counts accounted for 56.9% of all malignant false negatives. The patient-cluster bootstrap accordingly produced a wide 95% confidence interval for malignant sensitivity (0.2417-0.6136). This uncertainty is part of the result rather than a defect to be concealed: it reflects the limited number of independent patients available in the external cohort and supports patient-aware uncertainty estimation.

## 4.1 Clinical implications

The present findings do not support use of the evaluated frozen model as a standalone clinical decision tool on data represented by this external cohort. The concern is not only the low malignant sensitivity but also the fact that many malignant misses were assigned to the benign class with substantial softmax confidence. A system that remains confident when wrong presents a different reliability problem from one that consistently signals uncertainty before clinically important errors.

This result should not be interpreted as evidence that breast-ultrasound AI is inherently unreliable. Larger multicenter studies have reported stronger diagnostic performance under different development and validation designs [@gu2022multicenter; @xiang2023multivendor]. Clinical breast assessment is also not ordinarily based on a single isolated ultrasound image. In an international multicenter study of 1,288 women, Pfob et al. found that adding clinical and demographic information to ultrasound features improved machine-learning classification performance relative to unimodal ultrasound information [@pfob2022multimodal]. The present model is an image-level classifier and should therefore not be equated with a complete clinical diagnostic pathway.

For clinical engineering and deployment review, the practical lesson is that public availability and strong source-domain performance are not sufficient evidence of local reliability. A minimum predeployment evaluation framework should include independent local data representative of the intended use setting; patient-level analysis with uncertainty intervals; prespecified safety-relevant endpoints such as sensitivity for clinically important classes; calibration and high-confidence failure analysis; checks across relevant devices, sites, and subgroups; and explicit validation of any uncertainty, referral, or out-of-distribution mechanism that is intended to act as a safeguard. After deployment, performance-drift monitoring and periodic reassessment should be planned rather than assumed unnecessary. These principles are consistent with lifecycle-oriented guidance for trustworthy healthcare AI, including the FUTURE-AI framework [@lekadir2025futureai]. The present study does not prescribe a universal validation sample size because the required cohort size depends on the intended use, prevalence, target metric, and desired statistical precision.

## 4.2 Limitations

BUS-UCLM provides a genuine independent external test, which is a strength of the study design. Its 683 images, however, originate from only 38 patients. The number of independent patient-level observations is therefore limited, and repeated images from the same patient should not be interpreted as equivalent to an external cohort of 683 independent participants. Patient-cluster bootstrap confidence intervals were used to account for within-patient dependence, but the relatively small number of patients limits the precision of performance estimates, particularly for malignant cases and patient-level subgroup patterns.

Likewise, one independent cohort was sufficient to reveal a concrete transfer failure on BUS-UCLM, but it does not establish how the model would behave across all hospitals, scanners, populations, or acquisition protocols. The findings therefore define a demonstrated limitation in this external setting rather than a universal estimate of breast-ultrasound generalizability. Multi-institution external evaluation would be required to characterize the breadth and consistency of the observed failure pattern.

The prediction unit was the individual ultrasound image. Some patients contributed multiple images, and the model itself produces image-level rather than patient-level outputs. The study therefore does not establish patient-level diagnostic performance or clinical decision accuracy.

The evaluated model was treated as a fixed public artifact. The study did not optimize thresholds, retrain the model, perform domain adaptation, or fit a recalibration method on BUS-UCLM. This was intentional because the objective was external testing, but it means the analysis does not determine whether performance could be improved through adaptation or recalibration.

Maximum-softmax confidence and top-two prediction margin are simple uncertainty proxies rather than comprehensive uncertainty estimates. No independently validated out-of-distribution detector was evaluated. The study therefore does not establish that confidence thresholds can identify inappropriate inputs or all clinically important errors.

Image metadata such as marks, Doppler content, and combined images were examined descriptively. Associations between these characteristics and error patterns were not tested as causal mechanisms, and the study was not powered for definitive subgroup comparisons.

Finally, lesion masks were not used as classifier inputs and attribution analysis was not part of the core evaluation. Consequently, the study identifies failure patterns at the prediction and metadata levels but cannot determine which image regions, lesion characteristics, training-set properties, or model representations caused individual errors. The proposed explanations for malignant-to-benign confusion and external degradation should therefore be treated as hypotheses for future testing.

## 4.3 Future directions

The next step should be evaluation across larger, multi-institution and multivendor external cohorts with sufficient numbers of independent patients to estimate clinically important error rates more precisely. Such studies should preserve patient-level separation and, where possible, report performance across acquisition sites and devices.

A second direction is to test whether the observed ranking signal can be translated into more reliable operating behavior through prespecified recalibration or threshold-selection procedures using data that are separate from the final external test cohort. Domain-adaptation methods may also be investigated, but these would answer a different question from the frozen-model external test performed here.

Future studies should compare maximum-softmax confidence with stronger uncertainty and failure-detection approaches, including explicit out-of-distribution methods. These approaches should be evaluated on independent in-domain and out-of-domain data rather than assumed to be reliable from their formulation alone. Where appropriate datasets are available, radiologist-comparison or reader-assistance studies would help determine whether a model provides useful complementary information rather than merely stand-alone image classification.

Architecture-specific hypotheses also warrant direct testing. The evaluated model is a ViT-B/16 architecture that represents a 224 × 224 image as 16 × 16 patches. Standard Vision Transformers impose less image-specific local inductive bias than conventional convolutional architectures [@dosovitskiy2021vit]. It is therefore reasonable to test whether architecture, patch size, or image resolution affects preservation of subtle local sonographic features under external shift. The present study cannot determine that ViT patch tokenization caused the observed errors; a controlled comparison among CNNs, ViTs with different patch sizes, and otherwise matched preprocessing would be required.

Finally, mechanistic analysis could investigate why malignant lesions were predominantly assigned to the benign class. Lesion-level attribution, representation analysis, controlled class-balance experiments, and carefully designed comparisons between development and external datasets could test hypotheses related to sonographic feature overlap, acquisition shift, learned class priors, or dataset-specific cues. Such mechanisms should be tested directly rather than inferred from error counts alone.

## 4.4 Conclusion

Independent external testing of a frozen public breast-ultrasound Vision Transformer revealed important limitations that were not captured by discrimination metrics alone. On BUS-UCLM, malignant sensitivity was low, most malignant false negatives were assigned to the benign class, and a substantial proportion of these errors occurred with high maximum-softmax confidence. The model nevertheless retained useful ranking information, as reflected by malignant ROC-AUC and risk-coverage behavior.

Confidence-based abstention reduced error among accepted predictions, but the improvement required substantial loss of coverage and did not eliminate high-confidence malignant misses. Maximum-softmax confidence was also poorly calibrated with respect to top-class correctness on the external dataset.

These findings support external, patient-aware evaluation that examines calibration, uncertainty, selective prediction, and failure patterns in addition to conventional discrimination metrics. For released medical-imaging AI systems, useful ranking performance should not be assumed to imply reliable categorical decisions or calibrated confidence under dataset shift. The present study should be interpreted as a reproducible external test of one public model on one independent cohort, rather than as clinical validation or a universal assessment of breast-ultrasound AI.
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

# Declarations

## Ethics and consent
No new patient recruitment or prospective data collection was performed. This study used the publicly released BUS-UCLM dataset and a publicly released pretrained model. The final journal-specific ethics and consent wording should be aligned with the BUS-UCLM source publication and the target journal's requirements before submission.

## Data availability
BUS-UCLM is publicly available from its official repository. The manuscript should cite the canonical dataset publication and repository/DOI in the final journal format.

## Code availability
Analysis code, reproducibility materials, and project documentation are maintained in the project repository: `https://github.com/Abdooorl/breast-ultrasound-ai-validation`. A tagged release and archived version should be created before final submission.

## Funding
[TO COMPLETE: state funding source, or "The authors received no specific funding for this work" if accurate.]

## Competing interests
[TO COMPLETE.]

## Author contributions
[TO COMPLETE using the target journal's required taxonomy, such as CRediT.]

## Acknowledgements
[TO COMPLETE if applicable.]

# References

References are managed in `paper/references.bib` using citation keys in the manuscript. The bibliography should be rendered in the target journal's required style during submission formatting.
