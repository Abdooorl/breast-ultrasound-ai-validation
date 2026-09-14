# 2. Materials and Methods

## 2.1 Study design and prespecified protocol

This study was designed as an independent external testing study of a publicly available breast-ultrasound image classifier. The objective was to evaluate whether a pretrained Vision Transformer (ViT), released through Hugging Face, generalized to an independently acquired breast-ultrasound dataset without retraining, fine-tuning, calibration fitting, or other model adaptation on the external data. A secondary objective was to characterize model uncertainty, calibration, selective-classification behavior, and clinically relevant error patterns, with particular emphasis on malignant false-negative predictions.

The analysis plan was prespecified in Research Protocol v1.0 and frozen before complete external inference on BUS-UCLM. The protocol defined the primary outcome, dataset inclusion rules, external-testing sequence, patient-cluster bootstrap procedure, uncertainty measures, abstention thresholds, calibration analysis, failure-analysis priorities, and the distinction between core and optional analyses. Methodological changes after protocol freeze were required to remain traceable and to be identified as post hoc where applicable. The study was conducted as a research benchmark and was not intended to evaluate the model as a standalone clinical diagnostic device. Reporting was guided by principles from the Checklist for Artificial Intelligence in Medical Imaging (CLAIM) 2024 Update.

No new patient recruitment or prospective data collection was performed. The study used an openly released, previously collected breast-ultrasound dataset and a publicly released pretrained model.

## 2.2 Pretrained model

The model under evaluation was `Parveshiiii/breast-cancer-detector`, obtained from Hugging Face and pinned to revision `7c4c1ac11f5f80d382cc641ec6f2d3989e7fa73c` before complete external inference. The released model is a `ViTForImageClassification` classifier fine-tuned from `google/vit-base-patch16-224-in21k`. It performs three-class breast-ultrasound image classification with the label mapping defined in the released model configuration: class 0, benign; class 1, malignant; and class 2, normal.

The architecture uses 224 × 224 pixel inputs with 16 × 16 image patches, a hidden dimension of 768, 12 transformer encoder layers, and 12 attention heads. Model weights were kept frozen throughout all external analyses. No BUS-UCLM image, class label, patient identifier, or derived result was used to update model parameters, select alternative weights, or fine-tune the classifier.

The original model documentation reports training on a breast-ultrasound image dataset derived from the public BUSI/Al-Dhabyani dataset family. Developer-reported performance was treated only as contextual background because the original data, evaluation design, and inclusion rules were not identical to the present external test.

## 2.3 External test dataset

External testing was performed on BUS-UCLM, a publicly released breast-ultrasound dataset collected at Ciudad Real General University Hospital / University of Castilla-La Mancha, Spain. The dataset contains 683 ultrasound images from 38 patients, comprising 419 normal images, 174 benign images, and 90 malignant images. Images were acquired using a Siemens ACUSON S2000 ultrasound system during 2022–2023. The dataset includes expert annotations and separate lesion-segmentation masks; malignant lesions were reported by the dataset authors as biopsy-confirmed.

The image was the prediction unit because the model produces one classification per image. However, because multiple images may originate from the same patient, observations were not assumed to be statistically independent for uncertainty estimation. Patient identifiers supplied by the dataset were therefore retained in the analysis manifest and used for patient-clustered bootstrap confidence intervals.

The official BUS-UCLM class labels were used as the reference standard without modification based on model output. For the binary malignancy analysis, malignant images were coded as positive and normal plus benign images were combined into a non-malignant negative class.

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
