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
