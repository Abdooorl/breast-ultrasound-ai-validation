# Breast Ultrasound AI Validation

**Owner:** [Abdooorl](https://github.com/Abdooorl)  
**Target repository:** `Abdooorl/breast-ultrasound-ai-validation`  
**Project deadline:** **30 October 2026**

External validation and uncertainty-aware evaluation of a pretrained Vision Transformer for breast ultrasound classification.

## Research objective

Evaluate whether a publicly available pretrained breast-ultrasound classifier generalizes to an independently acquired dataset (BUS-UCLM), and assess whether uncertainty-aware abstention can improve reliability without retraining the model.

## Working paper title

> **Do Public Breast Ultrasound AI Models Generalize? External Validation and Uncertainty-Aware Evaluation of a Pretrained Vision Transformer**

## Status

**Phase 1 — Foundation / repository setup**

Research project in progress. Target completion: **30 October 2026**.

## Intended use

Research, education, and portfolio use only. **Not intended for clinical diagnosis, patient management, or medical decision-making.**

## Primary experiment

The pretrained model `Parveshiiii/breast-cancer-detector` will be evaluated on BUS-UCLM **without retraining or fine-tuning** for the primary external-validation experiment.

## Planned outputs

- External validation on BUS-UCLM
- Three-class and malignant-vs-non-malignant evaluation
- Patient-aware confidence intervals
- Calibration analysis
- Selective classification / abstention analysis
- Failure analysis, with emphasis on malignant false negatives
- Optional attribution / lesion-mask analysis
- Research-only FastAPI demo
- Reproducible manuscript and codebase
- Portfolio case study

## Repository structure

```text
breast-ultrasound-ai-validation/
├── app/
│   ├── api/                 # Research demo API
│   └── web/                 # Frontend added after core research
├── config/                  # Experiment/model configuration
├── data/
│   ├── raw/                 # Local dataset only; not committed
│   ├── processed/           # Derived local data
│   └── README.md
├── docs/
│   ├── dataset.md           # Dataset provenance/audit
│   ├── methodology.md       # Study methodology
│   └── model.md             # Pretrained-model audit
├── figures/                 # Publication-ready figures
├── notebooks/               # Exploratory analysis only
├── paper/                   # Manuscript + bibliography
├── results/
│   ├── metrics/
│   ├── predictions/
│   └── tables/
├── scripts/                 # Reproducible executable workflows
├── src/
│   ├── evaluation/
│   ├── inference/
│   ├── preprocessing/
│   └── utils/
├── tests/
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── RESEARCH_PROTOCOL.md
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the starter test suite:

```bash
pytest -q
```

Run the research API health check locally:

```bash
uvicorn app.api.main:app --reload
```

Then open `http://127.0.0.1:8000/health`.

## Data policy

The repository does **not** include BUS-UCLM or other medical image datasets. Raw images should remain local unless redistribution is explicitly permitted by the applicable licence.

## Reproducibility principle

The original pretrained model will first be evaluated exactly as an external model. Any later experimental adaptations must be clearly separated from the primary analysis.

## Responsible-use statement

This project evaluates the behaviour and generalization of an existing research model. Model scores must not be interpreted as validated cancer probabilities or used to diagnose or exclude disease.
