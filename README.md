# Breast Cancer Detection Using Wisconsin Diagnostic Data

**CLA-3 Machine Learning Case Study**  
**Student:** Hemagna Chowdary M  
**Registration No.:** AP24110012070  
**Problem type:** Binary classification

## Project overview
This project compares Logistic Regression and Random Forest classifiers on the Wisconsin Diagnostic Breast Cancer dataset. The task is to classify a sample as malignant or benign using 30 numeric features computed from digitized fine-needle aspirate images.

## Dataset
The project uses the Wisconsin Diagnostic Breast Cancer dataset distributed with scikit-learn. It contains 569 samples and 30 numeric features. In scikit-learn's label encoding, `0 = malignant` and `1 = benign`.

Original dataset reference: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

## Models
- Logistic Regression with StandardScaler in a pipeline.
- Random Forest with 300 trees and balanced class weights.

## Reproducibility
- Train/test split: 80% / 20%
- Stratified split
- `random_state=42`
- Test samples: 114

## Run locally
```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python breast_cancer_classification.py
```

The script prints classification reports and creates:
- `outputs/model_comparison.csv`
- `outputs/confusion_matrices.png`

## Results from the supplied experiment
| Model | Accuracy | Precision (benign) | Recall (benign) | F1 (benign) | Malignant recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9762 |\n| Random Forest | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9286 |\n
## Important limitation
This is a classroom demonstration using a small, historical dataset. Its predictions must not be used for diagnosis, treatment decisions, or patient care. Real clinical use requires independent validation, appropriate clinical oversight, and regulatory review.
