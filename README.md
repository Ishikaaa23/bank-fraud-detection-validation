# Bank Fraud Detection using Machine Learning and NLP

## Project Overview

This project explores fraud detection in banking transactions using a combination of structured transaction data and transaction description text. The workflow covers data exploration, feature engineering, NLP-based feature extraction, class imbalance handling, model development, explainability, and validation.

The goal was not only to build a fraud detection model, but also to evaluate its effectiveness through multiple validation techniques and understand the limitations imposed by the underlying dataset.

---

## Dataset Summary

- Total Transactions: 200,000
- Legitimate Transactions: 189,912
- Fraudulent Transactions: 10,088
- Fraud Rate: 5.04%

The dataset contains:

- Customer demographics
- Transaction information
- Merchant details
- Device and location attributes
- Transaction descriptions
- Fraud labels (`Is_Fraud`)

> Note: The dataset is not included in this repository.

---

## Project Workflow

### Exploratory Data Analysis

Performed analysis on:

- Class distribution
- Transaction amount patterns
- Fraud rate by transaction hour
- Merchant category risk
- Device type risk
- Transaction type risk
- Transaction description frequencies

### Feature Engineering

Created features such as:

- Log transaction amount
- Log account balance
- Amount-to-balance ratio
- Post-transaction balance
- Overdraft indicators
- Weekend and night transaction flags
- High-risk device indicators
- Interaction features

Structured Features: **38**

### NLP Features

Transaction descriptions were converted into numerical features using:

#### Keyword-Based Features

- Risk score generation
- Fraud keyword indicators
- Description length metrics

#### TF-IDF Features

- Top 30 TF-IDF features extracted from transaction descriptions

NLP Features: **30**

Total Model Features: **68**

---

## Models Evaluated

### 1. Baseline XGBoost

Standard XGBoost classifier without imbalance handling.

### 2. XGBoost + Class Weighting

Used `scale_pos_weight` to account for fraud class imbalance.

### 3. XGBoost + SMOTE

Applied Synthetic Minority Oversampling Technique on the training data.

### 4. XGBoost + ADASYN

Applied Adaptive Synthetic Sampling to generate minority-class observations.

---

## Test Set Performance

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 Score |
|---------|---------|---------|---------|---------|---------|
| Baseline | 0.5090 | 0.0524 | 0.0000 | 0.0000 | 0.0000 |
| XGBoost + Class Weighting | 0.5022 | 0.0520 | 0.0633 | 0.0431 | 0.0513 |
| XGBoost + SMOTE | 0.5119 | 0.0534 | 0.0698 | 0.0203 | 0.0315 |
| XGBoost + ADASYN | 0.5186 | 0.0547 | 0.0726 | 0.0198 | 0.0311 |

Best performing model based on PR-AUC:

**XGBoost + ADASYN**

---

## Cross Validation

5-Fold Stratified Cross Validation was performed to assess model stability.

| Metric | Score |
|----------|----------|
| ROC-AUC | 0.4950 ± 0.0063 |
| PR-AUC | 0.0505 ± 0.0009 |
| F1 Score | 0.0799 ± 0.0018 |

---

## Model Explainability

To understand model behaviour, explainability techniques were applied:

- Feature Importance Analysis
- SHAP Summary Plot
- SHAP Feature Ranking
- SHAP Waterfall Analysis
- NLP Feature Contribution Analysis

These analyses helped identify which structured and text-based features influenced fraud predictions.

---

## Business Impact Analysis

A simple cost-benefit framework was used to estimate the operational impact of fraud detection.

### Baseline Model

- Recovered Fraud Amount: ₹0
- Missed Fraud Loss: ₹110.99 Million
- Net Benefit: -₹110.99 Million

### Best Model (ADASYN)

- Recovered Fraud Amount: ₹93.61 Million
- Missed Fraud Loss: ₹17.38 Million
- False Alert Cost: ₹6.28 Million
- Net Benefit: ₹69.95 Million

---

## Key Takeaways

- Multiple imbalance-handling strategies were evaluated, including class weighting, SMOTE, and ADASYN.
- NLP-derived transaction description features were incorporated alongside structured transaction attributes.
- Model explainability techniques were used to understand prediction drivers.
- Despite extensive feature engineering and resampling strategies, predictive performance remained close to random classification.
- The project highlights the importance of data quality, validation, and explainability in fraud analytics workflows.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn
- SHAP
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Project Structure

```text
bank_fraud_detection/
│
├── fraud_detection.ipynb
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_generator.py
│   ├── feature_engineering.py
│   ├── nlp_features.py
│   └── evaluation.py
│
├── data/      (excluded)
├── models/    (excluded)
└── plots/     (excluded)
```

---
