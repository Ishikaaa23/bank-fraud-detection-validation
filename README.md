# Bank Transaction Fraud Detection

End-to-end machine learning pipeline for detecting fraudulent bank transactions on a **highly imbalanced dataset** (~0.17% fraud rate).

---

## Results

| Model | ROC-AUC | PR-AUC | F1 (Fraud) | Precision | Recall |
|---|---|---|---|---|---|
| Baseline XGBoost | ~0.90 | ~0.55 | ~0.35 | ~0.80 | ~0.22 |
| **+ scale_pos_weight** | **1.00** | **1.00** | **0.985** | **1.00** | **0.969** |
| + SMOTE | 1.00 | 0.99 | 0.981 | 0.99 | 0.972 |
| + ADASYN | 1.00 | 0.99 | 0.979 | 0.99 | 0.969 |

**5-Fold Stratified CV**: ROC-AUC `1.0000 ± 0.0000` | PR-AUC `0.9970 ± 0.0024` | F1 `0.9789 ± 0.0085`

---

## Why This Problem Is Hard

A model that always predicts "Legitimate" achieves **99.83% accuracy** while catching **zero fraud**. Accuracy is a misleading metric for imbalanced data. We optimise for:

- **PR-AUC** — gold standard for imbalanced binary classification
- **F1 (fraud class)** — harmonic mean of precision and recall
- **Recall** — fraction of actual fraud cases caught
- **ROC-AUC** — overall discrimination ability

---

## What We Do

### 1. Feature Engineering (`src/feature_engineering.py`)
Adds 13 domain-meaningful derived features on top of the original 16:
- `amount_to_avg_ratio` — transaction vs. 7-day average (anomaly signal)
- `no_security` — no chip AND no PIN (weakest auth)
- `weak_auth_high_amount` — interaction of auth weakness × amount
- `total_distance`, `log_distance_*` — spatial anomaly signals
- `is_night`, `night_x_foreign` — time-based risk flags
- `new_account` — accounts < 90 days old are higher risk
- `high_velocity` — > 10 transactions in 24h

### 2. Imbalance Handling (3 strategies compared)
| Strategy | Mechanism |
|---|---|
| `scale_pos_weight` | Upweights fraud gradients in XGBoost training |
| SMOTE | Synthesises new fraud samples via KNN interpolation |
| ADASYN | Adaptive synthesis — more samples near decision boundary |

### 3. Threshold Optimisation
The default 0.5 decision threshold is wrong for imbalanced data. We find the threshold that maximises F1 on the training distribution — exposing a key **business lever**: shift the threshold to trade precision for recall based on fraud cost vs. false positive cost.

### 4. Stratified Cross-Validation
5-Fold StratifiedKFold preserves the fraud rate in every fold, giving reliable performance bounds rather than a lucky single split.

### 5. Business Impact Analysis
Translates model metrics into dollar terms using configurable cost assumptions (avg fraud amount, false positive cost, missed fraud cost).

---

## Project Structure

```
├── fraud_detection.ipynb       # Main notebook — run this
├── src/
│   ├── data_generator.py       # Synthetic dataset generator (16 features)
│   ├── feature_engineering.py  # Domain feature engineering
│   └── evaluation.py           # Metrics, plots, business impact
├── models/
│   ├── fraud_detector_final.pkl
│   └── model_metadata.json
├── plots/                      # All generated visualisations
│   ├── eda_overview.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── metrics_comparison.png
│   ├── pr_roc_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── threshold_sensitivity.png
│   └── cv_scores.png
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```

To use your own dataset, replace the data loading cell with:
```python
df = pd.read_csv('data/transactions.csv')
```
Ensure your CSV has the 16 feature columns listed in `src/data_generator.py` and an `is_fraud` target column.

---

## Key Takeaways for Production

- **Real-time inference**: XGBoost predictions are sub-millisecond per transaction
- **Threshold is a business decision**: align with fraud cost vs. customer friction cost
- **Model drift**: fraud patterns evolve — schedule regular retraining on recent data
- **Feature store**: velocity features (`num_txns_last_24h`, `avg_amount_last_7d`) require real-time aggregation pipelines (e.g. Redis, Flink)
