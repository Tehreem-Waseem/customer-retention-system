# 🛡️ Customer Retention & Decision Support System

<p align="center">
  <a href="https://customer-retention-system-8zyvt6uptwnh9cfrdmeagw.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Live App">
  </a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/PyTest-Passing-46A2F1?style=flat&logo=pytest&logoColor=white" alt="PyTest">
</p>

<p align="center">
  An end-to-end Machine Learning system that converts raw customer churn probabilities into actionable operational risk tiers and targeted business retention strategies.
</p>

<p align="center">
  🚀 <b><a href="https://customer-retention-system-8zyvt6uptwnh9cfrdmeagw.streamlit.app/">CLICK HERE TO LAUNCH LIVE DASHBOARD DEMO</a></b> 🚀
</p>

---

## 📊 Business Impact & ROI Highlights

> **The $6,390 Bottom-Line Advantage:**
> Under a fixed outreach budget of 200 customers ($10 contact cost, $300/year retention value):

| Strategy | Actual Churners Reached | Net Saved Revenue |
| :--- | :---: | :---: |
| **Random Outreach (Baseline)** | 59 Customers | Baseline Profit |
| **ML-Prioritized Selection** | **130 Customers** | **+$6,390 Saved Revenue** |



## Repository Layout

```
customer_retention_system/
├── artifacts/
│   └── retention_pipeline.joblib   # Serialized, fitted scikit-learn Pipeline
├── data/
│   └── Telco-Customer-Churn.csv    # Source dataset
├── notebooks/
│   └── rough_work_eda.py           # Phase 0: exploratory data analysis
├── src/
│   ├── train_pipeline.py           # Phase 1: preprocessing, benchmarking, tuning
│   └── decision_engine.py          # Phase 2: risk rules, actions, ROI simulation
├── tests/
│   └── test_system.py              # Phase 3: automated unit tests (pytest)
├── app.py                          # Phase 3: Streamlit dashboard
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt

# Phase 0: exploratory analysis (generates plots in notebooks/)
python notebooks/rough_work_eda.py

# Phase 1: train, tune, and save the model
python src/train_pipeline.py

# Phase 2: risk tiers + ROI simulation demo
python src/decision_engine.py

# Phase 3: run the automated tests
pytest tests/test_system.py -v

# Phase 3: launch the dashboard
streamlit run app.py
```

Run these from the `customer_retention_system` folder — the scripts use
relative paths (`data/...`, `artifacts/...`).

## Architecture & Design Decisions

**Pipeline over manual preprocessing.** `train_pipeline.py` wraps a
`ColumnTransformer` (median imputation + scaling for numeric features,
most-frequent imputation + one-hot encoding for categorical features) and the
classifier inside a single `Pipeline`. Fitting is done once, on
`pipeline.fit(X_train, y_train)`, so every transformer only ever learns
statistics from the training fold. This prevents data leakage during both
cross-validation and the final test evaluation, and it's why `predict.py`-style
scripts never need to manually re-apply scaling or encoding — the saved
pipeline already knows how.

**Model selection is benchmarked, not assumed.** Four candidate models
(Logistic Regression, KNN, Decision Tree, Random Forest) are compared with
5-fold cross-validation on the training set only, using ROC-AUC, F1, and
PR-AUC. The winner is then tuned with `GridSearchCV` (also training-set-only,
5-fold). The held-out test set is touched exactly once, at the very end, for
an honest final evaluation.

**Accuracy is not the target metric.** With ~26% of customers churning, a
model that always predicts "no churn" would score ~74% accuracy while being
useless. `class_weight='balanced'` is used on the linear/tree ensemble
candidates, and ROC-AUC / PR-AUC are the metrics that actually guide model
selection and threshold decisions.

**The Decision Engine exists because a probability isn't a decision.**
`decision_engine.py` maps `P(Churn)` to a risk tier (Red/Orange/Green),
attaches rules-based recommended actions from the customer's own attributes
(e.g. month-to-month contract → offer an annual discount), and runs a
simplified ROI simulation comparing ML-prioritized outreach against random
outreach under a fixed budget. This is the layer that turns a model output
into something a non-technical retention team can act on.

**Separation of concerns.** Training logic, business/decision logic, tests,
and the UI live in separate files/modules by design. This means the model can
be retrained without touching the business rules, the business rules can be
changed without retraining, and the UI can be swapped out (or a second UI
added) without touching either.

## Known Data Quirk

`TotalCharges` in the raw CSV is stored as text and contains 11 rows with a
literal blank space instead of a number (all belong to brand-new customers
with `tenure=0`, i.e. customers who haven't been billed yet). This is not
caught by a standard `.isnull()` check. It's converted to a proper numeric
column with `pd.to_numeric(..., errors='coerce')` before splitting, and the
resulting `NaN` values are handled by `SimpleImputer(median)` **inside** the
pipeline — not filled manually beforehand — to avoid leaking information
across the train/test split.
