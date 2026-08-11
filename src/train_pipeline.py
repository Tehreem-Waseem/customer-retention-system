"""
Phase 1: Multi-Model Benchmarking, Hyperparameter Tuning & Pipeline Assembly
Customer Retention & Decision Support System
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report
)
import joblib
import os

DATA_PATH = "data/Telco-Customer-Churn.csv"
ARTIFACT_PATH = "artifacts/retention_pipeline.joblib"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load & lightly prepare data (dtype fixes only - NOT filling/imputing here,
# that happens inside the pipeline so it's leakage-safe)
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["customerID"])

# Fix the TotalCharges whitespace bug found in Phase 0: convert to numeric,
# blanks become NaN. We do NOT fill them here - SimpleImputer(median) inside
# the pipeline will handle that using only the training fold's statistics.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Encode target: Yes -> 1, No -> 0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in df.columns if c not in numerical_features + ["Churn"]]

X = df[numerical_features + categorical_features]
y = df["Churn"]

print("=" * 60)
print("FEATURE SETUP")
print("=" * 60)
print(f"Numerical features   : {numerical_features}")
print(f"Categorical features : {categorical_features}")
print(f"Target balance        : {y.value_counts(normalize=True).round(3).to_dict()}")

# ---------------------------------------------------------------------------
# 1.1 Stratified train/test split & preprocessing
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# ---------------------------------------------------------------------------
# 1.2 Algorithmic benchmarking (TRAINING SET ONLY, 5-fold CV)
# ---------------------------------------------------------------------------
candidates = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["roc_auc", "f1", "average_precision"]

print("\n" + "=" * 60)
print("PHASE 1.2: 5-FOLD CV BENCHMARKING (training set only)")
print("=" * 60)

cv_results = {}
for name, model in candidates.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results[name] = scores
    print(f"\n{name}")
    print(f"  ROC-AUC : {scores['test_roc_auc'].mean():.4f} (+/- {scores['test_roc_auc'].std():.4f})")
    print(f"  F1      : {scores['test_f1'].mean():.4f} (+/- {scores['test_f1'].std():.4f})")
    print(f"  PR-AUC  : {scores['test_average_precision'].mean():.4f} (+/- {scores['test_average_precision'].std():.4f})")

# Pick the best candidate by mean ROC-AUC (primary metric for this benchmarking pass)
best_name = max(cv_results, key=lambda n: cv_results[n]["test_roc_auc"].mean())
print(f"\n>>> Best candidate by mean ROC-AUC: {best_name}")

# ---------------------------------------------------------------------------
# 1.3 Hyperparameter optimization (GridSearchCV, 5-fold CV on training data)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"PHASE 1.3: GRIDSEARCHCV TUNING - {best_name}")
print("=" * 60)

param_grids = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10],
    },
    "K-Nearest Neighbors": {
        "classifier__n_neighbors": [5, 11, 21, 31],
        "classifier__weights": ["uniform", "distance"],
    },
    "Decision Tree": {
        "classifier__max_depth": [3, 5, 8, None],
        "classifier__min_samples_leaf": [1, 5, 10],
    },
    "Random Forest": {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_leaf": [1, 5],
    },
}

best_model = candidates[best_name]
best_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", best_model)])

grid_search = GridSearchCV(
    best_pipeline,
    param_grid=param_grids[best_name],
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print(f"Best params : {grid_search.best_params_}")
print(f"Best CV ROC-AUC : {grid_search.best_score_:.4f}")

final_pipeline = grid_search.best_estimator_

# ---------------------------------------------------------------------------
# 1.4 Final evaluation on held-out test set (ONE TIME ONLY)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 1.4: FINAL EVALUATION ON HELD-OUT TEST SET")
print("=" * 60)

y_pred = final_pipeline.predict(X_test)
y_proba = final_pipeline.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ---------------------------------------------------------------------------
# 1.5 Serialize the final tuned pipeline
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
joblib.dump(final_pipeline, ARTIFACT_PATH)
print(f"Saved final pipeline to {ARTIFACT_PATH}")