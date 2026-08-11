"""
Phase 0: Rough Work & Exploratory Data Analysis
Customer Retention & Decision Support System

This file is the exploratory scratchpad for Phase 0.
It performs:
1. Data ingestion and auditing
2. TotalCharges cleaning
3. Exploratory visualizations
4. Initial observations
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. DATA INGESTION
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)

# Get the folder where this Python file is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_csv(DATA_URL)


# ============================================================
# 2. BASIC DATA AUDITING
# ============================================================

print("=" * 60)
print("SHAPE & DTYPES")
print("=" * 60)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

if missing.sum() == 0:
    print("No standard NaN values found.")
else:
    print(missing[missing > 0])


# ============================================================
# 4. TOTALCHARGES WHITESPACE BUG
# ============================================================

print("\n" + "=" * 60)
print("TotalCharges WHITESPACE AUDIT")
print("=" * 60)

print("TotalCharges dtype before fix:")
print(df["TotalCharges"].dtype)


# Try converting TotalCharges to numeric
total_charges_numeric = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# Find values that cannot be converted
invalid_total_charges = total_charges_numeric.isna()

print(
    "Rows that fail numeric conversion:",
    invalid_total_charges.sum()
)


print("\nProblematic rows:")

print(
    df.loc[
        invalid_total_charges,
        [
            "customerID",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]
    ]
)


# ============================================================
# 5. FIX TOTALCHARGES
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTotalCharges dtype after conversion:")
print(df["TotalCharges"].dtype)

print(
    "Missing TotalCharges after conversion:",
    df["TotalCharges"].isnull().sum()
)


# ============================================================
# 6. DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ROW AUDIT")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)


# ============================================================
# 7. CHURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

print(df["Churn"].value_counts())

print("\nChurn Percentage:")

print(
    df["Churn"].value_counts(normalize=True) * 100
)


# ============================================================
# 8. EDA VISUALIZATION
#    TENURE + MONTHLY CHARGES
# ============================================================

print("\n" + "=" * 60)
print("CREATING NUMERICAL FEATURE PLOTS")
print("=" * 60)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


for churn_value in ["Yes", "No"]:

    subset = df[df["Churn"] == churn_value]

    # Tenure
    axes[0].hist(
        subset["tenure"],
        bins=30,
        alpha=0.6,
        label=f"Churn = {churn_value}"
    )

    # Monthly Charges
    axes[1].hist(
        subset["MonthlyCharges"],
        bins=30,
        alpha=0.6,
        label=f"Churn = {churn_value}"
    )


# Tenure graph
axes[0].set_title(
    "Tenure Distribution by Churn"
)

axes[0].set_xlabel(
    "Tenure (Months)"
)

axes[0].set_ylabel(
    "Number of Customers"
)

axes[0].legend()


# Monthly Charges graph
axes[1].set_title(
    "Monthly Charges Distribution by Churn"
)

axes[1].set_xlabel(
    "Monthly Charges"
)

axes[1].set_ylabel(
    "Number of Customers"
)

axes[1].legend()


plt.tight_layout()


# Save inside notebooks folder
distribution_path = (
    SCRIPT_DIR / "eda_distributions.png"
)

plt.savefig(
    distribution_path,
    dpi=120
)

print(
    f"Saved: {distribution_path}"
)

plt.show()

plt.close()


# ============================================================
# 9. CATEGORICAL FEATURE CHURN RATES
# ============================================================

print("\n" + "=" * 60)
print("CREATING CATEGORICAL FEATURE PLOTS")
print("=" * 60)


categorical_cols = [
    "Contract",
    "InternetService",
    "PaymentMethod"
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)


for ax, col in zip(
    axes,
    categorical_cols
):

    # Calculate churn rate
    churn_rate = (
        df.groupby(col)["Churn"]
        .apply(
            lambda x: (x == "Yes").mean()
        )
        .sort_values(
            ascending=False
        )
    )

    # Create bar chart
    ax.bar(
        churn_rate.index,
        churn_rate.values
    )

    ax.set_title(
        f"Churn Rate by {col}"
    )

    ax.set_ylabel(
        "Churn Rate"
    )

    ax.set_xlabel(
        col
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )


plt.tight_layout()


# Save inside notebooks folder
categorical_path = (
    SCRIPT_DIR /
    "eda_churn_rate_by_category.png"
)

plt.savefig(
    categorical_path,
    dpi=120
)

print(
    f"Saved: {categorical_path}"
)

plt.show()

plt.close()


# ============================================================
# 10. SCRATCHPAD OBSERVATIONS
# ============================================================

print("\n" + "=" * 60)
print("SCRATCHPAD OBSERVATIONS")
print("=" * 60)


# Overall churn rate
overall_churn_rate = (
    df["Churn"] == "Yes"
).mean()


print(
    f"Overall churn rate: "
    f"{overall_churn_rate:.2%}"
)


# Churn rate by contract
contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(
        lambda x: (x == "Yes").mean()
    )
)


print(
    "\nChurn rate by contract type:"
)

print(contract_churn)


# Mean tenure of churned customers
mean_tenure_churned = (
    df[df["Churn"] == "Yes"]["tenure"]
    .mean()
)


# Mean tenure of non-churned customers
mean_tenure_not_churned = (
    df[df["Churn"] == "No"]["tenure"]
    .mean()
)


print(
    f"\nMean tenure of churned customers: "
    f"{mean_tenure_churned:.1f} months"
)


print(
    f"Mean tenure of non-churned customers: "
    f"{mean_tenure_not_churned:.1f} months"
)


# ============================================================
# 11. FINAL OBSERVATIONS
# ============================================================
print("\n" + "=" * 60)
print("INITIAL EDA OBSERVATIONS")
print("=" * 60)

print(
    "1. The dataset contains 7,043 customers "
    "and 21 features."
)

print(
    "2. The overall churn rate is approximately "
    "26.54%, indicating class imbalance."
)

print(
    "3. Month-to-month contract customers have "
    "a much higher churn rate than one-year "
    "and two-year contract customers."
)

print(
    "4. Churned customers have lower average "
    "tenure than non-churned customers."
)

print(
    "5. TotalCharges contained 11 values that "
    "could not initially be converted to numeric."
)

print(
    "6. Numerical and categorical features show "
    "different patterns between churned and "
    "non-churned customers."
)

print(
    "7. The dataset requires preprocessing before "
    "machine learning because it contains both "
    "numerical and categorical features."
)


# ============================================================
# 12. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"\nGraphs saved in:\n{SCRIPT_DIR}"
)

# ============================================================
#  INITIAL MODEL EXPERIMENTS
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# INITIAL MODEL EXPERIMENTS

# Model                  Initial Observation
# ------------------------------------------------
# Logistic Regression    Strong baseline
# KNN                    Sensitive to feature scaling
# Decision Tree          Captures non-linear relationships
# Random Forest          Strong non-linear baseline

# Initial conclusion:
# All four models should be benchmarked using the same
# preprocessing strategy and 5-fold cross-validation in
# the production training pipeline.

# The final model should not be selected using the held-out
# test set.

### Critical Insights From Rough Work & Data Auditing

# Based on the exploratory data analysis (EDA) conducted above, here are the core patterns that will drive our machine learning and decision-making system:

# 1. **Temporal & Financial Risks:** Customer churn risk is heavily concentrated within the first 5 months of tenure. Furthermore, a significant churn spike is visible for customers with high monthly bills between $70 and $100.
# 2. **Contractual Vulnerability:** Month-to-Month contracts exhibit an alarmingly high churn rate (above 40%), whereas one-year and two-year commitments significantly stabilize retention.
# 3. **Service & Payment Friction:** Fiber Optic internet plans show unexpected high churn, indicating potential pricing or quality issues. Additionally, manual 'Electronic Check' payment users churn far more than those on automatic autopay channels.

# **Conclusion for Next Phase:** These insights strongly justify our Phase 2 Decision Engine rules: targeting Month-to-Month contracts, troubleshooting Fiber Optic plans, and incentivizing Electronic Check users to switch to Automatic Autopay.
