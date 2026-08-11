"""
Phase 2: Business Decision Rules & Simplified ROI Engine
Customer Retention & Decision Support System

This module turns a raw churn probability into something a retention team
can actually act on: a risk tier, a priority flag, recommended actions,
and (in the simulation) a dollar estimate of campaign value.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 2.1 Risk Stratification
# ---------------------------------------------------------------------------
def get_risk_tier(probability):
    """Map a churn probability P(Churn|X) to a risk tier and priority flag."""
    if probability >= 0.65:
        return "High Risk", "RED"
    elif probability >= 0.35:
        return "Medium Risk", "ORANGE"
    else:
        return "Low Risk", "GREEN"


# ---------------------------------------------------------------------------
# 2.2 Action Recommendation Engine
# ---------------------------------------------------------------------------
def recommend_actions(customer, priority_flag):
    """
    Rules-based recommendations linking individual customer risk drivers
    to concrete retention actions. `customer` is a dict-like row of raw
    customer attributes (e.g. a row from the raw dataframe or a JSON record).
    """
    actions = []

    if priority_flag == "RED":
        actions.append("Immediate direct outreach call from retention team")
    elif priority_flag == "ORANGE":
        actions.append("Automated email re-engagement campaign")
    else:
        actions.append("No intervention - standard operational flow")
        return actions  # low risk customers get no further targeted offers

    if customer.get("Contract") == "Month-to-month":
        actions.append("Offer discount for switching to an annual contract")

    if customer.get("InternetService") == "Fiber optic" and customer.get("TechSupport") == "No":
        actions.append("Offer a free trial of the Tech Support add-on")

    if customer.get("OnlineSecurity") == "No":
        actions.append("Highlight the Online Security add-on")

    if customer.get("PaymentMethod") == "Electronic check":
        actions.append("Encourage switching to automatic payment")

    return actions


# ---------------------------------------------------------------------------
# 2.3 Simplified ROI Simulation
# ---------------------------------------------------------------------------
def simulate_roi(y_true, y_proba, n_sample=1000, budget=200,
                  cost_per_contact=10, retention_value=300,
                  conversion_rate=0.30, random_state=42):
    """
    Compare ML-Prioritized Selection vs Random Selection outreach strategies
    on a sample of test customers.

    y_true: ground-truth churn labels (1 = churned) for the test set
    y_proba: model's predicted churn probability for the same customers
    """
    rng = np.random.RandomState(random_state)

    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_proba": np.asarray(y_proba)})

    n_sample = min(n_sample, len(df))
    sample = df.sample(n=n_sample, random_state=random_state).reset_index(drop=True)

    # --- Random selection strategy ---
    random_contacted = sample.sample(n=budget, random_state=random_state)
    random_true_churners_contacted = random_contacted["y_true"].sum()
    random_retained = random_true_churners_contacted * conversion_rate
    random_net_saved = (random_retained * retention_value) - (budget * cost_per_contact)

    # --- ML-prioritized selection strategy ---
    ml_contacted = sample.sort_values("y_proba", ascending=False).head(budget)
    ml_true_churners_contacted = ml_contacted["y_true"].sum()
    ml_retained = ml_true_churners_contacted * conversion_rate
    ml_net_saved = (ml_retained * retention_value) - (budget * cost_per_contact)

    return {
        "n_sample": n_sample,
        "budget": budget,
        "random_true_churners_contacted": int(random_true_churners_contacted),
        "random_estimated_retained": round(random_retained, 1),
        "random_net_saved_revenue": round(random_net_saved, 2),
        "ml_true_churners_contacted": int(ml_true_churners_contacted),
        "ml_estimated_retained": round(ml_retained, 1),
        "ml_net_saved_revenue": round(ml_net_saved, 2),
        "improvement_vs_random": round(ml_net_saved - random_net_saved, 2),
    }


# ---------------------------------------------------------------------------
# Demonstration when run directly (this is the "process/rough work" trail)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import joblib
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/Telco-Customer-Churn.csv")
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [c for c in df.columns if c not in numerical_features + ["Churn"]]
    X = df[numerical_features + categorical_features]
    y = df["Churn"]

    # Same split as train_pipeline.py so this test set matches what the
    # model was evaluated on (not retrained here, just re-used).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = joblib.load("artifacts/retention_pipeline.joblib")
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("SAMPLE RISK STRATIFICATION (first 5 test customers)")
    print("=" * 60)
    X_test_reset = X_test.reset_index(drop=True)
    for i in range(5):
        customer = X_test_reset.iloc[i].to_dict()
        prob = y_proba_test[i]
        tier, flag = get_risk_tier(prob)
        actions = recommend_actions(customer, flag)
        print(f"\nCustomer {i}: P(Churn)={prob:.3f} -> {tier} [{flag}]")
        print(f"  Contract={customer['Contract']}, InternetService={customer['InternetService']}, "
              f"PaymentMethod={customer['PaymentMethod']}")
        for a in actions:
            print(f"  - {a}")

    print("\n" + "=" * 60)
    print("ROI SIMULATION: ML-Prioritized vs Random Selection")
    print("=" * 60)
    results = simulate_roi(y_test.values, y_proba_test)
    for k, v in results.items():
        print(f"{k:35s}: {v}")

    print(f"\n>>> ML-prioritized outreach nets ${results['improvement_vs_random']:.2f} "
          f"more than random selection, on the same {results['budget']}-customer budget.")