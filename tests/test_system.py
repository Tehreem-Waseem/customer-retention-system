"""
Phase 3: Automated Quality Assurance
Customer Retention & Decision Support System

Run from the project root with:
    pytest tests/test_system.py -v
"""

import sys
import os
import pytest
import pandas as pd
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from decision_engine import get_risk_tier, recommend_actions

# Resolve paths relative to THIS file's location, not the current working
# directory. This means the tests work whether pytest is run from the
# project root, from tests/, or with a full path from anywhere else.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "artifacts", "retention_pipeline.joblib")


# ---------------------------------------------------------------------------
# 3.2a Probability thresholds correctly map to Risk Tiers and Priority flags
# ---------------------------------------------------------------------------
class TestRiskTierMapping:

    def test_high_risk_boundary(self):
        tier, flag = get_risk_tier(0.65)
        assert tier == "High Risk"
        assert flag == "RED"

    def test_high_risk_above_boundary(self):
        tier, flag = get_risk_tier(0.90)
        assert tier == "High Risk"
        assert flag == "RED"

    def test_medium_risk_lower_boundary(self):
        tier, flag = get_risk_tier(0.35)
        assert tier == "Medium Risk"
        assert flag == "ORANGE"

    def test_medium_risk_mid_range(self):
        tier, flag = get_risk_tier(0.50)
        assert tier == "Medium Risk"
        assert flag == "ORANGE"

    def test_medium_risk_just_below_high(self):
        tier, flag = get_risk_tier(0.6499)
        assert tier == "Medium Risk"
        assert flag == "ORANGE"

    def test_low_risk(self):
        tier, flag = get_risk_tier(0.10)
        assert tier == "Low Risk"
        assert flag == "GREEN"

    def test_low_risk_just_below_medium(self):
        tier, flag = get_risk_tier(0.3499)
        assert tier == "Low Risk"
        assert flag == "GREEN"

    def test_zero_probability(self):
        tier, flag = get_risk_tier(0.0)
        assert flag == "GREEN"

    def test_full_probability(self):
        tier, flag = get_risk_tier(1.0)
        assert flag == "RED"


class TestActionRecommendations:

    def test_green_gets_no_intervention_only(self):
        customer = {"Contract": "Two year", "PaymentMethod": "Credit card (automatic)"}
        actions = recommend_actions(customer, "GREEN")
        assert len(actions) == 1
        assert "No intervention" in actions[0]

    def test_red_gets_immediate_outreach(self):
        customer = {"Contract": "Two year", "PaymentMethod": "Credit card (automatic)"}
        actions = recommend_actions(customer, "RED")
        assert any("Immediate direct outreach" in a for a in actions)

    def test_month_to_month_triggers_annual_offer(self):
        customer = {"Contract": "Month-to-month", "PaymentMethod": "Credit card (automatic)"}
        actions = recommend_actions(customer, "RED")
        assert any("annual contract" in a for a in actions)

    def test_electronic_check_triggers_autopay_suggestion(self):
        customer = {"Contract": "Two year", "PaymentMethod": "Electronic check"}
        actions = recommend_actions(customer, "ORANGE")
        assert any("automatic payment" in a for a in actions)


# ---------------------------------------------------------------------------
# 3.2b Raw input dictionaries pass through preprocessor + model without
# throwing schema errors
# ---------------------------------------------------------------------------
class TestPipelineSchemaCompatibility:

    @pytest.fixture(scope="class")
    @staticmethod
    def pipeline():
        return joblib.load(ARTIFACT_PATH)

    def test_typical_customer_predicts_without_error(self, pipeline):
        customer = {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 24, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check", "MonthlyCharges": 55.0, "TotalCharges": 1320.0,
        }
        X_new = pd.DataFrame([customer])
        proba = pipeline.predict_proba(X_new)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0, 1] <= 1.0

    def test_new_customer_zero_tenure_predicts_without_error(self, pipeline):
        # Mirrors the TotalCharges whitespace-bug rows found in Phase 0:
        # brand new customers with tenure=0. Confirms the pipeline's imputer
        # handles a missing/edge-case TotalCharges without crashing.
        customer = {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 0, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": None,
        }
        X_new = pd.DataFrame([customer])
        proba = pipeline.predict_proba(X_new)
        assert proba.shape == (1, 2)

    def test_unseen_category_does_not_crash(self, pipeline):
        # OneHotEncoder(handle_unknown='ignore') should absorb a category
        # value it never saw during training rather than raising an error.
        customer = {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 10, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Lifetime",  # unseen category
            "PaperlessBilling": "No", "PaymentMethod": "Mailed check",
            "MonthlyCharges": 60.0, "TotalCharges": 600.0,
        }
        X_new = pd.DataFrame([customer])
        proba = pipeline.predict_proba(X_new)
        assert proba.shape == (1, 2)

    def test_batch_prediction_shape(self, pipeline):
        customers = [
            {"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
             "tenure": 24, "PhoneService": "Yes", "MultipleLines": "No",
             "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
             "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
             "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "No",
             "PaymentMethod": "Mailed check", "MonthlyCharges": 55.0, "TotalCharges": 1320.0},
            {"gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
             "tenure": 1, "PhoneService": "Yes", "MultipleLines": "Yes",
             "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
             "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
             "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
             "PaymentMethod": "Electronic check", "MonthlyCharges": 100.5, "TotalCharges": 100.5},
        ]
        X_batch = pd.DataFrame(customers)
        proba = pipeline.predict_proba(X_batch)
        assert proba.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])