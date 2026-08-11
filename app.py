"""
Phase 3: Streamlit Web Application (Enhanced Executive UI)
Customer Retention & Decision Support System

Run from root:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import sys
import os

# Import decision_engine from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from decision_engine import get_risk_tier, recommend_actions


# Page Config
st.set_page_config(
    page_title="Customer Retention Decision Support",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">🛡️ Customer Retention & Risk Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Churn Risk Assessment & Real-Time Decision Support Dashboard</div>', unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return joblib.load("artifacts/retention_pipeline.joblib")

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"⚠️ Model Artifact Error: Could not load pipeline from 'artifacts/retention_pipeline.joblib'. Check path. Error: {e}")
    st.stop()

# Layout: 2 Main Columns (Left: Inputs | Right: Results)
col_input, col_display = st.columns([1.2, 1], gap="large")

with col_input:
    st.subheader("📋 Customer Profile Input")
    
    # Input Tabs for clean layout
    tab1, tab2, tab3 = st.tabs(["👤 Account & Demographics", "🌐 Services Subscribed", "💳 Billing & Contract"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
        with c2:
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        with c2:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
        with c2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
            default_total = round(monthly_charges * tenure, 2)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(default_total), step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("📊 Evaluate Customer Risk")

# Prediction & Result Display Column
with col_display:
    st.subheader("📊 Risk Assessment Output")
    
    if predict_btn:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        X_new = pd.DataFrame([customer])
        probability = float(pipeline.predict_proba(X_new)[0, 1])
        risk_tier, priority_flag = get_risk_tier(probability)
        actions = recommend_actions(customer, priority_flag)

        # Priority Color Schema
        color_theme = {
            "RED": {"color": "#DC2626", "bg": "#FEF2F2", "border": "#FCA5A5", "icon": "🔴"},
            "ORANGE": {"color": "#D97706", "bg": "#FFFBEB", "border": "#FCD34D", "icon": "🟠"},
            "GREEN": {"color": "#16A34A", "bg": "#F0FDF4", "border": "#86EFAC", "icon": "🟢"}
        }
        theme = color_theme.get(priority_flag, color_theme["GREEN"])

        # Display Result Metric Box
        st.markdown(f"""
            <div style="background-color: {theme['bg']}; border: 2px solid {theme['border']}; border-radius: 12px; padding: 1.2rem; text-align: center;">
                <h4 style="color: #475569; margin: 0; font-size: 0.95rem;">PREDICTED CHURN RISK</h4>
                <h1 style="color: {theme['color']}; font-size: 3rem; margin: 0.2rem 0;">{probability:.1%}</h1>
                <p style="font-weight: 700; color: {theme['color']}; font-size: 1.1rem; margin: 0;">
                    {theme['icon']} {risk_tier.upper()} ({priority_flag} PRIORITY)
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Risk Score Progress Bar
        st.caption("Risk Threshold Spectrum")
        st.progress(probability)

        # Action Recommendations Section
        st.markdown("### 🎯 Recommended Retention Actions")
        
        if priority_flag == "RED":
            st.error("🚨 **Immediate Direct Outreach Required**")
        elif priority_flag == "ORANGE":
            st.warning("⚠️ **Automated Re-engagement Targeted**")
        else:
            st.success("✅ **Standard Operational Flow (No Action Needed)**")

        for action in actions:
            st.info(f"👉 {action}")

    else:
        # Initial Placeholder - Professional Dashboard Look
        st.markdown("""
            <div style="background-color: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 12px; padding: 2.5rem; text-align: center; margin-top: 1rem;">
                <h3 style="color: #475569; margin-bottom: 0.5rem;">👈 Ready for Risk Analysis</h3>
                <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 0;">
                    Fill out the customer attributes in the form and click <b>'Assess Churn Risk'</b> to trigger the AI decision engine.
                </p>
            </div>
        """, unsafe_allow_html=True)