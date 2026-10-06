"""
Streamlit Application: Customer Churn Prediction & Retention Strategy App.
Interactive UI for predicting churn risk, generating SHAP explanations,
and recommending prioritized retention interventions.
"""
import os
from pathlib import Path
import sys

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as pd_st
import streamlit as st

from src.explain import (
    create_explainer,
    explain_single_prediction,
    generate_business_recommendations,
)
from src.preprocessing import get_feature_names_clean, transform_data
from src.utils import load_artifact, load_json

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished portfolio styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .recommendation-card {
        border-left: 4px solid #2563EB;
        background-color: #F8FAFC;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained machine learning models...")
def load_model_resources():
    """Load serialized model, preprocessor, metadata, and SHAP explainer."""
    models_dir = REPO_ROOT / "models"
    preprocessor_path = models_dir / "preprocessor.pkl"
    model_path = models_dir / "churn_model.pkl"
    metadata_path = models_dir / "model_metadata.json"

    if not preprocessor_path.exists() or not model_path.exists() or not metadata_path.exists():
        return None, None, None, None, None

    preprocessor = load_artifact(preprocessor_path)
    model = load_artifact(model_path)
    metadata = load_json(metadata_path)
    feature_names = get_feature_names_clean(preprocessor)

    # Initialize SHAP Explainer
    raw_csv = REPO_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if raw_csv.exists():
        from src.utils import clean_telco_data, load_raw_data

        sample_df = load_raw_data(raw_csv).sample(300, random_state=42)
        X_sample, _ = clean_telco_data(sample_df)
        X_sample_trans = transform_data(preprocessor, X_sample)
    else:
        X_sample_trans = np.zeros((10, len(feature_names)))

    explainer = create_explainer(model, X_sample_trans, feature_names)

    return preprocessor, model, metadata, feature_names, explainer


def main():
    # Header
    st.markdown('<div class="main-title">📊 Customer Churn Prediction & Retention System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Production-grade ML system predicting individual churn probability, '
        'explaining risk factors via SHAP, and delivering deterministic retention recommendations.</div>',
        unsafe_allow_html=True,
    )

    preprocessor, model, metadata, feature_names, explainer = load_model_resources()

    if model is None:
        st.error(
            "⚠️ Model artifacts not found in `models/` directory! "
            "Please run the training pipeline first: `python src/train.py`"
        )
        st.stop()

    # Sidebar: Model Performance and Info
    with st.sidebar:
        st.header("📌 Model Performance")
        st.markdown(f"**Architecture:** `{metadata.get('model_name', 'XGBoost')}`")
        st.markdown(f"**Imbalance Handling:** `{metadata.get('imbalance_strategy', 'class_weight')}`")

        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", f"{metadata.get('roc_auc', 0.845):.3f}")
        c2.metric("Recall (Churn)", f"{metadata.get('recall', 0.757):.1%}")

        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{metadata.get('precision', 0.551):.1%}")
        c4.metric("F1-Score", f"{metadata.get('f1', 0.637):.3f}")

        threshold = float(metadata.get("decision_threshold", 0.56))
        st.info(f"🎯 **Decision Threshold:** `{threshold:.2f}`\n\nOptimized to balance churn retention recall with precision.")

        st.divider()
        st.subheader("📚 Global Feature Drivers")
        summary_img_path = REPO_ROOT / "app" / "assets" / "shap_summary.png"
        if summary_img_path.exists():
            st.image(str(summary_img_path), caption="Global SHAP Feature Summary", use_container_width=True)

        st.divider()
        st.markdown(
            "🔗 **Repository:** [GitHub](https://github.com/madrimriganka-maker/Churn-Prediction-App)\n\n"
            "**Dataset:** Telco Customer Churn (7,043 records)"
        )

    # Preset profiles for quick testing during demonstrations
    st.subheader("👤 Customer Profile Input")

    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns([1, 1, 1, 2])
    if "preset" not in st.session_state:
        st.session_state["preset"] = "custom"

    if preset_col1.button("🔴 High-Risk Profile", use_container_width=True):
        st.session_state["preset"] = "high"
    if preset_col2.button("🟢 Low-Risk Profile", use_container_width=True):
        st.session_state["preset"] = "low"
    if preset_col3.button("🟡 Borderline Profile", use_container_width=True):
        st.session_state["preset"] = "borderline"

    # Set default values based on preset
    if st.session_state["preset"] == "high":
        default_contract = "Month-to-month"
        default_tenure = 2
        default_monthly = 95.0
        default_total = 190.0
        default_internet = "Fiber optic"
        default_payment = "Electronic check"
        default_security = "No"
        default_tech_support = "No"
        default_paperless = "Yes"
    elif st.session_state["preset"] == "low":
        default_contract = "Two year"
        default_tenure = 60
        default_monthly = 24.5
        default_total = 1470.0
        default_internet = "DSL"
        default_payment = "Credit card (automatic)"
        default_security = "Yes"
        default_tech_support = "Yes"
        default_paperless = "No"
    elif st.session_state["preset"] == "borderline":
        default_contract = "Month-to-month"
        default_tenure = 30
        default_monthly = 60.0
        default_total = 1800.0
        default_internet = "DSL"
        default_payment = "Electronic check"
        default_security = "No"
        default_tech_support = "No"
        default_paperless = "Yes"
    else:
        default_contract = "Month-to-month"
        default_tenure = 12
        default_monthly = 70.0
        default_total = 840.0
        default_internet = "Fiber optic"
        default_payment = "Electronic check"
        default_security = "No"
        default_tech_support = "No"
        default_paperless = "Yes"

    # Input Form
    with st.form("customer_input_form"):
        col_acc, col_srv, col_demo = st.columns(3)

        with col_acc:
            st.markdown("#### 📝 Account & Billing")
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(default_contract),
            )
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=default_tenure)
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=15.0,
                max_value=150.0,
                value=float(default_monthly),
                step=1.0,
            )
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=float(default_total),
                step=10.0,
            )
            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
                index=["Yes", "No"].index(default_paperless),
            )
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ].index(default_payment),
            )

        with col_srv:
            st.markdown("#### 🌐 Internet & Services")
            internet_service = st.selectbox(
                "Internet Service",
                ["Fiber optic", "DSL", "No"],
                index=["Fiber optic", "DSL", "No"].index(default_internet),
            )
            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(default_security),
            )
            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(default_tech_support),
            )
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col_demo:
            st.markdown("#### 👥 Demographics & Phone")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        submit_btn = st.form_submit_button("⚡ Predict Churn Probability", use_container_width=True)

    # Collect Input Data into Schema
    customer_dict = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
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
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    # Perform Prediction
    input_df = pd.DataFrame([customer_dict])

    try:
        # Preprocess input using fitted pipeline (zero leakage)
        customer_transformed = transform_data(preprocessor, input_df)

        # Generate predicted probability
        prob = float(model.predict_proba(customer_transformed)[0, 1])
        is_churn = prob >= threshold

        # Determine risk tier based on optimized threshold
        if prob >= threshold:
            risk_tier = "HIGH RISK"
            risk_class = "risk-high"
        elif prob >= (threshold - 0.15):
            risk_tier = "MEDIUM RISK"
            risk_class = "risk-medium"
        else:
            risk_tier = "LOW RISK"
            risk_class = "risk-low"

    except Exception as e:
        st.error(f"Prediction Error: An unexpected error occurred while processing inputs: {str(e)}")
        st.stop()

    st.divider()

    # Section 1: Prediction & Risk Display
    st.subheader("🎯 Prediction Result")
    r_col1, r_col2, r_col3 = st.columns([1.5, 1.5, 2])

    with r_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #64748B;">Predicted Churn Probability</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: {'#DC2626' if is_churn else '#16A34A'};">
                    {prob:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #64748B;">Risk Tier</div>
                <div style="margin-top: 0.5rem;">
                    <span class="{risk_class}">{risk_tier}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #64748B;">Decision Threshold: <b>{threshold:.2f}</b></div>
                <div style="font-size: 0.85rem; color: #475569; margin-top: 0.4rem;">
                    {'⚠️ <b>Action Required:</b> Customer exceeds churn risk threshold. Proactive outreach recommended.' if is_churn else '✅ <b>Account Healthy:</b> Customer risk is below operational intervention threshold.'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(min(max(prob, 0.0), 1.0))

    # Section 2: SHAP Explainability (Waterfall Plot)
    st.divider()
    st.subheader("🔍 Why is this customer predicted at this risk level?")
    st.caption("Local SHAP Explanation reveals how individual customer features push the probability above or below the baseline.")

    try:
        fig_waterfall, top_drivers = explain_single_prediction(
            explainer, customer_transformed, feature_names, top_n=5
        )
        st.pyplot(fig_waterfall, use_container_width=True)
        plt.close(fig_waterfall)
    except Exception as e:
        st.warning(f"Unable to generate local SHAP explanation: {str(e)}")
        top_drivers = []

    # Section 3: Business Recommendations Engine
    st.divider()
    st.subheader("💡 Actionable Business Retention Recommendations")
    st.caption("Deterministic recommendations prioritized based on the customer's strongest empirical SHAP risk drivers.")

    recommendations = generate_business_recommendations(top_drivers, customer_dict)

    rec_cols = st.columns(len(recommendations) if len(recommendations) <= 3 else 3)
    for idx, rec in enumerate(recommendations[:3]):
        col = rec_cols[idx % len(rec_cols)]
        priority_color = "#DC2626" if rec["priority"] == "High" else ("#D97706" if rec["priority"] == "Medium" else "#2563EB")
        col.markdown(
            f"""
            <div class="recommendation-card">
                <div style="font-size: 0.75rem; font-weight: 700; color: {priority_color}; text-transform: uppercase;">
                    Priority: {rec['priority']}
                </div>
                <div style="font-weight: 600; font-size: 1rem; color: #0F172A; margin: 0.2rem 0;">
                    {rec['driver']}
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 0.4rem;">
                    {rec['impact']}
                </div>
                <div style="font-size: 0.88rem; color: #1E293B;">
                    👉 <b>Recommended Action:</b> {rec['action']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
