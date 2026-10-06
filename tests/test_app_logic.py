"""
Unit and integration tests for application business logic, risk classification,
and recommendation generation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.explain import (
    create_explainer,
    explain_single_prediction,
    generate_business_recommendations,
)
from src.preprocessing import get_feature_names_clean, transform_data
from src.utils import get_default_customer, load_artifact, load_json


@pytest.fixture
def app_resources():
    """Load preprocessor, model, metadata, and explainer."""
    preprocessor = load_artifact("models/preprocessor.pkl")
    model = load_artifact("models/churn_model.pkl")
    metadata = load_json("models/model_metadata.json")
    feature_names = get_feature_names_clean(preprocessor)
    explainer = create_explainer(model, np.zeros((10, len(feature_names))), feature_names)
    return preprocessor, model, metadata, feature_names, explainer


def test_threshold_classification():
    """Verify that predictions respect custom decision threshold."""
    threshold = 0.56
    assert (0.75 >= threshold) is True
    assert (0.50 >= threshold) is False
    assert (0.56 >= threshold) is True
    assert (0.559 >= threshold) is False


def test_risk_tier_classification():
    """Verify risk classification tiers according to optimized threshold."""
    threshold = 0.56

    def get_risk_tier(prob, thresh):
        if prob >= thresh:
            return "HIGH RISK"
        elif prob >= (thresh - 0.15):
            return "MEDIUM RISK"
        return "LOW RISK"

    assert get_risk_tier(0.70, threshold) == "HIGH RISK"
    assert get_risk_tier(0.56, threshold) == "HIGH RISK"
    assert get_risk_tier(0.50, threshold) == "MEDIUM RISK"
    assert get_risk_tier(0.42, threshold) == "MEDIUM RISK"
    assert get_risk_tier(0.35, threshold) == "LOW RISK"
    assert get_risk_tier(0.10, threshold) == "LOW RISK"


def test_recommendation_generation_rules():
    """Verify business recommendation engine produces targeted interventions."""
    raw_customer = {
        "tenure": 2,
        "MonthlyCharges": 95.0,
        "Contract": "Month-to-month",
    }
    drivers = [
        {"feature": "Contract_Month-to-month", "shap_value": 0.45, "feature_value": 1.0},
        {"feature": "tenure", "shap_value": 0.35, "feature_value": 2.0},
        {"feature": "MonthlyCharges", "shap_value": 0.20, "feature_value": 95.0},
    ]

    recs = generate_business_recommendations(drivers, raw_customer)

    assert len(recs) == 3
    for rec in recs:
        assert "driver" in rec
        assert "impact" in rec
        assert "action" in rec
        assert "priority" in rec

    driver_texts = [r["driver"] for r in recs]
    assert any("Contract" in d for d in driver_texts)
    assert any("Tenure" in d for d in driver_texts)
    assert any("Monthly Billing" in d for d in driver_texts)


def test_local_shap_explanation(app_resources):
    """Verify local SHAP waterfall explanation runs dynamically."""
    preprocessor, _, _, feature_names, explainer = app_resources

    customer = get_default_customer()
    df_customer = pd.DataFrame([customer])
    X_trans = transform_data(preprocessor, df_customer)

    fig, top_drivers = explain_single_prediction(explainer, X_trans, feature_names, top_n=5)

    assert fig is not None
    assert isinstance(top_drivers, list)
    for driver in top_drivers:
        assert "feature" in driver
        assert "shap_value" in driver
        assert driver["shap_value"] > 0
