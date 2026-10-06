"""
Unit and integration tests for preprocessing pipeline and data leakage prevention.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_preprocessor,
    fit_transform_train,
    get_feature_names_clean,
    transform_data,
)
from src.utils import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    clean_telco_data,
    get_default_customer,
    load_artifact,
)


@pytest.fixture
def sample_train_df():
    """Create a minimal representative DataFrame for pipeline testing."""
    records = [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85,
            "Churn": "No",
        },
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 34,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 56.95,
            "TotalCharges": 1889.5,
            "Churn": "No",
        },
        {
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.5,
            "TotalCharges": " ",  # Blank string to test robustness
            "Churn": "Yes",
        },
    ]
    return pd.DataFrame(records)


def test_clean_telco_data(sample_train_df):
    """Verify data cleaning correctly separates target and coerces blanks."""
    X, y = clean_telco_data(sample_train_df)

    assert "Churn" not in X.columns
    assert len(y) == len(sample_train_df)
    assert y.iloc[0] == 0
    assert y.iloc[2] == 1
    assert pd.isna(X.loc[2, "TotalCharges"])  # Blank coerced to NaN for imputer


def test_pipeline_transforms_and_imputes(sample_train_df):
    """Verify preprocessor fits, imputes NaNs, and standardizes properly."""
    X, _ = clean_telco_data(sample_train_df)
    preprocessor = build_preprocessor()

    X_trans, feature_names = fit_transform_train(preprocessor, X)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(sample_train_df)
    assert not np.isnan(X_trans).any(), "Transformed matrix contains NaNs!"
    assert len(feature_names) == X_trans.shape[1]


def test_handle_unseen_categorical_levels():
    """Verify preprocessor handles unseen categories without crashing."""
    preprocessor_path = Path("models/preprocessor.pkl")
    if preprocessor_path.exists():
        preprocessor = load_artifact(preprocessor_path)
    else:
        df = pd.DataFrame([get_default_customer()])
        preprocessor = build_preprocessor()
        preprocessor.fit(df)

    # Input customer with novel/unseen categorical values
    unseen_customer = get_default_customer()
    unseen_customer["InternetService"] = "Satellite (Unseen)"
    unseen_customer["Contract"] = "Three-Year-Ultra (Unseen)"
    unseen_customer["PaymentMethod"] = "Crypto (Unseen)"

    df_unseen = pd.DataFrame([unseen_customer])

    # Should transform cleanly without KeyError or ValueError
    transformed = transform_data(preprocessor, df_unseen)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_production_feature_count():
    """Verify trained production preprocessor outputs exact 46 features."""
    preprocessor_path = Path("models/preprocessor.pkl")
    if preprocessor_path.exists():
        preprocessor = load_artifact(preprocessor_path)
        feature_names = get_feature_names_clean(preprocessor)
        assert len(feature_names) == 46
        assert "tenure" in feature_names
        assert "MonthlyCharges" in feature_names
        assert "TotalCharges" in feature_names
