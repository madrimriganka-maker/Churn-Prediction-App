"""
Unit and integration tests for trained model loading, probability inference, and metadata validity.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import transform_data
from src.utils import get_default_customer, load_artifact, load_json


@pytest.fixture
def model_artifacts():
    """Load model, preprocessor, and metadata."""
    preprocessor = load_artifact("models/preprocessor.pkl")
    model = load_artifact("models/churn_model.pkl")
    metadata = load_json("models/model_metadata.json")
    return preprocessor, model, metadata


def test_artifacts_exist():
    """Verify serialized artifacts exist in models directory."""
    assert Path("models/preprocessor.pkl").exists()
    assert Path("models/churn_model.pkl").exists()
    assert Path("models/model_metadata.json").exists()


def test_model_prediction_range(model_artifacts):
    """Verify model outputs probabilities bounded strictly between 0 and 1."""
    preprocessor, model, _ = model_artifacts

    customer = get_default_customer()
    df_customer = pd.DataFrame([customer])
    X_trans = transform_data(preprocessor, df_customer)

    probs = model.predict_proba(X_trans)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0
    assert np.isclose(probs[0, 0] + probs[0, 1], 1.0)


def test_model_metadata_validity(model_artifacts):
    """Verify metadata contains required fields and valid performance metrics."""
    _, _, metadata = model_artifacts

    required_keys = [
        "model_name",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "decision_threshold",
        "imbalance_strategy",
        "feature_count",
    ]
    for key in required_keys:
        assert key in metadata, f"Missing required key: {key}"

    assert 0.75 <= metadata["roc_auc"] <= 1.0, f"ROC-AUC {metadata['roc_auc']} outside expected range."
    assert 0.50 <= metadata["recall"] <= 1.0, f"Recall {metadata['recall']} outside expected range."
    assert 0.10 <= metadata["decision_threshold"] <= 0.90, "Decision threshold out of bounds."
    assert metadata["feature_count"] == 46


def test_batch_prediction_consistency(model_artifacts):
    """Verify model can handle batch predictions consistently."""
    preprocessor, model, _ = model_artifacts

    batch = [get_default_customer(), get_default_customer()]
    batch[1]["tenure"] = 65
    batch[1]["Contract"] = "Two year"
    df_batch = pd.DataFrame(batch)

    X_trans = transform_data(preprocessor, df_batch)
    probs = model.predict_proba(X_trans)

    assert probs.shape == (2, 2)
    # Long tenure two-year customer should have lower churn probability
    assert probs[1, 1] < probs[0, 1]
