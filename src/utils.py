"""
Utility functions for data loading, schema management, artifact serialization, and logging.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd


def get_logger(name: str = "churn_app") -> logging.Logger:
    """Create a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


LOGGER = get_logger()

# Canonical feature schema
NUMERICAL_FEATURES: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load raw Telco churn dataset from CSV.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")
    df = pd.read_csv(path)
    LOGGER.info(f"Loaded dataset from {path} with shape {df.shape}")
    return df


def clean_telco_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean Telco churn dataset:
    - Drop customerID if present
    - Convert TotalCharges safely to numeric
    - Convert Churn target to binary (Yes: 1, No: 0)
    - Returns feature matrix X and target series y
    """
    df_clean = df.copy()

    # Drop identifier
    if ID_COLUMN in df_clean.columns:
        df_clean = df_clean.drop(columns=[ID_COLUMN])

    # Convert TotalCharges safely (blanks become NaN, which the pipeline imputes)
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    # Map target if present
    if TARGET_COLUMN in df_clean.columns:
        y = (df_clean[TARGET_COLUMN].astype(str).str.strip().str.lower() == "yes").astype(int)
        X = df_clean.drop(columns=[TARGET_COLUMN])
    else:
        y = None
        X = df_clean

    return X, y


def save_artifact(obj: Any, filepath: str | Path) -> None:
    """Serialize an object with joblib."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    LOGGER.info(f"Saved artifact to {path.resolve()}")


def load_artifact(filepath: str | Path) -> Any:
    """Load a serialized object with joblib."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path.resolve()}")
    return joblib.load(path)


def save_json(data: Dict[str, Any], filepath: str | Path) -> None:
    """Save dictionary to a formatted JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    LOGGER.info(f"Saved JSON metadata to {path.resolve()}")


def load_json(filepath: str | Path) -> Dict[str, Any]:
    """Load dictionary from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON metadata file not found at {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_default_customer() -> Dict[str, Any]:
    """
    Return baseline default values representing a typical customer.
    Used by the Streamlit application for any features not explicitly surfaced.
    """
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }
