"""
Production preprocessing pipeline for Customer Churn Prediction.
Encapsulates missing value imputation, scaling, and one-hot encoding
using scikit-learn Pipeline and ColumnTransformer to strictly prevent data leakage.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, LOGGER


def build_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Construct an unfitted scikit-learn ColumnTransformer pipeline.

    Numerical Pipeline:
      - SimpleImputer(strategy='median') -> Handles blank or missing values (e.g. TotalCharges)
      - StandardScaler() -> Standardizes continuous scales

    Categorical Pipeline:
      - OneHotEncoder(handle_unknown='ignore', sparse_output=False) -> One-hot encodes categories
        without prior manual dummy creation, guaranteeing zero leakage and robust handling
        of unseen categories at inference time.
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    LOGGER.info(
        f"Built ColumnTransformer with {len(numerical_features)} numerical features "
        f"and {len(categorical_features)} categorical features."
    )
    return preprocessor


def get_feature_names_clean(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract human-readable feature names from a fitted ColumnTransformer.
    Removes transformer prefixes so names are clean for SHAP explainability
    (e.g., 'Contract_Month-to-month', 'MonthlyCharges', 'tenure').
    """
    if not hasattr(preprocessor, "get_feature_names_out"):
        raise AttributeError("Preprocessor must be fitted before extracting feature names.")

    raw_names = preprocessor.get_feature_names_out()
    clean_names = []
    for name in raw_names:
        # Strip prefixes like 'num__' or 'cat__' if present
        clean_name = name.replace("num__", "").replace("cat__", "")
        clean_names.append(clean_name)

    return clean_names


def fit_transform_train(
    preprocessor: ColumnTransformer, X_train: pd.DataFrame
) -> Tuple[np.ndarray, List[str]]:
    """
    Fit preprocessor ONLY on the training split, and transform X_train.
    Returns transformed array and clean feature names.
    """
    X_train_transformed = preprocessor.fit_transform(X_train)
    feature_names = get_feature_names_clean(preprocessor)
    LOGGER.info(
        f"Fitted preprocessor on training data. Transformed shape: {X_train_transformed.shape} "
        f"across {len(feature_names)} features."
    )
    return X_train_transformed, feature_names


def transform_data(
    preprocessor: ColumnTransformer, X: pd.DataFrame
) -> np.ndarray:
    """
    Transform inference/test data using an already-fitted preprocessor.
    Ensures zero fitting occurs on test or runtime evaluation data.
    """
    return preprocessor.transform(X)
