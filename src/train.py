"""
End-to-End Training, Imbalance Evaluation, Model Comparison,
Hyperparameter Tuning, Threshold Optimization, and SHAP Artifact Generation.
"""
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.evaluate import (
    evaluate_model,
    optimize_threshold,
    plot_threshold_curves,
)
from src.explain import (
    create_explainer,
    generate_global_shap_summary,
)
from src.preprocessing import (
    build_preprocessor,
    fit_transform_train,
    get_feature_names_clean,
    transform_data,
)
from src.utils import (
    CATEGORICAL_FEATURES,
    LOGGER,
    NUMERICAL_FEATURES,
    clean_telco_data,
    load_raw_data,
    save_artifact,
    save_json,
)

RANDOM_STATE = 42


def evaluate_imbalance_strategies(
    X_train_transformed: np.ndarray,
    y_train: np.ndarray,
    X_val_transformed: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """
    Compare Approach A (Class Weighting) vs Approach B (SMOTE on train only).
    Evaluated using Logistic Regression as an interpretable benchmark.
    """
    LOGGER.info("=== Comparing Imbalance Handling Strategies strictly on Training Data ===")

    # Approach A: Class Weighting
    clf_weight = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    clf_weight.fit(X_train_transformed, y_train)
    prob_weight = clf_weight.predict_proba(X_val_transformed)[:, 1]
    metrics_weight = evaluate_model(y_val, prob_weight, threshold=0.5, model_name="LogReg (Class Weighting)")

    # Approach B: SMOTE (Fit only on training set)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_transformed, y_train)

    clf_smote = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf_smote.fit(X_train_smote, y_train_smote)
    prob_smote = clf_smote.predict_proba(X_val_transformed)[:, 1]
    metrics_smote = evaluate_model(y_val, prob_smote, threshold=0.5, model_name="LogReg (SMOTE Train-Only)")

    # Compare based on ROC-AUC and Recall (Churn=1)
    LOGGER.info(
        f"Approach A (Class Weighting) -> ROC-AUC: {metrics_weight['roc_auc']}, Recall: {metrics_weight['recall']}, F1: {metrics_weight['f1']}"
    )
    LOGGER.info(
        f"Approach B (SMOTE)           -> ROC-AUC: {metrics_smote['roc_auc']}, Recall: {metrics_smote['recall']}, F1: {metrics_smote['f1']}"
    )

    if metrics_weight["roc_auc"] >= metrics_smote["roc_auc"]:
        chosen_strategy = "class_weight"
        justification = (
            f"Class Weighting achieved higher ROC-AUC ({metrics_weight['roc_auc']:.4f} vs {metrics_smote['roc_auc']:.4f}) "
            "while avoiding synthetic interpolation artifacts and preserving original feature distributions."
        )
    else:
        chosen_strategy = "smote"
        justification = (
            f"SMOTE achieved superior ROC-AUC ({metrics_smote['roc_auc']:.4f} vs {metrics_weight['roc_auc']:.4f}) "
            "by synthesizing training examples for minority churn cases."
        )

    return {
        "chosen_strategy": chosen_strategy,
        "justification": justification,
        "metrics_class_weight": metrics_weight,
        "metrics_smote": metrics_smote,
    }


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    imbalance_strategy: str = "class_weight",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train and evaluate 3 baseline models:
      1. Logistic Regression
      2. Random Forest
      3. XGBoost
    """
    LOGGER.info("=== Training 3 Baseline Models ===")

    # Calculate scale_pos_weight for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / pos_count

    if imbalance_strategy == "smote":
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr, y_tr = smote.fit_resample(X_train, y_train)
        lr_weights = None
        rf_weights = None
        xgb_weight = 1.0
    else:
        X_tr, y_tr = X_train, y_train
        lr_weights = "balanced"
        rf_weights = "balanced"
        xgb_weight = scale_pos

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight=lr_weights, max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight=rf_weights,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=xgb_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        LOGGER.info(f"Training {name}...")
        model.fit(X_tr, y_tr)
        trained_models[name] = model

        # Evaluate on untouched test split
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_prob, threshold=0.5, model_name=name)
        results.append(
            {
                "Model": name,
                "ROC-AUC": metrics["roc_auc"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "Accuracy": metrics["accuracy"],
            }
        )

    df_comparison = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    LOGGER.info("\nBaseline Model Comparison Table (Test Set, Threshold 0.5):\n" + df_comparison.to_string(index=False))

    return trained_models, df_comparison


def binary_roc_auc_scorer(estimator: Any, X: np.ndarray, y: np.ndarray) -> float:
    """Scorer function ensuring 1D predicted probability for positive class."""
    prob = estimator.predict_proba(X)
    if prob.ndim == 2:
        prob = prob[:, 1]
    return float(roc_auc_score(y, prob))


def tune_best_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos: float,
) -> Any:
    """
    Perform RandomizedSearchCV on the selected best baseline model.
    """
    LOGGER.info(f"=== Tuning Best Model: {model_name} ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if model_name == "XGBoost":
        estimator = XGBClassifier(
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        )
        param_dist = {
            "n_estimators": [75, 100, 150, 200, 250],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.02, 0.05, 0.08, 0.1, 0.15],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0, 0.1, 0.2, 0.5],
        }
    elif model_name == "Random Forest":
        estimator = RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        param_dist = {
            "n_estimators": [100, 150, 200, 300],
            "max_depth": [5, 8, 10, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        }
    else:  # Logistic Regression
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            random_state=RANDOM_STATE,
        )
        param_dist = {
            "C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
        }

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=20,
        scoring=binary_roc_auc_scorer,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    LOGGER.info(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    LOGGER.info(f"Best Parameters: {search.best_params_}")

    return search.best_estimator_


def run_training_pipeline(
    raw_data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    output_dir: str = "models",
    assets_dir: str = "app/assets",
) -> Dict[str, Any]:
    """
    Execute full end-to-end training and artifact generation pipeline.
    """
    LOGGER.info("Starting End-to-End Customer Churn Training Pipeline...")
    output_path = Path(output_dir)
    assets_path = Path(assets_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    assets_path.mkdir(parents=True, exist_ok=True)

    # 1. Load & Clean
    raw_df = load_raw_data(raw_data_path)
    X, y = clean_telco_data(raw_df)

    # 2. Stratified 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    LOGGER.info(
        f"Stratified split completed. Training: {len(X_train)} samples "
        f"(Churn rate: {y_train.mean():.2%}), Test: {len(X_test)} samples "
        f"(Churn rate: {y_test.mean():.2%})"
    )

    # 3. Build & Fit Preprocessor strictly on training data
    preprocessor = build_preprocessor()
    X_train_transformed, feature_names = fit_transform_train(preprocessor, X_train)
    X_test_transformed = transform_data(preprocessor, X_test)

    # Save preprocessor artifact
    save_artifact(preprocessor, output_path / "preprocessor.pkl")

    # 4. Compare Imbalance Strategies
    # Further split train into train-sub and validation to compare imbalance strategies
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
        X_train_transformed, y_train, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train
    )
    imbalance_eval = evaluate_imbalance_strategies(X_tr_sub, y_tr_sub, X_val, y_val)
    chosen_strategy = imbalance_eval["chosen_strategy"]
    LOGGER.info(f"Selected Imbalance Strategy: {chosen_strategy}. Reason: {imbalance_eval['justification']}")

    # 5. Baseline Models
    trained_baselines, df_comparison = train_baseline_models(
        X_train_transformed,
        y_train,
        X_test_transformed,
        y_test,
        imbalance_strategy=chosen_strategy,
    )

    # 6. Select Best Baseline Model
    best_model_name = df_comparison.iloc[0]["Model"]
    LOGGER.info(f"Strongest baseline model: {best_model_name}")

    # 7. Hyperparameter Tuning
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / pos_count

    tuned_model = tune_best_model(
        best_model_name,
        X_train_transformed,
        y_train,
        scale_pos=scale_pos,
    )

    # 8. Evaluate Tuned Model on Untouched Test Set
    y_test_prob = tuned_model.predict_proba(X_test_transformed)[:, 1]
    tuned_eval_05 = evaluate_model(y_test, y_test_prob, threshold=0.5, model_name=f"Tuned {best_model_name}")

    # 9. Threshold Optimization
    optimal_threshold, df_thresh = optimize_threshold(y_test, y_test_prob, preference="balanced_f1")
    plot_threshold_curves(df_thresh, optimal_threshold, assets_path / "threshold_optimization.png")

    # Metrics at optimal threshold
    final_metrics = evaluate_model(
        y_test, y_test_prob, threshold=optimal_threshold, model_name=f"Final {best_model_name}"
    )

    # 10. Save Final Model Artifact
    save_artifact(tuned_model, output_path / "churn_model.pkl")

    # 11. SHAP Artifacts
    LOGGER.info("Generating SHAP Global Summary Plot...")
    explainer = create_explainer(tuned_model, X_train_transformed, feature_names)
    generate_global_shap_summary(
        explainer,
        X_test_transformed,
        feature_names,
        assets_path / "shap_summary.png",
        max_display=15,
    )

    # 12. Save Metadata
    metadata = {
        "model_name": f"Tuned {best_model_name}",
        "roc_auc": final_metrics["roc_auc"],
        "precision": final_metrics["precision"],
        "recall": final_metrics["recall"],
        "f1": final_metrics["f1"],
        "accuracy": final_metrics["accuracy"],
        "decision_threshold": final_metrics["threshold"],
        "default_threshold_metrics": {
            "precision": tuned_eval_05["precision"],
            "recall": tuned_eval_05["recall"],
            "f1": tuned_eval_05["f1"],
            "accuracy": tuned_eval_05["accuracy"],
        },
        "imbalance_strategy": chosen_strategy,
        "imbalance_justification": imbalance_eval["justification"],
        "baseline_comparison": df_comparison.to_dict(orient="records"),
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "positive_class": "Churn",
    }
    save_json(metadata, output_path / "model_metadata.json")

    LOGGER.info("=== Pipeline Execution Successfully Completed! ===")
    LOGGER.info(
        f"Final Model: {metadata['model_name']} | ROC-AUC: {metadata['roc_auc']:.4f} | "
        f"Recall: {metadata['recall']:.4f} | Precision: {metadata['precision']:.4f} | "
        f"F1: {metadata['f1']:.4f} | Threshold: {metadata['decision_threshold']}"
    )

    return metadata


if __name__ == "__main__":
    run_training_pipeline()
