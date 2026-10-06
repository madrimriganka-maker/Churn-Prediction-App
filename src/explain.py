"""
SHAP Explainability and Deterministic Business Recommendation Engine.
Provides global and local model explanations with genuine feature names,
and maps top risk drivers to actionable retention interventions.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils import LOGGER


def create_explainer(model: Any, X_train_transformed: np.ndarray, feature_names: List[str]) -> Any:
    """
    Initialize an appropriate SHAP explainer for the trained model.
    """
    model_type = type(model).__name__

    if "XGB" in model_type or "Forest" in model_type:
        LOGGER.info(f"Using shap.TreeExplainer for tree-based model: {model_type}")
        explainer = shap.TreeExplainer(model)
    else:
        LOGGER.info(f"Using shap.Explainer for model: {model_type}")
        explainer = shap.Explainer(model, X_train_transformed)

    return explainer


def generate_global_shap_summary(
    explainer: Any,
    X_test_transformed: np.ndarray,
    feature_names: List[str],
    save_path: str | Path,
    max_display: int = 15,
) -> None:
    """
    Generate and save a global SHAP summary plot with human-readable feature names.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    shap_values = explainer(X_test_transformed)

    # For binary classification where shap_values may have shape (N, features, 2)
    if len(shap_values.shape) == 3:
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    # Set feature names on Explanation object
    shap_values_class1.feature_names = feature_names

    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values_class1.values,
        X_test_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("Global Feature Importance (SHAP Summary)", fontsize=13, pad=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info(f"Saved global SHAP summary plot to {save_path.resolve()}")


def explain_single_prediction(
    explainer: Any,
    customer_transformed: np.ndarray,
    feature_names: List[str],
    top_n: int = 5,
) -> Tuple[plt.Figure, List[Dict[str, Any]]]:
    """
    Compute local SHAP explanation for an individual customer profile.

    Returns:
      - Matplotlib Figure of the waterfall plot
      - Ranked list of top positive churn risk drivers (feature name, shap value, feature value)
    """
    # Ensure 2D input
    if customer_transformed.ndim == 1:
        customer_transformed = customer_transformed.reshape(1, -1)

    shap_explanation = explainer(customer_transformed)

    if len(shap_explanation.shape) == 3:
        explanation_single = shap_explanation[0, :, 1]
    else:
        explanation_single = shap_explanation[0]

    explanation_single.feature_names = feature_names

    # Generate Waterfall Plot
    fig = plt.figure(figsize=(9, 5))
    shap.plots.waterfall(explanation_single, max_display=10, show=False)
    plt.title("Customer Churn Risk Breakdown (SHAP Waterfall)", fontsize=12, pad=12, fontweight="bold")
    plt.tight_layout()

    # Extract top positive contributors to churn risk
    shap_vals = explanation_single.values
    data_vals = explanation_single.data

    contributions = []
    for fname, sval, dval in zip(feature_names, shap_vals, data_vals):
        contributions.append(
            {
                "feature": fname,
                "shap_value": float(sval),
                "feature_value": float(dval),
            }
        )

    # Sort descending by SHAP value (positive pushes toward churn)
    contributions.sort(key=lambda x: x["shap_value"], reverse=True)
    top_risk_drivers = [c for c in contributions if c["shap_value"] > 0][:top_n]

    return fig, top_risk_drivers


def generate_business_recommendations(
    top_risk_drivers: List[Dict[str, Any]],
    raw_customer_dict: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Deterministic Business Recommendation Engine.
    Translates top empirical SHAP risk drivers and customer attributes
    into prioritized, actionable customer retention interventions.
    """
    recommendations = []
    seen_categories = set()

    for driver in top_risk_drivers:
        feat = driver["feature"]
        shap_val = driver["shap_value"]

        # 1. Contract risk
        if "Contract" in feat and "Month-to-month" in feat and "contract" not in seen_categories:
            recommendations.append(
                {
                    "driver": "Month-to-month Contract Commitment",
                    "impact": f"Increases churn risk (+{shap_val:.2f} SHAP contribution)",
                    "action": "Offer a targeted 15% discount for upgrading to a 1-year or 2-year contract.",
                    "priority": "High",
                }
            )
            seen_categories.add("contract")

        # 2. Tenure risk
        elif "tenure" == feat and "tenure" not in seen_categories:
            tenure_val = raw_customer_dict.get("tenure", 0)
            if tenure_val < 12:
                recommendations.append(
                    {
                        "driver": f"Early Customer Lifecycle (Tenure: {tenure_val} months)",
                        "impact": f"Increases churn risk (+{shap_val:.2f} SHAP contribution)",
                        "action": "Enroll in Customer Success onboarding check-in and 90-day satisfaction warranty.",
                        "priority": "High",
                    }
                )
                seen_categories.add("tenure")

        # 3. Monthly charges
        elif "MonthlyCharges" == feat and "charges" not in seen_categories:
            charges_val = raw_customer_dict.get("MonthlyCharges", 0.0)
            recommendations.append(
                {
                    "driver": f"Elevated Monthly Billing (${charges_val:.2f}/mo)",
                    "impact": f"Increases churn risk (+{shap_val:.2f} SHAP contribution)",
                    "action": "Review service plan; offer bundle discount or adjust unnecessary add-on subscriptions.",
                    "priority": "Medium",
                }
            )
            seen_categories.add("charges")

        # 4. Internet Service (Fiber optic)
        elif "InternetService_Fiber optic" in feat and "internet" not in seen_categories:
            recommendations.append(
                {
                    "driver": "Fiber Optic Service Plan",
                    "impact": f"Increases churn risk (+{shap_val:.2f} SHAP contribution)",
                    "action": "Proactively audit line connection stability and verify speed/router satisfaction.",
                    "priority": "Medium",
                }
            )
            seen_categories.add("internet")

        # 5. Payment Method (Electronic Check)
        elif "PaymentMethod_Electronic check" in feat and "payment" not in seen_categories:
            recommendations.append(
                {
                    "driver": "Electronic Check Payment Method",
                    "impact": f"Increases churn risk (+{shap_val:.2f} SHAP contribution)",
                    "action": "Incentivize enrollment in Credit Card or Bank Auto-Pay with a one-time $10 bill credit.",
                    "priority": "Low",
                }
            )
            seen_categories.add("payment")

        # 6. Online Security / Tech Support missing
        elif "TechSupport_No" in feat and "tech_support" not in seen_categories:
            recommendations.append(
                {
                    "driver": "No Dedicated Technical Support",
                    "impact": f"Increases churn risk (+{shap_val:.2f} SHAP contribution)",
                    "action": "Provide 3 months complimentary VIP Tech Support to increase customer stickiness.",
                    "priority": "Medium",
                }
            )
            seen_categories.add("tech_support")

        elif "OnlineSecurity_No" in feat and "security" not in seen_categories:
            recommendations.append(
                {
                    "driver": "No Online Security Protection",
                    "impact": f"Increases churn risk (+{shap_val:.2f} SHAP contribution)",
                    "action": "Offer complimentary 6-month trial of Online Security and automated antivirus suite.",
                    "priority": "Low",
                }
            )
            seen_categories.add("security")

    # Fallback default if no specific rule matched
    if not recommendations:
        recommendations.append(
            {
                "driver": "General Account Health",
                "impact": "Predicted churn risk exceeds baseline threshold.",
                "action": "Schedule a courtesy satisfaction inquiry to evaluate service quality and identify latent dissatisfaction.",
                "priority": "Standard",
            }
        )

    return recommendations
