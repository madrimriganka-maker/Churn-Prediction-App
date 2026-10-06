"""
Model evaluation, performance metrics computation, and decision threshold optimization.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import LOGGER


def evaluate_model(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Evaluate classification performance given true labels and predicted probabilities.

    Metrics computed:
      - ROC-AUC
      - Precision, Recall, F1, Accuracy at given threshold
      - Confusion matrix (TN, FP, FN, TP)
      - Class-specific metrics (Class 0: Retained, Class 1: Churned)
    """
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["Retained (0)", "Churned (1)"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model_name": model_name,
        "threshold": round(threshold, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "class_0_precision": round(report_dict["Retained (0)"]["precision"], 4),
        "class_0_recall": round(report_dict["Retained (0)"]["recall"], 4),
        "class_0_f1": round(report_dict["Retained (0)"]["f1-score"], 4),
        "class_1_precision": round(report_dict["Churned (1)"]["precision"], 4),
        "class_1_recall": round(report_dict["Churned (1)"]["recall"], 4),
        "class_1_f1": round(report_dict["Churned (1)"]["f1-score"], 4),
    }

    LOGGER.info(
        f"[{model_name} @ threshold={threshold:.2f}] ROC-AUC: {roc_auc:.4f} | "
        f"Recall(Churn=1): {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f} | Acc: {accuracy:.4f}"
    )

    return metrics


def scan_thresholds(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    start: float = 0.10,
    end: float = 0.90,
    step: float = 0.01,
) -> pd.DataFrame:
    """
    Evaluate precision, recall, f1, and false negatives across a range of decision thresholds.
    """
    thresholds = np.arange(start, end + 1e-5, step)
    records = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        records.append(
            {
                "threshold": round(thresh, 2),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "accuracy": round(acc, 4),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    return pd.DataFrame(records)


def optimize_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    preference: str = "balanced_f1",
) -> Tuple[float, pd.DataFrame]:
    """
    Identify business-optimal operating threshold.

    Business Rationale:
      - False Negative (FN): A customer actually churns, but model predicted stay.
        Business cost: Loss of Customer Lifetime Value (LTV ~$1,000+).
      - False Positive (FP): A customer would have stayed, but model predicted churn.
        Business cost: Small retention discount or proactive outreach call (~$15-$30).
      - Because FN is significantly more costly than FP, optimizing for Churn Recall
        while maintaining reasonable precision (F1 or balanced trade-off) yields a
        lower operating threshold than the default 0.50.
    """
    df_thresh = scan_thresholds(y_true, y_prob)

    if preference == "balanced_f1":
        # Operating point that maximizes F1 score
        best_idx = df_thresh["f1"].idxmax()
        optimal_threshold = float(df_thresh.loc[best_idx, "threshold"])
    elif preference == "high_recall":
        # Target recall >= 0.75 with highest precision
        candidates = df_thresh[df_thresh["recall"] >= 0.75]
        if not candidates.empty:
            best_idx = candidates["precision"].idxmax()
            optimal_threshold = float(candidates.loc[best_idx, "threshold"])
        else:
            best_idx = df_thresh["f1"].idxmax()
            optimal_threshold = float(df_thresh.loc[best_idx, "threshold"])
    else:
        best_idx = df_thresh["f1"].idxmax()
        optimal_threshold = float(df_thresh.loc[best_idx, "threshold"])

    LOGGER.info(
        f"Selected optimal threshold: {optimal_threshold:.2f} "
        f"(Recall: {df_thresh.loc[best_idx, 'recall']:.4f}, "
        f"Precision: {df_thresh.loc[best_idx, 'precision']:.4f}, "
        f"F1: {df_thresh.loc[best_idx, 'f1']:.4f})"
    )

    return optimal_threshold, df_thresh


def plot_threshold_curves(
    df_thresh: pd.DataFrame,
    selected_threshold: float,
    save_path: str | Path,
) -> None:
    """
    Plot Precision and Recall curves across thresholds with the selected operating point.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_thresh["threshold"], df_thresh["precision"], label="Precision", color="#1f77b4", linewidth=2.2)
    ax.plot(df_thresh["threshold"], df_thresh["recall"], label="Recall (Churn=1)", color="#d62728", linewidth=2.2)
    ax.plot(df_thresh["threshold"], df_thresh["f1"], label="F1 Score", color="#2ca02c", linestyle="--", linewidth=1.8)

    ax.axvline(
        selected_threshold,
        color="#7f7f7f",
        linestyle=":",
        linewidth=2,
        label=f"Selected Threshold ({selected_threshold:.2f})",
    )

    ax.set_title("Precision, Recall & F1 vs. Decision Threshold", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="center right", frameon=True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    LOGGER.info(f"Saved threshold optimization plot to {save_path.resolve()}")
