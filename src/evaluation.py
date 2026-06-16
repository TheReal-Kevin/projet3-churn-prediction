"""Model evaluation helpers: metrics, confusion matrix, ROC curve, feature importance."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)


def print_metrics(y_true, y_pred, model_name: str = "Model") -> None:
    """Print classification report."""
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["No Churn", "Churn"]))


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model", ax=None) -> None:
    """Plot a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    (ax.set_title if ax else plt.title)(f"Confusion Matrix — {model_name}")


def plot_roc_curves(models: dict, X_test, y_test) -> None:
    """
    Plot ROC curves for multiple models on the same axes.
    models: {name: fitted_estimator}
    """
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 15) -> None:
    """Plot top N feature importances for tree-based models."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        palette="Blues_r",
    )
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Build a comparison DataFrame.
    results: {model_name: {"accuracy": ..., "precision": ..., "recall": ..., "f1": ..., "auc": ...}}
    """
    return pd.DataFrame(results).T.round(4)
