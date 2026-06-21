"""
Evaluation utilities for imbalanced fraud detection.
Accuracy is misleading on imbalanced data — we focus on PR-AUC, F1, and ROC-AUC.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)
import warnings
warnings.filterwarnings("ignore")


def full_report(y_true, y_pred, y_prob, model_name: str = "Model") -> dict:
    """Print and return all key metrics for fraud detection."""
    metrics = {
        "ROC-AUC":       roc_auc_score(y_true, y_prob),
        "PR-AUC":        average_precision_score(y_true, y_prob),
        "F1 (fraud)":    f1_score(y_true, y_pred),
        "Precision":     precision_score(y_true, y_pred, zero_division=0),
        "Recall":        recall_score(y_true, y_pred),
        "Accuracy":      (y_true == y_pred).mean(),
    }
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Legit', 'Fraud'])}")
    return metrics


def find_optimal_threshold(y_true, y_prob, metric: str = "f1") -> float:
    """Find decision threshold that maximises F1 on the fraud class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precisions + recalls) == 0, 0,
        2 * precisions * recalls / (precisions + recalls + 1e-9)
    )
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def plot_comparison(results: dict, save_path: str = None):
    """Bar chart comparing key metrics across models."""
    metrics = ["ROC-AUC", "PR-AUC", "F1 (fraud)", "Precision", "Recall"]
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
    for i, (name, res) in enumerate(results.items()):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Fraud Detection — Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pr_roc_curves(curve_data: dict, save_path: str = None):
    """Side-by-side PR and ROC curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

    for i, (name, (y_true, y_prob)) in enumerate(curve_data.items()):
        c = colors[i % len(colors)]
        # PR Curve
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        axes[0].plot(rec, prec, color=c, lw=2, label=f"{name} (AP={pr_auc:.3f})")
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        axes[1].plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    axes[0].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[1].set_title("ROC Curve", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.suptitle("Model Evaluation — Fraud Detection", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrices(cm_data: dict, save_path: str = None):
    """Confusion matrices for all models."""
    n = len(cm_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    import itertools
    for ax, (name, (y_true, y_pred)) in zip(axes, cm_data.items()):
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legit", "Fraud"]); ax.set_yticklabels(["Legit", "Fraud"])
        thresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], ","),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
