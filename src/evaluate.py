import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from preprocessing import PLOTS_PATH

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def compute_metrics(y_true, y_pred, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "n_test": int(len(y_true)),
        "n_positive": int(sum(y_true)),
        "best_threshold": 0.5,
    }


def compute_threshold_curve(y_true, y_proba):
    precisions, recalls, thresholds = _pr_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def _pr_curve(y_true, y_proba):
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    return precisions, recalls, thresholds


def plot_confusion_matrix(y_true, y_pred, model_name, path=PLOTS_PATH):
    cm = confusion_matrix(y_true, y_pred)
    Path(path).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stay", "Leave"],
        yticklabels=["Stay", "Leave"],
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    out = Path(path) / "confusion_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_roc_curves(results, path=PLOTS_PATH, filename="roc_curves.png"):
    Path(path).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_true, y_proba in results:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = Path(path) / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_model_comparison(results_df, metric="f1", path=PLOTS_PATH):
    Path(path).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    order = results_df.sort_values(metric, ascending=True)
    bars = ax.barh(order["model"], order[metric], color="steelblue")
    for bar, val in zip(bars, order[metric]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )
    ax.set_xlim(0, order[metric].max() * 1.15)
    ax.set_title(f"Model Comparison - {metric.upper()}")
    ax.set_xlabel(metric.upper())
    fig.tight_layout()
    out = Path(path) / "model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_importance(importances, names, path=PLOTS_PATH, top_n=15, title="Top Feature Importance"):
    Path(path).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"], df["importance"], color="darkorange")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    out = Path(path) / "feature_importance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_metrics_json(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path