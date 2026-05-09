from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test, y_test, output_dir: Optional[str] = None) -> Dict[str, Any]:
    y_true = np.asarray(y_test).astype(int)
    y_proba = np.asarray(model.predict_proba(X_test))[:, 1]
    y_pred = (y_proba >= 0.4).astype(int)

    auroc = float(roc_auc_score(y_true, y_proba))
    f1 = float(f1_score(y_true, y_pred))
    sensitivity = float(recall_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(f"AUROC: {auroc:.4f}")
    print(f"F1 (thr=0.4): {f1:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("Confusion matrix (tn, fp, fn, tp):", (tn, fp, fn, tp))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _plot_roc_curve(y_true, y_proba, os.path.join(output_dir, "roc_curve.png"))
        _plot_confusion_matrix((tn, fp, fn, tp), os.path.join(output_dir, "confusion_matrix.png"))

    return {
        "auroc": auroc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, path: str) -> None:
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC={auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_confusion_matrix(cm_tuple, path: str) -> None:
    import matplotlib.pyplot as plt

    tn, fp, fn, tp = cm_tuple
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
