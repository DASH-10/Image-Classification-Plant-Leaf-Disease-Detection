"""
Bu dosya modelin tahminlerini ve performans metriklerini hesaplar.
Kisaca: modelin ne kadar iyi bildigini burada olcuyoruz.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize


def predict_torch(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Torch modeli ile tahmin yapar; gercek, tahmin ve olasiliklari dondurur."""
    model.eval()
    y_true = []
    y_pred = []
    y_proba = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
            y_proba.extend(probs.detach().cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_proba)


def predict_sklearn(model, X) -> Tuple[np.ndarray, np.ndarray]:
    """Sklearn modeli icin tahmin ve (varsa) skor/olasiliklari alir."""
    y_pred = model.predict(X)
    if hasattr(model, "decision_function"):
        y_proba = model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
    else:
        y_proba = None
    return y_pred, y_proba


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: Optional[List[str]] = None
) -> Dict:
    """Accuracy ve F1 gibi temel metrikleri hesaplar."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "report": classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        ),
    }
    return metrics


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Confusion matrix (karisiklik matrisi) olusturur."""
    return confusion_matrix(y_true, y_pred)


def compute_normalized_confusion(cm: np.ndarray) -> np.ndarray:
    """Confusion matrix'i normalize eder (satir bazinda oran)."""
    with np.errstate(all="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    return cm_norm


def compute_pr_curves(
    y_true: np.ndarray, y_scores: np.ndarray, num_classes: int
) -> Dict[int, Dict[str, np.ndarray]]:
    """Her sinif icin Precision-Recall egri noktalarini hesaplar (score bazli)."""
    curves = {}
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    y_scores = np.asarray(y_scores)
    if y_scores.ndim == 1 and num_classes == 2:
        y_scores = np.column_stack([-y_scores, y_scores])
    elif y_scores.ndim == 1:
        y_scores = y_scores.reshape(-1, 1)
    if y_scores.shape[1] != num_classes:
        raise ValueError("y_scores must have shape (n_samples, num_classes)")
    for class_idx in range(num_classes):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, class_idx], y_scores[:, class_idx]
        )
        curves[class_idx] = {"precision": precision, "recall": recall}
    return curves


def compute_micro_pr_curve(
    y_true: np.ndarray, y_scores: np.ndarray, num_classes: int
) -> Dict[str, np.ndarray]:
    """Tum siniflari birlikte (micro) Precision-Recall egrisi cikarir (score bazli)."""
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    y_scores = np.asarray(y_scores)
    if y_scores.ndim == 1 and num_classes == 2:
        y_scores = np.column_stack([-y_scores, y_scores])
    elif y_scores.ndim == 1:
        y_scores = y_scores.reshape(-1, 1)
    if y_scores.shape[1] != num_classes:
        raise ValueError("y_scores must have shape (n_samples, num_classes)")
    precision, recall, _ = precision_recall_curve(
        y_true_bin.ravel(), y_scores.ravel()
    )
    return {"precision": precision, "recall": recall}
