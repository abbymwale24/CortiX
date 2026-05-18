"""
CortiX Module 3 — Model Evaluation and Metrics Reports

Calculates exact metrics: overall Accuracy, False Positive Rate (FPR), 
False Negative Rate (FNR), F1-Score, and generates labelled confusion matrices.
"""

import logging
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from cortix.config import config
from cortix.classifier.model import CortixLSTMCNN

logger = logging.getLogger("cortix.classifier.evaluate")


def evaluate_loader(
    model: torch.nn.Module,
    data_loader,
    classes: list[str],
    device: str = "cpu",
) -> dict:
    """
    Run full model evaluation on a validation or test DataLoader.
    """
    model.eval()
    
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            probs = torch.softmax(logits, dim=-1)

            _, predicted = logits.max(1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # 1. Confusion Matrix Elements
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    # Calculate False Positive Rate (FPR) and False Negative Rate (FNR) per class
    # For a given class i:
    # TP = cm[i, i]
    # FN = sum(cm[i, :]) - TP
    # FP = sum(cm[:, i]) - TP
    # TN = sum(sum(cm)) - TP - FN - FP
    
    fpr_per_class = {}
    fnr_per_class = {}
    
    for i, class_name in enumerate(classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp
        
        fpr = fp / (fp + tn + 1e-8)
        fnr = fn / (fn + tp + 1e-8)
        
        fpr_per_class[class_name] = float(fpr)
        fnr_per_class[class_name] = float(fnr)

    # 2. Overall Accuracy
    accuracy = float(np.sum(y_true == y_pred) / len(y_true))
    
    # 3. Micro FPR / FNR (aggregated)
    # Define BENIGN as index 0 (assumes first class)
    # Any attack classed as benign = False Negative
    # Any benign classed as attack = False Positive
    benign_idx = 0
    
    tp_global = np.sum((y_true > 0) & (y_pred > 0))
    tn_global = np.sum((y_true == benign_idx) & (y_pred == benign_idx))
    fp_global = np.sum((y_true == benign_idx) & (y_pred > benign_idx))
    fn_global = np.sum((y_true > benign_idx) & (y_pred == benign_idx))
    
    fpr_global = fp_global / (fp_global + tn_global + 1e-8)
    fnr_global = fn_global / (fn_global + tp_global + 1e-8)

    report_dict = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )

    results = {
        "accuracy": accuracy,
        "fpr_global": float(fpr_global),
        "fnr_global": float(fnr_global),
        "fpr_per_class": fpr_per_class,
        "fnr_per_class": fnr_per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    logger.info("Evaluation Complete. Global Accuracy: %.2f%% | Global FPR: %.4f%%", accuracy * 100, fpr_global * 100)
    return results
