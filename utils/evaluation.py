from typing import Any, Dict, Tuple
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

def _unpack_eval_pred(eval_pred: Any) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        preds, labels = eval_pred
    return np.asarray(preds), np.asarray(labels)

def _softmax_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)

def _to_pos_scores(preds: np.ndarray) -> np.ndarray:

    if preds.ndim == 2 and preds.shape[1] == 2:
        row_sums = preds.sum(axis=1)
        if np.all(preds >= 0) and np.all(preds <= 1) and np.all(np.abs(row_sums - 1) < 1e-3):
            return preds[:, 1].astype(np.float64)
        probs = _softmax_2d(preds.astype(np.float64))
        return probs[:, 1]
    if preds.ndim == 1:
        return preds.astype(np.float64)
    raise ValueError(f"Unsupported preds shape: {preds.shape}")

def _to_hard_labels(preds: np.ndarray) -> np.ndarray:
    if preds.ndim == 2:
        return preds.argmax(axis=-1).astype(np.int64)
    
    if np.issubdtype(preds.dtype, np.floating):
        return (preds >= 0.5).astype(np.int64)
    return preds.astype(np.int64)

def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    preds, labels = _unpack_eval_pred(eval_pred)
    y_true = labels.astype(np.int64)
    y_true = np.clip(y_true, 0, 1)


    p_pos = _to_pos_scores(preds)
  
    y_pred = _to_hard_labels(preds)
    y_pred = np.clip(y_pred, 0, 1)

    out: Dict[str, float] = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))


    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    out["precision"] = float(p)
    out["recall"] = float(r)
    out["f1"] = float(f1)


    pM, rM, f1M, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    out["precision_macro"] = float(pM)
    out["recall_macro"] = float(rM)
    out["f1_macro"] = float(f1M)


    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        out["tn"] = float(tn)
        out["fp"] = float(fp)
        out["fn"] = float(fn)
        out["tp"] = float(tp)
    else:
        out["tn"] = out["fp"] = out["fn"] = out["tp"] = 0.0

  
    try:
        out["auc_roc"] = float(roc_auc_score(y_true, p_pos))
    except Exception:
        out["auc_roc"] = float("nan")

    try:
        out["auc_pr"] = float(average_precision_score(y_true, p_pos))
    except Exception:
        out["auc_pr"] = float("nan")

    return out