"""
metrics.py - Evaluation Metrics for Traffic Flow & Congestion Prediction
Computes both global performance (MAE, RMSE, MAPE) and specialized metrics
for the imbalanced congestion tail (Severe Congestion MAE, Peak Hour Error, and Detection F1).
"""

import numpy as np
import torch

def compute_regression_metrics(y_true, y_pred):
    """
    Computes MAE, RMSE, and MAPE across all nodes and horizons.
    y_true, y_pred: numpy arrays of shape (B, T_out, N) or flattened.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    # Global MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # Global RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Global MAPE (avoid division by near-zero)
    mask = y_true > 1.0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    else:
        mape = 0.0

    return {
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "MAPE": round(float(mape), 2)
    }

def compute_imbalance_regime_metrics(y_true, y_pred, severe_threshold=30.0, moderate_threshold=50.0):
    """
    Evaluates how the model performs across imbalanced traffic regimes:
    1. Free-flow (>= 50 mph) - Majority class
    2. Moderate Slowdown (30 - 50 mph)
    3. Severe Congestion / Bottleneck (< 30 mph) - Safety-critical minority class
    """
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()

    # Free flow mask
    ff_mask = y_true >= moderate_threshold
    mod_mask = (y_true >= severe_threshold) & (y_true < moderate_threshold)
    sev_mask = y_true < severe_threshold

    def get_regime_stats(mask):
        if np.sum(mask) == 0:
            return {"count": 0, "MAE": 0.0, "RMSE": 0.0}
        reg_mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
        reg_rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
        return {
            "count": int(np.sum(mask)),
            "pct_of_data": round(float(np.mean(mask) * 100), 2),
            "MAE": round(reg_mae, 3),
            "RMSE": round(reg_rmse, 3)
        }

    # Congestion Event Classification Metrics (Binary: Is it severely congested? < 30 mph)
    true_event = (y_true < severe_threshold).astype(int)
    pred_event = (y_pred < severe_threshold).astype(int)

    tp = np.sum((true_event == 1) & (pred_event == 1))
    fp = np.sum((true_event == 0) & (pred_event == 1))
    fn = np.sum((true_event == 1) & (pred_event == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        "free_flow_regime": get_regime_stats(ff_mask),
        "moderate_regime": get_regime_stats(mod_mask),
        "severe_congestion_regime": get_regime_stats(sev_mask),
        "bottleneck_detection": {
            "Precision": round(float(precision), 3),
            "Recall": round(float(recall), 3),
            "F1_Score": round(float(f1), 3)
        }
    }
