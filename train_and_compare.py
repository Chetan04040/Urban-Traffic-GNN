"""
train_and_compare.py - End-to-End Training & Comparative Benchmark Pipeline
Trains and compares:
1. Baseline: Pure Temporal LSTM (Spatial Blindness)
2. Base: Standard Spatio-Temporal GCN (MSE Loss)
3. Improved: Attention-Enhanced ST-GCN with Adaptive Topology & Congestion-Focal Loss
Saves metrics, checkpoints, and generated evaluation charts.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.dataset import UrbanTrafficGraphDataset, get_dataloaders
from src.metrics import compute_regression_metrics, compute_imbalance_regime_metrics
from models.baseline_lstm import PureTemporalLSTM
from models.base_stgcn import BaseSTGCN
from models.improved_stgcn import ImprovedSTGCN, CongestionFocalLoss

def train_epoch(model, dataloader, optimizer, criterion, device, adj_matrix=None):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x, adj_matrix)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device, mean_speed, std_speed, adj_matrix=None):
    model.eval()
    total_loss = 0.0
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x, adj_matrix)
            loss = criterion(pred, y)
            total_loss += loss.item() * len(x)

            # Un-normalize speeds back to physical mph
            pred_mph = pred.cpu().numpy() * std_speed + mean_speed
            target_mph = y.cpu().numpy() * std_speed + mean_speed
            preds_all.append(pred_mph)
            targets_all.append(target_mph)

    val_loss = total_loss / len(dataloader.dataset)
    y_pred = np.concatenate(preds_all, axis=0)
    y_true = np.concatenate(targets_all, axis=0)

    reg_metrics = compute_regression_metrics(y_true, y_pred)
    regime_metrics = compute_imbalance_regime_metrics(y_true, y_pred)
    return val_loss, reg_metrics, regime_metrics, y_pred, y_true

def run_experiment(num_days=21, epochs=12, batch_size=32, lr=0.003):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Setup] Using compute device: {device}")

    project_root = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Generate & Save Dataset
    print("\n[Step 1/5] Building Urban Traffic Network Graph & Imbalanced Dataset...")
    dataset = UrbanTrafficGraphDataset(num_nodes=25, num_days=num_days, seed=42)
    dataset.save_to_disk(data_dir)
    stats = dataset.get_imbalance_stats()
    print(f" Dataset Summary: {stats['skew_summary']}")

    train_loader, val_loader, test_loader, mean_arr, std_arr = get_dataloaders(
        dataset.raw_data, in_steps=12, out_steps=3, batch_size=batch_size
    )
    mean_speed = float(mean_arr[0, 0, 0])
    std_speed = float(std_arr[0, 0, 0])
    adj_matrix = dataset.adj_matrix

    models_to_train = {
        "Baseline_LSTM": {
            "model": PureTemporalLSTM(in_features=4, hidden_dim=64, num_layers=2, out_steps=3).to(device),
            "criterion": nn.MSELoss(),
            "use_adj": False,
            "desc": "Pure Temporal LSTM (No spatial awareness)"
        },
        "Base_STGCN": {
            "model": BaseSTGCN(num_nodes=25, in_features=4, hidden_dim=64, out_steps=3).to(device),
            "criterion": nn.MSELoss(),
            "use_adj": True,
            "desc": "Standard ST-GCN (Normalized Laplacian + MSE)"
        },
        "Improved_A_STGCN": {
            "model": ImprovedSTGCN(num_nodes=25, in_features=4, hidden_dim=64, out_steps=3, num_heads=4).to(device),
            "criterion": CongestionFocalLoss(alpha=4.0, gamma=2.0, mean_speed=mean_speed, std_speed=std_speed, free_flow_speed=65.0),
            "use_adj": True,
            "desc": "Improved A-STGCN (Adaptive Topology + Temporal Self-Attention + Focal Loss)"
        }
    }

    comparison_results = {}
    test_predictions = {}
    ground_truth_test = None

    for model_name, cfg in models_to_train.items():
        print(f"\n[Training] >>> {model_name}: {cfg['desc']}")
        model = cfg["model"]
        criterion = cfg["criterion"]
        adj_arg = adj_matrix if cfg["use_adj"] else None
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        start_time = time.time()
        best_val_loss = float("inf")
        best_state = None

        for ep in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, adj_arg)
            val_loss, val_reg, val_regime, _, _ = evaluate(model, val_loader, criterion, device, mean_speed, std_speed, adj_arg)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()

            if ep % 4 == 0 or ep == epochs:
                print(f"  Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val MAE: {val_reg['MAE']} mph | Congestion MAE: {val_regime['severe_congestion_regime']['MAE']} mph")

        elapsed = time.time() - start_time
        # Load best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(models_dir, f"{model_name}.pt"))

        # Final Test Evaluation
        eval_criterion = nn.MSELoss()  # standardized loss for comparison
        test_loss, reg_metrics, regime_metrics, preds, trues = evaluate(
            model, test_loader, eval_criterion, device, mean_speed, std_speed, adj_arg
        )
        ground_truth_test = trues
        test_predictions[model_name] = preds

        comparison_results[model_name] = {
            "description": cfg["desc"],
            "training_time_sec": round(elapsed, 1),
            "test_metrics": reg_metrics,
            "regime_metrics": regime_metrics
        }

    # Save comparative JSON summary
    summary_path = os.path.join(results_dir, "comparative_results.json")
    with open(summary_path, "w") as f:
        json.dump(comparison_results, f, indent=2)

    # Save test sample predictions for visualization
    np.save(os.path.join(results_dir, "ground_truth_test.npy"), ground_truth_test)
    for name, p in test_predictions.items():
        np.save(os.path.join(results_dir, f"{name}_preds.npy"), p)

    # 4. Generate Visual Comparisons & Plots
    print("\n[Step 4/5] Generating Visual Evaluation Figures...")
    generate_comparison_plots(comparison_results, ground_truth_test, test_predictions, results_dir)

    # 5. Display Final Benchmark Table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE (Test Set Across 25 Highway Nodes)")
    print("=" * 80)
    print(f"{'Model Architecture':<22} | {'Overall MAE':<11} | {'RMSE':<9} | {'Severe Jam MAE':<14} | {'Detect Recall':<13} | {'F1 Score':<8}")
    print("-" * 80)
    for name, res in comparison_results.items():
        m = res["test_metrics"]
        s_mae = res["regime_metrics"]["severe_congestion_regime"]["MAE"]
        det = res["regime_metrics"]["bottleneck_detection"]
        print(f"{name:<22} | {m['MAE']:<11} | {m['RMSE']:<9} | {s_mae:<14} | {det['Recall']:<13} | {det['F1_Score']:<8}")
    print("=" * 80)
    print(f"\n[Success] Results saved to:\n  - {summary_path}\n  - {os.path.join(results_dir, 'traffic_comparison.png')}")
    return comparison_results

def generate_comparison_plots(results, ground_truth, predictions, results_dir):
    """Creates a 4-panel publication-grade evaluation figure."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: Bar chart comparing Overall MAE vs Severe Congestion MAE
    ax1 = axes[0, 0]
    models = list(results.keys())
    clean_names = ["Baseline LSTM", "Base ST-GCN", "Improved A-STGCN"]
    overall_maes = [results[m]["test_metrics"]["MAE"] for m in models]
    jam_maes = [results[m]["regime_metrics"]["severe_congestion_regime"]["MAE"] for m in models]

    x_idx = np.arange(len(models))
    width = 0.35
    ax1.bar(x_idx - width/2, overall_maes, width, label="Overall MAE (mph)", color="#3498db")
    ax1.bar(x_idx + width/2, jam_maes, width, label="Severe Congestion MAE (mph)", color="#e74c3c")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(clean_names, fontweight="bold")
    ax1.set_ylabel("Mean Absolute Error (mph)", fontsize=11)
    ax1.set_title("Prediction Error: Overall vs. Critical Bottleneck Jams", fontsize=12, fontweight="bold")
    ax1.legend()

    # Panel 2: Bottleneck Detection Precision & Recall
    ax2 = axes[0, 1]
    recalls = [results[m]["regime_metrics"]["bottleneck_detection"]["Recall"] * 100 for m in models]
    f1s = [results[m]["regime_metrics"]["bottleneck_detection"]["F1_Score"] * 100 for m in models]
    ax2.bar(x_idx - width/2, recalls, width, label="Congestion Recall (%)", color="#2ecc71")
    ax2.bar(x_idx + width/2, f1s, width, label="Congestion F1 (%)", color="#9b59b6")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(clean_names, fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.set_title("Bottleneck Detection Performance (Imbalanced Minority)", fontsize=12, fontweight="bold")
    ax2.legend()

    # Panel 3: Time Series Trajectory on a critical bottleneck node (e.g. Node 12)
    ax3 = axes[1, 0]
    node_id = 12
    horizon_step = 0  # 15 mins ahead
    # View 120 steps = 10 hours
    t_slice = slice(100, 220)
    time_hours = np.linspace(0, 10, 120)

    ax3.plot(time_hours, ground_truth[t_slice, horizon_step, node_id], "k-", lw=2.2, label="Ground Truth (Actual)")
    ax3.plot(time_hours, predictions["Baseline_LSTM"][t_slice, horizon_step, node_id], "b--", lw=1.5, alpha=0.8, label="Baseline LSTM")
    ax3.plot(time_hours, predictions["Base_STGCN"][t_slice, horizon_step, node_id], "g-.", lw=1.5, alpha=0.8, label="Base ST-GCN")
    ax3.plot(time_hours, predictions["Improved_A_STGCN"][t_slice, horizon_step, node_id], "r-", lw=2.0, label="Improved A-STGCN")
    ax3.axhspan(0, 30, color="red", alpha=0.1, label="Severe Congestion Zone (<30 mph)")
    ax3.set_xlabel("Time (Hours)", fontsize=11)
    ax3.set_ylabel("Speed (mph)", fontsize=11)
    ax3.set_title(f"Speed Forecast Trajectory on Bottleneck Sensor Node #{node_id}", fontsize=12, fontweight="bold")
    ax3.legend(loc="lower left", fontsize=9)

    # Panel 4: Error Residual Distribution
    ax4 = axes[1, 1]
    lstm_res = (predictions["Baseline_LSTM"][:, horizon_step, :] - ground_truth[:, horizon_step, :]).flatten()
    improved_res = (predictions["Improved_A_STGCN"][:, horizon_step, :] - ground_truth[:, horizon_step, :]).flatten()
    ax4.hist(lstm_res, bins=40, range=(-25, 25), alpha=0.5, color="blue", label=f"Baseline LSTM (Std: {np.std(lstm_res):.2f})", density=True)
    ax4.hist(improved_res, bins=40, range=(-25, 25), alpha=0.6, color="red", label=f"Improved A-STGCN (Std: {np.std(improved_res):.2f})", density=True)
    ax4.axvline(0, color="black", linestyle="--")
    ax4.set_xlabel("Prediction Error (Predicted - Actual mph)", fontsize=11)
    ax4.set_ylabel("Density", fontsize=11)
    ax4.set_title("Error Residual Distribution (Tighter peak at 0 is better)", fontsize=12, fontweight="bold")
    ax4.legend()

    plt.tight_layout()
    chart_path = os.path.join(results_dir, "traffic_comparison.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved comparative figure to {chart_path}")

if __name__ == "__main__":
    run_experiment(num_days=21, epochs=12, batch_size=32, lr=0.003)
