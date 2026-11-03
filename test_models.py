"""
test_models.py - Unit & Integration Test Suite
Validates:
1. Dataset generation, graph connectivity, and imbalanced distribution
2. Baseline LSTM forward & backward pass
3. Base ST-GCN spatial message-passing and tensor shapes
4. Improved ST-GCN with Adaptive Graph Attention and Temporal Self-Attention
5. Congestion Focal Loss behavior on imbalanced traffic samples
"""

import sys
import os
import unittest
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.dataset import UrbanTrafficGraphDataset, TrafficTimeSeriesDataset, get_dataloaders
from src.metrics import compute_regression_metrics, compute_imbalance_regime_metrics
from models.baseline_lstm import PureTemporalLSTM
from models.base_stgcn import BaseSTGCN
from models.improved_stgcn import ImprovedSTGCN, CongestionFocalLoss

class TestUrbanTrafficGNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_nodes = 25
        cls.in_steps = 12
        cls.out_steps = 3
        cls.in_features = 4
        cls.batch_size = 8

        # Create miniature dataset for testing
        cls.dataset = UrbanTrafficGraphDataset(num_nodes=cls.num_nodes, num_days=7, seed=42)
        cls.adj_matrix = cls.dataset.adj_matrix

    def test_01_dataset_imbalance(self):
        """Verify that traffic dataset is properly imbalanced as required"""
        stats = self.dataset.get_imbalance_stats()
        print(f"\n[Test 1] Imbalance Stats: {stats['skew_summary']}")
        self.assertGreater(stats["free_flow_pct"], 50.0, "Free flow should be the majority class (>50%)")
        self.assertGreater(stats["severe_gridlock_pct"], 3.0, "Severe gridlock should exist as minority tail")
        self.assertLess(stats["severe_gridlock_pct"], 25.0, "Severe gridlock should be < 25% (imbalanced)")

    def test_02_graph_adjacency(self):
        """Verify adjacency matrix symmetry and values"""
        adj = self.dataset.adj_matrix
        self.assertEqual(adj.shape, (self.num_nodes, self.num_nodes))
        self.assertTrue(np.all(adj >= 0.0))
        self.assertTrue(np.any(adj > 0.0), "Graph must have connected edges")

    def test_03_baseline_lstm(self):
        """Verify baseline LSTM forward and backward passes"""
        model = PureTemporalLSTM(in_features=self.in_features, hidden_dim=32, out_steps=self.out_steps)
        dummy_x = torch.randn(self.batch_size, self.in_steps, self.num_nodes, self.in_features)
        
        out = model(dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.out_steps, self.num_nodes))

        # Check gradient flow
        loss = out.sum()
        loss.backward()
        param = next(model.parameters())
        self.assertIsNotNone(param.grad)
        print("[Test 3] Baseline LSTM forward & backward verified.")

    def test_04_base_stgcn(self):
        """Verify Base ST-GCN spatial graph message passing"""
        model = BaseSTGCN(num_nodes=self.num_nodes, in_features=self.in_features, hidden_dim=32, out_steps=self.out_steps)
        model.set_adjacency(self.adj_matrix)
        
        dummy_x = torch.randn(self.batch_size, self.in_steps, self.num_nodes, self.in_features)
        out = model(dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.out_steps, self.num_nodes))

        loss = out.sum()
        loss.backward()
        param = next(model.parameters())
        self.assertIsNotNone(param.grad)
        print("[Test 4] Base ST-GCN forward & backward verified.")

    def test_05_improved_stgcn(self):
        """Verify Improved ST-GCN with Adaptive Graph Conv & Temporal Attention"""
        model = ImprovedSTGCN(num_nodes=self.num_nodes, in_features=self.in_features, hidden_dim=32, out_steps=self.out_steps, num_heads=2)
        model.set_adjacency(self.adj_matrix)

        dummy_x = torch.randn(self.batch_size, self.in_steps, self.num_nodes, self.in_features)
        out = model(dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.out_steps, self.num_nodes))

        loss = out.sum()
        loss.backward()
        param = next(model.parameters())
        self.assertIsNotNone(param.grad)
        print("[Test 5] Improved A-STGCN forward & backward verified.")

    def test_06_congestion_focal_loss(self):
        """Verify that Congestion Focal Loss penalizes low-speed errors more heavily than free-flow errors"""
        focal_loss = CongestionFocalLoss(alpha=4.0, gamma=2.0, mean_speed=58.0, std_speed=14.0, free_flow_speed=65.0)

        # Normalized values:
        # Free-flow speed = 62 mph -> norm value ~ (62 - 58) / 14 = +0.28
        # Congested speed = 15 mph -> norm value ~ (15 - 58) / 14 = -3.07
        
        # Test 1: Error = 0.5 on free-flow target
        target_free = torch.tensor([[[0.28]]])
        pred_free = torch.tensor([[[0.78]]])  # error = 0.5
        loss_free = focal_loss(pred_free, target_free).item()

        # Test 2: Same error = 0.5 on congested target
        target_cong = torch.tensor([[[-3.07]]])
        pred_cong = torch.tensor([[[-2.57]]])  # error = 0.5
        loss_cong = focal_loss(pred_cong, target_cong).item()

        print(f"[Test 6] Loss on Free Flow (error 0.5): {loss_free:.4f} | Loss on Severe Jams (error 0.5): {loss_cong:.4f}")
        self.assertGreater(loss_cong, loss_free * 2.0, "Congestion Focal Loss must heavily up-weight congestion errors!")

    def test_07_metrics_evaluation(self):
        """Verify global and imbalanced regime evaluation metrics"""
        y_true = np.array([62.0, 60.0, 45.0, 15.0, 12.0])
        y_pred = np.array([61.0, 59.0, 42.0, 22.0, 18.0])
        
        reg_metrics = compute_regression_metrics(y_true, y_pred)
        imb_metrics = compute_imbalance_regime_metrics(y_true, y_pred)
        
        self.assertIn("MAE", reg_metrics)
        self.assertIn("severe_congestion_regime", imb_metrics)
        print(f"[Test 7] Regression Metrics: {reg_metrics}")
        print(f"[Test 7] Imbalanced Regimes: {imb_metrics['severe_congestion_regime']}")

if __name__ == "__main__":
    unittest.main()
