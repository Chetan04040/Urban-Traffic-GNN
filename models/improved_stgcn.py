"""
improved_stgcn.py - Attention-Enhanced Spatio-Temporal Graph Neural Network (A-STGCN)
with Adaptive Spatial-Graph Convolutions, Temporal Multi-Head Self-Attention,
and Congestion-Weighted Focal Loss for Imbalanced Traffic Prediction.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGraphAttentionConv(nn.Module):
    """
    Blends physical road network topology with dynamically learned spatial interactions:
    A_adaptive = Softmax(ReLU(E_src @ E_dst^T))
    A_total = alpha * A_physical + (1 - alpha) * A_adaptive
    """
    def __init__(self, in_channels, out_channels, num_nodes, node_embed_dim=16):
        super(AdaptiveGraphAttentionConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes

        # Learnable node embeddings for discovering latent road dependencies
        self.node_embed_src = nn.Parameter(torch.randn(num_nodes, node_embed_dim))
        self.node_embed_dst = nn.Parameter(torch.randn(num_nodes, node_embed_dim))
        self.alpha = nn.Parameter(torch.tensor(0.6))  # learned balance between physical and adaptive

        # Projection weights
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.layer_norm = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.node_embed_src)
        nn.init.xavier_uniform_(self.node_embed_dst)

    def forward(self, x, norm_adj):
        """
        x: [B, N, in_channels]
        norm_adj: [N, N] physical normalized road graph
        """
        B, N, C = x.shape

        # Compute adaptive spatial adjacency
        adaptive_adj = F.softmax(F.relu(torch.mm(self.node_embed_src, self.node_embed_dst.t())), dim=-1)

        # Blend physical topology with adaptive topology
        alpha = torch.sigmoid(self.alpha)
        combined_adj = alpha * norm_adj + (1.0 - alpha) * adaptive_adj

        # Graph Message Passing
        support = torch.matmul(x, self.weight)  # [B, N, out_channels]
        out = torch.matmul(combined_adj, support) + self.bias
        out = self.layer_norm(out)
        return out


class TemporalMultiHeadSelfAttention(nn.Module):
    """
    Self-Attention over temporal historical steps:
    Dynamically identifies which past hours/intervals signal upcoming congestion.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(TemporalMultiHeadSelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B * N, T, hidden_dim]
        """
        attn_out, _ = self.mha(x, x, x)
        out = self.layer_norm(x + self.dropout(attn_out))
        return out


class ImprovedSTGCN(nn.Module):
    """
    Improved Architecture:
    1. Adaptive Spatial-Graph Convolution (Physical + Latent road correlation)
    2. Bidirectional Temporal GRU
    3. Multi-Head Temporal Self-Attention
    4. Congestion-Focal Loss Integration
    """
    def __init__(self, num_nodes=25, in_features=4, hidden_dim=64, out_steps=3, num_heads=4, dropout=0.2):
        super(ImprovedSTGCN, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_steps = out_steps

        # Normalized physical adjacency buffer
        self.register_buffer("norm_adj", torch.eye(num_nodes))

        # 1. Feature Lift
        self.input_fc = nn.Linear(in_features, hidden_dim)

        # 2. Adaptive Spatial Graph Layers
        self.spatial_conv1 = AdaptiveGraphAttentionConv(hidden_dim, hidden_dim, num_nodes)
        self.spatial_conv2 = AdaptiveGraphAttentionConv(hidden_dim, hidden_dim, num_nodes)

        # 3. Temporal Modeling: Bi-GRU + Multi-Head Self-Attention
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.temporal_attn = TemporalMultiHeadSelfAttention(hidden_dim, num_heads=num_heads, dropout=dropout)

        # 4. Multi-Horizon Output Regressor with Residual Skip
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_steps)
        )

    def set_adjacency(self, adj_matrix):
        """Computes symmetric normalized Laplacian"""
        if isinstance(adj_matrix, torch.Tensor):
            adj = adj_matrix.clone().detach().cpu().numpy()
        else:
            adj = adj_matrix.copy()

        adj_hat = adj + np.eye(adj.shape[0], dtype=np.float32)
        degrees = np.sum(adj_hat, axis=1)
        d_inv_sqrt = np.zeros_like(degrees)
        mask = degrees > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)

        norm_adj = d_mat_inv_sqrt.dot(adj_hat).dot(d_mat_inv_sqrt)
        self.norm_adj = torch.tensor(norm_adj, dtype=torch.float32)

    def forward(self, x, adj_matrix=None):
        """
        Input x: [B, T_in, N, num_feat]
        Output: [B, T_out, N]
        """
        if adj_matrix is not None and (self.norm_adj is None or self.norm_adj.device != x.device):
            self.set_adjacency(adj_matrix)

        if self.norm_adj.device != x.device:
            self.norm_adj = self.norm_adj.to(x.device)

        B, T_in, N, num_feat = x.shape

        # Step 1: Feature Projection -> [B, T_in, N, hidden_dim]
        h = F.relu(self.input_fc(x))

        # Step 2: Spatial Convolutions at each time step with residual
        h_flat = h.view(B * T_in, N, self.hidden_dim)
        s1 = F.relu(self.spatial_conv1(h_flat, self.norm_adj))
        s2 = F.relu(self.spatial_conv2(s1, self.norm_adj))
        s_out = h_flat + s2  # Residual skip
        s_out = s_out.view(B, T_in, N, self.hidden_dim)

        # Step 3: Temporal GRU + Attention
        # Permute to (B * N, T_in, hidden_dim) to model per-sensor temporal trajectory
        t_in = s_out.permute(0, 2, 1, 3).contiguous().view(B * N, T_in, self.hidden_dim)
        gru_out, _ = self.temporal_gru(t_in)  # [B * N, T_in, hidden_dim]
        attn_out = self.temporal_attn(gru_out)  # [B * N, T_in, hidden_dim]

        # Aggregate temporal representations (weighted pool of last step and summary)
        context = attn_out[:, -1, :]  # [B * N, hidden_dim]

        # Step 4: Multi-Step Prediction Head
        out = self.head(context)  # [B * N, out_steps]
        out = out.view(B, N, self.out_steps).permute(0, 2, 1).contiguous()  # [B, out_steps, N]
        return out


class CongestionFocalLoss(nn.Module):
    """
    Imbalance-Aware Congestion Focal Loss:
    Real-world traffic is heavily imbalanced (~70% free flow, ~10% severe congestion).
    This loss dynamically up-weights errors during congested / bottleneck time-steps:
    Weight(y) = 1.0 + alpha * ( (y_free - y) / y_free )^gamma
    """
    def __init__(self, alpha=3.0, gamma=2.0, mean_speed=58.0, std_speed=14.0, free_flow_speed=65.0):
        super(CongestionFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mean_speed = mean_speed
        self.std_speed = std_speed
        self.free_flow_speed = free_flow_speed

    def forward(self, pred, target):
        """
        pred, target: [B, T_out, N] (normalized values)
        """
        # Convert target back to physical speed (mph) to calculate severity weight
        target_mph = target * self.std_speed + self.mean_speed
        
        # Calculate congestion severity: high when speed is low (< 35 mph)
        # Severity in [0, 1]
        severity = torch.clamp((self.free_flow_speed - target_mph) / self.free_flow_speed, min=0.0, max=1.0)
        
        # Imbalance weighting factor: normal free-flow gets weight 1.0; severe jams get weight up to (1.0 + alpha)
        weights = 1.0 + self.alpha * torch.pow(severity, self.gamma)

        squared_error = torch.square(pred - target)
        weighted_loss = torch.mean(weights * squared_error)
        return weighted_loss
