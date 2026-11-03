"""
base_stgcn.py - Standard Spatio-Temporal Graph Convolutional Network (ST-GCN)
Combines Spatial Graph Convolutions (Kipf & Welling GCN with normalized Laplacian)
with Temporal 1D Gated Convolutions / GRUs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGraphConv(nn.Module):
    """
    Standard GCN layer: H^(l+1) = D^(-0.5) * A_hat * D^(-0.5) * H^(l) * W
    where A_hat = A + I_N
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0.2)
        nn.init.zeros_(self.bias)

    def forward(self, x, norm_adj):
        """
        x: [B, N, in_channels]
        norm_adj: [N, N] normalized adjacency matrix
        returns: [B, N, out_channels]
        """
        # Linear projection: X * W -> [B, N, out_channels]
        support = torch.matmul(x, self.weight)
        # Message passing: norm_adj * support -> [B, N, out_channels]
        out = torch.matmul(norm_adj, support) + self.bias
        return out


class STGCNBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block:
    Temporal Gated Conv -> Spatial GCN -> Temporal Gated Conv
    """
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes, dropout=0.2):
        super(STGCNBlock, self).__init__()
        self.temporal_conv1 = nn.Conv2d(in_channels, spatial_channels, kernel_size=(3, 1), padding=(1, 0))
        self.gcn = SpatialGraphConv(spatial_channels, spatial_channels)
        self.temporal_conv2 = nn.Conv2d(spatial_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, norm_adj):
        """
        x: [B, C, T, N]
        norm_adj: [N, N]
        """
        # Temporal Conv 1
        t1 = F.relu(self.temporal_conv1(x))  # [B, C_spatial, T, N]

        # Spatial GCN applied at each time step
        B, C, T, N = t1.shape
        # Permute to [B * T, N, C]
        t1_perm = t1.permute(0, 2, 3, 1).contiguous().view(B * T, N, C)
        s_out = F.relu(self.gcn(t1_perm, norm_adj))  # [B * T, N, C]
        s_out = s_out.view(B, T, N, C).permute(0, 3, 1, 2).contiguous()  # [B, C, T, N]

        # Temporal Conv 2 + Residual connection if channel dimensions match
        t2 = self.dropout(F.relu(self.temporal_conv2(s_out)))
        out = self.batch_norm(t2)
        return out


class BaseSTGCN(nn.Module):
    """
    Base ST-GCN Architecture:
    Takes historical traffic graph features and predicts multi-step future speed.
    """
    def __init__(self, num_nodes=25, in_features=4, hidden_dim=64, out_steps=3, dropout=0.2):
        super(BaseSTGCN, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_steps = out_steps

        # Normalized adjacency placeholder (will register as buffer when supplied)
        self.register_buffer("norm_adj", torch.eye(num_nodes))

        # STGCN Blocks
        self.block1 = STGCNBlock(in_features, hidden_dim, hidden_dim, num_nodes, dropout)
        self.block2 = STGCNBlock(hidden_dim, hidden_dim, hidden_dim, num_nodes, dropout)

        # Output projection head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_steps)
        )

    def set_adjacency(self, adj_matrix):
        """
        Calculates and stores symmetric normalized Laplacian:
        A_hat = A + I
        D_hat = sum(A_hat)
        norm_adj = D_hat^(-1/2) * A_hat * D_hat^(-1/2)
        """
        if isinstance(adj_matrix, torch.Tensor):
            adj = adj_matrix.clone().detach().cpu().numpy()
        else:
            adj = adj_matrix.copy()

        # Add self-loops
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
        Input x: [B, T_in, N, F]
        Output: [B, T_out, N]
        """
        if adj_matrix is not None and (self.norm_adj is None or self.norm_adj.device != x.device):
            self.set_adjacency(adj_matrix)

        if self.norm_adj.device != x.device:
            self.norm_adj = self.norm_adj.to(x.device)

        # Convert to [B, F, T, N]
        x_in = x.permute(0, 3, 1, 2).contiguous()

        # Pass through ST-GCN blocks
        h1 = self.block1(x_in, self.norm_adj)
        h2 = self.block2(h1, self.norm_adj)  # [B, hidden_dim, T, N]

        # Pool or take final temporal step
        # Take temporal representation at last time step
        h_last = h2[:, :, -1, :]  # [B, hidden_dim, N]
        h_last = h_last.permute(0, 2, 1).contiguous()  # [B, N, hidden_dim]

        # Predict future horizon
        out = self.fc(h_last)  # [B, N, out_steps]
        out = out.permute(0, 2, 1).contiguous()  # [B, out_steps, N]
        return out
