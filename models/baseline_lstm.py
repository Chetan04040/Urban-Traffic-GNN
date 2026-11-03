"""
baseline_lstm.py - Pure Temporal Baseline Model (No Graph Convolutions)
Models each sensor node independently across time using LSTM.
Serves as the benchmark showing why temporal-only models fail to capture spatial road dependencies.
"""

import torch
import torch.nn as nn

class PureTemporalLSTM(nn.Module):
    def __init__(self, in_features=4, hidden_dim=64, num_layers=2, out_steps=3, dropout=0.2):
        super(PureTemporalLSTM, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_steps = out_steps

        # LSTM operates across time for each node
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_steps)
        )

    def forward(self, x, adj_matrix=None):
        """
        Input x: [B, T_in, N, F]
        Output: [B, T_out, N]
        """
        B, T_in, N, F = x.shape
        # Permute to treat each node as an independent sample: (B * N, T_in, F)
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * N, T_in, F)

        # Run LSTM
        lstm_out, _ = self.lstm(x_reshaped)  # [B * N, T_in, hidden_dim]
        last_hidden = lstm_out[:, -1, :]      # [B * N, hidden_dim]

        # Predict future horizon
        out = self.fc(last_hidden)            # [B * N, out_steps]
        
        # Reshape back to [B, N, out_steps] -> permute to [B, out_steps, N]
        out = out.view(B, N, self.out_steps).permute(0, 2, 1).contiguous()
        return out
