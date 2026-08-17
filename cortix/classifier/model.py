"""
CortiX Module 3 — PyTorch LSTM-CNN Model Architecture

A parallel hybrid architecture: Conv1D extracts spatial features from packet 
headers and flows, while BiLSTM captures temporal dependencies across subsequent 
flow sequences, finished with a learned Self-Attention readout.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from cortix.config import config

logger = logging.getLogger("cortix.classifier.model")


class Attention(nn.Module):
    """Simple Dot-product Self-Attention layer."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, hidden_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Q K^T / sqrt(d)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        weights = F.softmax(scores, dim=-1)
        
        # Weighted sum over sequence items
        context = torch.matmul(weights, V)
        # Global pooling over sequence steps
        return torch.mean(context, dim=1)


class CortixLSTMCNN(nn.Module):
    """
    Cortix hybrid LSTM-CNN network for flow classification.
    
    Expected input tensor: (batch_size, seq_len=10, num_features=40)
    """

    def __init__(self, num_classes: int | None = None, num_features: int | None = None, seq_len: int | None = None):
        super().__init__()
        self.num_classes = num_classes or config.CLASSIFIER_NUM_CLASSES
        self.seq_len = seq_len or config.CLASSIFIER_SEQ_LEN
        self.num_features = num_features or config.CLASSIFIER_NUM_FEATURES

        # CNN feature extractor (acts along the sequence features)
        # Input shape: (batch_size, num_features, seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=self.num_features,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(128)
        
        # Adaptive pooling so the model works with any seq_len (including 1
        # for independent tabular records like NSL-KDD where sequences are
        # artificial).  Output length = max(1, seq_len // 2).
        self._pool_out = max(1, self.seq_len // 2)
        self.pool = nn.AdaptiveMaxPool1d(self._pool_out)

        # BiLSTM Layer
        # Input: (batch_size, seq_len // 2, 128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        # Attention block over LSTM hidden states
        self.attention = Attention(hidden_dim=256)  # Bidirectional 128*2 = 256

        # Classification Dense Layers
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, self.num_classes)
        self.dropout = nn.Dropout(0.3)

        logger.info("CortixLSTMCNN Model architecture loaded successfully")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, num_features)
        batch_size = x.size(0)

        # 1. Prepare for CNN: shape (batch_size, num_features, seq_len)
        x_cnn = x.transpose(1, 2)

        # Conv Blocks
        x_cnn = F.relu(self.bn1(self.conv1(x_cnn)))
        x_cnn = F.relu(self.bn2(self.conv2(x_cnn)))
        x_cnn = self.pool(x_cnn)  # shape (batch_size, 128, seq_len // 2)

        # 2. Reshape back for LSTM: shape (batch_size, seq_len // 2, 128)
        x_lstm = x_cnn.transpose(1, 2)

        # BiLSTM output: shape (batch_size, seq_len // 2, 256)
        lstm_out, _ = self.lstm(x_lstm)

        # 3. Dynamic Attention & Pooling
        attention_out = self.attention(lstm_out)  # shape (batch_size, 256)

        # 4. Dense Head
        out = F.relu(self.fc1(self.dropout(attention_out)))
        logits = self.fc2(self.dropout(out))

        return logits
