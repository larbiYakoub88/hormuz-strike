import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CTTS(nn.Module):
    """
    CNN + Transformer for Time Series (CTTS)
    With BatchNorm before GELU in MLP head
    """
    def __init__(self,
                 input_dim,
                 seq_len=80,
                 cnn_kernel_size=16,
                 cnn_stride=8,
                 d_model=128,
                 nhead=4,
                 num_layers=4,
                 dropout=0.3,
                 num_classes=3):
        super().__init__()

        # 1D CNN to project local windows into tokens
        self.conv = nn.Conv1d(in_channels=input_dim,
                              out_channels=d_model,
                              kernel_size=cnn_kernel_size,
                              stride=cnn_stride)

        # Calculate number of tokens after convolution
        self.conv_out_len = (seq_len - cnn_kernel_size) // cnn_stride + 1

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.conv_out_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LayerNorm and Dropout (standard for Transformer)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # MLP head with BatchNorm before GELU
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),      # BatchNorm added
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)                # (batch, input_dim, seq_len)

        # CNN tokenization
        x = self.conv(x)                       # (batch, d_model, conv_out_len)
        x = x.permute(0, 2, 1)                 # (batch, conv_out_len, d_model)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)                # (batch, conv_out_len, d_model)

        # Global average pooling
        x = x.mean(dim=1)                      # (batch, d_model)
        x = self.norm(x)
        x = self.dropout(x)

        # Classification head
        logits = self.head(x)                  # (batch, num_classes)
        return logits
