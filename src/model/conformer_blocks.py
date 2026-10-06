import torch
import torch.nn as nn


class FeedForwardModule(nn.Module):
    def __init__(self, input_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvolutionModule(nn.Module):
    def __init__(self, input_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")

        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 2 * input_dim, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                input_dim,
                input_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=input_dim,
            ),
            nn.BatchNorm1d(input_dim),
            nn.SiLU(),
            nn.Conv1d(input_dim, input_dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm(x).transpose(1, 2)
        return self.net(x).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(input_dim, ffn_dim, dropout)
        self.attention_norm = nn.LayerNorm(input_dim)
        self.attention = nn.MultiheadAttention(
            input_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.conv = ConvolutionModule(input_dim, depthwise_conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(input_dim, ffn_dim, dropout)
        self.output_norm = nn.LayerNorm(input_dim)

    def forward(self, x, padding_mask=None):
        x = x + 0.5 * self.ffn1(x)
        norm = self.attention_norm(x)
        attended, _ = self.attention(
            norm, norm, norm, key_padding_mask=padding_mask, need_weights=False
        )
        x = x + self.attention_dropout(attended)
        x = x + self.conv(x)
        return self.output_norm(x + 0.5 * self.ffn2(x))


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, lengths):
        positions = torch.arange(x.size(1), device=x.device)
        padding_mask = positions.unsqueeze(0) >= lengths.to(x.device).unsqueeze(1)

        for layer in self.layers:
            x = layer(x, padding_mask)

        return x, lengths
