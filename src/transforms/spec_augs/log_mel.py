import torch
from torch import nn


class LogMelNormalization(nn.Module):
    """
    Log-Mel normalization module.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        spectrogram = torch.log(spectrogram.clamp_min(self.eps))

        mean = spectrogram.mean(dim=-1, keepdim=True)
        std = spectrogram.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)

        return (spectrogram - mean) / std
