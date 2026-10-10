import torch
from torchaudio.functional import add_noise


class AddNoise(torch.nn.Module):
    def __init__(self, min_snr: float = 20.0, max_snr: float = 30.0, p: float = 0.1):
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            snr = torch.rand(1) * (self.max_snr - self.min_snr) + self.min_snr
            noise = torch.randn_like(audio)
            return add_noise(audio, noise=noise, snr=snr)
        return audio
