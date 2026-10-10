import torch
import torchaudio


class SpeedPerturbation(torch.nn.Module):
    def __init__(self, sample_rate: int, factors: list[float]):
        super().__init__()
        self.sp = torchaudio.transforms.SpeedPerturbation(
            orig_freq=sample_rate, factors=factors
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        audio, _ = self.sp(audio)
        return audio
