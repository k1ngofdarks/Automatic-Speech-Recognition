import torch


class AudioPipeline:
    def __init__(
        self, feature_extractor, wave_transforms=None, spectrogram_transforms=None
    ):
        self.feature_extractor = feature_extractor
        self.wave_transforms = wave_transforms
        self.spectrogram_transforms = spectrogram_transforms

    def __call__(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.wave_transforms is not None:
            audio = self.wave_transforms(audio)
        spectrogram = self.feature_extractor(audio)
        if self.spectrogram_transforms is not None:
            spectrogram = self.spectrogram_transforms(spectrogram)
        return {"spectrogram": spectrogram, "audio": audio}
