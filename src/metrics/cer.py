from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer, calc_levenshtein_distance

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class CERMetric(BaseMetric):
    def __init__(self, text_encoder, decoder_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        # Key in config.decoders and batch["predictions"], e.g. "greedy" or "beam"
        self.decoder_key = decoder_key

    def __call__(self, predictions: dict[str, List[str]], text: List[str], **kwargs):
        cers = []
        for prediction, target in zip(predictions[self.decoder_key], text):
            target = self.text_encoder.normalize_text(target)
            cers.append(calc_cer(target, prediction))
        return {"value": sum(cers) / len(cers)}


class CorpusCERMetric(BaseMetric):
    def __init__(self, text_encoder, decoder_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        # Key in config.decoders and batch["predictions"], e.g. "greedy" or "beam"
        self.decoder_key = decoder_key

    def __call__(self, predictions: dict[str, List[str]], text: List[str], **kwargs):
        total_errors = 0
        total_chars = 0
        for prediction, target in zip(predictions[self.decoder_key], text):
            target = self.text_encoder.normalize_text(target)
            distance = calc_levenshtein_distance(target, prediction)
            total_errors += distance
            total_chars += len(target)
        return {
            "value": total_errors / total_chars if total_chars > 0 else 0,
            "n": total_chars,
        }
