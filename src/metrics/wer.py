from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, predictions: List[str], text: List[str], **kwargs):
        wers = []
        for prediction, target in zip(predictions, text):
            target = self.text_encoder.normalize_text(target)
            wers.append(calc_wer(target, prediction))
        return sum(wers) / len(wers)
