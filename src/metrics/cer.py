from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, predictions: List[str], text: List[str], **kwargs):
        cers = []
        for prediction, target in zip(predictions, text):
            target = self.text_encoder.normalize_text(target)
            cers.append(calc_cer(target, prediction))
        return sum(cers) / len(cers)
