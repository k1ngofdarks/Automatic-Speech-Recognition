import torch
from ctc_beam_decoder import build_ctcdecoder


class ExternalCTCDecoder:
    def __init__(
        self,
        text_encoder,
        kenlm_model_path=None,
        lm_uppercase=False,
        alpha=0.5,
        beta=1.0,
        beam_width=100,
        beam_prune_logp=-10.0,
        token_min_logp=-7.0,
    ):
        self.text_encoder = text_encoder
        labels = list(text_encoder.vocab)
        if lm_uppercase:
            labels = [
                token if token.startswith("<") else token.upper() for token in labels
            ]

        self.decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=kenlm_model_path,
            alpha=alpha,
            beta=beta,
        )
        self.beam_width = beam_width
        self.beam_prune_logp = beam_prune_logp
        self.token_min_logp = token_min_logp

    def decode(
        self, log_probs: torch.Tensor, log_probs_lengths: torch.Tensor
    ) -> list[str]:
        emissions = log_probs.detach().float().cpu().numpy()
        lengths = log_probs_lengths.detach().cpu().tolist()
        text_decoded = [
            self.decoder.decode(
                sample[:length],
                beam_width=self.beam_width,
                beam_prune_logp=self.beam_prune_logp,
                token_min_logp=self.token_min_logp,
            )
            for sample, length in zip(emissions, lengths)
        ]
        return [self.text_encoder.normalize_text(text) for text in text_decoded]
