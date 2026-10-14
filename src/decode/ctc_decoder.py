import torch


class GreedyCTCDecoder:
    decoder_name = "Argmax Greedy"

    def __init__(self, text_encoder):
        self.text_encoder = text_encoder

    def decode(
        self, log_probs: torch.Tensor, log_probs_lengths: torch.Tensor
    ) -> list[str]:
        """
        Decodes the output of a CTC model using greedy decoding.

        Args:
            log_probs (torch.Tensor): The log probabilities from the CTC model of shape (batch_size, max_seq_length, num_classes).
            log_probs_lengths (torch.Tensor): The lengths of the sequences in the batch of shape (batch_size,).

        Returns:
            list[str]: A list of decoded strings for each sequence in the batch.
        """

        max_probs = log_probs.argmax(-1).cpu().numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(max_probs, log_probs_lengths.cpu().tolist())
        ]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        return argmax_texts
