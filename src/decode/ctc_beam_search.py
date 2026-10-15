import numpy as np
import torch


class CTCBeamSearchDecoder:
    decoder_name = "Beam"

    def __init__(self, text_encoder, beam_width=100, blank_id=0):
        self.text_encoder = text_encoder
        self.beam_width = beam_width
        self.blank_id = blank_id

    def decode(
        self, log_probs: torch.Tensor, log_probs_lengths: torch.Tensor
    ) -> list[str]:
        log_probs = log_probs.detach().cpu().numpy()
        log_probs_lengths = log_probs_lengths.detach().cpu().tolist()

        result = []
        for log_prob, log_prob_length in zip(log_probs, log_probs_lengths):
            result.append(self._decode_one(log_prob, log_prob_length))
        return result

    def rank_beam_prefixes(self, beam_prefixes):
        """
        Ranks the beam prefixes based on their total log probabilities.

        Args:
            beam_prefixes (dict): A dictionary of beam prefixes and their corresponding log probabilities.

        Returns:
            list: A list of tuples containing the ranked beam prefixes and their total log probabilities.
        """
        ranked_beam_prefixes = []
        for prefix, (log_prob_b, log_prob_nb) in beam_prefixes.items():
            total_log_prob = np.logaddexp(log_prob_b, log_prob_nb)
            ranked_beam_prefixes.append((prefix, total_log_prob))
        ranked_beam_prefixes.sort(key=lambda x: x[1], reverse=True)
        return ranked_beam_prefixes[: self.beam_width]

    def _decode_one(self, log_prob: torch.Tensor, log_prob_length: int) -> str:
        """
        Decodes a single sequence of log probabilities using beam search decoding.

        Args:
            log_prob (torch.Tensor): The log probabilities for a single sequence of shape (max_seq_length, num_classes).
            log_prob_length (int): The length of the sequence.

        Returns:
            str: The decoded string for the sequence.
        """
        log_prob = log_prob[:log_prob_length]

        beam_prefixes = {
            tuple([]): (0.0, -np.inf)
        }  # {prefix: (log_prob_blank, log_prob_non_blank)}
        num_classes = log_prob.shape[-1]

        for t in range(log_prob_length):
            new_beam_prefixes = {}
            for prefix, (log_prob_b, log_prob_nb) in list(beam_prefixes.items()):
                total_log_prob = np.logaddexp(log_prob_b, log_prob_nb)
                for c in range(num_classes):
                    p_c = log_prob[t, c]

                    if c == self.blank_id:
                        new_log_prob_b = total_log_prob + p_c
                        old_log_prob_b, old_log_prob_nb = new_beam_prefixes.get(
                            prefix, (-np.inf, -np.inf)
                        )

                        new_beam_prefixes[prefix] = (
                            np.logaddexp(old_log_prob_b, new_log_prob_b),
                            old_log_prob_nb,
                        )
                    else:
                        new_prefix = prefix + (c,)

                        if prefix and prefix[-1] == c:
                            # 1. blank  + c : prefix + c
                            old_log_prob_b, old_log_prob_nb = new_beam_prefixes.get(
                                new_prefix, (-np.inf, -np.inf)
                            )

                            new_log_prob_nb = log_prob_b + p_c

                            new_beam_prefixes[new_prefix] = (
                                old_log_prob_b,
                                np.logaddexp(old_log_prob_nb, new_log_prob_nb),
                            )

                            # 2. non-blank + c : prefix
                            old_log_prob_b, old_log_prob_nb = new_beam_prefixes.get(
                                prefix, (-np.inf, -np.inf)
                            )

                            new_log_prob_nb = log_prob_nb + p_c

                            new_beam_prefixes[prefix] = (
                                old_log_prob_b,
                                np.logaddexp(old_log_prob_nb, new_log_prob_nb),
                            )
                        else:
                            new_log_prob_nb = total_log_prob + p_c

                            old_log_prob_b, old_log_prob_nb = new_beam_prefixes.get(
                                new_prefix, (-np.inf, -np.inf)
                            )

                            new_beam_prefixes[new_prefix] = (
                                old_log_prob_b,
                                np.logaddexp(old_log_prob_nb, new_log_prob_nb),
                            )

            ranked = self.rank_beam_prefixes(new_beam_prefixes)
            beam_prefixes = {prefix: new_beam_prefixes[prefix] for prefix, _ in ranked}

        best_prefix, _ = self.rank_beam_prefixes(beam_prefixes)[0]
        return self.text_encoder.decode(best_prefix)  # ctc_decode is alredy done
