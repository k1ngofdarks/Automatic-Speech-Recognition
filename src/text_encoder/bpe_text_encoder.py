import re

import sentencepiece as spm
import torch

# TODO add LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class BPECTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, model_file_path: str):
        self.model = spm.SentencePieceProcessor(model_file=model_file_path)
        self.blank_id = 0

        self.vocab = [self.EMPTY_TOK] + [
            self.model.id_to_piece(i) for i in range(self.model.vocab_size())
        ]

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        return self.vocab[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        ids = self.model.encode(text)
        return (torch.tensor(ids, dtype=torch.long) + 1).unsqueeze(0)

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        ctc_ids = [int(ind) for ind in inds]
        sp_ids = [id - 1 for id in ctc_ids if id != self.blank_id]
        return self.model.decode(sp_ids).strip()

    def ctc_decode(self, inds) -> str:
        """
        CTC decoding. Removes empty tokens and repetitions.

        Args:
            inds (list): list of tokens.
        Returns:
            ctc_text (str): text after CTC decoding.
        """
        if len(inds) == 0:
            return ""

        tokens = torch.as_tensor(inds, dtype=torch.long)
        mask = torch.cat(
            (torch.tensor([True], device=tokens.device), (tokens[1:] != tokens[:-1]))
        )

        tokens = tokens[mask]
        tokens = tokens[tokens != self.blank_id]
        return self.decode(tokens)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
