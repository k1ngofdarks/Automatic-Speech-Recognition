import argparse
import json
from pathlib import Path

import sentencepiece as spm

from src.text_encoder import CTCTextEncoder

TRAIN_PARTS = ("train-clean-100", "train-clean-360", "train-other-500")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/datasets/librispeech")
    parser.add_argument("--parts", nargs="+", required=True, choices=TRAIN_PARTS)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_sentences(index_dir, parts):
    sentences = []
    for part in parts:
        index_path = Path(index_dir) / f"{part}_index.json"
        records = json.loads(index_path.read_text())
        for record in records:
            text = CTCTextEncoder.normalize_text(record["text"])
            sentences.append(text)
    return sentences


def main():
    args = parse_args()
    sentences = load_sentences(args.index_dir, args.parts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.txt"
    with corpus_path.open("w") as file:
        for text in sentences:
            file.write(text + "\n")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(output_dir / "tokenizer"),
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
    )

    print(
        f"Finished training BPE on {len(sentences)} texts from {', '.join(args.parts)}."
    )
    print(f"Model: {output_dir / 'tokenizer.model'}")


if __name__ == "__main__":
    main()
