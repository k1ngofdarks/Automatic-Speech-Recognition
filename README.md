# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#download-resources">Resources</a> •
  <a href="#inference">Inference</a> •
  <a href="#standalone-cerwer-calculation">CER/WER</a> •
  <a href="#training">Training</a> •
  <a href="#tokenizer-and-lm-tuning">BPE and LM</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#results-and-experiments">Results</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

A project for training and evaluating English automatic speech recognition
models, developed for the [HSE DLA](https://github.com/markovka17/dla) course.
The main model is a custom FastConformer implementation trained with CTC loss.

- [Demo notebook](demo.ipynb)
- [Open the demo in Google Colab](https://colab.research.google.com/github/k1ngofdarks/Automatic-Speech-Recognition/blob/main/demo.ipynb)
- [Model weights and KenLM on Hugging Face](https://huggingface.co/k1ngofdarks/dla-asr)
- [Final model training on Comet](https://www.comet.com/alexander-stepanenko/dla-asr/e9ltoobnfmw1t4yob6xsdpaw121nxcai)
- [Experiment report and final results](https://www.comet.com/alexander-stepanenko/dla-asr/reports/T2Me7FLe16PcCB7Bvgnb0mz66)

> The published model was trained for 30 full epochs on LibriSpeech.
> Beam + LM achieves **14.90% WER** on test-other.

## About

The project includes Conformer and FastConformer models, character and BPE
text encoders, CTC loss, a custom CTC prefix beam search, and decoding with
KenLM. LM-based decoding uses the `ctc-beam-decoder` library.

Main model configuration:

| Parameter | Value |
| --- | --- |
| Training data | LibriSpeech 960h |
| Features | 128 mel bins, logarithmic compression and normalization |
| Encoder | 16 blocks, dimension 144, 4 attention heads, FFN dimension 576 |
| Convolutional frontend | 256 channels, 8x temporal subsampling |
| Tokenizer | SentencePiece BPE: 256 tokens + CTC blank |
| Parameter count | Approximately 8.5 million |
| Decoding | Greedy, custom beam search, beam search with KenLM |

The architectures are based on the
[Conformer](https://arxiv.org/abs/2005.08100) and
[FastConformer](https://arxiv.org/abs/2305.05084) papers.
The current implementation uses standard MultiheadAttention without relative
positional encoding.

## Installation

Run the commands below from the project root. The project uses Python 3.12.

```bash
git clone https://github.com/k1ngofdarks/Automatic-Speech-Recognition.git
cd Automatic-Speech-Recognition

conda create -n dla_env python=3.12
conda activate dla_env
python -m pip install -r requirements.txt
```

For development, you can install Git hooks:

```bash
pre-commit install
```

Inference supports both GPU and CPU. The device is selected automatically;
use `inferencer.device=cpu` to select CPU explicitly.
The main model was trained on an RTX 4060 with 8 GB of VRAM.

To train a model, configure Comet ML, for example through an environment variable:

```bash
export COMET_API_KEY="YOUR_COMET_API_KEY"
```

A Comet API key is not required for inference or metric calculation.

## Download resources

```bash
python -m scripts.download_resources
```

The file list is defined in
[`src/configs/download_resources.yaml`](src/configs/download_resources.yaml).
The script downloads resources using `huggingface_hub` and skips existing files.

| Resource | Local path |
| --- | --- |
| Final model, approximately 98 MiB | `saved/final/model.pth` |
| KenLM 4-gram, approximately 4.1 GiB | `data/kenlm/4-gram.arpa` |
| Full training log, epochs 1–30 | `saved/final/info.log` |
| BPE tokenizer included in the Git repository | `data/tokenizers/train-all/bpe-256/tokenizer.model` |

The same command downloads the complete final-model training log from
[Hugging Face](https://huggingface.co/k1ngofdarks/dla-asr/blob/main/info.log).

The external LibriSpeech language model uses the ARPA format and comes from
[OpenSLR](https://www.openslr.org/11/).

The commands below use the published final model at `saved/final/model.pth`.

## Inference

### LibriSpeech

Evaluate on dev-clean, dev-other, test-clean, and test-other using greedy
and Beam + LM decoding:

```bash
python inference.py -cn=inference_fastconformer \
  datasets=all_eval \
  inferencer.from_pretrained=saved/final/model.pth \
  inferencer.save_path=fastconformer/librispeech
```

Metrics are printed separately for each split and decoder.
The splits are downloaded and indexed automatically on the first run.
The data is stored in `data/datasets/librispeech/`.

Available dataset groups:

| Configuration | Splits |
| --- | --- |
| `datasets=all_eval` | dev-clean, dev-other, test-clean, test-other |
| `datasets=clean_eval` | dev-clean, test-clean |
| `datasets=other_eval` | dev-other, test-other |

The command above saves predictions to:

```text
data/saved/fastconformer/librispeech/
├── dev_clean/
│   ├── greedy/<utterance_id>.txt
│   └── beam_lm/<utterance_id>.txt
├── dev_other/
├── test_clean/
└── test_other/
```

### Custom dataset

Expected directory structure:

```text
custom_dataset/
├── audio/
│   ├── id1.flac
│   ├── id2.wav
│   └── id3.mp3
└── transcriptions/        # Optional directory.
    ├── id1.txt
    ├── id2.txt
    └── id3.txt
```

Supported audio extensions are `.wav`, `.flac`, and `.mp3`. Filenames without
the extension serve as utterance IDs. Each `.txt` file contains the reference
transcript for the corresponding recording.
Audio is loaded using SoundFile and resampled to 16 kHz if needed.

Example for a dataset stored in `data/datasets/custom_dataset`:

```bash
python inference.py -cn=inference_customdirdataset \
  custom_dir=data/datasets/custom_dataset \
  inferencer.from_pretrained=saved/final/model.pth \
  inferencer.save_path=custom_dataset
```

When reference transcripts are available, inference also calculates CER/WER.
If the `transcriptions/` directory is absent, only predictions are saved.

```text
data/saved/custom_dataset/customdataset/
├── greedy/<utterance_id>.txt
└── beam_lm/<utterance_id>.txt
```

A relative `inferencer.save_path` is resolved under `data/saved/`.
Absolute paths are also supported. The demo uses a separate `demo_output/` directory.

In the [demo](demo.ipynb), you can provide a public Google Drive folder link.
The notebook cells download the dataset, run inference, and calculate metrics.

### Decoder selection

Add the desired configuration overrides to the inference command:

| Decoder | Hydra overrides |
| --- | --- |
| Greedy | `decoders=greedy metrics=greedy` |
| Custom CTC beam search | `decoders=beam metrics=beam` |
| Beam + KenLM | `decoders=beam_lm metrics=beam_lm` |
| Greedy and Beam + LM, default | `decoders=two_decoders metrics=inference` |
| All three decoders | `decoders=compare metrics=compare` |

By default, the `beam_lm` configuration uses the downloaded 4-gram LM at
`data/kenlm/4-gram.arpa` with `lm_uppercase=True`.

You can change LM parameters through Hydra, for example:
`decoders.beam_lm.alpha=0.5 decoders.beam_lm.beta=1.0`.

## Standalone CER/WER calculation

The script reads two directories containing reference and predicted `.txt`
transcripts. Both directories must contain the same set of utterance IDs.

```bash
python -m scripts.calc_metrics \
  target_dir=data/datasets/custom_dataset/transcriptions \
  prediction_dir=data/saved/custom_dataset/customdataset/beam_lm \
  output_path=data/saved/custom_dataset/beam_lm_metrics.json
```

`output_path` is optional. Results are printed to the console and, if a path
is provided, saved as JSON with `CER` and `WER` fields.

The final evaluation uses CER/WER averaged over individual utterances.
Inference configurations use `batch_size=1`. Additional metrics with `Corpus`
in their names calculate the total number of errors divided by the total
number of reference characters or words. The text encoder normalizes reference
transcripts to lowercase English letters and spaces.

## Training

### Main run

Train from scratch on train-clean-100, train-clean-360, and train-other-500:

```bash
python train.py -cn=fc_big_run writer.run_name=your_run_name
```

Main settings in [`fc_big_run.yaml`](src/configs/fc_big_run.yaml):

- 30 full epochs, `trainer.epoch_len=null`;
- batch size 16 with AMP enabled;
- 256 frontend channels and encoder dimension 144;
- AdamW and OneCycleLR with a maximum learning rate of `3e-4` and 5% warmup;
- log-mel normalization and SpecAugment on the training split;
- logging every 100 batches and a checkpoint after each epoch.

The configuration, `info.log`, and checkpoints are saved to
`saved/your_run_name/`. Metrics and examples are also logged to Comet.
The final model was trained for 30 full epochs. The published `model.pth`
is the checkpoint originally saved as `checkpoint-epoch30.pth`.

For a short run on train-clean-100:

```bash
python train.py -cn=fastconformer \
  model.conv_channels=256 \
  trainer.n_epochs=1 \
  writer.run_name=your_run_name
```

### Resume

Example of resuming after completing epoch 20:

```bash
python train.py \
  --config-path saved/your_run_name \
  --config-name config \
  trainer.resume_from=checkpoint-epoch20.pth
```

This loads the saved configuration, model weights, optimizer, scheduler,
and AMP scaler states. Training continues from epoch 21 up to the total
number of epochs specified in the configuration. Any incomplete epoch after
the last checkpoint is restarted.

### Configuration

Configurations are stored in `src/configs/` and composed using Hydra.
The `defaults` list includes configuration groups, and command-line arguments
override individual settings. To inspect the composed configuration without
starting training:

```bash
python train.py -cn=fc_big_run --cfg job
```

### Augmentations

[`AudioPipeline`](src/transforms/audio_pipeline.py) applies waveform transforms,
computes a mel spectrogram, and then applies spectrogram transforms.
Log-mel normalization is used as feature preprocessing.

| `transforms/instance_transforms` configuration | Transform |
| --- | --- |
| `baseline` | Log-mel normalization without random augmentations |
| `noise` | Additive noise, SNR 20–30 dB |
| `speed_perturbation` | Speed factors of 0.9, 1.0, or 1.1 |
| `time_masking` | Time masking |
| `frequency_masking` | Frequency masking |
| `spec_augment` | Two frequency masks and two time masks |

For example, run a separate augmentation experiment:

```bash
python train.py -cn=fastconformer \
  transforms/instance_transforms=frequency_masking \
  writer.run_name=your_run_name
```

Augmentations are applied during training. Inference uses feature preprocessing.

## Tokenizer and LM tuning

For inference, use the tokenizer included in the repository that matches
the checkpoint. To train a new BPE tokenizer:

```bash
python -m scripts.train_tokenizer \
  --index-dir data/datasets/librispeech \
  --parts train-clean-100 train-clean-360 train-other-500 \
  --vocab-size 256 \
  --output-dir data/tokenizers/train-all/bpe-256-new
```

The script reads the `text` field from `<part>_index.json`; audio files are
not required. The LibriSpeech dataset loader creates these indexes.
The script saves `corpus.txt`, `tokenizer.model`, and `tokenizer.vocab` to
`--output-dir`. Set `text_encoder.model_file_path` to the new tokenizer path
when training a new model.

An Optuna script is available for tuning `alpha/beta` on dev-other:

```bash
python -m scripts.tune_kenlm \
  datasets=other_eval \
  inferencer.from_pretrained=saved/final/model.pth \
  tuning.study_name=kenlm_epoch30 \
  tuning.cache_path=data/kenlm/tuning/epoch30_dev_other.pt
```

Model outputs are cached and reused to evaluate different LM parameters.
Use a separate `cache_path` and `study_name` for each checkpoint.
Optuna results are stored in SQLite at `data/kenlm/tuning/study.sqlite3`.

## Project structure

```text
train.py                    # Training.
inference.py                # Evaluation and saving predictions.
demo.ipynb                  # Google Colab demo.
scripts/
├── download_resources.py    # Model weights, LM, and training log from Hugging Face.
├── calc_metrics.py          # CER/WER from saved transcripts.
├── train_tokenizer.py       # BPE tokenizer training.
└── tune_kenlm.py            # LM parameter tuning.
src/
├── configs/                 # Hydra configurations.
├── datasets/                # LibriSpeech, CustomDirDataset, and collate_fn.
├── model/                   # Conformer/FastConformer and shared blocks.
├── text_encoder/            # Character and BPE encoders.
├── decode/                  # Greedy, custom beam search, external beam + LM.
├── transforms/              # AudioPipeline, preprocessing, and augmentations.
├── trainer/                 # Training and inference.
├── metrics/                 # CER/WER and aggregation.
└── logger/                  # Comet ML and visualizations.
```

## Results and experiments

Results of the final epoch-30 checkpoint on four LibriSpeech splits.
All values are percentages averaged over utterances.

| Split | Greedy CER | Greedy WER | Beam + LM CER | Beam + LM WER |
| --- | --- | --- | --- | --- |
| dev-clean | 3.09 | 9.06 | 2.33 | 6.12 |
| dev-other | 8.76 | 20.50 | 6.84 | 14.04 |
| test-clean | 3.01 | 8.82 | 2.29 | 6.09 |
| test-other | 8.87 | 21.09 | 7.13 | 14.90 |

Analysis, speed comparisons, and final model evaluation are available in the
[Comet Report](https://www.comet.com/alexander-stepanenko/dla-asr/reports/T2Me7FLe16PcCB7Bvgnb0mz66).

### Before submission

- Verify the installation commands and demo in a fresh Google Colab session.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
