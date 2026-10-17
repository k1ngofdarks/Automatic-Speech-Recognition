from pathlib import Path

import hydra
import optuna
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.metrics.tracker import MetricTracker
from src.metrics.wer import WERMetric
from src.utils.init_utils import set_random_seed


def collect_outputs(config: DictConfig, text_encoder) -> list[dict]:
    set_random_seed(config.inferencer.seed)
    device = config.inferencer.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)

    checkpoint = torch.load(
        config.inferencer.from_pretrained, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint

    model = model.to(device)
    model.eval()

    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    cache = []
    with torch.inference_mode():
        for batch in dataloaders["val"]:
            for key in config.inferencer.device_tensors:
                batch[key] = batch[key].to(device)
            outputs = model(**batch)

            cache.append(
                {
                    "log_probs": outputs["log_probs"].float().cpu(),
                    "log_probs_length": outputs["log_probs_length"].cpu(),
                    "text": batch["text"],
                }
            )

    return cache


def evaluate(cache: list[dict], decoder, text_encoder) -> float:
    metric = WERMetric(text_encoder=text_encoder, decoder_key="beam_lm")
    metric_tracker = MetricTracker("WER_(Beam_LM)", writer=None)
    for batch in cache:
        decoded = decoder.decode(batch["log_probs"], batch["log_probs_length"])
        result = metric(predictions={"beam_lm": decoded}, text=batch["text"])
        metric_tracker.update("WER_(Beam_LM)", **result)

    return metric_tracker.avg("WER_(Beam_LM)")


def objective(trial, cache: list[dict], config: DictConfig, text_encoder) -> float:
    alpha = trial.suggest_float("alpha", 0.0, 2.0)
    beta = trial.suggest_float("beta", -1.0, 3.0)
    decoder = instantiate(
        config.decoders.beam_lm, text_encoder=text_encoder, alpha=alpha, beta=beta
    )
    return evaluate(cache, decoder, text_encoder)


@hydra.main(version_base=None, config_path="../src/configs", config_name="tune_kenlm")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    Path("data/kenlm/tuning").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=config.tuning.study_name,
        storage=config.tuning.storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    study.enqueue_trial(
        {"alpha": 0.5, "beta": 1.0}, skip_if_exists=True
    )  # baseline values

    text_encoder = instantiate(config.text_encoder)
    cache_path = Path(config.tuning.cache_path)
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        cache = collect_outputs(config, text_encoder)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)

    study.optimize(
        lambda trial: objective(trial, cache, config, text_encoder),
        n_trials=config.tuning.n_trials,
    )


if __name__ == "__main__":
    main()
