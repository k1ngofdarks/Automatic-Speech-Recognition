import json
from pathlib import Path

import hydra
from hydra.utils import instantiate

from src.metrics.tracker import MetricTracker
from src.text_encoder import CTCTextEncoder


def evaluate(config):
    target_dir = Path(config.target_dir)
    prediction_dir = Path(config.prediction_dir)

    if not target_dir.is_dir() or not prediction_dir.is_dir():
        raise NotADirectoryError("Both paths must be directories")

    targets = {path.stem: path for path in target_dir.glob("*.txt") if path.is_file()}
    predictions = {
        path.stem: path for path in prediction_dir.glob("*.txt") if path.is_file()
    }
    if not targets:
        raise ValueError("Target directory is empty or contains no .txt files")
    if not predictions:
        raise ValueError("Prediction directory is empty or contains no .txt files")
    if targets.keys() != predictions.keys():
        raise ValueError(
            "Target and prediction directories must contain the same set of utterance IDs"
        )

    text_encoder = CTCTextEncoder()

    metric_funcs = [
        instantiate(metric_config, text_encoder=text_encoder)
        for metric_config in config.metrics
    ]

    tracker = MetricTracker(*[metric.name for metric in metric_funcs])

    batch = {
        "text": [],
        "predictions": {
            "saved": []
        },  # "saved" is the key used in the config for the decoder.
    }

    for id in sorted(targets):
        target = targets[id].read_text().strip()
        prediction = predictions[id].read_text().strip()

        batch["text"].append(target)
        # "saved" is the key used in the config for the decoder.
        batch["predictions"]["saved"].append(prediction)

    for metric in metric_funcs:
        tracker.update(metric.name, **metric(**batch))

    return {**tracker.result()}


@hydra.main(version_base=None, config_path="../src/configs", config_name="calc_metrics")
def main(config):
    metrics = evaluate(config)
    result = json.dumps(metrics, indent=2)
    print(result)
    if config.output_path is not None:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as file:
            json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    main()
