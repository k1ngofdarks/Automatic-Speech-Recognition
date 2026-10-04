import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    texts = [item["text"] for item in dataset_items]
    audios = [item["audio"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]

    specs = [item["spectrogram"] for item in dataset_items]
    specs_length = [spec.shape[-1] for spec in specs]

    T_max = max(specs_length)
    spectrogram_padded = [F.pad(spec, (0, T_max - spec.shape[-1])) for spec in specs]
    spectrograms = torch.cat(spectrogram_padded, dim=0)
    spectrograms_length = torch.tensor(specs_length, dtype=torch.long)

    texts_encoded = [item["text_encoded"] for item in dataset_items]
    texts_encoded_length = [text.shape[-1] for text in texts_encoded]

    L_max = max(texts_encoded_length)
    text_encoded_padded = [
        F.pad(text, (0, L_max - text.shape[-1])) for text in texts_encoded
    ]
    texts_encoded = torch.cat(text_encoded_padded, dim=0)
    texts_encoded_length = torch.tensor(texts_encoded_length, dtype=torch.long)

    return {
        "text": texts,
        "audio": audios,
        "audio_path": audio_paths,
        "spectrogram": spectrograms,
        "spectrogram_length": spectrograms_length,
        "text_encoded": texts_encoded,
        "text_encoded_length": texts_encoded_length,
    }
