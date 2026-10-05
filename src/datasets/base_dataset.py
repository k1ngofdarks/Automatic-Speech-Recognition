import logging
import random

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from src.text_encoder import CTCTextEncoder

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index,
        audio_pipeline,
        text_encoder=None,
        target_sr=16000,
        limit=None,
        max_audio_length=None,
        max_text_length=None,
        shuffle_index=False,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            text_encoder (CTCTextEncoder): text encoder.
            target_sr (int): supported sample rate.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            max_audio_length (int): maximum allowed audio length.
            max_test_length (int): maximum allowed text length.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            audio_pipeline (AudioPipeline): audio pipeline for processing
                audio and spectrogram.
        """
        if audio_pipeline is None:
            raise ValueError(
                "Audio pipeline is not provided. Please provide an audio pipeline."
            )

        self._assert_index_is_valid(index)

        index = self._filter_records_from_dataset(
            index, max_audio_length, max_text_length
        )
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index: list[dict] = index

        self.text_encoder = text_encoder
        self.target_sr = target_sr

        self.audio_pipeline = audio_pipeline

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio = self.load_audio(audio_path)
        text = data_dict["text"]
        text_encoded = self.text_encoder.encode(text)

        processed = self.audio_pipeline(audio)

        instance_data = {
            "audio": processed["audio"],
            "spectrogram": processed["spectrogram"],
            "text": text,
            "text_encoded": text_encoded,
            "audio_path": audio_path,
        }

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
        max_audio_length,
        max_text_length,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        the desired max_test_length or max_audio_length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            max_audio_length (int): maximum allowed audio length.
            max_test_length (int): maximum allowed text length.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = (
                np.array([el["audio_len"] for el in index]) >= max_audio_length
            )
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if max_text_length is not None:
            exceeds_text_length = (
                np.array(
                    [len(CTCTextEncoder.normalize_text(el["text"])) for el in index]
                )
                >= max_text_length
            )
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total} ({_total / initial_size:.1%}) records  from dataset"
            )

        return index

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - object ground-truth transcription."
            )
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - length of the audio."
            )

    @staticmethod
    def _sort_index(index):
        """
        Sort index by audio length.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["audio_len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
