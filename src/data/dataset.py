"""Protein dataset class."""
import gzip
import os
import pickle
from pathlib import Path
from glob import glob
from typing import Optional, Union
from functools import lru_cache
import numpy as np
import pandas as pd
import torch

from src.data.transform import BioFeatureTransform


class BioTrainingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path_to_dataset: Union[Path, str],
                 transform: Optional[BioFeatureTransform] = None,
                 training: bool = True,
                 ):
        super().__init__()
        path_to_dataset = os.path.expanduser(path_to_dataset)

        assert path_to_dataset.endswith('.csv'), f"Invalid file extension: {path_to_dataset} (have to be .csv)"
        self._df = pd.read_csv(path_to_dataset)

        self._df = self._df[self._df['token_num'] > 20]
        self._df.sort_values('token_num', ascending=False)
        self._data = self._df['processed_path'].tolist()
        self._embedding = self._df['embedding_path'].tolist()

        self.data = np.asarray(self._data)
        self.embedding = np.asarray(self._embedding)
        self.transform = transform
        self.training = training  # not implemented yet

    @property
    def num_samples(self):
        return len(self.data)

    def len(self):
        return self.__len__()

    def __len__(self):
        return self.num_samples

    def get(self, idx):
        return self.__getitem__(idx)

    @lru_cache(maxsize=100)
    def __getitem__(self, idx):
        data_path = self.data[idx]
        accession_code = os.path.basename(data_path).split('.')[0]

        with gzip.open(data_path, 'rb') as f:
            data_object = pickle.load(f)

        data_object['plm_embedding'] = torch.load(self.embedding[idx])

        if self.transform is not None:
            data_object = self.transform(data_object)
        data_object['accession_code'] = accession_code
        return data_object


class BioInferenceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path_to_dataset: Union[Path, str],
                 suffix: str = 'pkl.gz',
                 transform: Optional[BioFeatureTransform] = None,
                 ):
        super().__init__()
        path_to_dataset = os.path.expanduser(path_to_dataset)

        if os.path.isfile(path_to_dataset):
            assert path_to_dataset.endswith('.csv'), f"Invalid file extension: {path_to_dataset} (have to be .csv)"
            self._df = pd.read_csv(path_to_dataset)
            self._df.sort_values('token_num', ascending=False)
            self._df = self._df[self._df['token_num'] <= 2048]
            self._df.sort_values('token_num', ascending=False)
            self._data = self._df['processed_path'].tolist()
            self._embedding = self._df['embedding'].tolist()

        elif os.path.isdir(path_to_dataset):
            self._data = glob(os.path.join(path_to_dataset, f'*.{suffix}'))

        self.data = np.asarray(self._data)
        self.embedding = np.asarray(self._embedding)
        self.transform = transform

    @property
    def num_samples(self):
        return len(self.data)

    def len(self):
        return self.__len__()

    def __len__(self):
        return self.num_samples

    def get(self, idx):
        return self.__getitem__(idx)

    @lru_cache(maxsize=100)
    def __getitem__(self, idx):
        data_path = self.data[idx]
        accession_code = os.path.basename(data_path).split('.')[0]

        if data_path.endswith('.pkl.gz'):
            with gzip.open(data_path, 'rb') as f:
                data_object = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        data_object['plm_embedding'] = torch.load(self.embedding[idx])

        if self.transform is not None:
            data_object = self.transform(data_object)
        data_object['accession_code'] = accession_code
        return data_object
