import os

import torch

from datasets import Dataset

from datasets import load_from_disk
from torch.utils.data import DataLoader

class CustomedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, key: str, dtype: torch.dtype):
        self.dataset = dataset
        self.key = key
        self.dtype = dtype

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx][self.key]
        return torch.tensor(data, dtype=self.dtype)

class MultiKeyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, keys: list, dtypes: list):
        self.dataset = dataset
        self.keys = keys
        self.dtypes = dtypes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_data = self.dataset[idx]
        data = [torch.tensor(raw_data[k], dtype=dt) for k, dt in zip(self.keys, self.dtypes)]
        return tuple(data)