import numpy as np
from torch.utils.data import Dataset
from .base_dataset import BaseDataset


class SubsampledDataset(Dataset):
    def __init__(self, dataset: BaseDataset, percent_subsample: float):
        self.dataset = dataset
        self.percent_subsample = percent_subsample

        self.length = int(np.round(self.percent_subsample * len(dataset)))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        return self.dataset[idx]
