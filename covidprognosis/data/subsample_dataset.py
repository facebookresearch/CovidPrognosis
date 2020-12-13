import numpy as np
from torch.utils.data import Dataset


class SubsampledDataset(Dataset):
    def __init__(self, dataset, percent_subsample):
        self.dataset = dataset
        self.percent_subsample = percent_subsample

        self.length = int(np.round(self.percent_subsample * len(dataset)))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.dataset[idx]
