import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ..utils import get_dataset_folder


class BaseDataset(Dataset):
    def __init__(
        self, dataset_name, directory, split, label_list, subselect, transform
    ):
        self.dataset_name = dataset_name

        if directory is None:
            directory = get_dataset_folder(self.dataset_name)

        split_list = ["train", "val", "test", "all"]

        if split not in split_list:
            raise ValueError("split {} not a valid split".format(split))

        self.directory = directory
        self.csv = None
        self.split = split
        self.label_list = label_list
        self.subselect = subselect
        self.transform = transform
        self.metadata_keys = []

    def preproc_csv(self, csv, subselect):
        if subselect is not None:
            csv = csv.query(subselect)

        return csv

    def open_image(self, path):
        with open(path, "rb") as f:
            image = Image.open(f).convert("F")
            image.load()

        return image

    @property
    def calc_pos_weights(self):
        pos = (self.csv[self.label_list] == 1).sum()
        neg = (self.csv[self.label_list] == 0).sum()

        neg_pos_ratio = (neg / np.maximum(pos, 1)).values.astype(np.float)

        return neg_pos_ratio

    def retrieve_metadata(self, idx, filename, exam):
        metadata = dict()
        metadata["dataset_name"] = self.dataset_name
        metadata["dataloader class"] = self.__class__.__name__
        metadata["idx"] = idx
        for key in self.metadata_keys:
            # cast to string due to typing issues with dataloader
            metadata[key] = str(exam[key])
        metadata["filename"] = str(filename)

        metadata["label_list"] = self.label_list

        return metadata

    def __repr__(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    @property
    def labels(self):
        return self.label_list
