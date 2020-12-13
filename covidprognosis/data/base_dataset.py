import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        directory: Union[str, os.PathLike],
        split: str,
        label_list: Union[str, List[str]],
        subselect: Optional[str],
        transform: Optional[Callable],
    ):
        self.dataset_name = dataset_name

        split_list = ["train", "val", "test", "all"]

        if split not in split_list:
            raise ValueError("split {} not a valid split".format(split))

        self.directory = Path(directory)
        self.csv = None
        self.split = split
        self.label_list = label_list
        self.subselect = subselect
        self.transform = transform
        self.metadata_keys: List[str] = []

    def preproc_csv(self, csv: pd.DataFrame, subselect: str) -> pd.DataFrame:
        if subselect is not None:
            csv = csv.query(subselect)

        return csv

    def open_image(self, path: Union[str, os.PathLike]) -> Image:
        with open(path, "rb") as f:
            image = Image.open(f).convert("F")
            image.load()

        return image

    def __len__(self) -> int:
        return 0

    @property
    def calc_pos_weights(self) -> float:
        if self.csv is None:
            return 0.0

        pos = (self.csv[self.label_list] == 1).sum()
        neg = (self.csv[self.label_list] == 0).sum()

        neg_pos_ratio = (neg / np.maximum(pos, 1)).values.astype(np.float)

        return neg_pos_ratio

    def retrieve_metadata(
        self, idx: int, filename: Union[str, os.PathLike], exam: pd.Series
    ) -> Dict:
        metadata = {}
        metadata["dataset_name"] = self.dataset_name
        metadata["dataloader class"] = self.__class__.__name__
        metadata["idx"] = idx  # type: ignore
        for key in self.metadata_keys:
            # cast to string due to typing issues with dataloader
            metadata[key] = str(exam[key])
        metadata["filename"] = str(filename)

        metadata["label_list"] = self.label_list  # type: ignore

        return metadata

    def __repr__(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    @property
    def labels(self) -> Union[str, List[str]]:
        return self.label_list
