"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
from typing import Callable, List, Optional, Union

from .base_dataset import BaseDataset
from .chexpert import CheXpertDataset
from .mimic_cxr import MimicCxrJpgDataset
from .nih_chest_xrays import NIHChestDataset


class CombinedXrayDataset(BaseDataset):
    """
    Combine several x-ray datasets into one.

    Args:
        directory_list: List of paths for directories for each dataset.
        dataset_list: List of datasets to load. Current options include:
                'all': Include all datasets.
                'chexpert': Include CheXpert dataset (223,414 in 'train').
                'nih-chest-xrays': Include NIH Chest x-rays (112,120 images in
                    'train').
        split_list: List of strings specifying split. If a string is passed
            (e.g., 'train'), that split will be broacast to all
            sub-dataloaders.
            options include:
                'all': Include all splits.
                'train': Include training split.
                'val': Include validation split.
        label_list: String specifying labels to include. Default is 'all',
            which loads all labels from all datasets.
        transform_list: A list of composed transforms. If a single composed
            transform is passed, it will be broadcast to all sub-dataloaders.
    """

    def __init__(
        self,
        directory_list: List[Union[str, os.PathLike]],
        dataset_list: Union[str, List[str]] = "all",
        split_list: Union[str, List[str]] = "train",
        label_list: Union[str, List[str]] = "all",
        subselect_list: Optional[List[str]] = None,
        transform_list: Optional[List[Optional[Callable]]] = None,
    ):
        self.dataset_name = "combined-xray-dataset"
        if dataset_list == "all":
            dataset_list = ["chexpert_v1", "nih-chest-xrays", "mimic-cxr"]
            self.dataset_list = dataset_list
        elif isinstance(dataset_list, str):
            raise RuntimeError("Unrecognized dataset list string.")
        else:
            self.dataset_list = dataset_list  # type:ignore

        self.directory_list = directory_list = self.to_list(directory_list)
        self.split_list = split_list = self.to_list(split_list)
        self.subselect_list = self.to_list(subselect_list)
        self.transform_list = transform_list = self.to_list(transform_list)

        # find all possible labels if using 'all'
        if label_list == "all":
            self.label_list = self.fetch_label_list(self.dataset_list)
        else:
            if isinstance(label_list, str):
                raise ValueError(
                    "If inputting label_list, label_list must not be a string"
                )
            self.label_list = label_list

        self.datasets = []
        for (dataset_name, directory, split, subselect, transform) in zip(
            self.dataset_list,
            self.directory_list,
            self.split_list,
            self.subselect_list,
            self.transform_list,
        ):
            self.datasets.append(
                self.fetch_dataset(
                    dataset_name,
                    directory,
                    split,
                    self.label_list,
                    subselect,
                    transform,
                )
            )

    def to_list(self, item):
        if not isinstance(item, list):
            item = [item] * len(self.dataset_list)

        assert len(item) == len(self.dataset_list)

        return item

    def fetch_label_list(self, dataset_name_list: List[str]) -> List[str]:
        label_list: List[str] = []
        for dataset_name in dataset_name_list:
            if dataset_name == "chexpert_v1":
                label_list = label_list + CheXpertDataset.default_labels()
            elif dataset_name == "nih-chest-xrays":
                label_list = label_list + NIHChestDataset.default_labels()
            elif dataset_name == "mimic-cxr":
                label_list = label_list + MimicCxrJpgDataset.default_labels()

        # remove duplicates
        label_list = list(set(label_list))

        return label_list

    def fetch_dataset(
        self,
        dataset_name: str,
        directory: Union[str, os.PathLike],
        split: str,
        label_list: Union[str, List[str]],
        subselect: str,
        transform: Callable,
    ) -> BaseDataset:
        dataset: BaseDataset

        if dataset_name == "chexpert_v1":
            dataset = CheXpertDataset(
                directory=directory,
                split=split,
                label_list=label_list,
                subselect=subselect,
                transform=transform,
            )
        elif dataset_name == "nih-chest-xrays":
            dataset = NIHChestDataset(
                directory=directory,
                split=split,
                label_list=label_list,
                subselect=subselect,
                transform=transform,
            )
        elif dataset_name == "mimic-cxr":
            dataset = MimicCxrJpgDataset(
                directory=directory,
                split=split,
                label_list=label_list,
                subselect=subselect,
                transform=transform,
            )
        else:
            raise RuntimeError(f"Data set {dataset_name} not found.")

        return dataset

    def __len__(self) -> int:
        count = 0
        for dataset in self.datasets:
            count = count + len(dataset)

        return count

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self) + idx

        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            else:
                idx = idx - len(dataset)
