import logging

from .base_dataset import BaseDataset
from .chexpert import CheXpertDataset
from .mimic_cxr import MimicCxrJpgDataset
from .nih_chest_xrays import NIHChestDataset


class CombinedXrayDataset(BaseDataset):
    """Combine several x-ray datasets into one.

    Args:
        directory_list (list, default=None): List of pathlib.Path objects for
            directories for each dataset. If None, it uses the default
            directory for every sub-dataloader.
        dataset_list (list, default='all'): List of datasets to load. Current
            options include:
                'all': Include all datasets.
                'chexpert': Include CheXpert dataset (223,414 in 'train').
                'nih-chest-xrays': Include NIH Chest x-rays (112,120 images in
                    'train').
        split_list (list, default='train'): List of strings specifying split.
            If a string is passed (e.g., 'train'), that split will be
            broacast to all sub-dataloaders.
            options include:
                'all': Include all splits.
                'train': Include training split.
                'val': Include validation split.
        label_list (list, default='all'): String specifying labels to include.
            Default is 'all', which loads all labels from all datasets.
        transform_list (list, default=None): A list of composed transforms. If
            a single composed transform is passed, it will be broadcast to all
            sub-dataloaders.
    """

    def __init__(
        self,
        directory_list=None,
        dataset_list="all",
        split_list="train",
        label_list="all",
        subselect_list=None,
        transform_list=None,
    ):
        self.dataset_name = "combined-xray-dataset"
        if dataset_list == "all":
            dataset_list = ["chexpert_v1", "nih-chest-xrays", "mimic-cxr"]
            self.dataset_list = dataset_list
        else:
            self.dataset_list = dataset_list

        self.directory_list = directory_list = self.to_list(directory_list)
        self.split_list = split_list = self.to_list(split_list)
        self.subselect_list = self.to_list(subselect_list)
        self.transform_list = transform_list = self.to_list(transform_list)
        self.label_list = label_list

        # find all possible labels if using 'all'
        if label_list == "all":
            label_list = []
            for dataset_name in dataset_list:
                dataset = self.dataset_fetcher(dataset_name, 0)
                label_list = label_list + dataset.label_list

            # remove duplicates
            label_list = list(dict.fromkeys(label_list))
            self.label_list = label_list

        self.datasets = []
        for i, dataset_name in enumerate(dataset_list):
            dataset = self.dataset_fetcher(dataset_name, i)

            if dataset is not None:
                self.datasets.append(dataset)
            else:
                logging.info("dataset %s unrecognized, skipping..." % dataset_name)

    def to_list(self, item):
        if not isinstance(item, list):
            item = [item] * len(self.dataset_list)

        assert len(item) == len(self.dataset_list)

        return item

    def dataset_fetcher(self, dataset_name, idx):
        dataset = None
        kwargs = {
            "directory": self.directory_list[idx],
            "split": self.split_list[idx],
            "label_list": self.label_list,
            "subselect": self.subselect_list[idx],
            "transform": self.transform_list[idx],
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if dataset_name == "chexpert_v1":
            return CheXpertDataset(**kwargs)
        elif dataset_name == "nih-chest-xrays":
            return NIHChestDataset(**kwargs)
        elif dataset_name == "mimic-cxr":
            return MimicCxrJpgDataset(**kwargs)

        return dataset

    def __len__(self):
        count = 0
        for dataset in self.datasets:
            count = count + len(dataset)

        return count

    def __getitem__(self, idx):
        sub_idx = idx

        dataset_idx = 0
        dataset = self.datasets[dataset_idx]

        while sub_idx >= len(dataset):
            sub_idx = sub_idx - len(dataset)

            dataset_idx = dataset_idx + 1
            dataset = self.datasets[dataset_idx]

        # no need to transform - this is taken care by subloaders
        item = dataset[sub_idx]

        return item
