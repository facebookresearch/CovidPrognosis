"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
from argparse import ArgumentParser
from typing import Callable, List, Optional, Union

import covidprognosis as cp
import numpy as np
import pytorch_lightning as pl
import torch


class TwoImageDataset(torch.utils.data.Dataset):
    """
    Wrapper for returning two augmentations of the same image.

    Args:
        dataset: Pre-initialized data set to return multiple samples from.
    """

    def __init__(self, dataset: cp.data.BaseDataset):
        assert isinstance(dataset, cp.data.BaseDataset)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # randomness handled via the transform objects
        # this requires the transforms to sample randomness from the process
        # generator
        item0 = self.dataset[idx]
        item1 = self.dataset[idx]

        sample = {
            "image0": item0["image"],
            "image1": item1["image"],
            "label": item0["labels"],
        }

        return sample


def fetch_dataset(
    dataset_name: str,
    dataset_dir: Union[List[Union[str, os.PathLike]], Union[str, os.PathLike]],
    split: str,
    transform: Optional[Callable],
    two_image: bool = False,
    label_list="all",
):
    """Dataset fetcher for config handling."""

    assert split in ("train", "val", "test")
    dataset: Union[cp.data.BaseDataset, TwoImageDataset]

    # determine the dataset
    if dataset_name == "nih":
        assert not isinstance(dataset_dir, list)
        dataset = cp.data.NIHChestDataset(
            directory=dataset_dir,
            split=split,
            transform=transform,
            label_list=label_list,
            resplit=True,
        )
    if dataset_name == "mimic":
        assert not isinstance(dataset_dir, list)
        dataset = cp.data.MimicCxrJpgDataset(
            directory=dataset_dir,
            split=split,
            transform=transform,
            label_list=label_list,
        )
    elif dataset_name == "chexpert":
        assert not isinstance(dataset_dir, list)
        dataset = cp.data.CheXpertDataset(
            directory=dataset_dir,
            split=split,
            transform=transform,
            label_list=label_list,
        )
    elif dataset_name == "mimic-chexpert":
        assert isinstance(dataset_dir, list)
        dataset = cp.data.CombinedXrayDataset(
            dataset_list=["chexpert_v1", "mimic-cxr"],
            directory_list=dataset_dir,
            transform_list=[transform, transform],
            label_list=[label_list, label_list],
            split_list=[split, split],
        )
    else:
        raise ValueError(f"dataset {dataset_name} not recognized")

    if two_image is True:
        dataset = TwoImageDataset(dataset)

    return dataset


def worker_init_fn(worker_id):
    """Handle random seeding."""
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed % (2 ** 32 - 1)  # pylint: disable=no-member

    np.random.seed(seed)


class XrayDataModule(pl.LightningDataModule):
    """
    X-ray data module for training models with PyTorch Lightning.

    Args:
        dataset_name: Name of the dataset.
        dataset_dir: Location of the data.
        label_list: Labels to load for training.
        batch_size: Training batch size.
        num_workers: Number of workers for dataloaders.
        use_two_images: Whether to return two augmentations of same image from
            dataset (for MoCo pretraining).
        train_transform: Transform for training loop.
        val_transform: Transform for validation loop.
        test_transform: Transform for test loop.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_dir: Union[List[Union[str, os.PathLike]], Union[str, os.PathLike]],
        label_list: Union[str, List[str]] = "all",
        batch_size: int = 1,
        num_workers: int = 4,
        use_two_images: bool = False,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = fetch_dataset(
            self.dataset_name,
            self.dataset_dir,
            "train",
            train_transform,
            label_list=label_list,
            two_image=use_two_images,
        )
        self.val_dataset = fetch_dataset(
            self.dataset_name,
            self.dataset_dir,
            "val",
            val_transform,
            label_list=label_list,
            two_image=use_two_images,
        )
        self.test_dataset = fetch_dataset(
            self.dataset_name,
            self.dataset_dir,
            "test",
            test_transform,
            label_list=label_list,
            two_image=use_two_images,
        )

        if isinstance(self.train_dataset, TwoImageDataset):
            self.label_list = None
        else:
            self.label_list = self.train_dataset.label_list

    def __dataloader(self, split: str) -> torch.utils.data.DataLoader:
        assert split in ("train", "val", "test")
        shuffle = False
        if split == "train":
            dataset = self.train_dataset
            shuffle = True
        elif split == "val":
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
        )

        return loader

    def train_dataloader(self):
        return self.__dataloader(split="train")

    def val_dataloader(self):
        return self.__dataloader(split="val")

    def test_dataloader(self):
        return self.__dataloader(split="test")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--dataset_name", default="mimic", type=str)
        parser.add_argument("--dataset_dir", default=None, type=str)
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--num_workers", default=4, type=int)

        return parser
