"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml
from covidprognosis.data import (
    CheXpertDataset,
    CombinedXrayDataset,
    MimicCxrJpgDataset,
    NIHChestDataset,
)
from PIL import Image

DATA_CONFIG = "configs/data.yaml"


def create_input(shape, label_count=12):
    image = np.arange(np.product(shape)).reshape(shape).astype(np.uint8)
    image = Image.fromarray(image)

    labels = []
    for _ in range(label_count):
        if np.random.normal() < 0.1:
            labels.append(np.nan)
        elif np.random.normal() < 0.2:
            labels.append(-1)
        elif np.random.normal() < 0.6:
            labels.append(0)
        else:
            labels.append(1)

    labels = np.array(labels)

    return {"image": image, "labels": labels, "metadata": {}}


def fetch_dataset(dataset_name, transform):
    dataset_name, split = dataset_name.split("_")

    with open(DATA_CONFIG, "r") as f:
        paths = yaml.load(f, Loader=yaml.SafeLoader)["paths"]

        if dataset_name == "combined":
            data_path = [paths["chexpert"], paths["nih"], paths["mimic"]]
            split = [split, split, split]
            transform = [transform, transform, transform]
        else:
            data_path = paths[dataset_name]

    if dataset_name == "combined":
        for path in data_path:
            if not Path(path).exists():
                pytest.skip()
    elif not Path(data_path).exists():
        return None

    if dataset_name == "nih":
        dataset = NIHChestDataset(directory=data_path, split=split, transform=transform)
    elif dataset_name == "chexpert":
        dataset = CheXpertDataset(directory=data_path, split=split, transform=transform)
    elif dataset_name == "mimic":
        dataset = MimicCxrJpgDataset(
            directory=data_path, split=split, transform=transform
        )
    elif dataset_name == "combined":
        dataset = CombinedXrayDataset(
            directory_list=data_path,
            dataset_list=["chexpert_v1", "nih-chest-xrays", "mimic-cxr"],
            split_list=split,
            transform_list=transform,
        )

    return dataset


@pytest.fixture
def dataset_length_dict():
    datalengths = {
        "nih_train": 112120,
        "nih_all": 112120,
        "chexpert_train": 223414,
        "chexpert_val": 234,
        "chexpert_all": 223648,
        "mimic_train": 368960,
        "mimic_val": 2991,
        "mimic_test": 5159,
        "mimic_all": 377110,
        "combined_train": 704494,
        "combined_val": 3225,
        "combined_test": 5159,
        "combined_all": 712878,
    }

    return datalengths
