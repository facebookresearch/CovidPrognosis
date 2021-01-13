"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import pytest
import torchvision.transforms as tvt
from covidprognosis.data.transforms import Compose

from .conftest import fetch_dataset


@pytest.mark.parametrize(
    "dataset_name",
    [
        "nih_train",
        "nih_all",
        "chexpert_train",
        "chexpert_val",
        "chexpert_all",
        "mimic_train",
        "mimic_val",
        "mimic_test",
        "mimic_all",
        "combined_train",
        "combined_val",
        "combined_test",
        "combined_all",
    ],
)
def test_dataset_lengths(dataset_name, dataset_length_dict):
    transform = Compose([tvt.ToTensor()])

    dataset = fetch_dataset(dataset_name, transform)

    if dataset is None:
        pytest.skip()
    else:
        assert len(dataset) == dataset_length_dict[dataset_name]


@pytest.mark.parametrize(
    "dataset_name",
    [
        "nih_train",
        "nih_all",
        "chexpert_train",
        "chexpert_val",
        "chexpert_all",
        "mimic_train",
        "mimic_val",
        "mimic_test",
        "mimic_all",
        "combined_train",
        "combined_val",
        "combined_test",
        "combined_all",
    ],
)
def test_dataset_getitem(dataset_name):
    transform = Compose([tvt.ToTensor()])

    dataset = fetch_dataset(dataset_name, transform)

    if dataset is None:
        pytest.skip()
    else:
        item1 = dataset[0]
        item2 = dataset[-1]

        assert item1 is not None
        assert item2 is not None


def test_combined_loader():
    transform = Compose([tvt.ToTensor()])
    dataset = fetch_dataset("combined_all", transform=transform)

    sample = dataset[0]
    assert "CheXpert" in str(sample["metadata"]["filename"])

    sample = dataset[300000]
    assert "nih-chest-xrays" in str(sample["metadata"]["filename"])

    sample = dataset[600000]
    assert "mimic-cxr-jpg" in str(sample["metadata"]["filename"])
