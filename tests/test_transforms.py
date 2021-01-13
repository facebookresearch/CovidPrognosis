"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import covidprognosis.data.transforms as cpt
import numpy as np
import pytest
import torch
import torchvision.transforms as tvt
from scipy.ndimage import gaussian_filter

from .conftest import create_input


@pytest.mark.parametrize("shape", [[32, 32, 3], [45, 16, 3]])
def test_compose(shape):
    sample = create_input(shape)
    transform = cpt.Compose(
        [tvt.RandomHorizontalFlip(), tvt.ToTensor(), cpt.RandomGaussianBlur()]
    )

    sample = transform(sample)

    assert sample["image"] is not None


@pytest.mark.parametrize("shape, label_idx", [[[32, 32, 3], 0], [[45, 16, 3], 5]])
def test_nan_to_int(shape, label_idx):
    sample = create_input(shape)
    transform = cpt.Compose([tvt.ToTensor(), cpt.NanToInt(5)])

    sample["labels"][label_idx] = np.nan

    sample = transform(sample)

    assert sample["labels"][label_idx] == 5


@pytest.mark.parametrize(
    "shape, label_idx, start_label, end_label",
    [[[32, 32, 3], 2, -1, 0], [[45, 16, 3], 10, 1, 0]],
)
def test_remap_label(shape, label_idx, start_label, end_label):
    sample = create_input(shape)
    transform = cpt.Compose([tvt.ToTensor(), cpt.RemapLabel(start_label, end_label)])

    sample["labels"][label_idx] = start_label

    sample = transform(sample)

    assert sample["labels"][label_idx] == end_label


@pytest.mark.parametrize("shape", [[32, 32, 3], [45, 16, 3]])
def test_histnorm(shape):
    """Test this to guard against an implementation change."""
    sample = create_input(shape)
    transform = cpt.Compose([tvt.ToTensor(), cpt.HistogramNormalize()])

    image = np.transpose(
        torch.tensor(np.array(sample["image"]), dtype=torch.float).numpy(), (2, 0, 1)
    )
    # get image histogram
    image_histogram, bins = np.histogram(
        image.flatten(), transform.transforms[1].number_bins, density=True
    )
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    image_equalized.reshape(image.shape)

    image = torch.tensor(image_equalized.reshape(image.shape)).to(torch.float)

    sample = transform(sample)

    assert torch.allclose(sample["image"], image)


@pytest.mark.parametrize("shape", [[32, 32, 3], [45, 16, 3]])
def test_rand_gauss_blur(shape):
    """Test this to guard against an implementation change."""
    seed = 123

    sample = create_input(shape)
    transform = cpt.Compose([tvt.ToTensor(), cpt.RandomGaussianBlur(p=1)])

    # run the custom blur
    np.random.seed(seed)
    image = tvt.functional.to_tensor(sample["image"]) * 1
    sigma = np.random.uniform(
        transform.transforms[1].sigma_range[0], transform.transforms[1].sigma_range[1]
    )

    image = torch.tensor(gaussian_filter(image.numpy(), sigma), dtype=image.dtype,)

    # transform blur
    transform = cpt.Compose(
        [tvt.ToTensor(), cpt.RandomGaussianBlur(p=1, sigma_range=(sigma, sigma))]
    )
    sample = transform(sample)

    assert torch.allclose(sample["image"], image)

    # retest for 0 probability
    sample = create_input(shape)
    transform = cpt.Compose([tvt.ToTensor(), cpt.RandomGaussianBlur(p=-0.1)])

    # run the custom blur
    image = tvt.functional.to_tensor(sample["image"]) * 1

    # transform blur
    sample = transform(sample)

    assert torch.allclose(sample["image"], image)


@pytest.mark.parametrize("shape", [[32, 32, 3], [45, 16, 3]])
def test_add_noise(shape):
    """Test this to guard against an implementation change."""
    seed = 456

    sample = create_input(shape)
    transform = cpt.Compose([tvt.ToTensor(), cpt.AddGaussianNoise(p=1)])

    # run the custom noise
    np.random.seed(seed)
    image = tvt.functional.to_tensor(sample["image"]) * 1
    np.random.uniform()
    snr_level = np.random.uniform(
        low=transform.transforms[1].snr_range[0],
        high=transform.transforms[1].snr_range[1],
    )
    signal_level = np.mean(image.numpy())

    image = image + (signal_level / snr_level) * torch.tensor(
        np.random.normal(size=tuple(image.shape)), dtype=image.dtype,
    )

    # transform blur
    np.random.seed(seed)
    sample = transform(sample)

    assert torch.allclose(sample["image"], image)

    # retest for 0 probability
    sample = create_input(shape)
    transform = cpt.Compose([tvt.ToTensor(), cpt.AddGaussianNoise(p=-0.1)])

    # run the custom blur
    image = tvt.functional.to_tensor(sample["image"]) * 1

    # transform blur
    sample = transform(sample)

    assert torch.allclose(sample["image"], image)


@pytest.mark.parametrize("shape", [[32, 32, 3], [45, 16, 3]])
def test_tensor_to_rgb(shape):
    sample = create_input(shape)
    transform = cpt.Compose([tvt.ToTensor(), cpt.TensorToRGB()])

    image = tvt.functional.to_tensor(sample["image"]) * 1
    expands = list()
    for i in range(image.ndim):
        if i == 0:
            expands.append(3)
        else:
            expands.append(-1)

    image = image.expand(*expands)

    sample = transform(sample)

    assert torch.allclose(sample["image"], image)
