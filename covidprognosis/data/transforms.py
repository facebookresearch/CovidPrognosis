"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


class XRayTransform:
    """XRayTransform base class."""

    def __repr__(self):
        return "XRayTransform: {}".format(self.__class__.__name__)


class Compose(XRayTransform):
    """
    Compose a list of transforms into one.

    Args:
        transforms: The list of transforms.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            if isinstance(t, XRayTransform):
                sample = t(sample)
            else:
                # assume torchvision transform
                sample["image"] = t(sample["image"])

        return sample


class NanToInt(XRayTransform):
    """
    Convert an np.nan label to an integer.

    Args:
        num: Integer to convert to.
    """

    def __init__(self, num: int = -100):
        self.num = num

    def __call__(self, sample: Dict) -> Dict:
        sample["labels"][np.isnan(sample["labels"])] = self.num

        return sample


class RemapLabel(XRayTransform):
    """
    Convert a label from one value to another.

    Args:
        input_val: Value to convert from.
        output_val: Value to convert to.
    """

    def __init__(self, input_val: Union[float, int], output_val: Union[float, int]):
        self.input_val = input_val
        self.output_val = output_val

    def __call__(self, sample: Dict) -> Dict:
        labels = np.copy(sample["labels"])

        labels[labels == self.input_val] = self.output_val

        sample["labels"] = labels

        return sample


class HistogramNormalize(XRayTransform):
    """
    Apply histogram normalization.

    Args:
        number_bins: Number of bins to use in histogram.
    """

    def __init__(self, number_bins: int = 256):
        self.number_bins = number_bins

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"].numpy()

        # get image histogram
        image_histogram, bins = np.histogram(
            image.flatten(), self.number_bins, density=True
        )
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
        image_equalized.reshape(image.shape)

        sample["image"] = torch.tensor(image_equalized.reshape(image.shape)).to(
            sample["image"]
        )

        return sample


class RandomGaussianBlur(XRayTransform):
    """
    Random Gaussian blur transform.

    Args:
        p: Probability to apply transform.
        sigma_range: Range of sigma values for Gaussian kernel.
    """

    def __init__(self, p: float = 0.5, sigma_range: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, sample: Dict) -> Dict:
        if np.random.uniform() <= self.p:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

            sample["image"] = torch.tensor(
                gaussian_filter(sample["image"].numpy(), sigma),
                dtype=sample["image"].dtype,
            )

        return sample


class AddGaussianNoise(XRayTransform):
    """
    Gaussian noise transform.

    Args:
        p: Probability of adding Gaussian noise.
        snr_range: SNR range for Gaussian noise addition.
    """

    def __init__(self, p: float = 0.5, snr_range: Tuple[float, float] = (2.0, 8.0)):
        self.p = p
        self.snr_range = snr_range

    def __call__(self, sample: Dict) -> Dict:
        if np.random.uniform() <= self.p:
            snr_level = np.random.uniform(low=self.snr_range[0], high=self.snr_range[1])
            signal_level = np.mean(sample["image"].numpy())

            # use numpy to keep things consistent on numpy random seed
            sample["image"] = sample["image"] + (
                signal_level / snr_level
            ) * torch.tensor(
                np.random.normal(size=tuple(sample["image"].shape)),
                dtype=sample["image"].dtype,
            )

        return sample


class TensorToRGB(XRayTransform):
    """
    Convert Tensor to RGB by replicating channels.

    Args:
        num_output_channels: Number of output channels (3 for RGB).
    """

    def __init__(self, num_output_channels: int = 3):
        self.num_output_channels = num_output_channels

    def __call__(self, sample: Dict) -> Dict:
        expands = list()
        for i in range(sample["image"].ndim):
            if i == 0:
                expands.append(self.num_output_channels)
            else:
                expands.append(-1)
        sample["image"] = sample["image"].expand(*expands)

        return sample
