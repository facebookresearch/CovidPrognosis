import logging

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


class XRayTransform(object):
    def __repr__(self):
        return "XRayTransform: {}".format(self.__class__.__name__)


class Compose(XRayTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            if isinstance(t, XRayTransform):
                sample = t(sample)
            else:
                # assume torchvision transform
                sample["image"] = t(sample["image"])

        return sample


class PathToNan(XRayTransform):
    def __init__(self, pathology):
        self.pathology = pathology

    def __call__(self, sample):
        idx = sample["metadata"]["label_list"].index(self.pathology)

        sample["labels"][idx] = np.nan

        return sample


class NanToInt(XRayTransform):
    def __init__(self, num=-100):
        self.num = num

    def __call__(self, sample):
        sample["labels"][np.isnan(sample["labels"])] = self.num

        return sample


class RemapLabel(XRayTransform):
    def __init__(self, input_val, output_val):
        self.input_val = input_val
        self.output_val = output_val

    def __call__(self, sample):
        labels = np.copy(sample["labels"])

        labels[labels == self.input_val] = self.output_val

        sample["labels"] = labels

        return sample


class LabelToTensor(XRayTransform):
    def __call__(self, sample):
        sample["labels"] = torch.tensor(sample["labels"])

        return sample


class ToRGB(XRayTransform):
    def __call__(self, sample):
        img = sample["image"]
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        sample["image"] = Image.fromarray(img, mode="RGB")

        return sample


class MedMadNormalize(XRayTransform):
    def __init__(self, eps=1e-05):
        self.eps = eps
        warn_msg = "MedMadNormalize is being deprecated due to name ambiguity, "
        warn_msg = warn_msg + "use MedianMaximumAbsoluteDifferenceNormalize instead"
        logging.warn(warn_msg)

    def __call__(self, sample):
        median = torch.median(sample["image"])
        mad = torch.max(sample["image"]) - torch.min(sample["image"])

        sample["image"] = (sample["image"] - median) / (mad + self.eps)

        return sample


class MedianMaximumAbsoluteDifferenceNormalize(XRayTransform):
    def __init__(self, eps=1e-05):
        self.eps = eps

    def __call__(self, sample):
        median = torch.median(sample["image"])
        mad = torch.max(sample["image"]) - torch.min(sample["image"])

        sample["image"] = (sample["image"] - median) / (mad + self.eps)

        return sample


class MedianMeanAbsoluteDifferenceNormalize(XRayTransform):
    def __init__(self, eps=1e-05):
        self.eps = eps

    def __call__(self, sample):
        median = torch.median(sample["image"])
        mad = torch.mean(torch.abs(sample["image"] - median))

        sample["image"] = (sample["image"] - median) / (mad + self.eps)

        return sample


class HistogramNormalize(XRayTransform):
    def __init__(self, number_bins=256):
        self.number_bins = number_bins

    def __call__(self, sample):
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
    def __init__(self, p=0.5, sigma_range=[0.1, 2.0]):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, sample):
        if np.random.uniform() <= self.p:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

            sample["image"] = torch.from_numpy(
                gaussian_filter(sample["image"].numpy(), sigma)
            )

        return sample


class AddGaussianNoise(XRayTransform):
    def __init__(self, snr_range=(2, 8), p=0.5):
        self.snr_range = snr_range
        self.p = p

    def __call__(self, sample):
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


class AddPoissonNoise(XRayTransform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.uniform() <= self.p:
            # use numpy to keep things consistent on numpy random seed
            sample["image"] = torch.tensor(
                np.random.poisson(lam=np.absolute(sample["image"].numpy())),
                dtype=sample["image"].dtype,
            )

        return sample


class AffineTransform(XRayTransform):
    def __init__(self, slope_std_param=1.0, offset_std_param=1.0, p=0.5):
        self.slope_std_param = slope_std_param
        self.offset_std_param = offset_std_param
        self.p = p

    def __call__(self, sample):
        if np.random.uniform() <= self.p:
            ny, nx = (
                sample["image"].shape[-2],
                sample["image"].shape[-1],
            )
            image_range = np.max(sample["image"].numpy()) - np.min(
                sample["image"].numpy()
            )

            # use numpy to keep things consistent on numpy random seed
            slope = np.random.normal(
                scale=self.slope_std_param * image_range
            ) / np.maximum(ny, nx)
            offset = np.random.normal(scale=self.offset_std_param * image_range)
            angle = np.random.uniform(high=np.pi)

            yy, xx = np.meshgrid(np.arange(ny) - ny / 2, np.arange(nx) - nx / 2)

            sample["image"] = sample["image"] + torch.tensor(
                slope * (xx * np.cos(angle) + yy * np.sin(angle)) + offset,
                dtype=sample["image"].dtype,
            )

        return sample


class TensorToRGB(XRayTransform):
    def __init__(self, num_output_channels=3):
        self.num_output_channels = num_output_channels

    def __call__(self, sample):
        expands = list()
        for i in range(sample["image"].ndim):
            if i == 0:
                expands.append(self.num_output_channels)
            else:
                expands.append(-1)
        sample["image"] = sample["image"].expand(*expands)

        return sample
