import os
from typing import Callable, Dict, List, Optional, Union

from torchvision.datasets import ImageFolder

from .base_dataset import BaseDataset


class ImageNetDataset(BaseDataset):
    """This is a wrapper for ImageNet to allow X-Ray dataset semantics."""

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        split: str = "train",
        label_list: Union[str, List[str]] = "all",
        subselect: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__("imagenet", directory, split, label_list, subselect, transform)

        self.dataset = ImageFolder(self.directory / self.split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        sample = {"image": sample[0], "labels": sample[1], "metadata": {}}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
