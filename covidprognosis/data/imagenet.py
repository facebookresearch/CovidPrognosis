from .base_dataset import BaseDataset
from torchvision.datasets import ImageFolder


class ImageNetDataset(BaseDataset):
    """This is a wrapper for ImageNet to allow X-Ray dataset semantics."""

    def __init__(
        self,
        directory=None,
        split="train",
        label_list="all",
        subselect=None,
        transform=None,
    ):
        super().__init__("imagenet", directory, split, label_list, subselect, transform)

        self.dataset = ImageFolder(self.directory / self.split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        sample = {"image": sample[0], "labels": sample[1], "metadata": {}}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
