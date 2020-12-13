import covidprognosis as cp
import pytorch_lightning as pl
import torch


class TwoImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        assert isinstance(dataset, cp.data.BaseDataset)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # randomness handled via the transform objects
        item0 = self.dataset[idx]
        item1 = self.dataset[idx]

        sample = {
            "image0": item0["image"],
            "image1": item1["image"],
            "label": item0["labels"],
        }

        return sample


def fetch_dataset(
    dataset_name, dataset_dir, split, transform, two_image=True, label_list="all"
):
    assert split in ("train", "val", "test")

    # determine the dataset
    if dataset_name == "nih":
        dataset = cp.data.NIHChestDataset(
            directory=dataset_dir,
            split=split,
            transform=transform,
            label_list=label_list,
            resplit=True,
        )
    if dataset_name == "mimic":
        dataset = cp.data.MimicCxrJpgDataset(
            directory=dataset_dir,
            split=split,
            transform=transform,
            label_list=label_list,
        )
    elif dataset_name == "chexpert":
        dataset = cp.data.CheXpertDataset(
            directory=dataset_dir,
            split=split,
            transform=transform,
            label_list=label_list,
        )
    elif dataset_name == "mimic-chexpert":
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


class XrayDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        dataset_dir,
        batch_size=128,
        num_workers=4,
        im_size=224,
        random_resized_crop_scale=(0.2, 1.0),
        snr_range=(4, 8),
        uncertain_label=0,
        nan_label=0,
        train_transform_list=None,
        val_transform_list=None,
        test_transform_list=None,
    ):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform_list = train_transform_list
        self.val_transform_list = val_transform_list
        self.test_transform_list = test_transform_list

    def __dataloader(self, split):
        assert split in ("train", "val", "test")
        if split == "train":
            transform = self.train_transform_list
        elif split == "val":
            transform = self.val_transform_list
        else:
            transform = self.test_transform_list

        dataset = fetch_dataset(self.dataset_name, self.dataset_dir, split, transform)

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

        return loader

    def train_dataloader(self):
        return self.__dataloader(split="train")

    def val_dataloader(self):
        return self.__dataloader(split="val")

    def test_dataloader(self):
        return self.__dataloader(split="test")
