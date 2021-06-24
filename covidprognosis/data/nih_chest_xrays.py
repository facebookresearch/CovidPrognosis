"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import os
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset


class NIHChestDataset(BaseDataset):
    """
    Data loader for NIH data set.

    Args:
        directory: Base directory for data set.
        split: String specifying split.
            options include:
                'all': Include all splits.
                'train': Include training split.
        label_list: String specifying labels to include. Default is 'all',
            which loads all labels.
        transform: A composible transform list to be applied to the data.
    """

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        split: str = "train",
        label_list: Union[str, List[str]] = "all",
        subselect: Optional[str] = None,
        transform: Optional[Callable] = None,
        resplit: bool = False,
        resplit_seed: int = 2019,
        
    ):
        if resplit_ratios is None: 44            
            resplit_ratios = [0.7, 0.2, 0.1]
        super().__init__(
            "nih-chest-xrays", directory, split, label_list, subselect, transform
        )

        if label_list == "all":
            self.label_list = self.default_labels()
        else:
            self.label_list = label_list

        self.metadata_keys = [
            "Image Index",
            "Follow-up #",
            "Patient ID",
            "Patient Age",
            "Patient Gender",
            "View Position",
        ]

        if resplit:
            rg = np.random.default_rng(resplit_seed)

            self.csv_path = self.directory / "Data_Entry_2017.csv"
            csv = pd.read_csv(self.csv_path)
            patient_list = csv["Patient ID"].unique()

            rand_inds = rg.permutation(len(patient_list))

            train_count = int(np.round(resplit_ratios[0] * len(patient_list)))
            val_count = int(np.round(resplit_ratios[1] * len(patient_list)))

            grouped = csv.groupby("Patient ID")

            if self.split == "train":
                patient_list = patient_list[rand_inds[:train_count]]
                self.csv = pd.concat([grouped.get_group(pat) for pat in patient_list])
            elif self.split == "val":
                patient_list = patient_list[
                    rand_inds[train_count : train_count + val_count]
                ]
                self.csv = pd.concat([grouped.get_group(pat) for pat in patient_list])
            elif self.split == "test":
                patient_list = patient_list[rand_inds[train_count + val_count :]]
                self.csv = pd.concat([grouped.get_group(pat) for pat in patient_list])
            else:
                logging.warning(
                    "split {} not recognized for dataset {}, "
                    "not returning samples".format(split, self.__class__.__name__)
                )
        else:
            if self.split == "train":
                self.csv_path = self.directory / "Data_Entry_2017.csv"
                self.csv = pd.read_csv(self.csv_path)
            elif self.split == "all":
                self.csv_path = self.directory / "Data_Entry_2017.csv"
                self.csv = pd.read_csv(self.csv_path)
            else:
                logging.warning(
                    "split {} not recognized for dataset {}, "
                    "not returning samples".format(split, self.__class__.__name__)
                )

        self.csv = self.preproc_csv(self.csv, self.subselect)

    @staticmethod
    def default_labels() -> List[str]:
        return [
            "Atelectasis",
            "Consolidation",
            "Infiltration",
            "Pneumothorax",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Effusion",
            "Pneumonia",
            "Pleural_Thickening",
            "Cardiomegaly",
            "Nodule",
            "Mass",
            "Hernia",
        ]

    def preproc_csv(self, csv: pd.DataFrame, subselect: Optional[str]) -> pd.DataFrame:
        if csv is not None:

            def format_view(s):
                return "frontal" if s in ("AP", "PA") else None

            csv["view"] = csv["View Position"].apply(format_view)

            if subselect is not None:
                csv = csv = csv.query(subselect)

        return csv

    def __len__(self) -> int:
        length = 0
        if self.csv is not None:
            length = len(self.csv)

        return length

    def __getitem__(self, idx: int) -> Dict:
        assert self.csv is not None
        exam = self.csv.iloc[idx]

        filename = self.directory / "images" / exam["Image Index"]
        image = self.open_image(filename)

        metadata = self.retrieve_metadata(idx, filename, exam)

        # example: exam['Finding Labels'] = 'Pneumonia|Cardiomegaly'
        # goal here is to see if label is a substring of
        # 'Pneumonia|Cardiomegaly' for each label in self.label_list
        labels = [
            1 if label in exam["Finding Labels"] else 0 for label in self.label_list
        ]
        labels = np.array(labels).astype(np.float)

        sample = {"image": image, "labels": labels, "metadata": metadata}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
