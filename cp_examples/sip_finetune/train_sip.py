"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from warnings import warn

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from covidprognosis.data.transforms import (
    Compose,
    HistogramNormalize,
    NanToInt,
    RemapLabel,
    TensorToRGB,
)
from covidprognosis.plmodules import XrayDataModule
from torchvision import transforms

from sip_finetune import SipModule


def build_args(arg_defaults=None):
    pl.seed_everything(1234)
    data_config = Path.cwd() / "../../configs/data.yaml"
    tmp = arg_defaults
    arg_defaults = {
        "accelerator": "ddp",
        "batch_size": 32,
        "max_epochs": 5,
        "gpus": 1,
        "num_workers": 10,
        "callbacks": [],
    }
    if tmp is not None:
        arg_defaults.update(tmp)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--im_size", default=224, type=int)
    parser.add_argument("--uncertain_label", default=np.nan, type=float)
    parser.add_argument("--nan_label", default=np.nan, type=float)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = XrayDataModule.add_model_specific_args(parser)
    parser = SipModule.add_model_specific_args(parser)
    parser.set_defaults(**arg_defaults)
    args = parser.parse_args()

    if args.default_root_dir is None:
        args.default_root_dir = Path.cwd()

    if args.pretrained_file is None:
        warn("Pretrained file not specified, training from scratch.")
    else:
        logging.info(f"Loading pretrained file from {args.pretrained_file}")

    if args.dataset_dir is None:
        with open(data_config, "r") as f:
            paths = yaml.load(f, Loader=yaml.SafeLoader)["paths"]

        if args.dataset_name == "nih":
            args.dataset_dir = paths["nih"]
        if args.dataset_name == "mimic":
            args.dataset_dir = paths["mimic"]
        elif args.dataset_name == "chexpert":
            args.dataset_dir = paths["chexpert"]
        elif args.dataset_name == "mimic-chexpert":
            args.dataset_dir = [paths["chexpert"], paths["mimic"]]
        else:
            raise ValueError("Unrecognized path config.")

    if args.dataset_name in ("chexpert", "mimic", "mimic-chexpert"):
        args.val_pathology_list = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
        ]
    elif args.dataset_name == "nih":
        args.val_pathology_list = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Effusion",
        ]
    else:
        raise ValueError("Unrecognized dataset.")

    # ------------
    # checkpoints
    # ------------
    checkpoint_dir = Path(args.default_root_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    elif args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    args.callbacks.append(
        pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, verbose=True)
    )

    return args


def fetch_pos_weights(dataset_name, csv, label_list, uncertain_label, nan_label):
    if dataset_name == "nih":
        pos = [(csv["Finding Labels"].str.contains(lab)).sum() for lab in label_list]
        neg = [(~csv["Finding Labels"].str.contains(lab)).sum() for lab in label_list]
        pos_weights = torch.tensor((neg / np.maximum(pos, 1)).astype(np.float))
    else:
        pos = (csv[label_list] == 1).sum()
        neg = (csv[label_list] == 0).sum()

        if uncertain_label == 1:
            pos = pos + (csv[label_list] == -1).sum()
        elif uncertain_label == -1:
            neg = neg + (csv[label_list] == -1).sum()

        if nan_label == 1:
            pos = pos + (csv[label_list].isna()).sum()
        elif nan_label == -1:
            neg = neg + (csv[label_list].isna()).sum()

        pos_weights = torch.tensor((neg / np.maximum(pos, 1)).values.astype(np.float))

    return pos_weights


def cli_main(args):
    # ------------
    # data
    # ------------
    train_transform_list = [
        transforms.Resize(args.im_size),
        transforms.CenterCrop(args.im_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        HistogramNormalize(),
        TensorToRGB(),
        RemapLabel(-1, args.uncertain_label),
        NanToInt(args.nan_label),
    ]
    val_transform_list = [
        transforms.Resize(args.im_size),
        transforms.CenterCrop(args.im_size),
        transforms.ToTensor(),
        HistogramNormalize(),
        TensorToRGB(),
        RemapLabel(-1, args.uncertain_label),
    ]
    data_module = XrayDataModule(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=Compose(train_transform_list),
        val_transform=Compose(val_transform_list),
        test_transform=Compose(val_transform_list),
    )

    # ------------
    # model
    # ------------
    pos_weights = fetch_pos_weights(
        dataset_name=args.dataset_name,
        csv=data_module.train_dataset.csv,
        label_list=data_module.label_list,
        uncertain_label=args.uncertain_label,
        nan_label=args.nan_label,
    )
    model = SipModule(
        arch=args.arch,
        num_classes=len(data_module.label_list),
        pretrained_file=args.pretrained_file,
        label_list=data_module.label_list,
        val_pathology_list=args.val_pathology_list,
        learning_rate=args.learning_rate,
        pos_weights=pos_weights,
        epochs=args.max_epochs,
    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    args = build_args()
    cli_main(args)
