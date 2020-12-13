import sys
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
from PIL import ImageFilter
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
from torch.utils.data import ConcatDataset, random_split

sys.path.append("../../../")  # noqa: E402

from moco_model import MoCo
from utils import GaussianBlur, accuracy, fetch_dataset, fetch_transform_list


class MoCoTrainer(LightningModule):
    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super().__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size
        self.batch_log_idx = 0
        self.batch_log_period = 50

        # build model
        self.__build_model()

        self.loss_fn = torch.nn.CrossEntropyLoss()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """
        self.model = MoCo(
            models.__dict__[self.hparams.arch],
            self.hparams.feature_dim,
            K=self.hparams.queue_size,
            mlp=self.hparams.mlp,
        )

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, image0, image1):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        return self.model(image0, image1)

    def loss(self, output, target):
        return self.loss_fn(output, target)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # extract images
        image0, image1 = batch["image0"], batch["image1"]

        # forward pass
        output, target = self(image0, image1)

        # calculate loss
        loss_val = self.loss(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {"train_loss": loss_val, "acc1": acc1, "acc5": acc5}
        log_dict = dict()
        # prevent tensorboard from getting too big
        if self.batch_log_idx > self.batch_log_period:
            log_dict.update(tqdm_dict)
            self.batch_log_idx = 0
        else:
            self.batch_log_idx = self.batch_log_idx + 1

        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": log_dict}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        # extract images
        image0, image1 = batch["image0"], batch["image1"]

        # forward pass
        output, target = self(image0, image1)

        # calculate loss
        loss_val = self.loss(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # for concatenation purposes
        loss_val = loss_val.unsqueeze(0).to(loss_val)
        acc1 = acc1.unsqueeze(0).to(loss_val)
        acc5 = acc5.unsqueeze(0).to(loss_val)

        output = OrderedDict({"val_loss": loss_val, "acc1": acc1, "acc5": acc5,})

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # manually handle aggregation across batch dimensions
        val_loss = []
        acc1 = []
        acc5 = []

        for output in outputs:
            val_loss.append(output["val_loss"])
            acc1.append(output["acc1"])
            acc5.append(output["acc5"])

        # aggregate across batch dims
        val_loss_mean = torch.mean(torch.cat(val_loss))
        acc1_mean = torch.mean(torch.cat(acc1))
        acc5_mean = torch.mean(torch.cat(acc5))

        # compile output dictionaries
        log_dict = {
            "val_loss": val_loss_mean,
            "val_acc1": acc1_mean,
            "val_acc5": acc5_mean,
        }
        tqdm_dict = {"val_loss": val_loss_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": log_dict,
            "val_loss": val_loss_mean,
        }

        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        if "ramp" in self.hparams.dataset_name:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, self.hparams.num_epochs_two_dataset_ramp
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.hparams.epochs
            )

        return [optimizer], [scheduler]

    def __dataloader(self, split):
        # this is neede when you want some info about dataset before binding to trainer
        self.prepare_data()

        log.info(f"dataset_name: {self.hparams.dataset_name}, split: {split}")

        # transforms
        if "ramp" in self.hparams.dataset_name:
            transform_imagenet = fetch_transform_list(
                "imagenet_grayscale", hparams=self.hparams, split=split
            )
            transform_mimic = fetch_transform_list(
                "mimic",
                hparams=self.hparams,
                split=split,
                idx=self.hparams.transform_idx,
            )
        elif "imagenet" in self.hparams.dataset_name:
            transform = fetch_transform_list(
                self.hparams.dataset_name, split=split, hparams=self.hparams
            )
        else:
            transform = fetch_transform_list(
                self.hparams.dataset_name,
                split=split,
                hparams=self.hparams,
                idx=self.hparams.transform_idx,
            )

        # init data generators
        if "ramp" in self.hparams.dataset_name:
            # ramp up to full-mimic over num_epochs_two_dataset_ramp
            percent = np.minimum(
                self.current_epoch / self.hparams.num_epochs_two_dataset_ramp, 1
            )
            log.info(f"mimic percent: {percent}")
            dataset_imagenet = fetch_dataset("imagenet", split, transform_imagenet)
            dataset_mimic = fetch_dataset("mimic", split, transform_mimic)

            if percent == 0:
                dataset = dataset_imagenet
            else:
                imagenet_count = np.minimum(
                    int(len(dataset_mimic) / percent * (1 - percent)),
                    len(dataset_imagenet),
                )
                log.info(f"imagenet_count: {imagenet_count}")
                dataset_imagenet = random_split(
                    dataset_imagenet,
                    [imagenet_count, len(dataset_imagenet) - imagenet_count],
                )[0]

                dataset = ConcatDataset([dataset_mimic, dataset_imagenet])
        else:
            dataset = fetch_dataset(self.hparams.dataset_name, split, transform)

        batch_size = self.hparams.batch_size

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

        return loader

    def train_dataloader(self):
        log.info(f"epoch: {self.current_epoch}, training data loader called.")
        return self.__dataloader(split="train")

    def val_dataloader(self):
        log.info(f"epoch: {self.current_epoch}, validation data loader called.")
        return self.__dataloader(split="train")

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument("--arch", default="densenet121", type=str)
        parser.add_argument("--im_size", default=224, type=int)
        parser.add_argument("--feature_dim", default=256, type=int)
        parser.add_argument("--queue_size", default=65536, type=int)
        parser.add_argument("--learning_rate", default=1.0, type=float)
        parser.add_argument("--momentum", default=0.9, type=float)
        parser.add_argument("--weight-decay", default=1e-4, type=float)
        parser.add_argument("--mlp", default=False, type=bool)

        # training params (opt)
        parser.add_argument("--dataset_name", default="imagenet", type=str)
        parser.add_argument("--transform_idx", default=5, type=int)
        parser.add_argument("--norm_func", default="hist", type=str)
        parser.add_argument("--use_color_input", default=True, type=bool)
        parser.add_argument("--num_epochs_two_dataset_ramp", default=50, type=int)
        parser.add_argument("--epochs", default=200, type=int)
        parser.add_argument("--optimizer_name", default="adam", type=str)
        parser.add_argument("--batch_size", default=64, type=int)

        return parser
