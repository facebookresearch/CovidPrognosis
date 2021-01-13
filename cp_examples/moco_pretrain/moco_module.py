"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from argparse import ArgumentParser

import covidprognosis as cp
import pytorch_lightning as pl
import torch
import torchvision.models as models


class MoCoModule(pl.LightningModule):
    def __init__(
        self,
        arch,
        feature_dim,
        queue_size,
        use_mlp=False,
        learning_rate=1.0,
        momentum=0.9,
        weight_decay=1e-4,
        epochs=1,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs

        # build model
        self.model = cp.models.MoCo(
            encoder_q=models.__dict__[arch](num_classes=feature_dim),
            encoder_k=models.__dict__[arch](num_classes=feature_dim),
            dim=feature_dim,
            K=queue_size,
            mlp=use_mlp,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def forward(self, image0, image1):
        return self.model(image0, image1)

    def training_step(self, batch, batch_idx):
        image0, image1 = batch["image0"], batch["image1"]

        output, target = self(image0, image1)

        # metrics
        loss_val = self.loss_fn(output, target)
        self.train_acc(output, target)
        self.log("train_metrics/loss", loss_val)
        self.log("train_metrics/accuracy", self.train_acc, on_step=True, on_epoch=False)

        return loss_val

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--arch", default="densenet121", type=str)
        parser.add_argument("--feature_dim", default=256, type=int)
        parser.add_argument("--queue_size", default=65536, type=int)
        parser.add_argument("--use_mlp", default=False, type=bool)
        parser.add_argument("--learning_rate", default=1.0, type=float)
        parser.add_argument("--momentum", default=0.9, type=float)
        parser.add_argument("--weight-decay", default=1e-4, type=float)

        return parser
