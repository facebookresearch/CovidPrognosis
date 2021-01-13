"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import requests
import torch
import torchvision.models as models
from tqdm import tqdm


def filter_nans(logits, labels):
    logits = logits[~torch.isnan(labels)]
    labels = labels[~torch.isnan(labels)]

    return logits, labels


def validate_pretrained_model(state_dict, pretrained_file):
    # sanity check to make sure we're not altering weights
    pretrained_dict = torch.load(pretrained_file, map_location="cpu")["state_dict"]
    model_dict = dict()
    for k, v in pretrained_dict.items():
        if "model.encoder_q" in k:
            model_dict[k[len("model.encoder_q.") :]] = v

    for k in list(model_dict.keys()):
        # only ignore fc layer
        if "classifier.weight" in k or "classifier.bias" in k:
            continue
        if "fc.weight" in k or "fc.bias" in k:
            continue

        assert (
            state_dict[k].cpu() == model_dict[k]
        ).all(), f"{k} changed in linear classifier training."


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


class SipModule(pl.LightningModule):
    def __init__(
        self,
        arch,
        num_classes,
        label_list,
        val_pathology_list,
        pretrained_file=None,
        learning_rate=1e-3,
        pos_weights=None,
        epochs=5,
    ):
        super().__init__()

        pretrained_file = str(pretrained_file)

        self.label_list = label_list
        self.val_pathology_list = val_pathology_list
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.pretrained_file = pretrained_file

        # load the pretrained model
        if pretrained_file is not None:
            self.pretrained_file = str(self.pretrained_file)

            # download the model if given a url
            if "https://" in pretrained_file:
                url = self.pretrained_file
                self.pretrained_file = Path.cwd() / pretrained_file.split("/")[-1]
                download_model(url, self.pretrained_file)

            pretrained_dict = torch.load(self.pretrained_file)["state_dict"]
            state_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith("model.encoder_q."):
                    k = k.replace("model.encoder_q.", "")
                    state_dict[k] = v

            if "model.encoder_q.classifier.weight" in pretrained_dict.keys():
                feature_dim = pretrained_dict[
                    "model.encoder_q.classifier.weight"
                ].shape[0]
                in_features = pretrained_dict[
                    "model.encoder_q.classifier.weight"
                ].shape[1]

                self.model = models.__dict__[arch](num_classes=feature_dim)
                self.model.load_state_dict(state_dict)
                del self.model.classifier
                self.model.add_module(
                    "classifier", torch.nn.Linear(in_features, num_classes)
                )
            elif "model.encoder_q.fc.weight" in pretrained_dict.keys():
                feature_dim = pretrained_dict["model.encoder_q.fc.weight"].shape[0]
                in_features = pretrained_dict["model.encoder_q.fc.weight"].shape[1]

                self.model = models.__dict__[arch](num_classes=feature_dim)
                self.model.load_state_dict(state_dict)
                del self.model.fc
                self.model.add_module("fc", torch.nn.Linear(in_features, num_classes))
            else:
                raise RuntimeError("Unrecognized classifier.")
        else:
            self.model = models.__dict__[arch](num_classes=num_classes)

        # loss function
        if pos_weights is None:
            pos_weights = torch.ones(num_classes)
        self.register_buffer("pos_weights", pos_weights)
        print(self.pos_weights)

        # metrics
        self.train_acc = torch.nn.ModuleList(
            [pl.metrics.Accuracy() for _ in val_pathology_list]
        )
        self.val_acc = torch.nn.ModuleList(
            [pl.metrics.Accuracy() for _ in val_pathology_list]
        )

    def on_epoch_start(self):
        if self.pretrained_file is not None:
            self.model.eval()

    def forward(self, image):
        return self.model(image)

    def loss(self, output, target):
        counts = 0
        loss = 0
        for i in range(len(output)):
            pos_weights, _ = filter_nans(self.pos_weights, target[i])
            loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=pos_weights, reduction="sum"
            )
            bind_logits, bind_labels = filter_nans(output[i], target[i])
            loss = loss + loss_fn(bind_logits, bind_labels)
            counts = counts + bind_labels.numel()

        counts = 1 if counts == 0 else counts
        loss = loss / counts

        return loss

    def training_step(self, batch, batch_idx):
        # forward pass
        output = self(batch["image"])
        target = batch["labels"]

        # calculate loss
        loss_val = self.loss(output, target)

        # metrics
        self.log("train_metrics/loss", loss_val)
        for i, path in enumerate(self.val_pathology_list):
            j = self.label_list.index(path)
            logits, labels = filter_nans(output[:, j], target[:, j])
            self.train_acc[i](logits, labels)
            self.log(
                f"train_metrics/accuracy_{path}",
                self.train_acc[i],
                on_step=True,
                on_epoch=False,
            )

        return loss_val

    def validation_step(self, batch, batch_idx):
        # forward pass
        output = self(batch["image"])
        target = batch["labels"]

        # calculate loss
        loss_val = self.loss(output, target)

        # metrics
        result_logits = {}
        result_labels = {}
        self.log("val_metrics/loss", loss_val)
        for path in self.val_pathology_list:
            j = self.label_list.index(path)
            logits, labels = filter_nans(output[:, j], target[:, j])

            result_logits[path] = logits
            result_labels[path] = labels

        return {"logits": result_logits, "targets": result_labels}

    def validation_epoch_end(self, outputs):
        # make sure we didn't change the pretrained weights
        if self.pretrained_file is not None:
            validate_pretrained_model(self.model.state_dict(), self.pretrained_file)

        auc_vals = []
        for i, path in enumerate(self.val_pathology_list):
            logits = []
            targets = []
            for output in outputs:
                logits.append(output["logits"][path].flatten())
                targets.append(output["targets"][path].flatten())

            logits = torch.cat(logits)
            targets = torch.cat(targets)
            print(f"path: {path}, len: {len(logits)}")

            self.val_acc[i](logits, targets)
            try:
                auc_val = pl.metrics.functional.auroc(torch.sigmoid(logits), targets)
                auc_vals.append(auc_val)
            except ValueError:
                auc_val = 0

            print(f"path: {path}, auc_val: {auc_val}")

            self.log(
                f"val_metrics/accuracy_{path}",
                self.val_acc[i],
                on_step=False,
                on_epoch=True,
            )
            self.log(f"val_metrics/auc_{path}", auc_val)

        self.log("val_metrics/auc_mean", sum(auc_vals) / len(auc_vals))

    def configure_optimizers(self):
        if self.pretrained_file is None:
            model = self.model
        else:
            if hasattr(self.model, "classifier"):
                model = self.model.classifier
            elif hasattr(self.model, "fc"):
                model = self.model.fc
            else:
                raise RuntimeError("Unrecognized classifier.")

        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--arch", default="densenet121", type=str)
        parser.add_argument("--num_classes", default=14, type=int)
        parser.add_argument("--pretrained_file", default=None, type=str)
        parser.add_argument("--val_pathology_list", nargs="+")
        parser.add_argument("--learning_rate", default=1e-2, type=float)
        parser.add_argument("--pos_weights", default=None, type=float)

        return parser
