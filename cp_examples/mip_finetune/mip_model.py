"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from argparse import ArgumentParser
from pathlib import Path
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels


class DenseNet(tvmodels.DenseNet):
    def forward(self, x):
        features = self.features(x)
        return F.relu(features, inplace=True)


def filter_nans(logits, labels):
    logits = logits[~torch.isnan(labels)]
    labels = labels[~torch.isnan(labels)]
    return logits, labels


def load_pretrained_model(arch, pretrained_file):
    pretrained_dict = torch.load(pretrained_file)["state_dict"]
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith("model.encoder_q."):
            k = k.replace("model.encoder_q.", "")
            state_dict[k] = v

    if arch.startswith("densenet"):
        num_classes = pretrained_dict["model.encoder_q.classifier.weight"].shape[0]
        model = DenseNet(num_classes=num_classes)
        model.load_state_dict(state_dict)
        feature_dim = pretrained_dict["model.encoder_q.classifier.weight"].shape[1]
        del model.classifier
    else:
        raise ValueError(f"Model architecture {arch} is not supported.")

    return model, feature_dim


class ContinuousPosEncoding(nn.Module):
    def __init__(self, dim, drop=0.1, maxtime=360):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        position = torch.arange(0, maxtime, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(maxtime, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, xs, times):
        ys = xs
        times = times.long()
        for b in range(xs.shape[1]):
            ys[:, b] += self.pe[times[b]]
        return self.dropout(ys)


class MIPModel(nn.Module):
    def __init__(
        self,
        image_model,
        feature_dim,
        projection_dim,
        num_classes,
        num_heads,
        feedforward_dim,
        drop_transformer,
        drop_cpe,
        pooling,
        image_shape=(7, 7),
    ):
        super().__init__()

        self.image_shape = image_shape
        self.pooling = pooling
        self.image_model = image_model
        self.group_norm = nn.GroupNorm(32, feature_dim)
        self.projection = nn.Conv2d(feature_dim, projection_dim, (1, 1))

        transformer_dim = projection_dim * image_shape[0] * image_shape[1]
        self.pos_encoding = ContinuousPosEncoding(transformer_dim, drop=drop_cpe)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            dim_feedforward=feedforward_dim,
            nhead=num_heads,
            dropout=drop_transformer,
        )
        self.classifier = nn.Linear(feature_dim + projection_dim, num_classes)

    def _apply_transformer(self, image_feats: torch.Tensor, times, lens):
        B, N, C, H, W = image_feats.shape
        image_feats = image_feats.flatten(start_dim=2).permute(
            [1, 0, 2]
        )  # [N, B, C * H * W]
        image_feats = self.pos_encoding(image_feats, times)
        image_feats = self.transformer(image_feats)
        return image_feats.permute([1, 0, 2]).reshape([B, N, C, H, W])

    def _pool(self, image_feats, lens):
        if self.pooling == "last_timestep":
            pooled_feats = []
            for b, l in enumerate(lens.tolist()):
                pooled_feats.append(image_feats[b, int(l) - 1])
        elif self.pooling == "sum":
            pooled_feats = []
            for b, l in enumerate(lens.tolist()):
                pooled_feats.append(image_feats[b, : int(l)].sum(0))
        else:
            raise ValueError(f"Unkown pooling method: {self.pooling}")

        pooled_feats = torch.stack(pooled_feats)
        pooled_feats = F.adaptive_avg_pool2d(pooled_feats, (1, 1))
        return pooled_feats.squeeze(3).squeeze(2)

    def forward(self, images, times, lens):
        B, N, C, H, W = images.shape
        images = images.reshape([B * N, C, H, W])
        # Apply Image Model
        image_feats = self.image_model(images)
        image_feats = F.relu(self.group_norm(image_feats))
        # Apply transformer
        image_feats_proj = self.projection(image_feats).reshape(
            [B, N, -1, *self.image_shape]
        )
        image_feats_trans = self._apply_transformer(image_feats_proj, times, lens)
        # Concat and apply classifier
        image_feats = image_feats.reshape([B, N, -1, *self.image_shape])
        image_feats_combined = torch.cat([image_feats, image_feats_trans], dim=2)
        image_feats_pooled = self._pool(image_feats_combined, lens)
        return self.classifier(image_feats_pooled)


class MIPModule(pl.LightningModule):
    def __init__(
        self, args, label_list, pos_weights=None,
    ):
        super().__init__()

        self.args = args
        self.label_list = label_list
        self.val_pathology_list = args.val_pathology_list
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs

        # loss function
        pos_weights = pos_weights or torch.ones(args.num_classes)
        self.register_buffer("pos_weights", pos_weights)

        # metrics
        self.train_acc = torch.nn.ModuleList(
            [pl.metrics.Accuracy() for _ in args.val_pathology_list]
        )
        self.val_acc = torch.nn.ModuleList(
            [pl.metrics.Accuracy() for _ in args.val_pathology_list]
        )

        image_model, feature_dim = load_pretrained_model(
            args.arch, args.pretrained_file
        )
        self.model = MIPModel(
            image_model,
            feature_dim,
            args.projection_dim,
            args.num_classes,
            args.num_heads,
            args.feedforward_dim,
            args.drop_transformer,
            args.drop_cpe,
            args.pooling,
            args.image_shape,
        )

    def forward(self, images, times, lens):
        return self.model(images, times, lens)

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
        output = self(batch["images"], batch["times"], batch["lens"])
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
        output = self(batch["images"], batch["times"], batch["lens"])
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
        optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--pretrained_file", type=Path, required=True)
        parser.add_argument("--arch", default="densenet121", type=str)
        parser.add_argument("--num_classes", default=14, type=int)
        parser.add_argument("--val_pathology_list", nargs="+")
        parser.add_argument("--pos_weights", default=None, type=float)
        # Training params
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        parser.add_argument("--epochs", default=50, type=int)
        # Model params
        parser.add_argument("--projection_dim", type=int, default=64)
        parser.add_argument("--num_heads", type=int, default=2)
        parser.add_argument("--feedforward_dim", type=int, default=128)
        parser.add_argument("--drop_transformer", type=float, default=0.5)
        parser.add_argument("--drop_cpe", type=float, default=0.5)
        parser.add_argument(
            "--pooling", choices=["last_timestep", "sum"], default="last_timestep"
        )
        parser.add_argument("--image_shape", default=(7, 7))
        return parser
