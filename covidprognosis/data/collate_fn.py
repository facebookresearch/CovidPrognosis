"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    """Collate function to handle X-ray metadata."""
    metadata = []
    for el in batch:
        metadata.append(el["metadata"])
        del el["metadata"]

    batch = default_collate(batch)

    batch["metadata"] = metadata

    return batch
