from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    metadata = []
    for el in batch:
        metadata.append(el["metadata"])
        del el["metadata"]

    batch = default_collate(batch)

    batch["metadata"] = metadata

    return batch
