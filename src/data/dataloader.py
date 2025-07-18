from typing import Any, Dict, Optional, Tuple, List, Sequence

import torch
from torch.utils.data import random_split, DataLoader, DistributedSampler


class BatchTensorConverter:
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, target_keys: Optional[List] = None):
        self.target_keys = target_keys

    def __call__(self, raw_batch: Sequence[Dict[str, object]]):
        B = len(raw_batch)
        # Only do for Tensor
        target_keys = self.target_keys \
            if self.target_keys is not None else [k for k, v in raw_batch[0].items() if torch.is_tensor(v)]
        # Non-array, for example string, int
        non_array_keys = [k for k in raw_batch[0] if k not in target_keys]
        collated_batch = dict()
        for k in target_keys:
            collated_batch[k] = self.collate_dense_tensors([d[k] for d in raw_batch], pad_v=0.0)
        for k in non_array_keys:  # return non-array keys as is
            collated_batch[k] = [d[k] for d in raw_batch]
        return collated_batch

    @staticmethod
    def collate_dense_tensors(samples: Sequence, pad_v: float = 0.0):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result


def get_training_dataloader(
        dataset: torch.utils.data.Dataset,
        batch_size: int = 64,
        prop_train: float = 0.95,
        shuffle: bool = False,
        distributed: bool = False,
        num_workers: int = 8,
        pin_memory: bool = False,
        seed: int = 42,
):
    data_train, data_val = random_split(
        dataset=dataset,
        lengths=(prop_train, 1 - prop_train),
        generator=torch.Generator().manual_seed(seed),
    )
    batch_collator = BatchTensorConverter()

    if distributed:
        training_sampler = DistributedSampler(
            data_train, seed=seed, shuffle=shuffle, drop_last=True
        )
        train_loader = DataLoader(
            dataset=data_train,
            collate_fn=batch_collator,
            batch_size=batch_size,  # batch size on single device
            sampler=training_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        validation_sampler = DistributedSampler(
            data_val, seed=seed, shuffle=shuffle, drop_last=True
        )
        val_loader = DataLoader(
            dataset=data_val,
            collate_fn=batch_collator,
            batch_size=batch_size,
            sampler=validation_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            dataset=data_train,
            collate_fn=batch_collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            dataset=data_val,
            collate_fn=batch_collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader


def get_inference_dataloader(
        dataset: torch.utils.data.Dataset,
        batch_size: int = 64,
        distributed: bool = False,
        num_workers: int = 8,
        pin_memory: bool = False,
):
    batch_collator = BatchTensorConverter()

    if distributed:
        inference_sampler = DistributedSampler(
            dataset, shuffle=False, drop_last=False
        )
        inference_loader = DataLoader(
            dataset=dataset,
            collate_fn=batch_collator,
            batch_size=batch_size,
            sampler=inference_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        inference_loader = DataLoader(
            dataset=dataset,
            collate_fn=batch_collator,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return inference_loader