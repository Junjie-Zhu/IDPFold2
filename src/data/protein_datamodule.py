from typing import Any, Dict, Optional, Tuple, List, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, WeightedRandomSampler
from lightning import LightningDataModule
from hydra.utils import instantiate


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
            if self.target_keys is not None else [k for k,v in raw_batch[0].items() if torch.is_tensor(v)]
        # Non-array, for example string, int
        non_array_keys = [k for k in raw_batch[0] if k not in target_keys]
        collated_batch = dict()
        for k in target_keys:
            collated_batch[k] = self.collate_dense_tensors([d[k] for d in raw_batch], pad_v=0.0)
        for k in non_array_keys:    # return non-array keys as is
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


class ProteinDataModule(LightningDataModule):
    """DataModule for a protein dataset with weighted sampling for training."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 64,
        generator_seed: int = 42,
        train_val_split: Tuple[float, float] = (0.95, 0.05),
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        idr_dataset: Optional[torch.utils.data.Dataset] = None,
        idr_weight: float = 0.3,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.dataset = dataset
        self.idr_dataset = idr_dataset
        self.idr_weight = idr_weight

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if stage == 'fit' and not self.data_train and not self.data_val:
            if self.idr_dataset is None:
                self.data_train, self.data_val = random_split(
                    dataset=self.dataset,
                    lengths=self.hparams.train_val_split,
                    generator=torch.Generator().manual_seed(self.hparams.generator_seed),
                )
            else:
                self.data_train, self.data_val = random_split(
                    dataset=self.idr_dataset,
                    lengths=self.hparams.train_val_split,
                    generator=torch.Generator().manual_seed(self.hparams.generator_seed),
                )
        elif stage in ('predict', 'test'):
            self.data_test = self.dataset
        else:
            raise NotImplementedError(f"Stage {stage} not implemented.")

        # Add weighted sampling for both datasets if idr_dataset is provided
        if self.idr_dataset:
            # Create weights for the datasets (same weight for each sample in the dataset)
            weight_dataset = torch.full((len(self.dataset),), 1 - self.idr_weight)
            weight_idr_dataset = torch.full((len(self.data_train),), self.idr_weight)

            # Combine weights from both datasets
            combined_weights = torch.cat([weight_dataset, weight_idr_dataset])

            # Create samplers for both datasets
            self.train_sampler = WeightedRandomSampler(combined_weights, len(combined_weights), replacement=True)

    def _dataloader_template(
            self,
            dataset: Dataset[Any],
            sampler: Optional[WeightedRandomSampler] = None
    ) -> DataLoader[Any]:
        """Create a dataloader from a dataset.

        :param dataset: The dataset.
        :return: The dataloader.
        """
        batch_collator = BatchTensorConverter()    # list of dicts -> dict of tensors
        return DataLoader(
            dataset=dataset,
            collate_fn=batch_collator,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle,
        )
    
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        # Combine the datasets for training
        if self.idr_dataset:
            combined_dataset = torch.utils.data.ConcatDataset([self.dataset, self.data_train])
            return self._dataloader_template(combined_dataset, self.train_sampler)
        else:
            return self._dataloader_template(self.data_train)
           

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self._dataloader_template(self.data_val)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self._dataloader_template(self.data_test)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


