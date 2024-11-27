import os
from contextlib import nullcontext
from typing import Any, Dict, Tuple, Optional
from random import random

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

from src.models.components.generator import (
    InferenceNoiseScheduler,
    TrainingNoiseSampler,
    sample_diffusion,
    sample_diffusion_training,
)
from src.models.optimizer import get_optimizer, get_lr_scheduler, is_loss_nan_check
from src.models.ema import EMAWrapper
from src.utils.torch_utils import autocasting_disable_decorator
from src.data.components.dataset import convert_atom_name_id
from src.common.pdb_utils import write_pdb_raw


class IDPFoldMultimer(LightningModule):
    """Example of a `LightningModule` for denoising diffusion training.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        ema_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        diffusion_sample: int,
        inference: Dict[str, Any],
        compile: bool = False,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # network and diffusion module
        self.net = net
        self.loss = loss

        if ema_config.get("ema_decay", -1) > 0:
            assert ema_config.ema_decay < 1
            self.ema_wrapper = EMAWrapper(
                self.net,
                ema_config.ema_decay,
                ema_config.ema_mutable_param_keywords,
            )
        self.ema_wrapper.register()
        torch.cuda.empty_cache()

        self.optimizer = get_optimizer(optimizer_config, self.net)
        self.train_noise_sampler = TrainingNoiseSampler()

        self.ema_config = ema_config
        self.lr_scheduler_config = optimizer_config
        self.N_sample = diffusion_sample

        self.init_scheduler()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def init_scheduler(self, **kwargs):
        self.lr_scheduler = get_lr_scheduler(self.lr_scheduler_config, self.optimizer, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`. 
        (Not actually used)

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(
        self, input_feature_dict):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        N_sample = self.N_sample
        s_inputs = input_feature_dict["seq_emb"].unsqueeze(1).expand(-1, N_sample, -1, -1)

        label_dict = {
            "coordinate": input_feature_dict["coordinate"],
            "coordinate_mask": input_feature_dict["coordinate_mask"],
            "lddt_mask": input_feature_dict["lddt_mask"],
        }
        input_feature_dict.pop("coordinate")
        input_feature_dict.pop("coordinate_mask")
        input_feature_dict.pop("lddt_mask")

        _, x_denoised, x_noise_level = autocasting_disable_decorator(
            True
        )(sample_diffusion_training)(
            noise_sampler=self.train_noise_sampler,
            denoise_net=self.net,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            N_sample=N_sample,
            diffusion_chunk_size=None,
        )

        pred_dict = {
            "coordinate": x_denoised,
            "noise_level": x_noise_level,
        }
        loss = self.loss(input_feature_dict, pred_dict, label_dict)
        return loss

    def training_step(
        self, input_feature_dict, label_dict, batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(input_feature_dict, label_dict)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        """Lightning hook that is called after every training batch."""
        if hasattr(self, "ema_wrapper"):
            self.ema_wrapper.update()

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, input_feature_dict, label_dict, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(input_feature_dict, label_dict)

        # update and log metrics
        self.val_loss(loss) # update
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_start(self) -> None:
        "Lightning hook that is called when a validation epoch starts."
        if hasattr(self, "ema_wrapper"):
            self.ema_wrapper.apply_shadow()

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        _vall = self.val_loss.compute()  # get current val acc
        self.val_loss_best(_vall)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        if hasattr(self, "ema_wrapper"):
            self.ema_wrapper.restore()
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        raise NotImplementedError("Test step not implemented.")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(
        self, batch
    ) -> str:
        """Perform a prediction step on a batch of data from the dataloader.

        This prediction step will sample `n_replica` copies from the forward-backward process,
            repeated for each delta-T in the range of [delta_min, delta_max] with step size
            `delta_step`. If `backward_only` is set to True, then only backward process will be
            performed, and `n_replica` will be multiplied by the number of delta-Ts.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # extract hyperparams for inference
        n_replica = self.hparams.inference.n_replica
        output_dir = self.hparams.inference.output_dir
        batch_size = self.hparams.inference.replica_per_batch

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        restypes = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        ]

        rep_index = 1
        for replica in range(int(n_replica / batch_size)):
            for k, v in batch.items():
                # check if v is tensor
                if torch.is_tensor(v):
                    batch[k] = v.expand(batch_size, *v.shape[1:])
                else:
                    continue

            output_dict = self.net.sample_diffusion(batch)

            output_coords = output_dict['final_atom_positions'].cpu()

            # atom_name and residue_name
            output_atom_name = [convert_atom_name_id(i) for i in batch['ref_atom_name_chars'][0]]
            output_residue_name = [restypes[batch['aatype'][0][i]] for i in batch['ref_token2atom_idx'][0]]

            # save output
            write_pdb_raw(output_atom_name, output_residue_name, batch['ref_token2atom_idx'][0],
                          output_coords, os.path.join(output_dir, str(replica)), batch['accession_code'][0])

            rep_index += 1

        return output_dict

    def state_dict(self):
        # Include EMA parameters in the state_dict
        state = super().state_dict()
        if hasattr(self, "ema_wrapper"):
            state['ema_params'] = {k: v.clone() for k, v in self.ema_wrapper.ema_params.items()}
        return state

    def load_state_dict(self, state_dict):
        # Load EMA parameters from the state_dict
        if 'ema_params' in state_dict:
            self.ema.ema_params = state_dict['ema_params']
            del state_dict['ema_params']  # Remove EMA from the standard model state_dict
        super().load_state_dict(state_dict)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
        }


if __name__ == "__main__":
    _ = IDPFoldMultimer(None, None, None, None, None)
