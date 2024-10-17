import os
from typing import Any, Dict, Tuple, Optional
from random import random
from copy import deepcopy

import numpy as np
import torch
from PIL.ImageOps import scale
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

from src.models.components.transport import R3Diffuser
from src.models.loss import ScoreMatchingLoss
from src.common.rigid_utils import Rigid
from src.common.all_atom import compute_backbone
from src.common.pdb_utils import atom37_to_pdb, merge_pdbfiles


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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        diffuser: R3Diffuser,
        loss: Dict[str, Any],
        compile: bool,
        inference: Optional[Dict[str, Any]] = None,
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
        self.diffuser = diffuser
        
        # loss function
        self.loss = ScoreMatchingLoss(config=self.hparams.loss)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

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
        self, representations, batch, training: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # preprocess by augmenting additional feats
        x_0 = batch['target_structure']
        batch_size = x_0.shape[0]
        t = (1.0 - self.diffuser.min_t) * torch.rand(batch_size, device=x_0.device) + self.diffuser.min_t
        perturb_feats = self.diffuser.forward_marginal(
            x_0=x_0,
            t=t,
            diffuse_mask=None,
        )
        patch_feats = {
            't': t,
            'x_0': x_0,
        }
        batch.update({**perturb_feats, **patch_feats})
        
        # probably add self-conditioning (recycle once)
        if self.net.embedder.self_conditioning and random() > 0.5:
            with torch.no_grad():
                batch['sc_coords'] = self.net(batch)  # b, n, 3

        # feedforward
        out = self.net(batch)
        
        # postprocess by add score computation
        pred_scores = self.diffuser.score(
            x_0=batch['x_0'],
            x_t=out,
            t=t,
            scale=True,
            mask=batch['residue_mask'],
        )
        out.update(pred_scores)

        ##### revised to this line

        # calculate losses
        loss, loss_bd = self.loss(out, batch, _return_breakdown=True)
        return loss, loss_bd

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, loss_bd = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for k,v in loss_bd.items():
            if k == 'loss': continue
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_bd = self.model_step(batch, training=False)

        # update and log metrics
        self.val_loss(loss) # update
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        _vall = self.val_loss.compute()  # get current val acc
        self.val_loss_best(_vall)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
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
        self, representations, batch
    ) -> str:
        coordinates = self.net.sample_diffusion(representations, batch)

        return coordinates

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
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = IDPFoldMultimer(None, None, None, None, None)
