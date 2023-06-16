import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar

import torchmetrics
from torchmetrics.functional import accuracy, f1_score

sys.path.append("./")
from src.train.config import StaticLearningParameter as slp
from src.train.config import StaticDataset as sd


class LitEfficientNet(LightningModule):
    """LightningModule for EfficientNet."""

    def __init__(self, model, lr=slp.LR.value, gamma=slp.GAMMA.value):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.num_classes = len(sd.ACTIVITY_MAP.value.items())
        # self.accuracy = torchmetrics.Accuracy(
        #     task="multiclass", num_classes=self.num_classes
        # )

    def forward(self, x):
        """Forward propagation."""
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: batch data
            batch_idx: batch index
        """
        inputs, labels = batch
        logits = self(inputs)
        loss = F.nll_loss(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        """
        Evaluation step.

        Args:
            batch: batch data
            stage: stage name
        """
        inputs, labels = batch
        logits = self(inputs)
        loss = F.nll_loss(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)  # self.accuracy(preds, labels)
        f1 = f1_score(preds, labels)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_f1", f1, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: batch data
            batch_idx: batch index
        """
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: batch data
            batch_idx: batch index
        """
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        """Configure optimizers."""

        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            # weight_decay=5e-4,
        )
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

        # criterion = nn.CrossEntropyLoss()  # loss function

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
