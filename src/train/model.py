import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from torchmetrics.functional import accuracy

from src.train.config import StaticLearningParameter as slp


class LitEfficientNet(LightningModule):
    """LightningModule for EfficientNet."""
    def __init__(self, model, lr=slp.LR, gamma=slp.GAMMA):
        super().__init__()

        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.gamma = gamma

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
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        """
        Evaluation step.

        Args:
            batch: batch data
            stage: stage name
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

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
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr
            )
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=self.gamma
            ) 

        criterion = nn.CrossEntropyLoss()  # loss function
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "criterion": criterion}