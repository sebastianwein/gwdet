from pytorch_lightning import LightningModule
import torch
from torch import nn as nn


class Model(LightningModule):
    def __init__(self, input_size=128, learning_rate=0.02):
        super().__init__()
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.loss_fn = nn.BCELoss()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Linear(32, 16),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.seq(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss)