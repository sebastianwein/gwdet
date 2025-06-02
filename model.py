from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_summary import summarize
import torch
from torchmetrics.classification import BinaryAccuracy 
import torch.nn as nn


class Conv1dModel(LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.loss_fn = nn.BCELoss()
        self.accuracy = BinaryAccuracy()

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(1, 16, (1, 16), padding="same")
        # self.bn1 = nn.BatchNorm1d(16)
        self.max_pool1 = nn.MaxPool2d((1, 4))
        self.conv2 = nn.Conv2d(16, 32, (1, 8), padding="same")
        # self.bn2 = nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool2d((1, 4))
        self.conv3 = nn.Conv2d(32, 64, (1, 8), padding="same")
        # self.bn3 = nn.BatchNorm1d(64)
        self.max_pool3 = nn.MaxPool2d((1, 4))
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.unflatten(x, 0, (-1, 1))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool3(x)
        # ReLU stack
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigm(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        acc = self.accuracy(y_pred, y)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss)
        acc = self.accuracy(y_pred, y)
        self.log("val_acc", acc)


if __name__ == "__main__":
    model = Conv1dModel(learning_rate=0.01)