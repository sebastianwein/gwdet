import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_summary import summarize
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix
import torch.nn as nn


class Conv1dModel(LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.loss_fn = nn.BCELoss()
        self.accuracy = BinaryAccuracy()
        self.conf_mat = lambda threshold: \
        BinaryConfusionMatrix(threshold).to(self.device)

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
        self.log("train_loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True)
        acc = self.accuracy(y_pred, y)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True)
        acc = self.accuracy(y_pred, y)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, 
                 logger=True)

    def test_step(self, batch, batch_idx):
        x, y, parameters = batch
        y_pred = self(x)
        if batch_idx == 0:
            self.test_samples = x
            self.test_labels = y
            self.test_predictions = y_pred
            self.test_parameters = parameters
        else: 
            self.test_samples = torch.cat((self.test_samples, x))
            self.test_labels = torch.cat((self.test_labels, y))
            self.test_predictions = torch.cat((self.test_predictions, y_pred))
            for key in self.test_parameters.keys():
                self.test_parameters[key] \
                = torch.cat((self.test_parameters[key], parameters[key]))

    def on_test_end(self):
        x = self.test_samples
        y = self.test_labels
        y_pred = self.test_predictions
        parameters = self.test_parameters

        # Confusion matrix
        threshold = 0.5
        mat = self.conf_mat(threshold)(y_pred, y)
        mat = mat.cpu().numpy()
        mat = mat / np.sum(mat)
        label = np.array([["TN", "FP"], ["FN", "TP"]])
        fig, ax = plt.subplots()
        ax.pcolormesh(mat, vmin=0, vmax=1)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j+0.5, i+0.5, f"{label[i,j]} = {100*mat[i,j]:.1f}%", 
                        color="white", ha="center", va="center")
        ax.set_xticks([0.5, 1.5], [0, 1])
        ax.set_yticks([0.5, 1.5], [0, 1])
        ax.set_xlabel("Predicted")
        ax.xaxis.set_label_position("top") 
        ax.set_ylabel("Target")
        ax.xaxis.set_inverted(True)
        ax.tick_params(top=True, labeltop=True, bottom=False, 
                       labelbottom=False)
        fig.savefig("conf_mat.png")

        # ROC
        n = 1000
        tpr, fpr = np.empty(n), np.empty(n)
        for i, threshold in enumerate(np.linspace(0, 1, n)):
            mat = self.conf_mat(threshold)(y_pred, y)
            mat = mat.cpu().numpy()
            mat = mat / np.sum(mat)
            (tn, fp), (fn, tp) = mat
            tpr[i] = tp / (tp + fn)
            fpr[i] = fp / (fp + tn)
        auc = np.trapz(tpr[::-1], x=fpr[::-1])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], ls="--")
        ax.text(0.75, 0.25, f"ROC AUC = {auc:.3f}", ha="center", va="center")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        fig.savefig("roc.png")

        # Accuracy vs SNR
        injection_snr = parameters["injection_snr"].cpu().numpy()
        n = 10
        acc = np.empty(n)
        min_snr, max_snr = 5, 20
        snr = np.linspace(min_snr, max_snr, n, endpoint=False)
        delta_snr = (max_snr - min_snr) / n

        threshold = 0.5
        for i, r in enumerate(snr):
            if i==0: mask = injection_snr < r+delta_snr
            elif i==n-1: mask = r <= injection_snr
            else: mask = np.logical_and(r <= injection_snr,
                                        injection_snr < r+delta_snr)
            acc[i] = self.accuracy(y_pred[mask], y[mask]) \
                     if np.sum(mask)>0 else np.nan

        fig, ax = plt.subplots()
        ax.bar(snr, acc, width=delta_snr, align="edge", edgecolor="k")
        ax.set_xlim(min_snr, max_snr)
        ax.set_xlabel("SNR")
        ax.set_ylabel("Accuracy")
        fig.savefig("snr_acc.png")