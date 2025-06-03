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
        self.conf_mat = lambda threshold: BinaryConfusionMatrix(threshold)

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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        acc = self.accuracy(y_pred, y)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        acc = self.accuracy(y_pred, y)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_test_start(self):
        self.test_step_output \
            = np.empty((self.trainer.num_test_batches[0], 1001, 5))

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        threshold = 0.5
        mat = self.conf_mat(threshold)(y_pred, y)
        tn = mat[0][0].item() 
        fp = mat[0][1].item() 
        fn = mat[1][0].item() 
        tp = mat[1][1].item()
        self.test_step_output[batch_idx][0] \
            = np.array([threshold, tn, fp, fn, tp])
        # Vary threshold for ROC
        for i, threshold in enumerate(np.linspace(0, 1, 1000)):
            mat = self.conf_mat(threshold)(y_pred, y)
            tn = mat[0][0].item() 
            fp = mat[0][1].item() 
            fn = mat[1][0].item() 
            tp = mat[1][1].item()
            self.test_step_output[batch_idx][i+1] \
                = np.array([threshold, tn, fp, fn, tp])

    def on_test_end(self): 

        # Confusion matrix
        tn, fp, fn, tp = np.sum(self.test_step_output[:,:,1:], axis=0).T
        mat = np.array([[tn[0], fp[0]], [fn[0], tp[0]]])/(tn[0]+fp[0]+fn[0]+tp[0])
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
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        fig.savefig("conf_mat.png")

        # ROC
        tpr = tp[1:]/(tp[1:]+fn[1:])
        fpr = fp[1:]/(fp[1:]+tn[1:])
        auc = np.trapz(tpr[::-1], x=fpr[::-1])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], ls="--")
        ax.text(0.75, 0.25, f"ROC AUC = {auc:.3f}", ha="center", va="center")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        fig.savefig("roc.png")
        
