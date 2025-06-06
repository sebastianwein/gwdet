import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch_lightning.callbacks import Callback    
import torch


class ConfusionMatrix(Callback):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def on_test_end(self, trainer, module):
        x = module.test_samples
        y = module.test_labels
        y_pred = module.test_predictions
        parameters = module.test_parameters

        # Confusion matrix
        mat = module.conf_mat(self.threshold)(y_pred, y)
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

        fig_path = os.path.join(trainer.log_dir, "conf_mat.png")
        fig.savefig(fig_path)


class ROC(Callback):
    def __init__(self, threshold=0.5, n=1000):
        self.threshold = threshold
        self.n = n

    def on_test_end(self, trainer, module):
        x = module.test_samples
        y = module.test_labels
        y_pred = module.test_predictions
        parameters = module.test_parameters

        tpr, fpr = np.empty(self.n), np.empty(self.n)
        for i, threshold in enumerate(np.linspace(0, 1, self.n)):
            mat = module.conf_mat(self.threshold)(y_pred, y)
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

        fig_path = os.path.join(trainer.log_dir, "roc.png")
        fig.savefig(fig_path)


class SNRAccuracy(Callback):
    def __init__(self, snr_range=(5, 20), bins=10):
        self.min_snr, self.max_snr = snr_range
        self.bins = bins

    def on_test_end(self, trainer, module):
        x = module.test_samples
        y = module.test_labels
        y_pred = module.test_predictions
        parameters = module.test_parameters

        injection_snr = parameters["injection_snr"].cpu().numpy()
        acc = np.empty(self.bins)
        snr = np.linspace(self.min_snr, self.max_snr, self.bins, 
                          endpoint=False)
        delta_snr = (self.max_snr - self.min_snr) / self.bins

        for i, r in enumerate(snr):
            if i==0: mask = injection_snr < r+delta_snr
            elif i==self.bins-1: mask = r <= injection_snr
            else: mask = np.logical_and(r <= injection_snr,
                                        injection_snr < r+delta_snr)
            acc[i] = module.accuracy(y_pred[mask], y[mask]) \
                     if np.sum(mask)>0 else np.nan

        fig, ax = plt.subplots()
        ax.bar(snr, acc, width=delta_snr, align="edge", edgecolor="k")
        ax.set_xlim(self.min_snr, self.max_snr)
        ax.set_xlabel("SNR")
        ax.set_ylabel("Accuracy")

        fig_path = os.path.join(trainer.log_dir, "snr_acc.png")
        fig.savefig(fig_path)