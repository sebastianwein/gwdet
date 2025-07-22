import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch_lightning.callbacks import Callback    
import torch
import torch.nn as nn


class Summary(Callback):
    def on_fit_start(self, trainer, module):
        shape = trainer.datamodule.out_shape
        module.example_input_array = torch.empty(shape)


class ConfusionMatrix(Callback):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def on_test_end(self, trainer, module):
        y = module.test_labels
        y_pred = module.test_predictions

        # Confusion matrix
        mat = module.conf_mat(self.threshold)(y_pred, y)
        mat = np.array(mat.cpu().tolist(), dtype=float)
        mat[0] /= np.sum(mat[0])
        mat[1] /= np.sum(mat[1])
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
    def __init__(self, num_points=1000):
        self.num_points = num_points

    def on_test_end(self, trainer, module):
        y = module.test_labels
        y_pred = module.test_predictions

        tpr, fpr = np.empty(self.num_points), np.empty(self.num_points)
        for i, threshold in enumerate(np.linspace(0, 1, self.num_points)):
            mat = module.conf_mat(threshold)(y_pred, y)
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
        y = module.test_labels
        y_pred = module.test_predictions
        parameters = module.test_parameters

        nomf_snr = parameters["nomf_snr"].cpu().numpy()
        acc = np.empty(self.bins)
        snr = np.linspace(self.min_snr, self.max_snr, self.bins, 
                          endpoint=False)
        delta_snr = (self.max_snr - self.min_snr) / self.bins

        for i, r in enumerate(snr):
            if i==0: mask = nomf_snr < r+delta_snr
            elif i==self.bins-1: mask = r <= nomf_snr
            else: mask = np.logical_and(r <= nomf_snr,
                                        nomf_snr < r+delta_snr)
            acc[i] = module.accuracy(y_pred[mask], y[mask]) \
                     if np.sum(mask)>0 else np.nan

        fig, ax = plt.subplots()
        ax.bar(snr, acc, width=delta_snr, align="edge", edgecolor="k")
        ax.set_xlim(self.min_snr, self.max_snr)
        ax.set_xlabel("SNR")
        ax.set_ylabel("True positive rate")

        fig_path = os.path.join(trainer.log_dir, "snr_acc.png")
        fig.savefig(fig_path)


# TODO: Think of less ugly name
# class SNRAccuracyVarBins(Callback):
#     def __init__(self, bins=10):
#         self.bins = bins

#     def on_test_end(self, trainer, module):
#         y = module.test_labels.cpu().numpy()
#         y_pred = module.test_predictions.cpu().numpy()
#         parameters = module.test_parameters
#         snr = parameters["nomf_snr"].cpu().numpy()

#         mask = np.isfinite(snr)
#         y = y[mask]
#         y_pred = y_pred[mask]
#         snr = snr[mask] 

#         samples_per_bin = len(y)/self.bins
#         batch_idxs = np.int64(np.arange(self.bins) * samples_per_bin)
#         acc = np.empty(self.bins)

#         sorted_idxs = np.argsort(snr)
#         snr_sorted = snr[sorted_idxs]
#         y_sorted = y[sorted_idxs]
#         y_pred_sorted = y_pred[sorted_idxs]

#         y_batches = np.split(y_sorted, batch_idxs[1:])
#         y_pred_batches = np.split(y_pred_sorted, batch_idxs[1:])

#         for i, (y_batch, y_pred_batch) in \
#         enumerate(zip(y_batches, y_pred_batches)):
#             acc[i] = module.accuracy(torch.Tensor(y_pred_batch), 
#                                      torch.Tensor(y_batch))

#         fig, ax = plt.subplots()
#         ax.bar(np.arange(self.bins), acc, width=1, align="edge", edgecolor="k")
#         tick_labels = np.append(snr_sorted[batch_idxs], snr_sorted[-1])
#         ax.set_xticks(np.arange(self.bins+1), 
#                       labels=[f"{x:.1f}" for x in tick_labels], 
#                       rotation="vertical")
#         ax.set_xlabel("SNR")
#         ax.set_ylabel("Accuracy")

#         fig_path = os.path.join(trainer.log_dir, "snr_acc_var_bins.png")
#         fig.savefig(fig_path)