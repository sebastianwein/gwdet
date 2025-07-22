import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch_lightning.callbacks import Callback    
import torch
import torch.nn as nn


class AttentionHeatMap(Callback):
    def on_test_end(self, trainer, module):
        parameters = module.test_parameters
        weights = parameters["weights"].cpu().numpy()
        y = module.test_labels.squeeze().cpu().numpy()
        signal_weights = weights[np.bool(y)]
        noise_weights = weights[np.logical_not(np.bool(y))]  
        mean_signal_weights = np.mean(signal_weights, axis=0)
        mean_noise_weights = np.mean(noise_weights, axis=0)
        num_tokens = weights.shape[-1]

        fig, axs = plt.subplots(3, sharex=True)
        for i, ax in enumerate(axs):
            signal = ax.stairs(mean_signal_weights[i], np.arange(num_tokens+1))
            noise = ax.stairs(mean_noise_weights[i], np.arange(num_tokens+1))
            ax.set_ylabel(f"Weight (ET{i+1})")
        axs[-1].set_xlabel("Token")
        axs[0].legend([signal, noise], ["Signal", "Noise"], frameon=False)
        fig.subplots_adjust(hspace=0)

        fig_path = os.path.join(trainer.log_dir, "attention_heat_map.png")
        fig.savefig(fig_path)
        

class SNRBinnedAttentionHeatMap(Callback):
    def __init__(self, bins: int = 5):
        self.bins = bins

    def on_test_end(self, trainer, module):
        parameters = module.test_parameters
        snr = parameters["nomf_snr"].cpu().numpy()
        weights = parameters["weights"].cpu().numpy()

        mask = np.isfinite(snr)
        min_snr, max_snr = np.min(snr[mask]), np.max(snr[mask])
        snr_linspace = np.linspace(min_snr, max_snr, self.bins+1)

        mean_weights = np.empty((self.bins, *weights.shape[1:]))
        labels = [""] * self.bins

        for i, (left, right) in \
        enumerate(zip(snr_linspace[:-1], snr_linspace[1:])):
            if i==0: 
                mask = snr < right
                label = f"SNR < {right:.1f}"
            elif i==self.bins-1: 
                mask = left <= snr
                label = f"{left:.1f} <= SNR"
            else: 
                mask = np.logical_and(left <= snr, snr < right)
                label = f"{left:.1f} <= SNR < {right:.1f}"
            mean_weights[i] = np.mean(weights[mask], axis=0)
            labels[i] = label
            
        num_tokens = weights.shape[-1]

        fig, axs = plt.subplots(3, sharex=True)
        handles = list()
        for weights in mean_weights:
            for i, ax in enumerate(axs):
                handle = ax.stairs(weights[i], np.arange(num_tokens+1))
                ax.set_ylabel(f"Weight (ET{i+1})")
            handles.append(handle)
        axs[-1].set_xlabel("Token")
        axs[0].legend(handles, labels, frameon=False)
        fig.subplots_adjust(hspace=0)

        fig_path = os.path.join(trainer.log_dir, 
                                "snr_binned_attention_heat_map.png")
        fig.savefig(fig_path)


class VisualizeAttention(Callback):
    def __init__(self, min_snr: float=10):
        self.min_snr = min_snr

    def on_test_end(self, trainer, module):
        parameters = module.test_parameters
        snr = parameters["nomf_snr"].cpu().numpy()
        idx = np.random.choice(np.flatnonzero(snr>self.min_snr))
        snr = snr[idx]
        weights = parameters["weights"][idx].cpu().numpy()
        x = module.test_samples[idx].cpu().numpy()
        y = module.test_labels[idx].squeeze().cpu().numpy()
        y_pred = module.test_predictions[idx].squeeze().cpu().numpy()

        token_len = module.token_length
        fig, axs = plt.subplots(3, sharex=True)
        for i, ax in enumerate(axs):
            for j, w in enumerate(weights[i]):
                t = (np.arange(token_len)+j*token_len) 
                ax.plot(t, x[i][t])
                ax.text(t[len(t)//2], 0, int(100*w), c="k", ha="center", 
                        va="center")
            ax.set_ylabel(f"Strain (ET{i+1})")
        axs[-1].set_xlabel("Time step")
        fig.suptitle(f"Target={int(y)}, prediction={y_pred:.2f}, snr={snr:.1f}")
        fig.subplots_adjust(hspace=0)

        fig_path = os.path.join(trainer.log_dir, "attention.png")
        fig.savefig(fig_path)