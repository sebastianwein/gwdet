import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch_lightning.callbacks import Callback    
import torch
import torch.nn as nn


class AttentionHeatMap(Callback):
    def on_test_end(self, trainer, module):
        num_maps = 32
        idxs = np.random.choice(len(module.test_samples), size=num_maps)
        y = module.test_labels[idxs].squeeze().cpu().numpy()
        y_pred = module.test_predictions[idxs].squeeze().cpu().numpy()
        weights = module.test_weights[idxs].cpu().numpy()
        weights = np.mean(weights, axis=1)
        merger_time = module.test_parameters["event_time"][idxs].cpu().numpy()
        snr =  module.test_parameters["nomf_snr"][idxs].cpu().numpy()
        
        fig_dir = os.path.join(trainer.log_dir, "attention_heat_maps") 
        if not os.path.exists(fig_dir): os.makedirs(fig_dir)

        for idx in range(num_maps):
            fig, ax = plt.subplots()
            pc = ax.pcolormesh(weights[idx])
            fig.colorbar(pc, ax=ax, label="Attention weights")
            ax.set_xlabel("Token")
            ax.set_ylabel("Token")
            title = f"Target={int(y[idx])}, prediction={y_pred[idx]:.2f}"
            if int(y[idx]):
                title += f", merger_time={merger_time[idx]:.1f}, " \
                         + f"SNR={snr[idx]:.1f}"
            fig.suptitle(title)
            fig_path = os.path.join(fig_dir, f"{idx}.png")
            fig.savefig(fig_path)


class PositionalEncoding(Callback):
    def on_test_end(self, trainer, module):

        pos_enc = module.pos_enc.cpu().numpy()
        cos_sim = [[np.sum(vec1*vec2)/np.sqrt(np.sum(vec1**2)*np.sum(vec2**2)) 
                    for vec1 in pos_enc] for vec2 in pos_enc]

        fig, ax = plt.subplots()
        pc = ax.pcolormesh(pos_enc)
        fig.colorbar(pc, ax=ax, label="Positinal encoding")
        ax.set_xlabel("Token")
        ax.set_ylabel("Token")

        fig_path = os.path.join(trainer.log_dir, "pos_enc.png")
        fig.savefig(fig_path)

        fig, ax = plt.subplots()
        pc = ax.pcolormesh(cos_sim)
        fig.colorbar(pc, ax=ax, label="Cosine similarity")
        ax.set_xlabel("Token")
        ax.set_ylabel("Token")

        fig_path = os.path.join(trainer.log_dir, "pos_enc_cos_sim.png")
        fig.savefig(fig_path)