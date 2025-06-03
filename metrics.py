from data import Data
import matplotlib.pyplot as plt
from model import Conv1dModel
import numpy as np
from pytorch_lightning import Trainer


metrics_path = "/home/s/swein/gwdet/main/lightning_logs/version_4/metrics.csv"
conv = lambda x: float(x) if x!="" else None
epoch, step, train_acc, train_loss, val_acc, val_loss \
    = np.loadtxt(metrics_path, delimiter=",", skiprows=1, converters=conv, 
                 unpack=True)

def plot_ignore_nan(ax, x, y, *args, **kwargs):
    mask = np.isfinite(y)
    ax.plot(x[mask], y[mask], *args, **kwargs)

# Learning curve
fig, ax = plt.subplots()
plot_ignore_nan(ax, step, train_loss)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
fig.suptitle("Learning curve")
fig.savefig("learning_curve.png")

# Accuracy
fig, ax = plt.subplots()
plot_ignore_nan(ax, step, train_acc, label="Train data")
plot_ignore_nan(ax, step, val_acc, label="Train data")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend(frameon=False)
fig.savefig("accuracy.png")

batch_size = 64
data = Data("/scratch/tmp/swein/ggwd/output/bbh.hdf", batch_size=batch_size, num_workers=4)
checkpoint_path = "/home/s/swein/gwdet/main/lightning_logs/version_4/checkpoints/epoch=19-step=4100.ckpt"
model = Conv1dModel.load_from_checkpoint(checkpoint_path)
trainer = Trainer(logger=False)
trainer.test(model, datamodule=data)
