import os
import matplotlib.pyplot as plt
import numpy as np


def plot_ignore_nan(ax, x, y, *args, **kwargs):
    mask = np.isfinite(y)
    ax.plot(x[mask], y[mask], *args, **kwargs)

def step_ignore_nan(ax, y, *args, **kwargs):
    mask = np.isfinite(y)
    y = y[mask] 
    x = np.arange(0, np.sum(mask)+1)
    y = np.append(y, y[-1])
    ax.plot(x, y, drawstyle="steps-post", *args, **kwargs)

def main():
    version = 0
    log_dir = f"/home/s/swein/gwdet/main/logs/version_{version}"
    metrics_path = os.path.join(log_dir, "metrics.csv")
    conv = lambda x: float(x) if x!="" else None
    epoch, step, train_acc_epoch, train_acc_step, train_loss_epoch, \
    train_loss_step, val_acc, val_loss \
    = np.loadtxt(metrics_path, delimiter=",", skiprows=1, converters=conv, 
                unpack=True)

    # Learning curve
    fig, ax_epoch = plt.subplots()
    mask = np.isfinite(train_loss_epoch)
    steps_per_epoch = step[mask][1] - step[mask][0]
    ax_step = ax_epoch.secondary_xaxis("top", 
                                    functions=(lambda s: s*steps_per_epoch, 
                                                lambda s: s/steps_per_epoch))
    step_ignore_nan(ax_epoch, train_loss_epoch)       
    plot_ignore_nan(ax_epoch, step/steps_per_epoch, train_loss_step)
    ax_epoch.set_xlabel("Epoch")
    ax_step.set_xlabel("Step")
    ax_epoch.set_ylabel("Loss")
    fig.suptitle("Learning curve")
    fig_path = os.path.join(log_dir, "learning_curve.png")
    fig.savefig(fig_path)

    # Accuracy
    fig, ax = plt.subplots()
    step_ignore_nan(ax, train_acc_epoch, label="Train data")
    step_ignore_nan(ax, val_acc, label="Validation data")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False)
    fig_path = os.path.join(log_dir, "validation_curve.png")
    fig.savefig(fig_path)


if __name__ == "__main__":
    main()