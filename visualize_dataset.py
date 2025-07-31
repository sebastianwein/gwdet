import h5py 
import matplotlib.pyplot as plt
import numpy as np
import os


def main():

    plot_dir = "dataset"
    os.makedirs(plot_dir, exist_ok=True)

    file_path = "/scratch/tmp/swein/ggwd/output/longer/data/0.hdf"
    file = h5py.File(file_path, "r")
    group = "parameters"

    for dataset in file[group]:
            
            array = np.ndarray(file[group][dataset].shape)
            file[group][dataset].read_direct(array)
            mask = np.isfinite(array)
            array = array[mask]

            if array.size > 0:
                fig, ax = plt.subplots()
                ax.hist(array, bins=100)
                ax.set_xlabel(f"{group}/{dataset}")
                fig_path = os.path.join(plot_dir, f"{group}_{dataset}.png")
                fig.savefig(fig_path)

    return 0


if __name__ == "__main__":
    main()