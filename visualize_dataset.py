import h5py 
import matplotlib.pyplot as plt
import numpy as np
import os


def main():

    plot_dir = "dataset"
    os.makedirs(plot_dir, exist_ok=True)

    file_path = "/scratch/tmp/swein/ggwd/output/ggwd_longer.hdf"
    file = h5py.File(file_path, "r")
    for key in file["normalization_parameters"].attrs.keys():
        print(key, file["normalization_parameters"].attrs[key])

    for group in file:
        for dataset in file[group]:
            if len(file[group][dataset].shape) == 1 \
            and np.issubdtype(file[group][dataset].dtype, float):
                
                array = np.ndarray(file[group][dataset].shape)
                file[group][dataset].read_direct(array)

                fig, ax = plt.subplots()
                ax.hist(array, bins=100)
                ax.set_xlabel(f"{group}/{dataset}")
                fig_path = os.path.join(plot_dir, f"{group}_{dataset}.png")
                fig.savefig(fig_path)

    return 0


if __name__ == "__main__":
    main()