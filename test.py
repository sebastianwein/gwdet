import h5py
import numpy as np
import torch

file_path = "/scratch/tmp/swein/ggwd/output/bbh.hdf"
file = h5py.File(file_path, "r")
injection = file["injection_samples"]["e1_strain"]
noise = file["noise_samples"]["e1_strain"]
a = torch.as_tensor((injection[1], injection[2]))
print(a)

targets = np.concatenate((np.ones(len(injection)), np.zeros(len(noise))))[:,np.newaxis]