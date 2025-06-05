import numpy as np
import h5py
from pytorch_lightning import LightningDataModule
import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset


class GGWDDataset(Dataset):
    def __init__(self, file_path):
        file = h5py.File(file_path, "r")
        self.injection_e1 = file["injection_samples"]["e1_strain"]
        self.injection_e2 = file["injection_samples"]["e2_strain"]
        self.injection_e3 = file["injection_samples"]["e3_strain"]
        self.noise_e1 = file["noise_samples"]["e1_strain"]
        self.noise_e2 = file["noise_samples"]["e2_strain"]
        self.noise_e3 = file["noise_samples"]["e3_strain"]
        self.sample_shape = (3, self.injection_e1.shape[1])

    def __len__(self):
        return len(self.injection_e1) + len(self.noise_e1)

    def __getitem__(self, idx):
        if idx < len(self.injection_e1): 
            sample = np.array([self.injection_e1[idx], 
                               self.injection_e2[idx], 
                               self.injection_e3[idx]])*1e+23
            target = np.array([1], dtype=np.float32)
        else:
            idx -= len(self.injection_e1)
            sample = np.array([self.noise_e1[idx], 
                               self.noise_e2[idx], 
                               self.noise_e3[idx]])*1e+23
            target = np.array([0], dtype=np.float32)
        return sample, target


class GGWDTestDataset(GGWDDataset):
    # Inherits GGWDDataset, but returns parameters dict as well in getitem()
    def __init__(self, file_path):
        super().__init__(file_path)
        file = h5py.File(file_path, "r")
        group = "injection_parameters"
        self.keys = list(file[group].keys())
        self.datasets = {k: file[group][k] for k in self.keys}

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        if idx < len(self.injection_e1): 
            parameters = {k: np.array(self.datasets[k][idx]) 
                          for k in self.keys}
        else:
            parameters = {k: np.full(self.datasets[k][0].shape, np.nan) 
                          for k in self.keys}
        return sample, target, parameters


class Data(LightningDataModule):
    def __init__(self, file_path, batch_size, num_workers):
        super().__init__()
        self.generator = torch.Generator().manual_seed(42)
        dataset = GGWDDataset(file_path)
        self.train_dataset, self.val_dataset \
        = random_split(dataset, (0.8, 0.2), generator=self.generator)
        test_dataset = GGWDTestDataset(file_path)
        _, self.test_dataset \
        = random_split(test_dataset, (0.8, 0.2), generator=self.generator) # I don't think this works
        self.sample_shape = dataset.sample_shape
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          generator=self.generator,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers)


