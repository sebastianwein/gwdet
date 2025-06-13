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
        attrs = file["normalization_parameters"].attrs
        self.mean_e1 = attrs["E1_mean"]
        self.std_e1 = attrs["E1_std"] if attrs["E1_std"]>0 else 10e-23
        self.mean_e2 = attrs["E2_mean"]
        self.std_e2 = attrs["E2_std"] if attrs["E1_std"]>0 else 10e-23
        self.mean_e3 = attrs["E3_mean"]
        self.std_e3 = attrs["E3_std"] if attrs["E1_std"]>0 else 10e-23
        self.sample_shape = (3, self.injection_e1.shape[1])

    def __len__(self):
        return len(self.injection_e1) + len(self.noise_e1)

    def __getitem__(self, idx):
        if idx < len(self.injection_e1): 
            sample = np.array([(self.injection_e1[idx]-self.mean_e1) \
                               / self.std_e1, 
                               (self.injection_e2[idx]-self.mean_e2) \
                               / self.std_e2,  
                               (self.injection_e3[idx]-self.mean_e3) \
                               / self.std_e3], dtype=np.float32)
            target = np.array([1], dtype=np.float32)
        else:
            idx -= len(self.injection_e1)
            sample = np.array([(self.noise_e1[idx]-self.mean_e1) \
                               / self.std_e1, 
                               (self.noise_e2[idx]-self.mean_e2) \
                               / self.std_e2,  
                               (self.noise_e3[idx]-self.mean_e3) \
                               / self.std_e3], dtype=np.float32)
            target = np.array([0], dtype=np.float32)
        return sample, target


class GGWDTestDataset(GGWDDataset):
    # Inherits GGWDDataset, but returns parameters dict as well in getitem()
    def __init__(self, file_path):
        super().__init__(file_path)
        file = h5py.File(file_path, "r")
        group = "injection_parameters"
        self.keys = list()
        for k in file[group].keys():
            if np.issubdtype(file[group][k].dtype, float):
                self.keys.append(k)
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


class GGWDData(LightningDataModule):
    def __init__(self, file_path, batch_size, num_workers):
        super().__init__()
        dataset = GGWDDataset(file_path)
        self.train_dataset, self.val_dataset \
        = random_split(dataset, (0.8, 0.2))
        test_dataset = GGWDTestDataset(file_path)
        self.test_dataset = Subset(test_dataset, self.val_dataset.indices)
        self.sample_shape = dataset.sample_shape
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
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

