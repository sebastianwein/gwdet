import glob
import h5py
import numpy as np
import os
from pytorch_lightning import LightningDataModule
import random
import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Type


class GGWDDataset(Dataset):
    def __init__(self, file_path: str) -> None:
        file = h5py.File(file_path, "r")
        self.injection_e1 = file["timeseries/samples/injection/e1"]
        self.injection_e2 = file["timeseries/samples/injection/e2"]
        self.injection_e3 = file["timeseries/samples/injection/e3"]
        self.noise_e1 = file["timeseries/samples/noise/e1"]
        self.noise_e2 = file["timeseries/samples/noise/e2"]
        self.noise_e3 = file["timeseries/samples/noise/e3"]
        attrs = file["timeseries/samples"].attrs
        self.mean_e1 = attrs["e1_mean"]
        self.std_e1 = attrs["e1_std"] if attrs["e1_std"]!=0 else 10e-23
        self.mean_e2 = attrs["e2_mean"]
        self.std_e2 = attrs["e2_std"] if attrs["e2_std"]!=0 else 10e-23
        self.mean_e3 = attrs["e3_mean"]
        self.std_e3 = attrs["e3_std"] if attrs["e3_std"]!=0 else 10e-23
        self.sample_shape = (3, self.injection_e1.shape[1])

    def __len__(self) -> int:
        return len(self.injection_e1) + len(self.noise_e1)

    def __getitem__(self, idx: int) -> tuple:
        if idx < len(self.injection_e1): 
            sample = np.array([(self.injection_e1[idx]-self.mean_e1) \
                               / self.std_e1, 
                               (self.injection_e2[idx]-self.mean_e2) \
                               / self.std_e2,  
                               (self.injection_e3[idx]-self.mean_e3) \
                               / self.std_e3], dtype=np.float32)
            target = np.array([1], dtype=np.float32)
        else:
            shifted_idx = idx-len(self.injection_e1)
            sample = np.array([(self.noise_e1[shifted_idx]-self.mean_e1) \
                               / self.std_e1, 
                               (self.noise_e2[shifted_idx]-self.mean_e2) \
                               / self.std_e2,  
                               (self.noise_e3[shifted_idx]-self.mean_e3) \
                               / self.std_e3], dtype=np.float32)
            target = np.array([0], dtype=np.float32)
        return sample, target


class GGWDTestDataset(GGWDDataset):
    # Inherits GGWDDataset, but returns parameters dict as well in getitem()
    def __init__(self, file_path: str):
        super().__init__(file_path)
        group = "parameters"
        self.keys = list()
        file = h5py.File(file_path, "r")
        for dataset in file[group]:
            self.keys.append(dataset)
        self.datasets = {key: file[group][key] for key in self.keys}

    def __getitem__(self, idx: int) -> tuple:
        sample, target = super().__getitem__(idx)
        parameters = {k: np.array(self.datasets[k][idx]) 
                         for k in self.keys}
        return sample, target, parameters
    

class LargeDataset(Dataset):
    def __init__(self, 
                 dataset_cls: Type[Dataset], 
                 file_paths: list[str]) -> None:
        # TODO: check all files being compatible
        self.dataset_cls = dataset_cls
        self.file_paths = file_paths
        self.file_idx = 0
        self.dataset = self.dataset_cls(self.file_paths[self.file_idx])
        # Assumes all files being of the same size as the first file
        self.file_size = len(self.dataset)
        self.sample_shape = self.dataset.sample_shape
    
    def __len__(self) -> int:
        return len(self.file_paths)*self.file_size
    
    def __getitem__(self, idx):
        file_idx = int(idx/self.file_size)
        if file_idx != self.file_idx:
            self.file_idx = file_idx
            self.dataset = self.dataset_cls(self.file_paths[self.file_idx])
        return self.dataset[idx-self.file_idx*self.file_size]


class LargeDatasetSampler():
    def __init__(self, dataset: LargeDataset, batch_size: int):
        self.num_files = len(dataset.file_paths)
        self.file_size = dataset.file_size
        self.batch_size = batch_size

        self.file_idx = 0
        self.batch_idx = 0
        self.randperm = torch.randperm(self.file_size)

    def __len__(self) -> int:
        return (self.file_size/self.batch_size).__ceil__() * self.num_files
    
    def __iter__(self):
        return self

    # TODO: ugly
    def __next__(self) -> torch.Tensor:
        randperm = self.randperm
        low = self.batch_idx*self.batch_size
        high = (self.batch_idx+1)*self.batch_size

        if high >= self.file_size:
            high = None
            self.file_idx += 1
            if self.file_idx >= self.num_files: 
                self.file_idx = 0
            self.batch_idx = 0
            self.randperm = torch.randperm(self.file_size) \
                            + self.file_idx*self.file_size
        else: 
            self.batch_idx += 1

        return randperm[low:high]


class GGWDData(LightningDataModule):
    def __init__(self, data_dirs: list[str], batch_size: int, num_workers: int):
        super().__init__()
        files = list()
        for dir in data_dirs:
            files.extend(glob.glob(os.path.join(dir, "*.hdf")))
        num_train_files = int(0.9*len(files)) - 1
        train_files = random.sample(files[:-1], num_train_files)
        val_files = list(set(files[:-1]) - set(train_files))
        self.train_dataset = LargeDataset(GGWDDataset, train_files)
        self.val_dataset = LargeDataset(GGWDDataset, val_files)
        indices = torch.randperm(self.train_dataset.file_size)[:2048]
        self.test_dataset = Subset(GGWDTestDataset(files[-1]), indices)

        self.batch_size = batch_size
        self.sampler = LargeDatasetSampler(self.train_dataset, 
                                           batch_size=self.batch_size)
        self.num_workers = num_workers
        self.sample_shape = self.train_dataset.sample_shape
        self.out_shape = (self.batch_size, *self.sample_shape)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, 
                          batch_sampler=self.sampler,
                          num_workers=self.num_workers)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers)