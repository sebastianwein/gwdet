import glob
import h5py
import numpy as np
import os
import pycbc
from pytorch_lightning import LightningDataModule
import random
import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Type


class GGWDDataset(Dataset):
    def __init__(self, 
                 file_path: str, 
                 length: int = None,
                 whiten: bool = False, 
                 segment_duration: float = None, 
                 max_filter_duration: float = None) -> None:
        
        file = h5py.File(file_path, "r")
        self.injection_e1 = file["timeseries/samples/injection/e1"]
        self.injection_e2 = file["timeseries/samples/injection/e2"]
        self.injection_e3 = file["timeseries/samples/injection/e3"]
        self.noise_e1 = file["timeseries/samples/noise/e1"]
        self.noise_e2 = file["timeseries/samples/noise/e2"]
        self.noise_e3 = file["timeseries/samples/noise/e3"]
        self.delta_t = file["timeseries"].attrs["delta_t"]
        if length is not None: self.length = length
        else: self.length = self.injection_e1.shape[1]
        self.sample_shape = (3, self.length)

        attrs = file["timeseries/samples"].attrs
        self.mean_e1 = attrs["e1_mean"]
        self.std_e1 = attrs["e1_std"] if attrs["e1_std"]!=0 else 10e-23
        self.mean_e2 = attrs["e2_mean"]
        self.std_e2 = attrs["e2_std"] if attrs["e2_std"]!=0 else 10e-23
        self.mean_e3 = attrs["e3_mean"]
        self.std_e3 = attrs["e3_std"] if attrs["e3_std"]!=0 else 10e-23

        self.whiten = whiten
        self.segment_duration = segment_duration
        self.max_filter_duration = max_filter_duration
        if self.whiten:
            if self.segment_duration is None:
                raise ValueError("Provide value for segment_duration")
            if self.max_filter_duration is None:
                raise ValueError("Provide value for max_filter_duration")

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
            shifted_idx = idx - len(self.injection_e1)
            sample = np.array([(self.noise_e1[shifted_idx]-self.mean_e1) \
                               / self.std_e1, 
                               (self.noise_e2[shifted_idx]-self.mean_e2) \
                               / self.std_e2,  
                               (self.noise_e3[shifted_idx]-self.mean_e3) \
                               / self.std_e3], dtype=np.float32)
            target = np.array([0], dtype=np.float32)
        if self.whiten:
            arr = [pycbc.types.TimeSeries(s, delta_t=self.delta_t) \
                   .whiten(segment_duration=self.segment_duration, 
                           max_filter_duration=self.max_filter_duration, 
                           remove_corrupted=False)
                   for s in sample]
            sample = np.array(arr)
        if self.length <= sample.shape[1]:
            sample = sample[:,:self.length]
        else:
            pad_width = self.length - sample.shape[1]
            before = random.randint(0, pad_width)
            after = pad_width - before
            sample = np.pad(sample, ((0, 0), (before, after)))
        return sample, target


class GGWDTestDataset(GGWDDataset):
    # Inherits GGWDDataset, but returns parameters dict as well in getitem()
    def __init__(self, file_path: str, **kwargs) -> None:
        super().__init__(file_path, **kwargs)
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
                 file_paths: list[str], 
                 **kwargs) -> None:
        self.dataset_cls = dataset_cls
        self.kwargs = kwargs
        self.file_paths = file_paths
        self.file_idx = 0
        self.datasets = [self.dataset_cls(file_path, **self.kwargs)
                         for file_path in self.file_paths]
        self.file_sizes = [len(dataset) for dataset in self.datasets]
        # Assumes all samples having the same shape as for the first file
        self.sample_shape = self.datasets[self.file_idx].sample_shape
    
    def __len__(self) -> int:
        return sum(self.file_sizes)
    
    def __getitem__(self, idx):
        cumsum = np.cumsum(self.file_sizes)
        file_idx = next(i for i, x in enumerate(cumsum) if x > idx)
        if file_idx != self.file_idx:
            self.file_idx = file_idx
            self.dataset = self.dataset_cls(self.file_paths[self.file_idx], 
                                            **self.kwargs)
        shift = 0 if self.file_idx == 0 else cumsum[self.file_idx-1]
        return self.datasets[self.file_idx][idx-shift]


class LargeDatasetSampler():
    def __init__(self, dataset: LargeDataset, batch_size: int):
        self.num_files = len(dataset.file_paths)
        self.file_sizes = dataset.file_sizes
        self.batch_size = batch_size

        self.file_idx = 0
        self.batch_idx = 0
        self.randperm = torch.randperm(self.file_sizes[self.file_idx])

    def __len__(self) -> int:
        iterations = [(file_size/self.batch_size).__ceil__() 
                      for file_size in self.file_sizes]
        return sum(iterations)
    
    def __iter__(self):
        return self

    # TODO: ugly
    def __next__(self) -> torch.Tensor:
        randperm = self.randperm
        low = self.batch_idx*self.batch_size
        high = (self.batch_idx+1)*self.batch_size

        if high >= self.file_sizes[self.file_idx]:
            high = None
            self.file_idx += 1
            if self.file_idx >= self.num_files: 
                self.file_idx = 0
            self.batch_idx = 0
            self.randperm = torch.randperm(self.file_sizes[self.file_idx]) \
                            + self.file_idx*self.file_sizes[self.file_idx]
        else: 
            self.batch_idx += 1

        return randperm[low:high]


class GGWDData(LightningDataModule):
    def __init__(self, 
                 data_dirs: list[str], 
                 batch_size: int, 
                 num_workers: int, 
                 **kwargs):
        super().__init__()
        files = list()
        for dir in data_dirs:
            files.extend(glob.glob(os.path.join(dir, "*.hdf")))
        num_train_files = int(0.9*len(files)) - 1
        train_files = random.sample(files[:-1], num_train_files)
        val_files = list(set(files[:-1]) - set(train_files))
        self.train_dataset = LargeDataset(GGWDDataset, train_files, **kwargs)
        self.val_dataset = LargeDataset(GGWDDataset, val_files, **kwargs)
        indices = torch.randperm(self.train_dataset.file_sizes[-1])[:2048]
        self.test_dataset = Subset(GGWDTestDataset(files[-1],**kwargs), indices)

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