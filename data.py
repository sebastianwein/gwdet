from pytorch_lightning import LightningDataModule
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


def Dataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class Data(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader():
        dataset = Dataset()
        return DataLoader(dataset, batch_size=self.batch_size)
    
    def val_dataloader():
        dataset = Dataset()
        return DataLoader(dataset, batch_size=self.batch_size)