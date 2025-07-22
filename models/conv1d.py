import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class Conv1dModel(LightningModule):
    def __init__(self, 
                 channels: list[int],
                 kernel_sizes: list[int],
                 pool_sizes: list[int]) -> None:
        super().__init__()

        activation = nn.ReLU()
        self.seq = nn.Sequential()
        for i, kernel_size in enumerate(kernel_sizes):
            in_channels, out_channels = channels[i], channels[i+1]
            pool_size = pool_sizes[i]
            layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size), 
                nn.BatchNorm1d(out_channels), 
                nn.MaxPool1d(pool_size), 
                activation
            )
            self.seq.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        return x  
    

class ResConv1dModel(nn.Module):
    def __init__(self, channels: int, kernel_size: int, 
                 num_layers: int) -> None:
        super().__init__()

        self.activation = nn.ReLU()
        self.seq = nn.Sequential()
        for _ in range(num_layers):
            skip = nn.Conv1d(channels, channels, 1)
            conv = nn.Conv1d(channels, channels, kernel_size, padding="same")
            layer = nn.Sequential(
                AddModel(skip, conv), 
                nn.BatchNorm1d(channels), 
                self.activation
            ) 
            self.seq.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        return x
    

class AddModel(nn.Module):
    def __init__(self, module1: nn.Module, module2: nn.Module) -> None:
        super().__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module1(x) + self.module2(x)