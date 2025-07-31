import torch
import torch.nn as nn


class MinMaxPool1d(nn.Module):
    def __init__(self, pool_size: int) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool1d(pool_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs = torch.abs(x)
        _, indices = self.maxpool(abs, return_indices=True)
        x = x[indices]
        return x


class Conv1dModel(nn.Module):
    def __init__(self, 
                 channels: list[int],
                 kernel_sizes: list[int],
                 strides: list[int], 
                 pool_sizes: list[int]) -> None:
        super().__init__()

        activation = nn.ReLU()
        self.seq = nn.Sequential()
        for i, kernel_size in enumerate(kernel_sizes):
            in_channels, out_channels = channels[i], channels[i+1]
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, strides[i])
            self.seq.append(conv) \
                    .append(nn.BatchNorm1d(out_channels)) \
                    .append(nn.MaxPool1d(pool_sizes[i])) \
                    .append(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        return x  
    

class ResConv1dModel(nn.Module):
    def __init__(self, 
                 channels: int, 
                 kernel_size: int, 
                 num_layers: int) -> None:
        super().__init__()

        activation = nn.ReLU()
        self.seq = nn.Sequential()
        for _ in range(num_layers):
            skip = nn.Conv1d(channels, channels, 1)
            conv = nn.Conv1d(channels, channels, kernel_size, padding="same")
            self.seq.append(AddModel(skip, conv)) \
                    .append(nn.BatchNorm1d(channels)) \
                    .append(activation)

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