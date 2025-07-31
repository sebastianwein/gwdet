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
                 strides: list[int]) -> None:
        super().__init__()

        activation = nn.ReLU()
        self.seq = nn.Sequential()
        for i, kernel_size in enumerate(kernel_sizes):
            in_channels, out_channels = channels[i], channels[i+1]
            layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, strides[i]), 
                nn.BatchNorm1d(out_channels),  
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

        activation = nn.ReLU()
        self.seq = nn.Sequential()
        for _ in range(num_layers):
            skip = nn.Conv1d(channels, channels, 1)
            conv = nn.Conv1d(channels, channels, kernel_size, padding="same")
            layer = nn.Sequential(
                AddModel(skip, conv), 
                nn.BatchNorm1d(channels), 
                activation
            ) 
            self.seq.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        return x
    

class DeepResConv1dModel(nn.Module):
    def __init__(self, 
                 channels: list[int], 
                 kernel_sizes: list[int],
                 strides: list[int],
                 depth: list[int], 
                 skip: list[bool]) -> None:
        
        super().__init__()

        activation = nn.ReLU()
        self.seq = nn.Sequential()

        in_channels = 1
        for i, num_channels in enumerate(channels):
            out_channels = num_channels

            conv_blocks = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_sizes[i], 
                          strides[i], padding=kernel_sizes[i]//2), 
                nn.BatchNorm1d(out_channels), 
            )
            for _ in range(depth[i]-1):
                conv_blocks.append(activation) \
                           .append(nn.Conv1d(out_channels, out_channels, 
                                            kernel_sizes[i], padding="same")) \
                           .append(nn.BatchNorm1d(out_channels))

            if skip[i]:
                self.seq.append(AddModel(nn.Conv1d(in_channels, out_channels, 1, 
                                                   strides[i]), 
                                         conv_blocks)) \
                        .append(activation)
            else:
                self.seq.append(conv_blocks) \
                        .append(activation)

            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> None:
        """
        (N, C, L) -> (N, channels[-1])
        """
        if len(x.shape) == 2:
            x = x.unsqeeze(1)
            x = self.seq(x)  # (B, C, L)
            x = x.mean(dim=-1)
        elif len(x.shape) == 3:
            batches, channels, _ = x.shape
            x = x.flatten(0, 1)                      # (B*C, L)
            x = x.unsqueeze(1)                       # (B*C, 1, L)
            x = self.seq(x)                          # (B*C, out_dim, L')
            x = x.unflatten(0, (batches, channels))  # (B, C, out_dim, L')
            x = x.transpose(1, 2)                    # (B, out_dim, C, L')
            x = x.flatten(2)                         # (B, out_dim, C*L')
            x = x.mean(dim=-1)                       # (B, out_dim)
            return x
    

class AddModel(nn.Module):
    def __init__(self, module1: nn.Module, module2: nn.Module) -> None:
        super().__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module1(x) + self.module2(x)