import torch
import torch.nn as nn


class Conv1dModel(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_dim: int, 
                 batch_norm: bool = True, 
                 dropout: float = 0, 
                 skip: bool = False) -> None:
        
        super().__init__()
        
        self.out_dim = out_dim
        self.batch_norm = batch_norm  
        self.dropout = dropout      
        self.skip = skip

        self.conv1 = nn.Conv1d(in_channels, 16, 16, padding="same")
        self.skip1 = nn.Conv1d(in_channels, 16, 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.max_pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(16, 32, 8, padding="same")
        self.skip2 = nn.Conv1d(16, 32, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(32, 64, 8, padding="same")
        self.skip3 = nn.Conv1d(32, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.max_pool3 = nn.MaxPool1d(4)
        self.relu = nn.ReLU()

        self.dp = nn.Dropout(p=self.dropout)
        self.fc = nn.LazyLinear(self.out_dim)

    def forward(self, x):  # (n, c, l)

        if self.skip: x = self.skip1(x) + self.conv1(x)
        else: x = self.conv1(x)
        if self.batch_norm: x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool1(x)

        if self.skip: x = self.skip2(x) + self.conv2(x)
        else: x = self.conv2(x)
        if self.batch_norm: x = self.bn2(x)
        x = self.relu(x)
        x = self.max_pool2(x)

        if self.skip: x = self.skip3(x) + self.conv3(x)
        else: x = self.conv3(x)
        if self.batch_norm: x = self.bn3(x)
        x = self.relu(x)
        x = self.max_pool3(x)

        x = x.flatten(1)
        x = self.dp(x)
        x = self.fc(x)
        
        return x  # (n, c*l)