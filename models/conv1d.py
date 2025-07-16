import torch
import torch.nn as nn


class Conv1dModel(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_dim: int, 
                 batch_norm: bool = True, 
                 dropout: float = 0) -> None:
        
        super().__init__()
        
        self.out_dim = out_dim
        self.batch_norm = batch_norm  
        self.dropout = dropout      

        self.conv1 = nn.Conv1d(in_channels, 16, 15)
        self.bn1 = nn.BatchNorm1d(16)
        self.max_pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(16, 32, 9)
        self.bn2 = nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(32, 64, 7)
        self.bn3 = nn.BatchNorm1d(64)
        self.max_pool3 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(64, 64, 3, padding="same")
        self.skip1 = nn.Conv1d(64, 64, 1)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, 3, padding="same")
        self.skip2 = nn.Conv1d(64, 64, 1)
        self.bn5 = nn.BatchNorm1d(64)

        self.activation = nn.ReLU()

        self.dp = nn.Dropout(p=self.dropout)
        self.fc = nn.LazyLinear(self.out_dim)

    def forward(self, x):  # (n, c, l)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.max_pool3(x)

        x = self.skip1(x) + self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)

        x = self.skip2(x) + self.conv5(x)
        x = self.bn5(x)
        x = self.activation(x)

        x = x.flatten(1)
        x = self.dp(x)
        x = self.fc(x)
        x = self.activation(x)
        
        return x  # (n, c*l)