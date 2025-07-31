import math
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn

class PosEncoding(nn.Module):
    def __init__(self, dim0: int, dim1: int, mode: str) -> None:
        super().__init__()
        if mode == "learnable":
            self.pos_enc = nn.Parameter(torch.normal(torch.zeros(dim0, dim1), 
                                                     1/math.sqrt(dim0*dim1)))
        elif mode == "static":
            self.pos_enc = torch.empty((dim0, dim1))
            for i in range(dim1):
                if i%2==0:
                    arr = [math.sin(pos/(10_000**(i/dim1))) 
                           for pos in range(dim0)]
                    self.pos_enc[:,i] = torch.Tensor(arr)
                else:
                    arr = [math.cos(pos/(10_000**((i-1)/dim1)))
                           for pos in range(dim0)]        
                    self.pos_enc[:,i] = torch.Tensor(arr)
        else: 
            raise ValueError(f"Unknown value {mode} for positional encoding" 
                             + "mode")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batches = x.size(0)
        pos_enc = self.pos_enc.unsqueeze(0).repeat(batches, 1, 1)
        pos_enc = pos_enc.to(x)
        x = x + pos_enc
        return x


class LazyPosEncoding(torch.nn.modules.lazy.LazyModuleMixin, nn.Module):
    def __init__(self, mode: str) -> None: 
        super().__init__()
        self.mode = mode
        if self.mode == "learnable":
            self.pos_enc = nn.UninitializedParameter()
        elif self.mode != "static": 
            raise ValueError(f"Unknown value {self.mode} for positional " \
                             + "encoding mode")
        self.initialized = False

    def initialize_parameters(self, input: torch.Tensor) -> None: 

        if self.initialized: 
            return
        
        if len(input.shape) != 3:
            raise ValueError("LazyPosEncodings expects input of " \
                             + "(BATCHES, TOKENS, EMBED_DIM), was given " \
                             + f"{input.shape} instead") 
        
        _, dim0, dim1 = input.shape
        if self.mode == "learnable":
            self.pos_enc.materialize((dim0, dim1))
            self.pos_enc = nn.Parameter(torch.normal(torch.zeros(dim0, dim1), 
                                                     1/math.sqrt(dim0*dim1)))
        elif self.mode == "static":
            self.pos_enc = torch.empty((dim0, dim1))
            for i in range(dim1):
                if i%2==0:
                    arr = [math.sin(pos/(10_000**(i/dim1))) 
                           for pos in range(dim0)]
                    self.pos_enc[:,i] = torch.Tensor(arr)
                else:
                    arr = [math.cos(pos/(10_000**((i-1)/dim1)))
                           for pos in range(dim0)]        
                    self.pos_enc[:,i] = torch.Tensor(arr)

        self.initialized = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batches = x.size(0)
        pos_enc = self.pos_enc.unsqueeze(0).repeat(batches, 1, 1)
        pos_enc = pos_enc.to(x)
        x = x + pos_enc
        return x
    