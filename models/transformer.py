import torch
import torch.nn as nn


class TransformerEncoderModel(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 num_layers: int, 
                 dropout: float = 0,
                 batch_first: bool = False) -> None:
        
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.ff_dim = 2048

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, 
                                         num_heads=self.num_heads, 
                                         dropout=self.dropout,
                                         batch_first=self.batch_first)
        self.lin1 = nn.Linear(self.embed_dim, self.ff_dim)
        self.activation = nn.GELU()
        self.lin2 = nn.Linear(self.ff_dim, self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        return x

    def layer(self, 
              x: torch.Tensor, 
              need_weights: bool = False) -> torch.Tensor:
        # Pre layer normalization
        norm1 = self.norm1(x)
        q, k, v = norm1, norm1, norm1
        mha, weigths = self.mha(q, k, v, 
                                need_weights=need_weights, 
                                average_attn_weights=False)
        x = x + mha
        norm2 = self.norm2(x)
        ff = self.feed_forward(norm2)
        x = x + ff
        return x, weigths

    def forward(self, 
                x: torch.Tensor, 
                need_weights: bool = False) -> torch.Tensor:
        for _ in range(self.num_layers):
            x, weigths = self.layer(x, need_weights)
        return x, weigths
