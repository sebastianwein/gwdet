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

        self._norm1 = nn.LayerNorm(self.embed_dim)
        self._mha = nn.MultiheadAttention(embed_dim=self.embed_dim, 
                                          num_heads=self.num_heads, 
                                          dropout=self.dropout,
                                          batch_first=self.batch_first)
        self._lin1 = nn.Linear(self.embed_dim, self.ff_dim)
        self._activation = nn.GELU()
        self._lin2 = nn.Linear(self.ff_dim, self.embed_dim)
        self._norm2 = nn.LayerNorm(self.embed_dim)

    def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._lin1(x)
        x = self._activation(x)
        x = self._lin2(x)
        return x

    def _layer(self, x: torch.Tensor) -> torch.Tensor:
        # Pre layer normalization
        norm1 = self._norm1(x)
        q, k, v = norm1, norm1, norm1
        mha, _ = self._mha(q, k, v, need_weights=False)
        x = x + mha
        norm2 = self._norm2(x)
        ff = self._feed_forward(norm2)
        x = x + ff
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_layers):
            x = self._layer(x)
        return x
