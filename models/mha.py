import math
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix

from .conv1d import Conv1dModel
from .scheduler import CosineWarmupScheduler
from .transformer import TransformerEncoderModel
from .posenc import LazyPosEncoding


class MHAModel(LightningModule):
    def __init__(self, 
                 token_length: int, 
                 embed_dim: int, 
                 num_heads: int, 
                 num_layers: int,
                 ff_dim: int,
                 dropout: float, 
                 learning_rate: float, 
                 weight_decay: float, 
                 warmup: int, 
                 pos_enc: str) -> None:
        
        super().__init__()
        self.save_hyperparameters()

        self.loss_fn = nn.BCELoss()
        self.accuracy = BinaryAccuracy()

        self.conv = Conv1dModel(channels=[1, 8, 16, 32, 64], 
                                kernel_sizes=[15, 9, 7, 5], 
                                pool_sizes=[4, 4, 4, 4], 
                                strides=[1, 1, 1, 1])
        self.embed = nn.LazyLinear(self.hparams.embed_dim)

        self.pos_enc = LazyPosEncoding(mode=self.hparams.pos_enc)
        self.dropout = nn.Dropout(p=self.hparams.dropout)

        self.cls_token \
        = nn.Parameter(torch.normal(torch.zeros(self.hparams.embed_dim),
                                    1/math.sqrt(self.hparams.embed_dim)))
        
        self.transformer \
        = TransformerEncoderModel(embed_dim=self.hparams.embed_dim, 
                                  num_heads=self.hparams.num_heads, 
                                  num_layers=self.hparams.num_layers, 
                                  dropout=self.hparams.dropout,
                                  batch_first=True)

        self.cls_head = nn.Sequential(
            nn.LazyBatchNorm1d(),
            nn.Linear(self.hparams.embed_dim, self.hparams.ff_dim),
            nn.Dropout(p=self.hparams.dropout),
            nn.Tanh(),
            nn.Linear(self.hparams.ff_dim, 1),
            nn.Dropout(p=self.hparams.dropout),
            nn.Sigmoid()
        )

    def tokenize(self, x: torch.Tensor) -> torch.Tensor: 
        """ 
        (*, length) -> (*, tokens, token_length) 
        where num_tokens = length // token_length, i.e. data gets cropped
        """
        length = x.size(-1)
        tokens = length // self.hparams.token_length
        x = x[...,:int(tokens*self.hparams.token_length)] 
        x = x.unflatten(-1, (tokens, self.hparams.token_length))  
        return x
    
    def forward(self, 
                x: torch.Tensor, 
                need_weights: bool = False) -> torch.Tensor:

        x = self.tokenize(x)                          # (b, dets, t, token_length)
        batches, dets, tokens, _ = x.shape
        x = x.transpose(1, 2)                         # (b, t, dets, token_length)
        x = x.flatten(0, 2)                           # (b*t*dets, token_length)
        x = x.unsqueeze(1)                            # (b*t*dets, 1, token_length) 
        x = self.conv(x)                              # (b*t*dets, c', l')
        x = x.unflatten(0, (batches, tokens, dets))   # (b, t, dets, c', l')
        x = x.flatten(-3)                             # (b, t, det*c'*l')
        x = self.embed(x)                             # (b, t, embed_dim)    

        x = self.pos_enc(x)
        x = self.dropout(x)

        cls_token \
        = self.cls_token.unsqueeze(0).unsqueeze(0).repeat(batches, 1, 1)
        x = torch.cat((cls_token, x), 1)  # (b, t+1, embed_dim)

        x, weights = self.transformer(x, need_weights)
        x = x[:,0,:]  # (b, embed_dim)

        x = self.cls_head(x)

        if need_weights:
            return x, weights
        else: 
            return x
    
    def configure_optimizers(self) -> tuple:
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.hparams.learning_rate, 
                                     weight_decay=self.hparams.learning_rate)
        self.trainer.fit_loop.setup_data()
        iters = len(self.trainer.train_dataloader.dataset)
        max_iters = self.trainer.max_epochs*iters
        scheduler = CosineWarmupScheduler(optimizer, 
                                          warmup=self.hparams.warmup,
                                          max_iters=max_iters)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch: int, batch_idx: int) -> float:
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True)
        acc = self.accuracy(y_pred, y)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True) 
        return loss

    def validation_step(self, batch: int, batch_idx: int) -> None:
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True)
        acc = self.accuracy(y_pred, y)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, 
                 logger=True)

    def test_step(self, batch: int, batch_idx: int) -> None:
        x, y, parameters = batch
        y_pred, weights = self(x, need_weights=True)
        if batch_idx == 0:
            self.test_samples = x
            self.test_labels = y
            self.test_predictions = y_pred
            self.test_weights = weights
            self.test_parameters = parameters
        else: 
            self.test_samples = torch.cat((self.test_samples, x))
            self.test_labels = torch.cat((self.test_labels, y))
            self.test_predictions = torch.cat((self.test_predictions, y_pred))
            self.test_weights = torch.cat((self.test_weights, weights))
            for key in self.test_parameters:
                self.test_parameters[key] \
                = torch.cat((self.test_parameters[key], parameters[key]))