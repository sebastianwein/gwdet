from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix

from .conv1d import Conv1dModel
from .scheduler import CosineWarmupScheduler
from .transformer import TransformerEncoderModel


class MHAModel(LightningModule):
    def __init__(self, 
                 num_tokens: int, 
                 embed_dim: int, 
                 ff_dim: int,
                 learning_rate: float) -> None:
        
        super().__init__()
        self.save_hyperparameters()

        self.loss_fn = nn.BCELoss()
        self.accuracy = BinaryAccuracy()
        self.conf_mat = lambda threshold: \
        BinaryConfusionMatrix(threshold).to(self.device)

        self.conv1d = Conv1dModel(in_channels=3,
                                  out_dim=self.hparams.embed_dim, 
                                  dropout=0.5, 
                                  skip=True)
        
        self.cls_token = nn.Parameter(torch.zeros(self.hparams.embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(self.hparams.num_tokens+1, 
                                                  self.hparams.embed_dim))
        
        self.norm = nn.LayerNorm(self.hparams.embed_dim)
        
        self.transformer \
        = TransformerEncoderModel(embed_dim=self.hparams.embed_dim, 
                                  num_heads=8, 
                                  num_layers=8, 
                                  dropout=0.5,
                                  batch_first=True)

        self.cls_head = nn.Sequential(
            nn.LazyBatchNorm1d(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hparams.embed_dim, self.hparams.ff_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.ff_dim, 1),
            nn.Sigmoid()
        )

    def tokenize(self, x: torch.Tensor) -> torch.Tensor: 
        """ 
        (batches, dets, length) -> (batches, dets, tokens, token_length) 
        where tokens = length // token_length, i.e. data gets cropped
        """
        _, _, length = x.shape
        token_length = length // self.hparams.num_tokens
        x = x[:,:,:int(self.hparams.num_tokens*token_length)] 
        x = x.unflatten(-1, (self.hparams.num_tokens, token_length))  # (b, d, t, l) 
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.tokenize(x)
        batches, _, tokens, _ = x.shape
        x = x.transpose(1, 2).flatten(0, 1)    # (b*t, d, token_len)
        x = self.conv1d(x)                     # (b*t, embed_dim)
        x = x.unflatten(0, (batches, tokens))  # (b, t, embed_dim)

        cls_token \
        = self.cls_token.unsqueeze(0).unsqueeze(0).repeat(batches, 1, 1)
        x = torch.cat((cls_token, x), 1)  # (b, t+1, embed_dim)

        pos_embed = self.pos_embed.unsqueeze(0).repeat(batches, 1, 1)
        x = x + pos_embed

        x = self.norm(x)   
        x = self.transformer(x)
        x = x[:,0,:]  # (b, embed_dim)

        x = self.cls_head(x)

        return x
    
    def configure_optimizers(self) -> tuple:
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.hparams.learning_rate)
        scheduler = CosineWarmupScheduler(optimizer, 
                                          warmup=1000,
                                          max_iters=10000)
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
        y_pred = self(x)
        features = self.conv1d(self.tokenize(x))
        if batch_idx == 0:
            self.test_samples = x
            self.test_labels = y
            self.test_predictions = y_pred
            self.test_parameters = parameters
        else: 
            self.test_samples = torch.cat((self.test_samples, x))
            self.test_labels = torch.cat((self.test_labels, y))
            self.test_predictions = torch.cat((self.test_predictions, y_pred))
            for key in self.test_parameters:
                self.test_parameters[key] \
                = torch.cat((self.test_parameters[key], parameters[key]))