from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix


class SHAModel(LightningModule):
    def __init__(self, token_length, learning_rate):
        super().__init__()
        self.token_length = token_length
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.loss_fn = nn.BCELoss()
        self.accuracy = BinaryAccuracy()
        self.conf_mat = lambda threshold: \
        BinaryConfusionMatrix(threshold).to(self.device)

        # Hidden representation
        self.conv1 = nn.Conv1d(1, 16, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.max_pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(16, 32, 8)
        self.bn2 = nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(32, 64, 8)
        self.bn3 = nn.BatchNorm1d(64)
        self.max_pool3 = nn.MaxPool1d(4)

        # Scoring
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.Linear(64, 1)
        self.scoring_softmax = nn.Softmax(dim=2)

        # Fully connected layer
        self.dropout = nn.Dropout()
        self.bn4 = nn.LazyBatchNorm1d()
        self.fc3 = nn.LazyLinear(32)
        self.fc4 = nn.Linear(32, 1)
        self.sigm = nn.Sigmoid()

        # Activation functions
        self.relu = nn.ReLU()

    def tokenize(self, x): 
        """ 
        (batches, dets, length) -> (batches, dets, tokens, token_length) 
        where tokens = length // token_length, i.e. data gets cropped
        """
        _, _, length = x.shape
        tokens = length // self.token_length
        x = x[:,:,:int(tokens*self.token_length)] 
        x = torch.unflatten(x, -1, (tokens, self.token_length))  # (b, d, t, l) 
        return x

    def conv1d(self, x):  # (b, d, t, l) 
        batches, dets, tokens, _ = x.shape
        x = torch.flatten(x, 0, -2)  # flatten batches, dets and tokens (n=b*d*t, l)
        x = torch.unflatten(x, 1, (1, -1))  # (n, c=1, l)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.max_pool3(x)
        x = torch.flatten(x, 1)  # (n, (c=64)*l)
        x = torch.unflatten(x, 0, (batches, dets, tokens))  # (b, d, t, l)
        return x  # (b, d, t, l)
    
    def weight(self, x):  # (b, d, t, (c=64)*length)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.scoring_softmax(x)
        return x  # (b, d, t, 1)

    def forward(self, x):
        x = self.tokenize(x)
        features = self.conv1d(x)
        weights = self.weight(features)  
        prod = weights*features
        x = torch.sum(prod, dim=2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.bn4(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.sigm(x)
        return x  

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True)
        acc = self.accuracy(y_pred, y)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True)
        acc = self.accuracy(y_pred, y)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, 
                 logger=True)

    def test_step(self, batch, batch_idx):
        x, y, parameters = batch
        y_pred = self(x)
        features = self.conv1d(self.tokenize(x))
        weights = self.weight(features).squeeze()
        parameters["weights"] = weights
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