from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix


class TransformerModel(LightningModule):
    def __init__(self, token_width, learning_rate):
        super().__init__()
        self.token_width = token_width
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.loss_fn = nn.BCELoss()
        self.accuracy = BinaryAccuracy()
        self.conf_mat = lambda threshold: \
        BinaryConfusionMatrix(threshold).to(self.device)

        # Hidden representation
        self.conv1 = nn.Conv2d(1, 16, (1, 16), padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.max_pool1 = nn.MaxPool2d((1, 4))
        self.conv2 = nn.Conv2d(16, 32, (1, 8), padding="same")
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool2 = nn.MaxPool2d((1, 4))
        self.conv3 = nn.Conv2d(32, 64, (1, 8), padding="same")
        self.bn3 = nn.BatchNorm2d(64)
        self.max_pool3 = nn.MaxPool2d((1, 4))

        # Scoring
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=1)

        # Fully connected layer
        self.bn4 = nn.LazyBatchNorm1d()
        self.fc3 = nn.LazyLinear(64)
        self.fc4 = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigm = nn.Sigmoid()

        # Activation functions
        self.relu = nn.ReLU()

    def tokenize(self, x): 
        """ 
        (batches, height, width) -> (batches, tokens, height, token_width) 
        where tokens = width // token_width, i.e. data gets cropped
        """
        _, _, width = x.shape
        tokens = width // self.token_width
        x = x[:,:,:int(tokens*self.token_width)] 
        x = torch.unflatten(x, -1, (tokens, self.token_width))  # (b, h, t, w) 
        x = torch.transpose(x, 1, 2)  # (b, t, h, w)
        return x

    def conv1d(self, x):  # (b, t, h, w)
        batches, tokens, _, _ = x.shape
        x = torch.flatten(x, 0, 1)  # flatten batches and tokens
        x = torch.unflatten(x, 0, (-1, 1))  # (n=b*t, c=1, h, w)
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
        x = torch.flatten(x, 1)  # (n, c*h*w)
        x = torch.unflatten(x, 0, (batches, tokens)) 
        return x  # (b, t, (c=64)*(h=3)*width)
    
    def weight(self, x):  # (b, t, (c=64)*(h=3)*width)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x  # (b, t, 1)

    def forward(self, x):
        x = self.tokenize(x)
        features = self.conv1d(x)
        weights = self.weight(features)  
        x = torch.sum(weights*features, dim=1)
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