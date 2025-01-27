import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassConfusionMatrix, Accuracy
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class SimpleCNN(pl.LightningModule):
    def __init__(self, in_channels=1, dropout=0.05, n_classes=2, w=512, learning_rate=1e-3, log_dir="logs", class_weights=None):
        super(SimpleCNN, self).__init__()

        self.hyperparameters = {
            'in_channels': in_channels,
            'dropout': dropout,
            'n_classes': n_classes,
            'w': w,
            'learning_rate': learning_rate
        }

        self.class_weights = class_weights or torch.ones(n_classes)
        self.log_dir = log_dir

        # TensorBoard Logger
        self.writer = SummaryWriter(log_dir=log_dir)

        self.conv1 = Convolution(spatial_dims=2, in_channels=in_channels, out_channels=16, adn_ordering="ADN", dropout=dropout)
        self.conv2 = Convolution(spatial_dims=2, in_channels=16, out_channels=16, adn_ordering="ADN", dropout=dropout)
        self.conv3 = Convolution(spatial_dims=2, in_channels=16, out_channels=16, adn_ordering="ADN", dropout=dropout)
        self.mxpool = nn.MaxPool2d(4)

        self.out_head = nn.Sequential(
            nn.Linear(w // 64 * w // 64 * 16, 64),
            nn.PReLU(),
            nn.Linear(64, 64),
            nn.PReLU(),
            nn.Linear(64, n_classes)
        )

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        self.learning_rate = learning_rate

        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=n_classes)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        out = self.mxpool(self.conv1(x))
        out = self.mxpool(self.conv2(out))
        out = self.mxpool(self.conv3(out))
        out = nn.Flatten(start_dim=1)(out)
        out = self.out_head(out)
        return out

   
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
       
        preds = self(x)
        loss = self.criterion(preds, y)

      
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

