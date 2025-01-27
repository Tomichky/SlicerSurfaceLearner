from argparse import ArgumentParser
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchmetrics

class ImageClassifier(pl.LightningModule):
    def __init__(self, backbone, custom_device=None, criterion=nn.CrossEntropyLoss(), learning_rate=1e-3, metrics=["acc"]):
        super(ImageClassifier, self).__init__()
        self.save_hyperparameters(ignore=['backbone', 'criterion'])
        self.backbone = backbone
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.custom_device = custom_device
        self.train_metrics = {}
        self.val_metrics = {}
        self.metric_names = metrics
        for m in metrics:
            if m == "acc":
                train_metric = torchmetrics.Accuracy(task="binary").to(custom_device)
                val_metric = torchmetrics.Accuracy(task="binary").to(custom_device)
            elif m == "auc":
                train_metric = torchmetrics.AUROC(pos_label=1, task="binary").to(custom_device)
                val_metric = torchmetrics.AUROC(pos_label=1, task="binary").to(custom_device)
            elif m == "precision":
                train_metric = torchmetrics.Precision(task="binary").to(custom_device)
                val_metric = torchmetrics.Precision(task="binary").to(custom_device)
            elif m == "recall":
                train_metric = torchmetrics.Recall(task="binary").to(custom_device)
                val_metric = torchmetrics.Recall(task="binary").to(custom_device)
            self.train_metrics[m] = train_metric
            self.val_metrics[m] = val_metric

    def forward(self, x):
        return self.backbone(x)

    def common_step(self, batch, batch_idx, mode="train"):
        x, y = batch
        y = y.to(self.device)  
        x = x.to(self.device)
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        
        y_probs = nn.Softmax(dim=-1)(y_hat)
        if y_probs.shape[1] == 2:
            y_hat = y_probs[:, 1]
        else:
            y_hat = torch.argmax(y_probs, dim=1)
        
        if mode == "train":
            for m in self.metric_names:
                self.train_metrics[m](y_hat, y)
        elif mode == "valid":
            for m in self.metric_names:
                self.val_metrics[m](y_hat, y)
        return loss

    def on_train_epoch_end(self):
        for m in self.metric_names:
            self.log(f"train/{m}", self.train_metrics[m].compute())
            self.train_metrics[m].reset()

    def on_validation_epoch_end(self):
        for m in self.metric_names:
            self.log(f"validation/{m}", self.val_metrics[m].compute())
            self.val_metrics[m].reset()

    def training_step(self, batch, batch_idx):
        train_loss = self.common_step(batch, batch_idx, mode="train")
        self.log('train/train_loss', train_loss, on_step=True, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        valid_loss = self.common_step(batch, batch_idx, mode="valid")
        self.log('validation/valid_loss', valid_loss, on_step=True, on_epoch=True)
        return valid_loss

    def test_step(self, batch, batch_idx):
        test_loss = self.common_step(batch, batch_idx, mode="test")
        self.log('test/test_loss', test_loss, on_step=True, on_epoch=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--metrics', nargs='+', default=["acc"], help="List of metrics to track (e.g., acc, auc, precision, recall)")
        return parser