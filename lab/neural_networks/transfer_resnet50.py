import pytorch_lightning as L
import torchvision
from torch import nn
import torch
import torchmetrics


class TransferResnet50(L.LightningModule):
    def __init__(self, out_features: int, lr: float = 5e-2):
        super().__init__()
        self.net = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.net.fc = nn.Linear(2048, out_features)
        nn.init.xavier_uniform_(self.net.fc.weight)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.train_metrics = torchmetrics.MetricCollection({
            "loss": torchmetrics.MeanMetric(),
            "accuracy": torchmetrics.Accuracy(task='multiclass', num_classes=out_features)
        }, prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="validation_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.save_hyperparameters()

    def forward(self, X):
        return self.net(X)

    def configure_optimizers(self):
        high_lr_params = [i for i in self.net.named_parameters() if (
            i[0] == 'fc.weight' or i[0] == 'fc.bias')]
        low_lr_params = [i for i in self.net.named_parameters() if (
            i[0] != 'fc.weight' and i[0] != 'fc.bias')]
        optimizer = torch.optim.SGD([
            {"params": high_lr_params, "lr": self.lr},
            {"params": low_lr_params, "lr": (self.lr / 10)}
        ])
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, output, y = self.__common_step(batch, batch_idx)
        self.train_metrics['accuracy'](output, y)
        self.train_metrics['loss'](loss)
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output, y = self.__common_step(batch, batch_idx)
        self.val_metrics['accuracy'](output, y)
        self.val_metrics['loss'](loss)
        self.log_dict(self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        loss, output, y = self.__common_step(batch, batch_idx)
        self.test_metrics['accuracy'](output, y)
        self.test_metrics['loss'](loss)
        self.log_dict(self.test_metrics)
        return loss

    def __common_step(self, batch, batch_idx):
        X, y = batch
        output = self(X)
        loss = self.loss(output, y)
        return loss, output, y
