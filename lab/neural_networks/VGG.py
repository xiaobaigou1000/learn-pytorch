from torch import nn
import torch.nn.functional as F
import torch
import pytorch_lightning as L
import torchmetrics


class VGG(L.LightningModule):
    def __init__(self, out_features: int, in_channels: int = 3, lr: float = 0.1):
        super().__init__()

        conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]

        conv_blks = []

        for num_convs, out_channels in conv_arch:
            conv_blks.append(VGG.vgg_block(
                num_convs, in_channels, out_channels))
            in_channels = out_channels

        self.net = nn.Sequential(
            *conv_blks,
            nn.Flatten(),
            nn.Linear(out_channels*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, out_features)
        )

        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

        self.train_metrics = torchmetrics.MetricCollection({
            "loss": torchmetrics.MeanMetric(),
            "accuracy": torchmetrics.Accuracy(task='multiclass', num_classes=out_features)
        }, prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="validation_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, X):
        return self.net(X)

    def training_step(self, batch, batch_idx):
        loss, out, y = self.__common_step(batch, batch_idx)
        self.train_metrics['accuracy'](out, y)
        self.train_metrics['loss'](loss)
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, out, y = self.__common_step(batch, batch_idx)
        self.val_metrics['accuracy'](out, y)
        self.val_metrics['loss'](loss)
        self.log_dict(self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        loss, out, y = self.__common_step(batch, batch_idx)
        self.test_metrics['accuracy'](out, y)
        self.test_metrics['loss'](loss)
        self.log_dict(self.test_metrics)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

    def __common_step(self, batch, batch_idx):
        X, y = batch
        out = self(X)
        loss = self.loss(out, y)
        return loss, out, y

    def on_train_epoch_end(self):
        self.log("global_step", self.global_step)

    def vgg_block(num_conv, in_channels, out_channels):
        layers = []
        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
