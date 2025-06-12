from torch import nn
import torch.nn.functional as F
import torch
import pytorch_lightning as L
import torchmetrics


class NIN(L.LightningModule):
    def __init__(self, out_features: int, in_channels: int = 3, lr: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            NIN.NiN_block(in_channels, 96, kernel_size=11,
                          stride=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            NIN.NiN_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            NIN.NiN_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            NIN.NiN_block(384, out_features, kernel_size=3,
                          stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.net.apply(NIN.__init_vars)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()

        self.train_metrics = torchmetrics.MetricCollection({
            "loss": torchmetrics.MeanMetric(),
            "accuracy": torchmetrics.Accuracy(task='multiclass', num_classes=out_features)
        }, prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="validation_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def __init_vars(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight, 0.2)

    def forward(self, X):
        return self.net(X)

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

    def on_train_epoch_end(self):
        self.log("global_step", self.global_step)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def NiN_block(in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.ReLU()
        )
