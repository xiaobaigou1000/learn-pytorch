from torch import nn
import torch.nn.functional as F
import torch
import pytorch_lightning as L
import torchmetrics


class Inception(nn.Module):
    def __init__(self, in_channels, path1_channels, path2_channels, path3_channels, path4_channels):
        super().__init__()
        self.path1_block1 = nn.Conv2d(
            in_channels, path1_channels, kernel_size=(1, 1))
        self.path2_block1 = nn.Conv2d(
            in_channels, path2_channels[0], kernel_size=(1, 1))
        self.path2_block2 = nn.Conv2d(
            path2_channels[0], path2_channels[1], kernel_size=(3, 3), padding=1)
        self.path3_block1 = nn.Conv2d(
            in_channels, path3_channels[0], kernel_size=1)
        self.path3_block2 = nn.Conv2d(
            path3_channels[0], path3_channels[1], kernel_size=(5, 5), padding=2)
        self.path4_block1 = nn.MaxPool2d(
            kernel_size=(3, 3), stride=1, padding=1)
        self.path4_block2 = nn.Conv2d(
            in_channels, path4_channels, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.path1_block1(x))
        p2 = F.relu(self.path2_block2(F.relu(self.path2_block1(x))))
        p3 = F.relu(self.path3_block2(F.relu(self.path3_block1(x))))
        p4 = F.relu(self.path4_block2(F.relu(self.path4_block1(x))))
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(L.LightningModule):
    def __init__(self, out_features: int, in_channels: int = 3, lr=0.1):
        super().__init__()
        stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        stage2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        stage3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        stage4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        stage5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.net = nn.Sequential(
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            nn.Linear(1024, out_features)
        )

        self.net.apply(GoogLeNet.__init_vars)
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
