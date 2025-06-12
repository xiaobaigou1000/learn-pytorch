from torch import nn
import torchmetrics
import torch
import pytorch_lightning as L


class ResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, use1x1conv: bool = False, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.downsample_layer = None
        self.downsample_bn = None
        if use1x1conv:
            self.downsample_layer = nn.Conv2d(
                in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, padding=0)
            self.downsample_bn = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU()

    def forward(self, X):
        identity = X.clone()
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu(X)
        X = self.conv3(X)
        X = self.bn3(X)

        if self.downsample_layer != None:
            identity = self.downsample_layer(identity)
            identity = self.downsample_bn(identity)

        X += identity
        X = self.relu(X)

        return X


class ResNet50(L.LightningModule):
    def __init__(self, out_features: int, input_channels: int = 3, lr=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResNet50.__make_layer(64, 64, 3),
            ResNet50.__make_layer(
                64*ResidualBlock.expansion, 128, 4, stride=2),
            ResNet50.__make_layer(
                128*ResidualBlock.expansion, 256, 6, stride=2),
            ResNet50.__make_layer(
                256*ResidualBlock.expansion, 512, 3, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512*ResidualBlock.expansion, out_features)
        )
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

        self.train_metrics = torchmetrics.MetricCollection({
            "loss": torchmetrics.MeanMetric(),
            "accuracy": torchmetrics.Accuracy(task='multiclass', num_classes=out_features)
        }, prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="validation_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        self.net.apply(ResNet50.__init_layers)

    def __init_layers(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight, 0.1)
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, 0.1)

    def __make_layer(in_channels, out_channels, num_residual_blocks, stride=1):
        layers = []

        if stride != 1 or in_channels != out_channels * ResidualBlock.expansion:
            layers.append(ResidualBlock(
                in_channels, out_channels, True, stride=stride))
        else:
            layers.append(ResidualBlock(in_channels, out_channels))

        in_channels = out_channels*ResidualBlock.expansion

        for i in range(num_residual_blocks-1):
            layers.append(ResidualBlock(in_channels, out_channels))

        return nn.Sequential(*layers)

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

    def on_train_epoch_end(self):
        self.log("global_step", self.global_step)

    def __common_step(self, batch, batch_idx):
        X, y = batch
        output = self(X)
        loss = self.loss(output, y)
        return loss, output, y

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
