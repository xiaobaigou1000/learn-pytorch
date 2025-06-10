import torch
import torchvision
from torch.utils import data
from torchvision.transforms import v2
import pytorch_lightning as L


class MnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 256, image_size: tuple[int, int] = (28, 28), num_workers: int = 16):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trans = v2.Compose([v2.ToImage(), v2.ToDtype(
            torch.float32, True), v2.Resize(image_size)])

    def prepare_data(self):
        torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=True, download=True)
        torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=True, transform=self.trans, download=False)

        train_dataset, validation_dataset = data.random_split(
            entire_dataset, [0.8, 0.2])
        self.train_ds = train_dataset
        self.val_ds = validation_dataset

        self.test_ds = torchvision.datasets.FashionMNIST(
            root=self.data_dir, transform=self.trans, train=False, download=True)

    def train_dataloader(self):
        return data.DataLoader(self.train_ds, self.batch_size, True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_ds, self.batch_size, False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return data.DataLoader(self.test_ds, self.batch_size, False, num_workers=self.num_workers, persistent_workers=True)


class ImagenetDatasetSize224(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 8):
        super().__init__()
        self.train_dataset_path = "./data/imagenet/training_images/"
        self.test_dataset_path = "./data/imagenet/validation_images/"
        self.image_transforms = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, True), v2.Resize((224, 224))])
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        return super().prepare_data()

    def setup(self, stage):
        all_train_data = torchvision.datasets.ImageFolder(
            self.train_dataset_path, self.image_transforms)
        train_dataset, validation_dataset = data.random_split(
            all_train_data, [0.8, 0.2])
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        test_dataset = torchvision.datasets.ImageFolder(
            self.test_dataset_path, self.image_transforms)
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, self.batch_size, True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(self.validation_dataset, self.batch_size, False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, self.batch_size, False, num_workers=self.num_workers, persistent_workers=True)
