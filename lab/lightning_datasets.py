import torch
import torchvision
from torch.utils import data
from torchvision.transforms import v2
import pytorch_lightning as L
import pandas
import PIL.Image
from pathlib import Path


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


class CIFAR10(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 256, image_size: tuple[int, int] = (28, 28), image_augumentation: bool = True, num_workers: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if image_augumentation:
            self.transform = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, True),
                v2.Resize(image_size)
            ])
        else:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, True),
                v2.Resize(image_size)
            ])

        self.val_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, True),
            v2.Resize(image_size)
        ])

    def prepare_data(self):
        torchvision.datasets.CIFAR10(
            self.data_dir, True, self.transform, download=True)
        torchvision.datasets.CIFAR10(
            self.data_dir, False, self.transform, download=True)

    def setup(self, stage):
        self.train_dataset = torchvision.datasets.CIFAR10(
            self.data_dir, True, self.transform, download=False)
        self.validation_dataset = torchvision.datasets.CIFAR10(
            self.data_dir, False, self.val_transform, download=False)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, self.batch_size, True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(self.validation_dataset, self.batch_size, False, num_workers=self.num_workers, persistent_workers=True)


class BananaDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, transform: v2.Transform = None):
        super().__init__()
        self.labels = pandas.read_csv(Path(data_dir) / Path('label.csv'))
        self.image_root = Path(data_dir) / Path("images")
        self.transform = transform

    def __getitem__(self, index):
        label = self.labels.loc[index]
        image = PIL.Image.open(self.image_root / Path(label['img_name']))
        if self.transform != None:
            image = self.transform(image)
        label = torch.tensor([0, label['xmin'] / 256, label['ymin'] /
                             256, label['xmax'] / 256, label['ymax'] / 256]).reshape(1, 5)
        return image, label

    def __len__(self):
        return len(self.labels)


class BananaDetection(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data/banana-detection/", batch_size: int = 128, image_size: tuple[int, int] = (256, 256), num_workers: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, True),
            v2.Resize(self.image_size)
        ])

    def prepare_data(self):
        return super().prepare_data()

    def setup(self, stage):
        self.train_dataset = BananaDetectionDataset(
            Path(self.data_dir) / Path("bananas_train"), self.transform)
        self.valiation_dataset = BananaDetectionDataset(
            Path(self.data_dir) / Path("bananas_val"), self.transform)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, self.batch_size, True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(self.valiation_dataset, self.batch_size, False, num_workers=self.num_workers, persistent_workers=True)
