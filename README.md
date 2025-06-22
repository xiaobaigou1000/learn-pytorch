# Learn Pytorch with Pytorch Lightning Framework

一个基于[《动手学深度学习》 - 李沐](https://zh.d2l.ai/)的神经网络学习记录。

使用`Pytorch Lightning`框架替代了原书中的`d2l`模块，可以支持更高版本的Python。

## Improve Points

* 支持Python 3.13.2
* 使用`mlflow`记录training log，可视化功能更加丰富
* 使用`uv`管理依赖，可以通过`uv sync`同步python版本和依赖

## Usage

1. 参考[uv官方文档](https://docs.astral.sh/uv/)，安装uv
2. 克隆项目
3. 使用以下命令安装依赖

```powershell
> uv sync
```

4. 使用以下命令运行示例

```powershell
> cd lab
> uv run 27_01_LT_AlexNet_CIFAR10.py
```

5. 在额外的terminal中运行以下命令开启mlflow server以监控训练参数

```powershell
> cd lab
> uv run python -m mlflow server
```

## Datasets

`MNIST`及`CIFAR10`数据集为官方提供数据集，但是`ImageNet`和`banana-detection`等dataset需要额外下载。

### ImageNet

**该数据集下载和配置都较为繁琐，不推荐使用**

ImageNet官方下载渠道需要非公开邮箱的验证，这里采用Hugging Face提供的坂本。[Hugging Face ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k)

下载后，不推荐使用Hugging Face提供的`imagenet1k.py`文件读取，即使数据集完好，使用该方式读取仍然有无法读取的图片。

推荐解压后，将图片按照分类存储成以下格式，使用Pytorch自带的`ImageFolder`进行读取。在`lab/lightning_datasets.py`中，是使用`ImageFolder`读取，并转换为Lightning Framework需要的LightningDataset示例。

```
.
.
└── lab
    └── data
        └── imagenet
            ├── training_images
            │   ├── n01440764
            │   ├── n01443537
            │   ├── ......
            │   └── n15075141
            └── validation_images
                ├── n01440764
                ├── n01443537
                ├── ......
                └── n15075141
```

### Banana Detection

d2l教程中使用的简单目标检测数据集，因为不使用`d2l`库，需要额外下载，并解压至`lab/data`目录下。

[Banana Detection数据集](https://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip)

解压后目录如下：

```
└── lab
    └── data
        └── banana-detection
            ├── bananas_train
            │   ├── images
            │   │   ├── 0.png
            │   │   ├── 100.png
            │   │   ├── ......
            │   │   └── 9.png
            │   └── label.csv
            └── bananas_val
                ├── images
                │   ├── 0.png
                │   ├── 10.png
                │   ├── ......
                │   └── 9.png
                └── label.csv
```
