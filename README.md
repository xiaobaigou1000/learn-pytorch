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
