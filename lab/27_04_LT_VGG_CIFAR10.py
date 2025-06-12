from lightning_datasets import CIFAR10
from neural_networks import VGG
import pytorch_lightning as L
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    model = VGG(10)
    dataset = CIFAR10(batch_size=128, image_size=(224, 224))
    logger = MLFlowLogger(run_name="VGG-CIFAR10")
    callbacks = [
        ModelCheckpoint("./saved_checkpoints/vgg-cifar10/",
                        save_top_k=5, monitor="validation_loss", mode='min', every_n_epochs=1)
    ]
    trainer = L.Trainer('gpu', logger=logger,
                        callbacks=callbacks, max_epochs=20)
    trainer.fit(model, dataset)


if __name__ == "__main__":
    main()
