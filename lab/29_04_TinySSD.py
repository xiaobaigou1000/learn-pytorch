from lightning_datasets import BananaDetection
from neural_networks import TinySSD
import pytorch_lightning as L
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    model = TinySSD(1)
    dataset = BananaDetection(batch_size=32)
    logger = MLFlowLogger(
        experiment_name="object_detection", run_name="TinySSD-BananaDetection")

    callbacks = [
        ModelCheckpoint("./saved_checkpoints/tinyssd-bananadetection/",
                        save_top_k=5, monitor="validation_final_loss", mode='min', every_n_epochs=1)
    ]
    trainer = L.Trainer('gpu', logger=logger,
                        callbacks=callbacks, max_epochs=20, log_every_n_steps=1)

    trainer.fit(model, dataset)


if __name__ == "__main__":
    main()
