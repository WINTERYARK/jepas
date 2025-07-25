import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from .datamodule import VQADatamodule
from .model import MultimodalJEPA


if __name__ == "__main__":
    datamodule = VQADatamodule()
    model = MultimodalJEPA()

    checkpoint = ModelCheckpoint(dirpath="checkpoints", save_top_k=1, monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger("logs", name="multimodal_jepa")

    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[checkpoint, lr_monitor],
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)
