import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Assuming dataset and model are in a package named 'muilti_model'
from muilti_model.dataset import VQAMMDataModule
from muilti_model.model import MultiModalJEPA


def main():
    # ------------------------------------------------------------------
    # Hyper-parameters (feel free to edit / expose as CLI arg if needed)
    # ------------------------------------------------------------------
    MAX_EPOCHS = 10
    BATCH_SIZE = 32
    LR = 1e-4
    MODEL_SIZE = "image_bert"
    LOG_DIR = "logs"
    CHECKPOINT_DIR = "checkpoints"
    NUM_WORKERS = 8
    # --- NEW: Control streaming from here ---
    USE_STREAMING = True 

    # ------------------------------------------------------------------
    # Instantiate datamodule & model
    # ------------------------------------------------------------------
    dm = VQAMMDataModule(
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        streaming=USE_STREAMING  # Pass the flag to the datamodule
    )
    model = MultiModalJEPA(image_encoder_name=MODEL_SIZE, lr=LR)

    # ------------------------------------------------------------------
    # Callbacks & trainer
    # ------------------------------------------------------------------
    callbacks = [
        ModelCheckpoint(dirpath=CHECKPOINT_DIR, filename=MODEL_SIZE, save_top_k=1, monitor="val_loss", mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = TensorBoardLogger(save_dir=LOG_DIR, name="MMJEPA", version=MODEL_SIZE)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()