import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Assuming dataset and model are in a package named 'muilti_model'
from muilti_model.dataset import VQAMMDataModule
from muilti_model.model import MultiModalJEPA


def main():
    # ------------------------------------------------------------------
    # Hyper-parameters
    # ------------------------------------------------------------------
    MAX_EPOCHS = 50
    BATCH_SIZE = 128
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    PROJECTION_DIM = 512
    TEMPERATURE = 0.07
    MSE_WEIGHT = 0.1
    ACCUMULATE_GRAD_BATCHES = 2 # Increase effective batch size
    SEED = 42

    LOG_DIR = "logs"
    CHECKPOINT_DIR = "checkpoints"
    NUM_WORKERS = 8

    # ------------------------------------------------------------------
    # Instantiate datamodule & model
    # ------------------------------------------------------------------
    dm = VQAMMDataModule(
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        seed=SEED,
    )
    model = MultiModalJEPA(
        projection_dim=PROJECTION_DIM,
        temperature=TEMPERATURE,
        mse_weight=MSE_WEIGHT,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ------------------------------------------------------------------
    # Callbacks & trainer
    # ------------------------------------------------------------------
    # Early stop based on validation loss
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    callbacks = [
        ModelCheckpoint(dirpath=CHECKPOINT_DIR, filename="mm-jepa", save_top_k=1, monitor="val_loss", mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        early_stopping,
    ]

    logger = TensorBoardLogger(save_dir=LOG_DIR, name="MMJEPA", version="dual-encoder-pretrain")

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()