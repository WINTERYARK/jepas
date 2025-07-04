from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from configs import get_image_dataset_config

from .hf_image_dataset import HFImageDataset


dataset_config = get_image_dataset_config()


class HFImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_id: str,
        subset_name: Optional[str],
        train_split: str,
        val_split: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: int,
        shuffle: bool,
    ) -> None:
        super().__init__()
        self.dataset_id = dataset_id
        self.subset_name = subset_name
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = HFImageDataset(
            dataset_id=self.dataset_id,
            subset_name=self.subset_name,
            split=self.train_split,
        )
        self.val_dataset = HFImageDataset(
            dataset_id=self.dataset_id,
            subset_name=self.subset_name,
            split=self.val_split,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=self.shuffle,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )


def create_hf_image_datamodule(image_config: Dict[str, Any]) -> HFImageDataModule:
    dataset_cfg = image_config["dataset"]
    exp_cfg = image_config["experiment"]

    return HFImageDataModule(
        dataset_id=dataset_cfg["HF_DATASET_ID"],
        subset_name=dataset_cfg.get("HF_DATASET_NAME"),
        train_split=dataset_cfg.get("HF_TRAIN_SPLIT", "train"),
        val_split=dataset_cfg.get("HF_VAL_SPLIT", "val"),
        batch_size=exp_cfg["BATCH_SIZE"],
        num_workers=exp_cfg["NUM_WORKERS"],
        pin_memory=exp_cfg["PIN_MEMORY"],
        persistent_workers=exp_cfg["PERSISTENT_WORKERS"],
        prefetch_factor=exp_cfg["PREFETCH_FACTOR"],
        shuffle=dataset_cfg["SHUFFLE_DATASET"],
    )
