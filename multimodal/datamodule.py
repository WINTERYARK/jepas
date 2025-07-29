from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import VQADataset


class VQADatamodule(pl.LightningDataModule):
    """Lightning DataModule for the VQAv2 dataset."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        train_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.train_fraction = train_fraction
        self.train_dataset: Optional[VQADataset] = None
        self.val_dataset: Optional[VQADataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        image_transform = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor()]
        )
        full_train = VQADataset("train", image_transform=image_transform)
        num_train = int(len(full_train) * self.train_fraction)
        self.train_dataset, self.val_dataset = (
            torch.utils.data.random_split(
                full_train, [num_train, len(full_train) - num_train]
            )
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )
