import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer

class VQADataset(Dataset):
    """Simple wrapper around a Hugging Face VQAv2 split that returns transformed
    images and raw questions.
    """
    def __init__(self, hf_split, image_transform):
        self.split = hf_split
        self.image_transform = image_transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        sample = self.split[idx]
        image = sample["image"]  # PIL Image
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        question = sample["question"]
        return {"image": image, "question": question}


class VQAMMDataModule(pl.LightningDataModule):
    """Downloads `lmms-lab/VQAv2` from HuggingFace, with support for streaming.
    """

    def __init__(
        self,
        hf_dataset_name: str = "lmms-lab/VQAv2",
        tokenizer_name: str = "google-bert/bert-base-uncased",
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        max_text_length: int = 32,
        streaming: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def _wrap_streaming_dataset(self, hf_iterable_dataset):
        """Wraps a HuggingFace iterable dataset in a PyTorch IterableDataset."""
        class HFStreamWrapper(IterableDataset):
            def __init__(self, hf_dataset, transform):
                self.hf_dataset = hf_dataset
                self.transform = transform

            def __iter__(self):
                for sample in self.hf_dataset:
                    image = sample["image"]
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    transformed_image = self.transform(image)
                    question = sample["question"]
                    yield {"image": transformed_image, "question": question}
        return HFStreamWrapper(hf_iterable_dataset, self.image_transform)

    def setup(self, stage=None):
        # --- FIX: Changed split names to match the dataset ('validation' for train, 'test' for val) ---
        train_split_name = 'validation'
        val_split_name = 'test'
        
        if self.hparams.streaming:
            print(f"Setting up data in streaming mode. Using '{train_split_name}' for training.")
            
            # For streaming, we'll use the 'validation' split and reserve a part of it for our own validation
            full_stream = load_dataset(self.hparams.hf_dataset_name, split=train_split_name, streaming=True)
            
            # Use the first 5000 samples for validation
            val_stream = full_stream.take(5000)
            self.val_dataset = self._wrap_streaming_dataset(val_stream)

            # Use the rest of the stream for training
            train_stream = full_stream.skip(5000)
            self.train_dataset = self._wrap_streaming_dataset(train_stream)

        else:
            print(f"Setting up data in download mode. Using '{train_split_name}' for training and '{val_split_name}' for validation.")
            
            dataset_dict = load_dataset(self.hparams.hf_dataset_name)
            
            # Use the 'validation' split as our training data
            self.train_dataset = VQADataset(dataset_dict[train_split_name], self.image_transform)
            
            # Use the 'test' split as our validation data
            try:
                self.val_dataset = VQADataset(dataset_dict[val_split_name], self.image_transform)
            except KeyError:
                print(f"Warning: '{val_split_name}' split not found. Creating a validation set from the training data.")
                lengths = len(self.train_dataset)
                val_length = int(0.05 * lengths)
                train_length = lengths - val_length
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    self.train_dataset, [train_length, val_length]
                )

    def _collate_fn(self, batch):
        images = torch.stack([b["image"] for b in batch])
        questions = [b["question"] for b in batch]
        tokenized = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.hparams.max_text_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": images,
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
        }

    def train_dataloader(self):
        is_streaming = self.hparams.streaming
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=not is_streaming,
            num_workers=0 if is_streaming else self.hparams.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        is_streaming = self.hparams.streaming
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0 if is_streaming else self.hparams.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )