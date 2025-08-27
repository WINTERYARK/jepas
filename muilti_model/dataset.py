import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
import pytorch_lightning as pl
from datasets import load_dataset, concatenate_datasets
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
        # For VQA, we might need answers later, but for pre-training, we only need image-question pairs.
        # However, to ensure no leakage, we should probably keep track of the question_id and image_id if available.
        # For now, let's assume the combination of (image, question) is unique.
        return {"image": image, "question": question}


class VQAMMDataModule(pl.LightningDataModule):
    """Loads and splits the `lmms-lab/VQAv2` dataset from HuggingFace for pre-training.
    """

    def __init__(
        self,
        hf_dataset_name: str = "lmms-lab/VQAv2",
        tokenizer_name: str = "google-bert/bert-base-uncased",
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        max_text_length: int = 32,
        seed: int = 42,
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
        self.rng = torch.Generator().manual_seed(self.hparams.seed)


    def setup(self, stage=None):
        print("Setting up data...")
        
        # Load all available splits and concatenate them
        dataset_dict = load_dataset(self.hparams.hf_dataset_name)
        
        # The VQAv2 dataset from lmms-lab has 'train', 'validation', and 'test' splits.
        # We will concatenate them to create one large dataset before splitting.
        all_splits = [split for split in dataset_dict.keys()]
        print(f"Found splits: {all_splits}. Concatenating them.")
        
        # It's better to handle cases where some splits might be missing.
        # Let's concatenate all available splits.
        if len(all_splits) > 1:
            full_dataset_hf = concatenate_datasets([dataset_dict[s] for s in all_splits])
        else:
            full_dataset_hf = dataset_dict[all_splits[0]]

        full_dataset = VQADataset(full_dataset_hf, self.image_transform)
        
        # Split once: 80% train / 10% validation / 10% test
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        print(f"Total dataset size: {total_size}")
        print(f"Splitting into train: {train_size}, val: {val_size}, test: {test_size}")

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size], generator=self.rng
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
            "questions": questions, # Pass raw questions for logging
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )