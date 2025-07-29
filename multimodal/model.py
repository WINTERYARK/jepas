from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from model.vision.vit import vit_tiny


class MultimodalJEPA(pl.LightningModule):
    """Simple teacher–student JEPA aligning images with text."""

    def __init__(
        self,
        text_model_name: str = "google-bert/bert-base-uncased",
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        for p in self.text_model.parameters():
            p.requires_grad = False
        embed_dim = self.text_model.config.hidden_size
        self.image_encoder = vit_tiny(img_size=224, patch_size=16)
        self.proj = nn.Sequential(
            nn.Linear(self.image_encoder.embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, images: torch.Tensor, texts: list[str]):
        text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_outputs = self.text_model(**text_inputs)
            text_emb = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        img_feat = self.image_encoder.forward_vit(images)
        img_feat = img_feat.mean(dim=1)
        pred = self.proj(img_feat)
        loss = self.criterion(pred, text_emb)
        return loss, pred, text_emb

    def training_step(self, batch, batch_idx):
        images, texts = batch
        loss, _, _ = self(images, texts)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        loss, _, _ = self(images, texts)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
