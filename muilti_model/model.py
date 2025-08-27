import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
import math

class MultiModalJEPA(pl.LightningModule):
    """A dual-encoder model for learning joint image-text embeddings, inspired by
    CLIP and JEPA. It uses a pre-trained vision transformer (DINOv2) and a
    pre-trained text transformer (BERT) and aligns their representations using
    a contrastive (InfoNCE) loss and an auxiliary MSE loss.
    """

    def __init__(
        self,
        image_encoder_name: str = "facebook/dinov2-base",
        text_encoder_name: str = "google-bert/bert-base-uncased",
        projection_dim: int = 512,
        temperature: float = 0.07,
        mse_weight: float = 0.1, # Weight for the auxiliary MSE loss
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---------------- Image encoder ----------------
        self.image_encoder = AutoModel.from_pretrained(image_encoder_name)
        image_embedding_dim = self.image_encoder.config.hidden_size

        # ---------------- Text encoder ----------------
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        text_embedding_dim = self.text_encoder.config.hidden_size
        
        # --- Freeze the text encoder ---
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        
        # ---------------- Projection heads ----------------
        self.image_projection = nn.Linear(image_embedding_dim, projection_dim)
        self.text_projection = nn.Linear(text_embedding_dim, projection_dim)

        # Loss functions
        self.criterion_infonce = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()

    def encode_image(self, pixel_values):
        """Encodes images, extracts the CLS token, and projects to shared space."""
        outputs = self.image_encoder(pixel_values=pixel_values)
        # DINOv2 and many ViTs use the last hidden state of the CLS token
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        projected = self.image_projection(cls_embeddings)
        return projected

    def encode_text(self, input_ids, attention_mask):
        """Encodes text, extracts the CLS token, and projects to shared space."""
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        projected = self.text_projection(cls_embeddings)
        return projected

    def forward(self, pixel_values, input_ids, attention_mask):
        """Performs a forward pass through both encoders and normalizes the embeddings."""
        image_embeds_unnorm = self.encode_image(pixel_values)
        
        # Text encoder is frozen and in eval mode
        with torch.no_grad():
            text_embeds_unnorm = self.encode_text(input_ids, attention_mask)
        
        # L2 normalize embeddings for contrastive loss
        image_embeds = F.normalize(image_embeds_unnorm, p=2, dim=1)
        text_embeds = F.normalize(text_embeds_unnorm, p=2, dim=1)
        
        return image_embeds, text_embeds, image_embeds_unnorm, text_embeds_unnorm

    def _calculate_loss(self, image_embeds, text_embeds, image_embeds_unnorm, text_embeds_unnorm):
        """Calculates the total loss (InfoNCE + auxiliary MSE)."""
        batch_size = image_embeds.size(0)

        # --- InfoNCE Loss ---
        # Calculate cosine similarity matrix
        logits = (image_embeds @ text_embeds.T) / self.hparams.temperature
        
        # Create ground-truth labels for positive pairs
        labels = torch.arange(batch_size, device=self.device)
        
        # Symmetric loss
        loss_i = self.criterion_infonce(logits, labels)
        loss_t = self.criterion_infonce(logits.T, labels)
        loss_infonce = (loss_i + loss_t) / 2

        # --- Auxiliary MSE Loss ---
        # Note: We still need gradients for the image side of this loss.
        loss_mse = self.criterion_mse(image_embeds_unnorm, text_embeds_unnorm.detach())
        
        # --- Total Loss ---
        total_loss = loss_infonce + self.hparams.mse_weight * loss_mse
        
        return total_loss, loss_infonce, loss_mse

    def training_step(self, batch, batch_idx):
        image_embeds, text_embeds, image_embeds_unnorm, text_embeds_unnorm = self.forward(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        
        loss, loss_infonce, loss_mse = self._calculate_loss(
            image_embeds, text_embeds, image_embeds_unnorm, text_embeds_unnorm
        )
        
        self.log_dict({
            "train_loss": loss,
            "train_loss_infonce": loss_infonce,
            "train_loss_mse": loss_mse,
        }, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image_embeds, text_embeds, image_embeds_unnorm, text_embeds_unnorm = self.forward(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        
        loss, loss_infonce, loss_mse = self._calculate_loss(
            image_embeds, text_embeds, image_embeds_unnorm, text_embeds_unnorm
        )
        
        # Calculate cosine similarity for logging
        val_cos_sim = F.cosine_similarity(image_embeds, text_embeds).mean()
        
        self.log_dict({
            "val_loss": loss,
            "val_loss_infonce": loss_infonce,
            "val_loss_mse": loss_mse,
            "val_cosine_similarity": val_cos_sim
        }, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        # A scheduler can be added here if needed, e.g., CosineAnnealingLR
        return optimizer 