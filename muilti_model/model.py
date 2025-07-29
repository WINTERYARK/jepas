import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from torchvision import models
import math


class ImageBERT(nn.Module):
    """Minimal Vision Transformer ("Image-BERT") that outputs a CLS embedding.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2

        # CLS token & positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)

        # Param init
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Patch embed
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Encoder
        x = self.encoder(x)

        # Norm & return CLS embedding
        x = self.norm(x)
        return x[:, 0]  # (B, embed_dim)


class MultiModalJEPA(pl.LightningModule):
    """Predict text (BERT) embeddings from images using a teacher-student setup.
    The teacher (text encoder) is frozen BERT. The student is a learnable image
    encoder + predictor projecting into the same embedding space. Loss is MSE
    between predicted and ground-truth CLS embeddings.
    """

    def __init__(
        self,
        image_encoder_name: str = "resnet50",
        text_encoder_name: str = "google-bert/bert-base-uncased",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---------------- Text encoder (teacher) ----------------
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        for p in self.text_encoder.parameters():
            p.requires_grad = False  # frozen teacher
        self.text_encoder.eval()

        self.text_embedding_dim = self.text_encoder.config.hidden_size  # typically 768

        # ---------------- Image encoder (student) ----------------
        if image_encoder_name.lower() == "image_bert":
            self.image_encoder = ImageBERT(embed_dim=self.text_embedding_dim)
            img_dim = self.text_embedding_dim
        else:
            backbone_constructor = getattr(models, image_encoder_name)
            # Avoid SSL issues with weight download by allowing user to disable pretrained weights.
            if pretrained_backbone:
                backbone = backbone_constructor(weights="DEFAULT")
            else:
                backbone = backbone_constructor(weights=None)

            if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
                img_dim = backbone.fc.in_features
                backbone.fc = nn.Identity()
            elif hasattr(backbone, "classifier") and isinstance(backbone.classifier, nn.Linear):
                img_dim = backbone.classifier.in_features
                backbone.classifier = nn.Identity()
            else:
                raise ValueError("Unsupported backbone; unable to locate final linear layer to strip.")

            self.image_encoder = backbone

        # ---------------- Predictor head ----------------
        self.predictor = nn.Linear(img_dim, self.text_embedding_dim)

        # Loss
        self.criterion = nn.MSELoss()

    # ---------------- Forward helpers ----------------
    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_embeddings

    def forward(self, pixel_values):
        features = self.image_encoder(pixel_values)
        if features.dim() == 4:
            features = features.flatten(1)  # (B, C, 1, 1) -> (B, C)
        pred = self.predictor(features)
        return pred

    # ---------------- Lightning hooks ----------------
    def training_step(self, batch, batch_idx):
        preds = self(batch["pixel_values"])
        with torch.no_grad():
            target = self.encode_text(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(preds, target)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch["pixel_values"])
        with torch.no_grad():
            target = self.encode_text(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(preds, target)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler] 