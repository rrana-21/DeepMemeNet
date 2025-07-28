import torch
import torch.nn as nn
from transformers import AutoModel

class CLIPFusionClassifier(nn.Module):
    """
    Generic model to fuse CLIP image embeddings with transformer-based text embeddings (BERT, RoBERTa, XLM-R, etc.)
    using simple concatenation.
    """

    def __init__(self, text_encoder_name="bert-base-uncased", text_model_dim=768, img_embed_dim=512, dropout=0.2):
        super(CLIPFusionClassifier, self).__init__()

        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)

        self.fusion = nn.Sequential(
            nn.Linear(text_model_dim + img_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # Optional: freeze base transformer
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False

    def forward(self, img_embed, input_ids, attention_mask):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = text_out.last_hidden_state[:, 0, :]  # [CLS] token

        fused = torch.cat([img_embed, cls_embed], dim=1)
        logits = self.fusion(fused)
        return logits
