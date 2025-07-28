import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveFusion(nn.Module):
    """
    A simple attention-based fusion module that learns to attend over image and text embeddings.
    Inputs:
      - image_embed: (B, D_img)
      - text_embed: (B, D_text)
    Output:
      - fused: (B, fusion_dim)
    """
    def __init__(self, img_dim=512, text_dim=768, fusion_dim=768):
        super(AttentiveFusion, self).__init__()

        self.img_proj = nn.Linear(img_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        self.attn_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 1),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
        )

    def forward(self, img_embed, text_embed):
        img_proj = self.img_proj(img_embed)       # (B, fusion_dim)
        text_proj = self.text_proj(text_embed)     # (B, fusion_dim)

        # Concatenate for attention scoring
        concat = torch.cat([img_proj, text_proj], dim=1)  # (B, 2*fusion_dim)
        attn_weight = torch.sigmoid(self.attn_layer(concat))  # (B, 1)

        # Weighted fusion
        fused = attn_weight * img_proj + (1 - attn_weight) * text_proj
        fused_out = self.output_layer(torch.cat([fused, text_proj], dim=1))  # residual + attention

        return fused_out  # (B, fusion_dim)
