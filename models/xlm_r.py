import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from models.attention_module import AttentiveFusion

class PromptedXLMRModel(nn.Module):
    def __init__(self, prompt_template=None, fusion_dim=768, dropout=0.2, return_features=False):
        super(PromptedXLMRModel, self).__init__()
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        self.text_encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.return_features = return_features

        self.fusion = AttentiveFusion(img_dim=512, text_dim=768, fusion_dim=fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.prompt_template = (
            prompt_template or "Analyze the following meme. Text: '{text}'. Image description: '{caption}'. Is this meme hateful or not hateful? Answer:"
        )

        # Optional: Freeze backbone
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False

    def tokenize(self, text, caption):
        prompt = self.prompt_template.format(text=text, caption=caption)
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

    def forward(self, img_embed, texts, captions):
        """
        Inputs:
            - img_embed: (B, 512) CLIP image embeddings
            - texts: list of raw text strings
            - captions: list of BLIP image captions (strings)
        """
        batch = [self.tokenize(t, c) for t, c in zip(texts, captions)]
        input_ids = torch.cat([x["input_ids"] for x in batch], dim=0).to(img_embed.device)
        attn_mask = torch.cat([x["attention_mask"] for x in batch], dim=0).to(img_embed.device)

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        cls_embed = text_out.last_hidden_state[:, 0, :]  # [CLS]

        fused = self.fusion(img_embed, cls_embed)
        logits = self.classifier(fused)

        if self.return_features:
            return logits, fused  # for contrastive loss
        return logits
