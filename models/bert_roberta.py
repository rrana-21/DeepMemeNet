import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel

class CLIPTextFusion(nn.Module):
    def __init__(self, text_model_name="bert-base-uncased", img_embed_dim=512, fusion_dim=768, model_type="bert"):
        super(CLIPTextFusion, self).__init__()
        self.model_type = model_type.lower()

        if self.model_type == "bert":
            self.text_encoder = BertModel.from_pretrained(text_model_name)
        elif self.model_type == "roberta":
            self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        else:
            raise ValueError("model_type must be 'bert' or 'roberta'")

        self.freeze_text_encoder()
        self.fusion_layer = nn.Sequential(
            nn.Linear(img_embed_dim + fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
    
    def freeze_text_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, img_embed, input_ids, attention_mask):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        fusion = torch.cat((img_embed, cls_embed), dim=1)
        logits = self.fusion_layer(fusion)
        return logits
