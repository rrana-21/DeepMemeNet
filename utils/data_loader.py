import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer
import os

class MemeDataset(Dataset):
    def __init__(self, dataframe, img_features, tokenizer_name="bert-base-uncased", model_type="bert", max_len=77):
        self.data = dataframe
        self.img_features = img_features  # dictionary: id -> embedding
        self.model_type = model_type.lower()
        self.tokenizer = (BertTokenizer.from_pretrained(tokenizer_name) 
                          if self.model_type == "bert" 
                          else RobertaTokenizer.from_pretrained("roberta-base"))
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        meme_id = row["id"]
        text = row["text"]
        label = torch.tensor(row["label"], dtype=torch.float)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        img_embed = torch.tensor(self.img_features[meme_id], dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image_embedding": img_embed,
            "label": label,
        }

def load_data_and_embeddings(csv_path, embeddings_path):
    df = pd.read_csv(csv_path)  # expects 'id', 'text', 'label'
    img_features = torch.load(embeddings_path)  # expects a dict: id -> embedding vector
    return df, img_features
