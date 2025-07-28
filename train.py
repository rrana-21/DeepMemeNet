import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from utils.data_loader import MemeDataset, load_data_and_embeddings
from utils.captioning import BLIPCaptionGenerator
from utils.augmentation import TextAugmentor
from models.losses import SupervisedContrastiveLoss

# Import all model variants
from models.bert_roberta import CLIPTextFusion
from models.clip_fusion import CLIPFusionClassifier
from models.xlm_r import PromptedXLMRModel

def get_model(model_name, args):
    if model_name == "bert_roberta":
        return CLIPTextFusion(text_model_name=args.tokenizer, model_type=args.base_model)
    elif model_name == "clip_fusion":
        return CLIPFusionClassifier(text_encoder_name=args.tokenizer)
    elif model_name == "xlm_r":
        return PromptedXLMRModel(return_features=args.use_contrastive)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df, img_features = load_data_and_embeddings(args.csv, args.embeddings)

    # Optionally augment text
    text_aug = TextAugmentor(aug_prob=args.aug_prob) if args.use_aug else None

    if args.model == "xlm_r":
        # Generate BLIP captions if needed
        if args.captions:
            import json
            with open(args.captions, "r") as f:
                caption_dict = json.load(f)
        else:
            raise ValueError("XLM-R model requires --captions path to BLIP caption JSON")

        texts = [text_aug(row["text"]) if text_aug else row["text"] for _, row in df.iterrows()]
        captions = [caption_dict[row["id"]] for _, row in df.iterrows()]
        labels = df["label"].tolist()

        # XLM-R prompt model
        dataset = list(zip(img_features.values(), texts, captions, labels))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    else:
        # BERT/RoBERTa/CLIP-fusion style
        dataset = MemeDataset(df, img_features, tokenizer_name=args.tokenizer, model_type=args.base_model)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Load model
    model = get_model(args.model, args)
    model = model.to(device)

    # Losses
    bce_loss_fn = nn.BCEWithLogitsLoss()
    scl_loss_fn = SupervisedContrastiveLoss() if args.use_contrastive else None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            if args.model == "xlm_r":
                img_embed, texts, captions, labels = zip(*batch)
                img_embed = torch.stack(img_embed).to(device)
                labels = torch.tensor(labels).float().unsqueeze(1).to(device)

                logits, features = model(img_embed, texts, captions) if args.use_contrastive else (model(img_embed, texts, captions), None)
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                img_embed = batch["image_embedding"].to(device)
                labels = batch["label"].unsqueeze(1).to(device)

                logits = model(img_embed, input_ids, attention_mask)
                features = None

            loss = bce_loss_fn(logits, labels)
            if args.use_contrastive and features is not None:
                loss += args.scl_weight * scl_loss_fn(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

        # Save model
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.model}_epoch{epoch+1}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--captions", type=str, help="BLIP captions JSON (for xlm_r)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    parser.add_argument("--model", type=str, choices=["bert_roberta", "clip_fusion", "xlm_r"], default="bert_roberta")
    parser.add_argument("--base_model", type=str, choices=["bert", "roberta"], default="bert")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--use_aug", action="store_true")
    parser.add_argument("--aug_prob", type=float, default=0.5)

    parser.add_argument("--use_contrastive", action="store_true")
    parser.add_argument("--scl_weight", type=float, default=0.2)

    args = parser.parse_args()
    train(args)
