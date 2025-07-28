import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve, 
    confusion_matrix, ConfusionMatrixDisplay, auc
)
import numpy as np

from utils.data_loader import MemeDataset, load_data_and_embeddings
from models.bert_roberta import CLIPTextFusion

def plot_roc_curve(y_true, y_probs, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()

def plot_pr_curve(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {pr_auc:.2f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    df, img_features = load_data_and_embeddings(args.csv, args.embeddings)
    dataset = MemeDataset(df, img_features, tokenizer_name=args.tokenizer, model_type=args.model)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = CLIPTextFusion(text_model_name=args.tokenizer, model_type=args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    y_true = []
    y_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            img_embed = batch["image_embedding"].to(device)
            labels = batch["label"].to(device)

            logits = model(img_embed, input_ids, attention_mask)
            probs = torch.sigmoid(logits).squeeze(1)

            y_probs.extend(probs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_probs = np.array(y_probs)
    y_true = np.array(y_true)
    y_pred = (y_probs >= 0.5).astype(int)

    print("ROC AUC:", roc_auc_score(y_true, y_probs))
    plot_roc_curve(y_true, y_probs)
    plot_pr_curve(y_true, y_probs)
    plot_confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to image embeddings (.pt)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--model", type=str, choices=["bert", "roberta"], default="bert")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    evaluate(args)
