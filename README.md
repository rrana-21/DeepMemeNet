# DeepMemeNet – Multimodal Deep Learning for Hateful Meme Detection

**DeepMemeNet** is a multimodal deep learning framework for classifying potentially hateful memes by combining image and text understanding. This project explores multiple architectures — from CLIP + BERT/RoBERTa to advanced prompted fusion with XLM-R and BLIP — to detect subtle, sarcastic, and context-dependent hateful content in memes.

---

## Features

- **Multimodal Fusion**: Integrates CLIP image embeddings with transformer-based text models (BERT, RoBERTa, XLM-R)
- **Prompt Engineering**: XLM-R model uses natural language prompts for better alignment with hateful content detection
- **AttentiveFusion**: Custom attention-based fusion module for image-text alignment
- **BLIP Captioning**: Enhances text input using generated image captions from BLIP
- **Supervised Contrastive Learning**: Encourages class-aware representation separation
- **Evaluation Utilities**: ROC, PR, calibration curves, and confusion matrix plotting

---

## Project Structure
hateful-meme-detector/
├── models/
│ ├── bert_roberta.py # CLIP + BERT/RoBERTa fusion
│ ├── xlm_r.py # Prompted XLM-R + BLIP + AttentiveFusion
│ ├── clip_fusion.py # General CLIP + transformer concat
│ ├── attention_module.py # Attentive fusion layer
│ └── losses.py # Supervised Contrastive Loss
│
├── utils/
│ ├── data_loader.py # Dataset and tokenizer
│ ├── captioning.py # BLIP caption generator
│ ├── eval_utils.py # ROC, PR, Confusion, Calibration
│ └── augmentation.py # nlpaug + torchvision transforms
│
├── train.py # Unified training script
├── evaluate.py # Inference + metrics
├── requirements.txt # Python dependencies
├── README.md # This file
└── data/
├── sample_input.json # Example meme input
└── blip_captions.json # (Expected) caption cache

---

## Sample Input

{
  "id": "meme_0417.png",
  "text": "Oh great, another genius who figured out how to plug in a charger.",
  "caption": "A person holding a phone with a charging cable, smiling sarcastically",
  "label": 1
}

---

## Installation 

# Create a fresh virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

