import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import json
from tqdm import tqdm

class BLIPCaptionGenerator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def generate_caption(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=30)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

    def generate_captions_for_folder(self, folder_path, output_json="captions.json", force=False):
        if os.path.exists(output_json) and not force:
            with open(output_json, "r") as f:
                return json.load(f)

        captions = {}
        image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png", ".jpeg"))]
        for filename in tqdm(image_files, desc="Generating BLIP Captions"):
            image_path = os.path.join(folder_path, filename)
            caption = self.generate_caption(image_path)
            captions[filename] = caption

        with open(output_json, "w") as f:
            json.dump(captions, f, indent=2)
        return captions
