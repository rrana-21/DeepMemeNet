import torchvision.transforms as transforms
import nlpaug.augmenter.word as naw
import random

# Text Augmentation (nlp-aug)
class TextAugmentor:
    def __init__(self, aug_prob=0.5):
        self.aug = naw.SynonymAug(aug_src='wordnet')
        self.aug_prob = aug_prob

    def __call__(self, text):
        if random.random() < self.aug_prob:
            return self.aug.augment(text)
        return text

# Image Augmentation (torchvision)
def get_image_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
