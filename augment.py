import os

import pandas as pd
from PIL import Image
from torchvision import transforms

from utils import PREFIX, IMAGE_DIR

AUGMENTED_IMAGE_DIR = IMAGE_DIR + "/augmented/"

def augment():
    train_csv = f"{PREFIX}/train_labels.csv"
    train_df = pd.read_csv(train_csv)

    malignant_df = train_df[train_df['malignant'] == 1]
    os.makedirs(AUGMENTED_IMAGE_DIR, exist_ok=True)

    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    ])

    for _, row in malignant_df.iterrows():
        img_id = row['isic_id']
        img_path = os.path.join(IMAGE_DIR, img_id + ".jpg")
        image = Image.open(img_path).convert("RGB")

        for i in range(5):  # create 5 variants per image
            augmented = augment(image)
            new_filename = f"{img_id}_aug{i}.jpg"
            augmented.save(os.path.join(AUGMENTED_IMAGE_DIR, new_filename))

if __name__ == "__main__":
    augment()