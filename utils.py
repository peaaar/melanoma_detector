import os
import ssl

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

from ISICDataset import ISICDataset, PREFIX, IMAGE_ARCHIVE_DIR, ARCHIVE_PREFIX

# Config
BATCH_SIZE = 256
NUM_EPOCHS = 30
NUM_CLASSES = 2

MODEL_PATH = f"{PREFIX}/efficientnet_skin_cancer_checkpoint_weighted_random_erasing.pth"
LOSS_LOG_PATH = f"{PREFIX}/training_loss_log_weighted_random_erasing.csv"
BEST_MODEL_PATH = f"{PREFIX}/efficientnet_skin_cancer_best_model_weighted_random_erasing.pth"

def get_model():
    ssl._create_default_https_context = ssl._create_unverified_context

    # Load labels
    train_csv = f"{PREFIX}/train_labels.csv"
    val_csv = f"{PREFIX}/val_labels.csv"
    if os.path.exists(train_csv) and os.path.exists(val_csv):
        print("Reading existing train/val CSVs...")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
    else:
        print("Splitting dataset into train/val...")
        labels_df = pd.read_csv(f"{PREFIX}/ISIC_2024_Training_GroundTruth.csv")

        # 80% training, 20% validation
        train_df, val_df = train_test_split(
            labels_df,
            test_size=0.2,
            stratify=labels_df['malignant'],
            random_state=42
        )
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3))
    ])

    # Dataset and loader
    train_dataset = ISICDataset(train_df, transform=transform, preload=True)
    val_dataset = ISICDataset(val_df, transform=transform, preload=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Training setup: whole model fine tuning
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # Replace the classifier for 2-class output (binary classification)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    # Unfreeze all layers for full fine-tuning
    for param in model.features.parameters():
        param.requires_grad = True

    # Set optimizer for all trainable parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Device: use MPS on Mac or fallback to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # This makes the loss punish misclassified malignant examples far more than benign ones.
    class_counts = train_df['malignant'].value_counts().sort_index()
    weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float)  # inverse frequency
    weights = weights / weights.sum()  # normalize (optional but good)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    criterion.to(device)

    return model, device, criterion, optimizer, train_loader, val_loader, train_dataset
