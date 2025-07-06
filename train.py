import os
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Config
PREFIX = "/Users/jinghanli/personalSpace/research/isic 2024"
IMAGE_DIR = f"{PREFIX}/ISIC_2024_Training_Input"
BATCH_SIZE = 4
NUM_EPOCHS = 2
NUM_CLASSES = 2
MODEL_PATH = f"{PREFIX}/efficientnet_skin_cancer_checkpoint.pth"

# Load labels
labels_df = pd.read_csv(f"{PREFIX}/ISIC_2024_Training_GroundTruth.csv")


# Define dataset
class ISICDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_id = row['isic_id']
        label = int(row['malignant'])
        image_path = os.path.join(self.image_dir, image_id + ".jpg")

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset and loader
    dataset = ISICDataset(labels_df, IMAGE_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load model
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Resume from checkpoint if exists
    start_epoch = 0
    start_batch = 0
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint.get('batch', 0)
            print(f"Resumed from epoch {start_epoch}, batch {start_batch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for i, (images, labels) in enumerate(dataloader):
            if epoch == start_epoch and i < start_batch:
                continue

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if i % 10 == 0:
                print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}")

            # Save checkpoint after each batch
            checkpoint = {
                'epoch': epoch,
                'batch': i + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }
            torch.save(checkpoint, MODEL_PATH)

        epoch_loss = running_loss / len(dataset)
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Time: {elapsed_time:.2f}s")

        # Reset batch count after epoch
        start_batch = 0

    # Load model for inference
    inference_model = models.efficientnet_b0(pretrained=False)
    inference_model.classifier[1] = nn.Linear(inference_model.classifier[1].in_features, NUM_CLASSES)
    checkpoint = torch.load(MODEL_PATH)
    inference_model.load_state_dict(checkpoint['model_state_dict'])
    inference_model.eval()
    inference_model.to(device)

    # Sample inference
    sample_image_path = os.path.join(IMAGE_DIR, "ISIC_0015670.jpg")
    sample_image = Image.open(sample_image_path).convert("RGB")
    sample_tensor = transform(sample_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = inference_model(sample_tensor)
        predicted = torch.argmax(output, dim=1).item()
        print("Predicted:", "Malignant" if predicted == 1 else "Benign")


if __name__ == "__main__":
    main()
