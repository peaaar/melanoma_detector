import torch

from utils import get_model, MODEL_PATH
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels

if __name__ == "__main__":
    checkpoint = torch.load(MODEL_PATH)
    model, device, criterion, _, _, val_loader, _ = get_model()
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
    print(f"🧪 Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(labels, preds, target_names=["Benign", "Malignant"]))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

