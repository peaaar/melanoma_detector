import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from utils import get_model, BEST_MODEL_PATH
from isic_score import score

THRESHOLD = 0.2  # initial custom threshold


def evaluate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = F.softmax(outputs, dim=1)[:, 1]  # malignant probability
            predicted = (probs > THRESHOLD).long()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels, all_probs


def eval_pr_auc(true_labels, predicted_probs):
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
    pr_auc = average_precision_score(true_labels, predicted_probs)

    print(f"\nPR AUC: {pr_auc:.4f}")
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def sweep_thresholds(true_labels, predicted_probs, start=0.05, stop=0.51, step=0.01):
    thresholds = [round(t, 2) for t in np.arange(start, stop, step)]
    print(f"\n{'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1-Score':>10}")
    print("-" * 45)

    best_f1 = 0.0
    best_threshold = 0.0

    for t in thresholds:
        preds = (np.array(predicted_probs) > t).astype(int)

        prec = precision_score(true_labels, preds, zero_division=0)
        rec = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)

        print(f"{t:10.2f}  {prec:10.4f}  {rec:10.4f}  {f1:10.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"\nBest threshold for F1-score: {best_threshold:.2f} (F1 = {best_f1:.4f})")


if __name__ == "__main__":
    checkpoint = torch.load(BEST_MODEL_PATH)
    model, device, criterion, _, _, val_loader, _ = get_model()
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, val_acc, preds, labels, probs = evaluate(model, val_loader, criterion, device)
    print(f"\nValidation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Benign", "Malignant"]))

    # cm = confusion_matrix(labels, preds)
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
    #             xticklabels=["Benign", "Malignant"],
    #             yticklabels=["Benign", "Malignant"])
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Confusion Matrix")
    # plt.tight_layout()
    # plt.show()

    # eval_pr_auc(labels, probs)
    # sweep_thresholds(labels, probs)

    # Construct DataFrames
    solution_df = pd.DataFrame({
        "image_id": list(range(len(labels))),
        "target": labels
    })
    submission_df = pd.DataFrame({
        "image_id": list(range(len(probs))),
        "target": probs
    })

    # Compute ISIC pAUC score
    isic_auc = score(solution_df, submission_df, row_id_column_name="image_id", min_tpr=0.80)
    print(f"ISIC Challenge pAUC (TPR ≥ 0.8): {isic_auc:.4f}")