import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from ISICDataset import PREFIX
from evaluate import evaluate
from isic_score import score as pauc_score
from utils import get_model

def doit():
    # Paths to your best models
    model_paths = {
        "Weighted + ISIC Archive": "efficientnet_skin_cancer_best_model_weighted_isic_archive.pth",
        "Focal Loss": "efficientnet_focal_best_model.pth",
        "Weighted + Aug + Erasing": "efficientnet_skin_cancer_best_model_weighted_augmented_random_erasing.pth",
        "Weighted + Augmented": "efficientnet_skin_cancer_best_model_weighted_augmented.pth",
        "Weighted": "efficientnet_skin_cancer_best_model_weighted.pth",
        "Manual Weights": "efficientnet_skin_cancer_best_model.pth",
    }

    results = []

    for label, path in model_paths.items():
        print(f"Evaluating: {label}")

        # Load model and data
        model, device, criterion, _, _, val_loader, _ = get_model()
        checkpoint = torch.load(f"{PREFIX}/{path}", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        val_loss, val_acc, preds, labels, probs = evaluate(model, val_loader, criterion, device)

        # Extract class-wise metrics
        prec = precision_score(labels, preds, pos_label=1, zero_division=0)
        rec = recall_score(labels, preds, pos_label=1, zero_division=0)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)

        # Format pAUC: prepare submission and solution
        submission_df = pd.DataFrame({"isic_id": range(len(probs)), "malignant": probs})
        solution_df = pd.DataFrame({"isic_id": range(len(labels)), "malignant": labels})
        pauc = pauc_score(solution_df, submission_df, row_id_column_name="isic_id", min_tpr=0.80)

        # Save results
        results.append({
            "Model": label,
            "Accuracy": val_acc,
            "Malignant Precision": prec,
            "Malignant Recall": rec,
            "Malignant F1": f1,
            "pAUC": pauc
        })

    # Display
    df = pd.DataFrame(results)
    df = df.sort_values(by="pAUC", ascending=False)
    print("\nComparison Table:")
    print(df.to_string(index=False, float_format="%.4f"))

    df.to_csv("model_comparison.csv", index=False)

if __name__ == "__main__":
    doit()
