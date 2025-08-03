import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import os

# Load model from checkpoint
def load_model(checkpoint_path, num_classes=2, device='cpu'):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# Define default image preprocessing
def get_default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Single image prediction
def predict(model, image_path, transform=None, device='cpu'):
    if transform is None:
        transform = get_default_transform()

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.softmax(output, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()
        confidence = prob[0][pred_class].item()

    return pred_class, confidence

# Main function to run prediction
def main():
    parser = argparse.ArgumentParser(description="Predict skin lesion (malignant or benign)")
    parser.add_argument("--image", required=True,
                        help="Image filename only (e.g., ISIC_0051822.jpg). Will be looked up in IMAGE_DIR.")
    parser.add_argument("--checkpoint", default=MODEL_PATH,
                        help=f"Path to model checkpoint (.pth), default: {MODEL_PATH}")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cpu or cuda)")
    args = parser.parse_args()

    image_path = os.path.join(IMAGE_DIR, args.image)
    model = load_model(args.checkpoint, num_classes=2, device=args.device)
    transform = get_default_transform()

    pred_class, confidence = predict(model, image_path, transform=transform, device=args.device)

    label_map = {0: "Benign", 1: "Malignant"}
    print(f"Image: {args.image}")
    print(f"Prediction: {label_map[pred_class]} (class {pred_class}), Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
