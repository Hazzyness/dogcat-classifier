import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_PATH = "./model_output/resnet18_dogcat.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 64
ID2LABEL   = {0: "cat", 1: "dog"}

DEFAULT_SAMPLES = [
    {"image_path": "./dataset/test_set/dogs/9977.jpg", "true_label": 1},
    {"image_path": "./dataset/test_set/cats/9977.jpg", "true_label": 0},
    {"image_path": "./dataset/test_set/dogs/9978.jpg", "true_label": 1},
    {"image_path": "./dataset/test_set/cats/9978.jpg", "true_label": 0},
]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

def predict(image_path, model):
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return {"label": ID2LABEL[pred], "id": pred, "confidence": round(float(max(probs)), 4)}

def main():
    print("="*60)
    print("  Dog vs Cat Inference — Batch of 4")
    print("="*60)
    model = load_model()
    true_labels, pred_labels = [], []
    print(f"\n{'#':<3} {'Pred':<8} {'True':<8} {'Conf':>8}   File")
    print("-"*55)
    for i, s in enumerate(DEFAULT_SAMPLES):
        if not os.path.exists(s["image_path"]):
            print(f"{i+1:<3} ⚠️  File not found: {s['image_path']}")
            continue
        r = predict(s["image_path"], model)
        true_str = ID2LABEL[s["true_label"]]
        match    = "✅" if r["label"] == true_str else "❌"
        print(f"{i+1:<3} {r['label']:<8} {true_str:<8} {r['confidence']*100:>7.2f}%   {match} {os.path.basename(s['image_path'])}")
        true_labels.append(s["true_label"])
        pred_labels.append(r["id"])
    print("-"*55)
    print(f"\n📊 Accuracy : {accuracy_score(true_labels, pred_labels):.4f}")
    print(f"   F1 Score : {f1_score(true_labels, pred_labels, average='weighted', zero_division=0):.4f}")
    print(classification_report(true_labels, pred_labels, target_names=["Cat","Dog"], zero_division=0))

if __name__ == "__main__":
    main()