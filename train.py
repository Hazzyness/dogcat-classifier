"""
train.py
Fine-tune ResNet18 for Dog vs Cat Image Classification
Dataset: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
Expected folder structure:
    dataset/
        training_set/
            cats/   *.jpg
            dogs/   *.jpg
        test_set/
            cats/   *.jpg
            dogs/   *.jpg
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader, Subset

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR  = "./dataset"           # root folder of the Kaggle dataset
TRAIN_DIR    = os.path.join(DATASET_DIR, "training_set")
TEST_DIR     = os.path.join(DATASET_DIR, "test_set")
OUTPUT_DIR   = "./model_output"
MODEL_PATH   = os.path.join(OUTPUT_DIR, "resnet18_dogcat.pth")

IMAGE_SIZE   = 64
BATCH_SIZE   = 64
EPOCHS       = 2
LEARNING_RATE = 1e-4
DEVICE       = "cpu"

CLASS_NAMES  = ["cats", "dogs"]   # alphabetical = torchvision default
ID2LABEL     = {0: "cat", 1: "dog"}
LABEL2ID     = {"cat": 0, "dog": 1}


# ─────────────────────────────────────────────
# DATA TRANSFORMS
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225]),   # ImageNet std
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
def build_model(num_classes: int = 2):
    """Load pretrained ResNet18 and replace final layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze all layers except the final FC
    for param in model.parameters():
        param.requires_grad = False
    # Replace final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc       = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}  {'─'*40}")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            all_preds    = []
            all_labels   = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = torch.argmax(outputs, dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = accuracy_score(all_labels, all_preds)
            epoch_f1   = f1_score(all_labels, all_preds, average="weighted")

            print(f"  [{phase.upper():5}] Loss: {epoch_loss:.4f} | "
                  f"Acc: {epoch_acc:.4f} | F1: {epoch_f1:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc       = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n✅ Best Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ResNet18 Fine-Tuning — Dog vs Cat Classification")
    print("=" * 60)
    print(f"  Device      : {DEVICE}")
    print(f"  Train dir   : {TRAIN_DIR}")
    print(f"  Test dir    : {TEST_DIR}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  LR          : {LEARNING_RATE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Datasets ────────────────────────────
    print("\n[1/4] Loading datasets ...")
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(TEST_DIR,  transform=val_transforms)
     
    train_dataset = Subset(train_dataset, range(2000))
    val_dataset   = Subset(val_dataset,   range(500))
    
    print(f"      Classes       : {train_dataset.dataset.classes}")
    print(f"      Train samples : {len(train_dataset)}")
    print(f"      Val   samples : {len(val_dataset)}")

    dataloaders   = {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
        "val":   DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    }
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    # ── Model ───────────────────────────────
    print("\n[2/4] Building model (ResNet18 pretrained) ...")
    model = build_model(num_classes=2).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # ── Train ───────────────────────────────
    print("\n[3/4] Training ...")
    start = time.time()
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, EPOCHS)
    elapsed = time.time() - start
    print(f"\n      Training time : {elapsed/60:.1f} min")

    # ── Save ────────────────────────────────
    print("\n[4/4] Saving model ...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names":      train_dataset.dataset.classes,
        "id2label":         ID2LABEL,
        "label2id":         LABEL2ID,
        "image_size":       IMAGE_SIZE,
    }, MODEL_PATH)
    print(f"      Model saved to: {MODEL_PATH}")

    # ── Final metrics on test set ────────────
    print("\n--- Final Evaluation on Test Set ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            outputs = model(inputs.to(DEVICE))
            preds   = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Score : {f1:.4f}")
    print()
    print(classification_report(all_labels, all_preds))
    print("\n✅ Done!")


if __name__ == "__main__":
    main()