import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

# =====================================
# Reproducibility
# =====================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================
# Configuration
# =====================================
DATA_DIR = "data"   # <-- change this to your dataset root
BATCH_SIZE = 16
LR = 5e-4
NUM_CLASSES = 7
EPOCHS = 50
PATIENCE = 5
MODEL_NAME = "swin_tiny"

# =====================================
# Transforms
# =====================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================================
# Model
# =====================================
model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=NUM_CLASSES
)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# =====================================
# Training
# =====================================
best_val_loss = float("inf")
counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{MODEL_NAME}_best.pth")
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# =====================================
# Evaluation
# =====================================
model.load_state_dict(torch.load(f"{MODEL_NAME}_best.pth"))
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")

labels_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
macro_auc = roc_auc_score(labels_bin, all_probs, average="macro", multi_class="ovr")

print("\n=== Test Performance ===")
print("Accuracy:", accuracy)
print("Macro-F1:", macro_f1)
print("Macro-AUC:", macro_auc)
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
