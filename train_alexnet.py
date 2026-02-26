import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

# ==============================
# Reproducibility
# ==============================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# Configuration
# ==============================
DATA_DIR = "data"
BATCH_SIZE = 32
LR = 5e-4
NUM_CLASSES = 7
EPOCHS = 50
PATIENCE = 5
MODEL_NAME = "alexnet"

# ==============================
# Transforms
# ==============================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR,"train"), transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR,"val"), transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR,"test"), transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==============================
# Model
# ==============================
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ==============================
# Training
# ==============================
best_val_loss = float("inf")
counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out,y)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{MODEL_NAME}_best.pth")
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# ==============================
# Evaluation
# ==============================
model.load_state_dict(torch.load(f"{MODEL_NAME}_best.pth"))
model.eval()

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for x,y in test_loader:
        x = x.to(device)
        out = model(x)
        probs = torch.softmax(out,1)
        preds = torch.argmax(probs,1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())
        all_probs.extend(probs.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")

labels_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
macro_auc = roc_auc_score(labels_bin, all_probs, average="macro", multi_class="ovr")

print("Test Accuracy:", accuracy)
print("Macro-F1:", macro_f1)
print("Macro-AUC:", macro_auc)
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
