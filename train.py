import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from pretrained_model import CNN_Pretrained

# --------------------
# Config
# --------------------
DATA_DIR = "more_data"
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
MODEL_PATH = "bird_cnn_pretrained.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# --------------------
# Transforms
# --------------------
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------
# Datasets & Loaders
# --------------------
train_dataset = ImageFolder(
    root=os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = ImageFolder(
    root=os.path.join(DATA_DIR, "valid"),
    transform=val_transform
)

assert train_dataset.class_to_idx == val_dataset.class_to_idx, \
    "Class index mismatch between train and validation sets!"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)

# --------------------
# Model
# --------------------
model = CNN_Pretrained(num_classes).to(DEVICE)

# Freeze backbone initially
for param in model.backbone.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

scaler = torch.cuda.amp.GradScaler()

# --------------------
# Training Loop
# --------------------
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Unfreeze backbone after warmup
    if epoch == 5:
        print("Unfreezing backbone...")
        for param in model.backbone.features.parameters():
            param.requires_grad = True

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # --------------------
    # Validation
    # --------------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    scheduler.step()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )

    # --------------------
    # Save Best Model
    # --------------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {
                "model_state": model.state_dict(),
                "class_names": train_dataset.classes,
            },
            MODEL_PATH
        )
        print(f"Saved best model (val_acc={best_val_acc:.2f}%)")

print("Training complete!")
