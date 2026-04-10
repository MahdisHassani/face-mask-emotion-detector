from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from tqdm import tqdm

from dataset import MultiTaskDataset
from models.multitask_model import MultiTaskModel
from torchvision import transforms

# transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor()
])

# dataset
dataset = MultiTaskDataset(
    mask_dir="data/mask",
    emotion_img_dir="data/emotion/images",
    emotion_csv="data/emotion/legend.csv",
    transform=transform
)

# split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# loss
mask_loss_fn = nn.CrossEntropyLoss()
emotion_loss_fn = nn.CrossEntropyLoss()

def compute_loss(mask_pred, emotion_pred, labels, task_types):
    mask_idx = (task_types == 0)
    emotion_idx = (task_types == 1)

    loss = 0

    if mask_idx.sum() > 0:
        loss += mask_loss_fn(mask_pred[mask_idx], labels[mask_idx])

    if emotion_idx.sum() > 0:
        loss += emotion_loss_fn(emotion_pred[emotion_idx], labels[emotion_idx])

    return loss

# accuracy
def compute_accuracy(mask_pred, emotion_pred, labels, task_types):
    mask_idx = (task_types == 0)
    emotion_idx = (task_types == 1)

    mask_correct = 0
    emotion_correct = 0
    mask_total = 0
    emotion_total = 0

    if mask_idx.sum() > 0:
        preds = mask_pred[mask_idx].argmax(dim=1)
        mask_correct = (preds == labels[mask_idx]).sum().item()
        mask_total = mask_idx.sum().item()

    if emotion_idx.sum() > 0:
        preds = emotion_pred[emotion_idx].argmax(dim=1)
        emotion_correct = (preds == labels[emotion_idx]).sum().item()
        emotion_total = emotion_idx.sum().item()

    return mask_correct, mask_total, emotion_correct, emotion_total

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = MultiTaskModel().to(device)

for param in model.features[-3:].parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
best_val_loss = float("inf")

for epoch in range(epochs):

    # ===== TRAIN =====
    model.train()
    total_loss = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for images, labels, task_types in train_bar:
        images = images.to(device)
        labels = labels.to(device)
        task_types = task_types.to(device)

        optimizer.zero_grad()

        mask_out, emotion_out = model(images)

        loss = compute_loss(mask_out, emotion_out, labels, task_types)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    train_loss = total_loss / len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0

    mask_correct = 0
    mask_total = 0
    emotion_correct = 0
    emotion_total = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")

    with torch.no_grad():
        for images, labels, task_types in val_bar:
            images = images.to(device)
            labels = labels.to(device)
            task_types = task_types.to(device)

            mask_out, emotion_out = model(images)

            loss = compute_loss(mask_out, emotion_out, labels, task_types)
            val_loss += loss.item()

            mc, mt, ec, et = compute_accuracy(mask_out, emotion_out, labels, task_types)

            mask_correct += mc
            mask_total += mt
            emotion_correct += ec
            emotion_total += et

            val_bar.set_postfix(loss=loss.item())

    val_loss /= len(val_loader)
    
    mask_acc = mask_correct / mask_total if mask_total > 0 else 0
    emotion_acc = emotion_correct / emotion_total if emotion_total > 0 else 0

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Mask Acc: {mask_acc:.4f} | Emotion Acc: {emotion_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
    
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "mask_acc": mask_acc,
            "emotion_acc": emotion_acc
        }, "best_model.pth")
    
        print("Best model saved!")
    print("-" * 50)