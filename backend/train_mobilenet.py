"""
组员3 - MobileNetV3-Large 训练脚本
Stanford Cars 196 类微调
用法: python train_mobilenet.py --data_dir <数据集路径>
"""
import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import scipy.io as sio
from sklearn.metrics import classification_report
from copy import deepcopy

# ============================================================
# 配置
# ============================================================
parser = argparse.ArgumentParser(description="组员3 - MobileNetV3 车型识别训练")
parser.add_argument("--data_dir", default="data", help="Stanford Cars 数据集根目录")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--val_split", type=float, default=0.15)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--output_dir", default="weights")
parser.add_argument("--output_name", default="mobilenet_v3")
args, _ = parser.parse_known_args()

os.makedirs(args.output_dir, exist_ok=True)
NUM_CLASSES = 196


# ============================================================
# 数据集
# ============================================================
class StanfordCars(Dataset):
    """Stanford Cars 数据集"""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        img_dir = os.path.join(root, "cars_train")
        anno_file = os.path.join(root, "devkit", "cars_train_annos.mat")

        annos = sio.loadmat(anno_file)
        records = annos["annotations"][0]

        self.samples = []
        for rec in records:
            fname = str(rec["fname"][0])
            cls = int(rec["class"][0][0])
            self.samples.append((os.path.join(img_dir, fname), cls - 1))

        meta = sio.loadmat(os.path.join(root, "devkit", "cars_meta.mat"))
        self.class_names = [str(c[0]) for c in meta["class_names"][0]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ============================================================
# 模型: MobileNetV3-Large
# ============================================================
def build_model(num_classes=NUM_CLASSES, pretrained=True):
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        model.classifier[2],
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ============================================================
# 训练循环
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None

    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.4f}")

    return running_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return running_loss / len(loader), correct / total, all_preds, all_labels


def main():
    device = torch.device(args.device)
    print(f"[组员3 - MobileNetV3] Device: {device}")

    # 数据集
    full_ds = StanfordCars(args.data_dir, transform=None)
    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size

    generator = torch.Generator().manual_seed(456)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)
    train_ds.dataset.transform = get_transforms(args.img_size, True)

    val_dataset = deepcopy(full_ds)
    val_dataset.transform = get_transforms(args.img_size, False)
    val_subset = torch.utils.data.Subset(val_dataset, val_ds.indices)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    class_names = full_ds.class_names
    print(f"Train: {train_size} | Val: {val_size} | Classes: {len(class_names)}")

    # 模型 — MobileNetV3 用更大学习率，因为是轻量网络
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_acc = 0.0
    best_preds, best_labels = [], []
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n{'='*50}\nEpoch {epoch}/{args.epochs}\n{'='*50}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, preds, labels = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Time: {elapsed:.0f}s")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_preds, best_labels = preds, labels
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
                "history": history,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f"{args.output_name}.pt"))
            print(f"  -> Best model saved (acc={val_acc:.4f})")

    print(f"\n{'='*50}")
    print(f"Best accuracy: {best_acc:.4f}")

    if best_acc >= 0.80:
        report = classification_report(best_labels, best_preds, target_names=class_names, zero_division=0)
        report_path = os.path.join(args.output_dir, f"{args.output_name}_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Classification report saved to {report_path}")


if __name__ == "__main__":
    main()
