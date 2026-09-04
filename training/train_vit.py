import argparse
import csv
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from tqdm import tqdm

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from utils.models import get_model_essentials

# =====================
# Paths
# =====================
CHECKPOINTS_DIR = os.environ.get(
    "CHECKPOINTS_DIR",
     "path/to/save/checkpoints/tinyimagenet/"
     )

# =====================
# Transforms
# =====================
def build_transforms():

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.25)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, test_transform

# =====================
# Train
# =====================
def train_one_epoch(model, loader, criterion, optimizer, device, mixup_fn=None):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        # Mixup / CutMix
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy (handle soft labels)
        _, preds = outputs.max(1)

        if targets.ndim == 2:
            targets_cls = targets.argmax(dim=1)
        else:
            targets_cls = targets

        total += targets_cls.size(0)
        correct += preds.eq(targets_cls).sum().item()

        acc = 100. * correct / total

        pbar.set_postfix({
            "loss": f"{total_loss / (total + 1e-6):.4f}",
            "acc": f"{acc:.2f}%"
        })

    return total_loss / len(loader), acc

# =====================
# Eval
# =====================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item()

        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

    acc = 100. * correct / total
    return total_loss / len(loader), acc

# =====================
# Main
# =====================
def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================
    # Data
    # =====================
    print("==> Preparing TinyImageNet data...")
    train_tf, test_tf = build_transforms()

    DATA_DIR = "path/to/data/tinyimagenet"

    train_set = ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_set   = ImageFolder(os.path.join(DATA_DIR, "val"), transform=test_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # =====================
    # Mixup / CutMix
    # =====================
    mixup_fn = Mixup(
        mixup_alpha=0.3,
        cutmix_alpha=0.8,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=200
    )

    # =====================
    # Model
    # =====================
    print("==> Building model...")
    model = get_model_essentials(args.model_name)["model"]
    model.to(device)

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # =====================
    # Loss + Optimizer
    # =====================
    criterion = SoftTargetCrossEntropy()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.decay
    )

    # =====================
    # Scheduler (Warmup + Cosine)
    # =====================
    warmup_epochs = 10

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=args.epoch - warmup_epochs
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )

    # =====================
    # Logging
    # =====================
    log_dir = os.path.join(CHECKPOINTS_DIR, args.model_name,str(args.seed),str(args.seed))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "results.csv")

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_acc = 0

    # =====================
    # Training Loop
    # =====================
    for epoch in range(args.epoch):
        print(f"\nEpoch {epoch}/{args.epoch}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            mixup_fn
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            nn.CrossEntropyLoss(),   # IMPORTANT: normal CE for eval
            device
        )

        scheduler.step()

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(log_dir, "best.pth"))

        # Log
        with open(log_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    torch.save(model.state_dict(), os.path.join(log_dir, "last.pth"))

# =====================
# Args
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="vit_tinyimagenet", type=str)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--decay", default=0.05, type=float)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()
    main(args) 