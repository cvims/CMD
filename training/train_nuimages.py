import argparse
import csv
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)
import timm

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR
)

from utils.nuimages import NuImagesClassification

# =====================================================
# Paths
# =====================================================
CHECKPOINTS_ROOT = os.environ.get(
    "CHECKPOINTS_DIR",
    "path/to/save/checkpoints/nuImages/"
)

def build_model(
    model_name,
    num_classes,
    pretrained=True
):
    if model_name == "resnet18":
        model = resnet18(
            weights=ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

    elif model_name == "swin_tiny":
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=num_classes
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model

# =====================================================
# Transforms
# =====================================================
def build_transforms():

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.08, 1.0)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(
            transforms.AutoAugmentPolicy.IMAGENET
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(
            p=0.25
        )
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

# =====================================================
# Train one epoch
# =====================================================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    mixup_fn=None
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        loader,
        desc="Train",
        leave=False
    )

    for images, targets in pbar:
        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        if mixup_fn is not None:
            images, targets = mixup_fn(
                images,
                targets
            )

        outputs = model(images)

        loss = criterion(
            outputs,
            targets
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = outputs.max(1)

        if targets.ndim == 2:
            targets_cls = targets.argmax(dim=1)
        else:
            targets_cls = targets

        total += targets_cls.size(0)

        correct += preds.eq(
            targets_cls
        ).sum().item()

        acc = (
            100.0 *
            correct /
            max(total, 1)
        )

        pbar.set_postfix({
            "loss": f"{total_loss / max(len(loader), 1):.4f}",
            "acc": f"{acc:.2f}%"
        })

    return total_loss / len(loader), acc

# =====================================================
# Evaluation
# =====================================================
@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            targets
        )

        total_loss += loss.item()

        _, preds = outputs.max(1)

        total += targets.size(0)

        correct += preds.eq(
            targets
        ).sum().item()

    acc = (
        100.0 *
        correct /
        max(total, 1)
    )

    return total_loss / len(loader), acc

# =====================================================
# Main training function
# =====================================================
def main(args):
    # -----------------------------
    # Reproducibility
    # -----------------------------
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # =================================================
    # Dataset
    # =================================================
    print("\n==> Preparing nuImages data...")
    train_tf, test_tf = build_transforms()

    train_set = NuImagesClassification(
        dataroot=args.data_root,
        version="v1.0-train",
        transform=train_tf
    )

    val_set = NuImagesClassification(
        dataroot=args.data_root,
        version="v1.0-val",
        transform=test_tf,
        class_to_idx=train_set.class_to_idx
    )

    num_classes = train_set.num_classes
    print(
        f"\nNumber of classes: {num_classes}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # =================================================
    # Mixup / CutMix
    # =================================================
    mixup_fn = Mixup(
        mixup_alpha=0.3,
        cutmix_alpha=0.8,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=num_classes
    )

    # =================================================
    # Model
    # =================================================
    print("\n==> Building Model...")
    weights = ResNet18_Weights.DEFAULT
    model = build_model(
        args.model_name,
        num_classes,
        pretrained=True
    )
    model = model.to(device)
    if device == "cuda":

        model = torch.nn.DataParallel(
            model
        )

        cudnn.benchmark = True

    # =================================================
    # Loss
    # =================================================
    criterion = SoftTargetCrossEntropy()

    # =================================================
    # Optimizer
    # =================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.decay
    )

    # =================================================
    # Scheduler
    # =================================================
    warmup_epochs = min(
        3,
        args.epochs
    )
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            warmup,
            cosine
        ],
        milestones=[
            warmup_epochs
        ]
    )

    # =================================================
    # Checkpoints and logging
    # =================================================
    log_dir = os.path.join(
        CHECKPOINTS_DIR,
        str(args.seed)
    )

    os.makedirs(
        log_dir,
        exist_ok=True
    )

    log_file = os.path.join(
        log_dir,
        "results.csv"
    )

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc"
            ])

    best_acc = 0.0

    # =================================================
    # Training loop
    # =================================================
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

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
            nn.CrossEntropyLoss(),
            device
        )

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )
        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # -----------------------------
        # Save best model
        # -----------------------------
        if val_acc > best_acc:
            best_acc = val_acc

            save_path = os.path.join(
                log_dir,
                "best.pth"
            )

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                f"Saved best model: {best_acc:.2f}%"
            )

        # -----------------------------
        # CSV logging
        # -----------------------------
        with open(log_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                train_acc,
                val_loss,
                val_acc
            ])

    # =================================================
    # Save final model
    # =================================================
    torch.save(
        model.state_dict(),
        os.path.join(log_dir, "last.pth")
    )

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
# =====================================================
# Arguments
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Model on nuImages classification task"
    )

    # -----------------------------
    # Dataset
    # -----------------------------
    parser.add_argument(
        "--data_root",
        default="path/to/data/nuImagenes",
        type=str,
        help="Path to nuImages root directory"
    )

    # -----------------------------
    # Training
    # -----------------------------
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Training batch size"
    )

    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--lr",
        default=5e-4,
        type=float,
        help="Learning rate"
    )

    parser.add_argument(
        "--decay",
        default=0.05,
        type=float,
        help="Weight decay"
    )

    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        help="Number of dataloader workers"
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed"
    )

    parser.add_argument(
        "--model_name",
        default="resnet18",
        choices=[
            "resnet18",
            "swin_tiny"
        ],
        type=str
    )

    args = parser.parse_args()

    CHECKPOINTS_DIR = os.path.join(
        CHECKPOINTS_ROOT,
        args.model_name
    )

    main(args)