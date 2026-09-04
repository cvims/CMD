import argparse
import csv
import os
import random
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)
from utils.kitti import KITTIClassification

# ==========================================================
# Paths
# ==========================================================
CHECKPOINTS_ROOT = os.environ.get(
    "CHECKPOINTS_DIR",
    "path/to/save/checkpoints/KITTI"
)

# ==========================================================
# Model Factory
# ==========================================================
def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
):
    """
    Build a classification model.
    """

    if model_name == "resnet18":
        model = resnet18(
            weights=(
                ResNet18_Weights.DEFAULT
                if pretrained
                else None
            )
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

    elif model_name == "swin_tiny":
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# ==========================================================
# Data Transforms
# ==========================================================
def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.5, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(
            transforms.AutoAugmentPolicy.IMAGENET
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform

# ==========================================================
# Train One Epoch
# ==========================================================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    mixup_fn=None,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        if mixup_fn is not None:
            images, targets = mixup_fn(
                images,
                targets,
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            enabled=(device == "cuda")
        ):
            outputs = model(images)

            loss = criterion(
                outputs,
                targets,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        #
        # Accuracy
        # (only approximate under MixUp)
        #

        preds = outputs.argmax(dim=1)

        if targets.ndim == 2:
            labels = targets.argmax(dim=1)
        else:
            labels = targets

        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        avg_loss = running_loss / (batch_idx + 1)
        acc = 100.0 * correct / max(total, 1)

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{acc:.2f}%"
        )

    return (
        running_loss / len(loader),
        acc,
    )

# ==========================================================
# Validation
# ==========================================================
@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    class_names,
):
    model.eval()

    running_loss = 0.0
    total = 0
    correct = 0

    class_correct = {
        i: 0
        for i in range(len(class_names))
    }

    class_total = {
        i: 0
        for i in range(len(class_names))
    }

    for images, targets in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            targets,
        )

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)

        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

        for target, pred in zip(targets, preds):
            label = target.item()

            class_total[label] += 1

            if label == pred.item():
                class_correct[label] += 1

    overall_acc = (
        100.0 * correct /
        max(total, 1)
    )

    print("\nPer-class accuracy")
    print("-" * 50)

    mean_acc = 0.0
    valid_classes = 0

    for idx, name in enumerate(class_names):
        if class_total[idx] == 0:
            print(f"{name:20s} N/A")
            continue

        cls_acc = (
            100.0 *
            class_correct[idx] /
            class_total[idx]
        )

        mean_acc += cls_acc
        valid_classes += 1

        print(
            f"{name:20s}"
            f"{cls_acc:6.2f}% "
            f"({class_correct[idx]}/{class_total[idx]})"
        )

    print("-" * 50)

    if valid_classes > 0:
        mean_acc /= valid_classes

        print(
            f"Mean Class Accuracy : "
            f"{mean_acc:.2f}%"
        )

    return (
        running_loss / len(loader),
        overall_acc,
    )
# ==========================================================
# Main Training Function
# ==========================================================
def main(args):
    # ------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------
    print("\nPreparing KITTI dataset...")
    train_tf, val_tf = build_transforms()

    train_set = KITTIClassification(
        root=args.data_root,
        split="train",
        transform=train_tf,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    val_set = KITTIClassification(
        root=args.data_root,
        split="val",
        transform=val_tf,
        val_ratio=args.val_ratio,
        seed=args.seed,
        class_to_idx=train_set.class_to_idx,
    )

    num_classes = len(train_set.class_to_idx)
    class_names = [
        train_set.idx_to_class[i]
        for i in range(num_classes)
    ]

    print(f"\nNumber of classes : {num_classes}")
    print("\nClasses")

    for idx, name in enumerate(class_names):
        print(f"{idx:2d} : {name}")

    if hasattr(train_set, "print_class_distribution"):
        print("\nTraining distribution")
        train_set.print_class_distribution()

    # ------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    print(f"\nTraining samples   : {len(train_set)}")
    print(f"Validation samples : {len(val_set)}")

    # ------------------------------------------------------
    # MixUp
    # ------------------------------------------------------
    mixup_fn = Mixup(
        mixup_alpha=0.1,
        cutmix_alpha=0.0,
        prob=0.5,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=num_classes,
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------
    print(f"\nBuilding {args.model_name}...")
    model = build_model(
        args.model_name,
        num_classes=num_classes,
        pretrained=True,
    )
    model = model.to(device)
    if device == "cuda":
        model = nn.DataParallel(model)
        print(
            f"Using {torch.cuda.device_count()} GPU(s)"
        )

    # ------------------------------------------------------
    # Loss Functions
    # ------------------------------------------------------
    train_criterion = SoftTargetCrossEntropy()
    val_criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.decay,
    )

    # ------------------------------------------------------
    # LR Scheduler
    # ------------------------------------------------------
    warmup_epochs = min(
        3,
        max(args.epochs - 1, 1)
    )

    cosine_epochs = max(
        args.epochs - warmup_epochs,
        1,
    )

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs,
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            warmup,
            cosine,
        ],
        milestones=[
            warmup_epochs,
        ],
    )

    # ------------------------------------------------------
    # Automatic Mixed Precision
    # ------------------------------------------------------
    scaler = GradScaler(
        enabled=(device == "cuda")
    )

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------
    log_dir = os.path.join(
        CHECKPOINTS_ROOT,
        args.model_name,
        str(args.seed),
    )

    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    log_file = os.path.join(
        log_dir,
        "results.csv",
    )

    if not os.path.exists(log_file):

        with open(
            log_file,
            "w",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
            ])

    best_acc = 0.0

    # ------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------
    print("\nStarting training...")

    for epoch in range(args.epochs):
        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 70)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixup_fn=mixup_fn,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=val_criterion,
            device=device,
            class_names=class_names,
        )

        scheduler.step()

        print()
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.2f}%")

        # --------------------------------------------------
        # Save Best Model
        # --------------------------------------------------
        if val_acc > best_acc:

            best_acc = val_acc

            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )

            torch.save(
                state_dict,
                os.path.join(
                    log_dir,
                    "best.pth",
                ),
            )

            print(
                f"Best model saved "
                f"({best_acc:.2f}%)"
            )

        # --------------------------------------------------
        # CSV Logging
        # --------------------------------------------------
        with open(
            log_file,
            "a",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch + 1,
                current_lr,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            ])

    # ------------------------------------------------------
    # Save Final Model
    # ------------------------------------------------------
    final_state = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )

    torch.save(
        final_state,
        os.path.join(
            log_dir,
            "last.pth",
        ),
    )

    print("\nTraining completed.")
    print(
        f"Best Validation Accuracy: "
        f"{best_acc:.2f}%"
    )

# ==========================================================
# Command Line Interface
# ==========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train image classification models on KITTI object crops"
    )

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------
    parser.add_argument(
        "--data_root",
        type=str,
        default="path/to/data/KITTI",
        help="Path to the KITTI dataset root directory",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.20,
        help="Validation split ratio",
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "swin_tiny",
        ],
        help="Backbone architecture",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet pretrained weights",
    )

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Initial learning rate",
    )

    parser.add_argument(
        "--decay",
        type=float,
        default=0.05,
        help="Weight decay",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # ------------------------------------------------------
    # MixUp
    # ------------------------------------------------------
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.1,
        help="MixUp alpha (0 disables MixUp)",
    )

    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=0.5,
        help="Probability of applying MixUp",
    )

    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing",
    )

    # ------------------------------------------------------
    # Hardware
    # ------------------------------------------------------
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution",
    )

    # ------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------
    args = parser.parse_args()

    # ------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)

    for key, value in vars(args).items():
        print(f"{key:20s}: {value}")

    print("=" * 70)

    # ------------------------------------------------------
    # Create checkpoint directory
    # ------------------------------------------------------
    os.makedirs(
        CHECKPOINTS_ROOT,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Launch training
    # ------------------------------------------------------
    main(args)