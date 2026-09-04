import os
import argparse
import itertools
import json
import random

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data

from tqdm import tqdm

from torchvision import transforms
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

import timm

from utils.kitti import KITTIClassification
from utils.methods import get_method
from utils.eval import evaluate
from utils.helpers import (
    append_results_to_file
)

# =====================================================
# Paths
# =====================================================
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    "path/to/save/results/"
)

# =====================================================
# Reproducibility
# =====================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================================================
# Model builder
# =====================================================
def build_model(
        model_name,
        num_classes,
        pretrained=False
):
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
            f"Unknown model {model_name}"
        )

    return model

# =====================================================
# Transforms
# =====================================================
def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, test_transform

# =====================================================
# Device helper
# =====================================================
def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

# =====================================================
# KITTI Dataset preparation
# =====================================================
def load_kitti_dataset(args):
    print("\n==> Loading KITTI dataset...")

    _, test_transform = build_transforms()
    # -----------------------------------------
    # Training split
    # -----------------------------------------
    train_set = KITTIClassification(
        root=args.data_root,
        split="train",
        transform=test_transform,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # -----------------------------------------
    # Validation split
    # Use identical class mapping
    # -----------------------------------------
    val_set = KITTIClassification(
        root=args.data_root,
        split="val",
        transform=test_transform,
        val_ratio=args.val_ratio,
        seed=args.seed,
        class_to_idx=train_set.class_to_idx
    )

    num_classes = len(train_set.class_to_idx)

    print("--------------------------------")
    print(f"Classes      : {num_classes}")
    print(train_set.class_to_idx)
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples  : {len(val_set)}")
    print("--------------------------------")

    return train_set, val_set, num_classes

# =====================================================
# DataLoader creation
# =====================================================
def create_loader(
        dataset,
        args,
        shuffle=False
):
    loader_args = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.workers,
        "pin_memory": True,
    }

    # Faster loading when workers exist
    if args.workers > 0:
        loader_args.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })

    return torch.utils.data.DataLoader(
        **loader_args
    )


def prepare_dataloaders(
        train_set,
        val_set,
        args
):
    train_loader = create_loader(
        train_set,
        args,
        shuffle=False
    )

    val_loader = create_loader(
        val_set,
        args,
        shuffle=False
    )

    print("\nEvaluation loaders")
    print("--------------------------------")
    print(f"Calibration : {len(train_set)}")
    print(f"Validation  : {len(val_set)}")
    print("--------------------------------")

    return train_loader, val_loader

# =====================================================
# Checkpoint loading
# =====================================================
def load_model(
        args,
        num_classes
):
    print("\n==> Loading model...")

    model = build_model(
        args.model_name,
        num_classes,
        pretrained=False
    )

    checkpoint_path = os.path.join(
        args.checkpoint,
        "best.pth"
    )

    print(f"Checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    # Handle DataParallel checkpoints
    state_dict = {
        key.replace("module.", ""): value
        for key, value in state_dict.items()
    }

    # Verify classifier size
    classifier_keys = [
        key
        for key in state_dict.keys()
        if (
            "fc.weight" in key
            or "classifier.weight" in key
        )
    ]

    if len(classifier_keys) > 0:
        classifier_key = classifier_keys[0]

        checkpoint_classes = (
            state_dict[classifier_key].shape[0]
        )

        print(f"Checkpoint classes: {checkpoint_classes}")
        print(f"Dataset classes: {num_classes}")

        if checkpoint_classes != num_classes:
            raise RuntimeError(
                "Checkpoint and KITTI "
                "class count mismatch"
            )

    model.load_state_dict(state_dict)

    device = get_device()
    model = model.to(device)
    model.eval()

    print("Model loaded successfully")

    return model

# =====================================================
# Uncertainty evaluation
# =====================================================
def run_uncertainty_eval(
        model,
        train_loader,
        val_loader,
        test_loader,
        args,
        temperature,
        magnitude
):
    device = next(model.parameters()).device

    # -------------------------------------------------
    # Create uncertainty method
    # -------------------------------------------------
    method = get_method(
        args.method,
        temperature=temperature,
        model=model,
        lbd=args.lbd,
        num_classes=args.num_classes
    )

    print(
        f"\nFitting uncertainty method: "
        f"{args.method}"
    )

    method.fit(
        train_loader,
        val_loader
    )

    preds_all = []
    targets_all = []
    scores_all = []

    print("\nRunning KITTI inference...")

    # -------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------
    for images, targets in tqdm(
        test_loader,
        total=len(test_loader)
    ):
        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        prediction = None

        # =================================================
        # ODIN style input perturbation
        # =================================================
        if magnitude > 0:
            images.requires_grad_(True)

            model.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=device.type == "cuda"
            ):
                logits = model(images)

                prediction = torch.argmax(
                    logits,
                    dim=1
                )

                scores = method(logits) 
                scores = torch.log(
                    scores.clamp(min=1e-12)
                )

            scores.sum().backward()

            images = (
                images
                - magnitude
                * torch.sign(-images.grad)
            )

            images = images.detach()

        # =================================================
        # Normal inference
        # =================================================
        with torch.no_grad():
            with torch.cuda.amp.autocast(
                enabled=device.type == "cuda"
            ):
                logits = model(images)
                scores = method(logits)

        if prediction is None:
            prediction = torch.argmax(
                logits,
                dim=1
            )

        preds_all.append(
            prediction.detach().cpu()
        )

        targets_all.append(
            targets.detach().cpu()
        )

        scores_all.append(
            scores.detach().cpu()
        )

    preds = torch.cat(
        preds_all,
        dim=0
    )

    targets = torch.cat(
        targets_all,
        dim=0
    )

    scores = torch.cat(
        scores_all,
        dim=0
    )

    return preds, targets, scores

# =====================================================
# Metric computation
# =====================================================
def compute_metrics(
        preds,
        targets,
        scores,
        args,
        temperature,
        magnitude
):
    print("\nComputing uncertainty metrics...")

    (
        accuracy,
        roc_auc,
        fpr,
        aurc,
    ) = evaluate(
        preds,
        targets,
        scores
    )

    results = {
        "model_name": args.model_name,
        "dataset": "kitti",
        "method": args.method,
        "temperature": temperature,
        "magnitude": magnitude,
        "accuracy": accuracy,
        "auc": roc_auc,
        "fpr": fpr,
        "aurc": aurc,
        "seed": args.seed
    }

    print(
        json.dumps(
            results,
            indent=2
        )
    )

    return results

# =====================================================
# Save results
# =====================================================
def save_results(
        results,
        preds,
        targets,
        scores,
        args,
        temperature,
        magnitude
):
    result_dir = os.path.join(
        RESULTS_DIR,
        args.model_name,
        args.method
    )

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    csv_file = os.path.join(
        result_dir,
        f"{args.model_name}_{args.seed}.csv"
    )

    append_results_to_file(
        results,
        csv_file
    )

    print("\nSaved:")
    print(csv_file)

# =====================================================
# Main evaluation
# =====================================================
def main(
        temperature,
        magnitude
):
    device = get_device()

    print(f"\nUsing device: {device}")

    seed_everything(args.seed)

    # -----------------------------------------
    # Dataset
    # -----------------------------------------
    train_set, val_set, num_classes = load_kitti_dataset(
        args
    )

    args.num_classes = num_classes

    train_loader, val_loader = prepare_dataloaders(
        train_set,
        val_set,
        args
    )

    # Test loader = KITTI validation split
    # because KITTI has no public labels
    test_loader = create_loader(
        val_set,
        args,
        shuffle=False
    )

    # -----------------------------------------
    # Model
    # -----------------------------------------
    model = load_model(
        args,
        num_classes
    )

    # -----------------------------------------
    # Uncertainty inference
    # -----------------------------------------
    preds, targets, scores = run_uncertainty_eval(
        model,
        train_loader,
        val_loader,
        test_loader,
        args,
        temperature,
        magnitude
    )

    # -----------------------------------------
    # Metrics
    # -----------------------------------------
    results = compute_metrics(
        preds,
        targets,
        scores,
        args,
        temperature,
        magnitude
    )

    # -----------------------------------------
    # Save
    # -----------------------------------------
    save_results(
        results,
        preds,
        targets,
        scores,
        args,
        temperature,
        magnitude
    )

    return results

# =====================================================
# Entry point
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KITTI uncertainty evaluation"
    )

    parser.add_argument(
        "--model_name",
        default="resnet18",
        choices=[
            "resnet18",
            "swin_tiny"
        ]
    )

    parser.add_argument(
        "--checkpoint",
        default="path/to/checkpoints",
        type=str
    )

    parser.add_argument(
        "--data_root",
        default="path/to/data/KITTI",
        type=str
    )

    parser.add_argument(
        "--method",
        default="msp",
        type=str
    )

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int
    )

    parser.add_argument(
        "--workers",
        default=8,
        type=int
    )

    parser.add_argument(
        "--val_ratio",
        default=0.2,
        type=float
    )

    parser.add_argument(
        "--lbd",
        default=0.5,
        type=float
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int
    )

    parser.add_argument(
        "--num_classes",
        default=8,
        type=int
    )

    args = parser.parse_args()

    # -----------------------------------------
    # Hyperparameter sweep
    # -----------------------------------------
    TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2, 2.3]
    MAGNITUDES = [
        0.0,
        0.0005,
        0.001,
        0.0015,
        0.002
    ]

    # Methods without perturbation
    if args.method in ["msp"]:
        main(
            temperature=1.0,
            magnitude=0.0
        )
    else:
        for temperature, magnitude in itertools.product(TEMPERATURES,MAGNITUDES):
            main(
                temperature,
                magnitude
            )
