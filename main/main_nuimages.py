import os
import argparse
import itertools
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data

from torch.autograd import Variable
from tqdm import tqdm

from torchvision import transforms
from torchvision.models import resnet18
import timm

from utils.nuimages import NuImagesClassification
from utils.methods import get_method
from utils.eval import evaluate
from utils.helpers import append_results_to_file


RESULT_DIR = "path/to/save/results"

# =====================================================
# Models
# =====================================================
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
# Load model
# =====================================================
def get_model(args):
    print("\n==> Loading nuImages model...")

    model = build_model(
        args.model_name,
        args.num_classes,
        pretrained=False
    )

    checkpoint = torch.load(
        os.path.join(
            args.checkpoint,
            "best.pth"
        ),
        map_location="cpu"
    )

    checkpoint = {
        k.replace("module.", ""): v
        for k, v in checkpoint.items()
    }

    model.load_state_dict(checkpoint)

    print("Checkpoint loaded")

    return model

# =====================================================
# Dataset
# Prepare dataloaders
# =====================================================
def prepare_dataloaders(args, train_tf, test_tf):

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

    test_set = NuImagesClassification(
        dataroot=args.data_root,
        version="v1.0-test",
        transform=test_tf,
        class_to_idx=train_set.class_to_idx
    )

    print("--------------------------------")
    print(f"Train samples : {len(train_set)}")
    print(f"Val samples   : {len(val_set)}")
    print(f"Test samples  : {len(test_set)}")
    print("--------------------------------")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )

# =====================================================
# Uncertainty evaluation
# =====================================================
def evaluate_uncertainty(
        model,
        train_loader,
        val_loader,
        test_loader,
        args,
        temperature,
        magnitude
):
    device = next(model.parameters()).device

    method = get_method(
        args.method,
        temperature=temperature,
        model=model,
        lbd=args.lbd,
        num_classes=args.num_classes
    )

    print(
        f"\nFitting uncertainty method: {args.method}"
    )

    method.fit(
        train_loader,
        val_loader
    )

    preds = []
    targets_all = []
    scores_all = []

    print("\nRunning inference...")

    for images, targets in tqdm(
        test_loader,
        total=len(test_loader)
    ):
        images = images.to(device)
        targets = targets.to(device)

        pred = None

        # ---------------------------------
        # Gradient based perturbation
        # ---------------------------------
        if magnitude > 0:
            images = Variable(
                images,
                requires_grad=True
            )

            logits = model(images)

            pred = torch.argmax(
                logits,
                dim=1
            )

            scores = method(logits)

            scores = torch.log(scores)

            scores.sum().backward()

            images = (
                images -
                magnitude *
                torch.sign(
                    -images.grad
                )
            )

            images = Variable(
                images,
                requires_grad=False
            )

        # ---------------------------------
        # Prediction
        # ---------------------------------
        with torch.no_grad():
            logits = model(images)
            scores = method(logits)

        if pred is None:
            pred = torch.argmax(
                logits,
                dim=1
            )

        preds.append(
            pred.cpu()
        )

        targets_all.append(
            targets.cpu()
        )

        scores_all.append(
            scores.cpu()
        )

        preds = torch.cat(preds)
        targets_all = torch.cat(targets_all)
        scores_all = torch.cat(scores_all)

        return (
            preds,
            targets_all,
            scores_all
        )

# =====================================================
# Main
# =====================================================
def main(
    temperature,
    magnitude
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True

    # =================================================
    # Model
    # =================================================
    model = get_model(args)
    model = model.to(device)
    model.eval()

    # =================================================
    # Dataset
    # =================================================
    train_tf, test_tf = build_transforms()
    train_loader, val_loader, test_loader = prepare_dataloaders(
        args, train_tf, test_tf
    )

    # =================================================
    # Uncertainty inference
    # =================================================

    (
        preds,
        targets,
        scores
    ) = evaluate_uncertainty(
        model,
        train_loader,
        val_loader,
        val_loader,
        args,
        temperature,
        magnitude
    )

    # =================================================
    # Metrics
    # =================================================
    print("\nComputing metrics...")

    (accuracy, roc_auc, fpr, aurc) = evaluate(preds, targets, scores)
    results = {
        "model_name": args.model_name,
        "dataset": "nuimages",
        "version": args.version,
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

    # =================================================
    # Save csv
    # =================================================
    save_dir = os.path.join(
        RESULT_DIR,
        args.model_name,
        args.method
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    csv_file = os.path.join(
        save_dir,
        f"{args.model_name}_{args.seed}.csv"
    )

    append_results_to_file(
        results,
        csv_file
    )
    
    print("\nEvaluation completed.")

# =====================================================
# Arguments
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate nuImages ResNet18 uncertainty"
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

    parser.add_argument(
        "--checkpoint",
        default="path/to/checkpoints",
        type=str
    )

    parser.add_argument(
        "--data_root",
        default="path/to/data/nuImages",
        type=str
    )

    parser.add_argument(
        "--version",
        default="v1.0-val",
        type=str
    )

    parser.add_argument(
        "--method",
        default="msp",
        type=str
    )

    parser.add_argument(
        "--batch_size",
        default=256,
        type=int
    )

    parser.add_argument(
        "--workers",
        default=8,
        type=int
    )

    parser.add_argument(
        "--r",
        default=10,
        type=int
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
        default=24,
        type=int
    )

    args = parser.parse_args()

    TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2, 2.3]
    MAGNITUDES = [
        0.0,
        0.0005,
        0.001,
        0.0015,
        0.002,
    ]

    if args.method in ["msp"]:
        main(
            temperature=1.0,
            magnitude=0.0
        )
    else:
        for temperature, magnitude in itertools.product(
            TEMPERATURES,
            MAGNITUDES
        ):
            main(
                temperature,
                magnitude
            )
