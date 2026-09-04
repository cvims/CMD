import argparse
import itertools
import json
import os
import random

import numpy as np
import timm
import timm.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from utils.methods import get_method
from utils.eval import evaluate
from utils.helpers import append_results_to_file
from utils.models import get_model_essentials
# =====================================================
# Paths
# =====================================================
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    "path/to/save/results/"
)

DATA_DIR = "path/to/data/tinyimagenet"


def get_model_and_dataset(args):
    # Load model
    model_essentials = get_model_essentials(args.model_name)
    model = model_essentials["model"]
    test_transform = model_essentials["test_transforms"]

    try:
        weights_path = os.path.join(
            args.checkpoint,
            "best.pth",
        )
        weights = torch.load(weights_path, map_location="cpu")
    except:
        weights_path = os.path.join(
            args.checkpoint,
            "last.pt",
        )
        weights = torch.load(weights_path, map_location="cpu")

    weights = {
        key.replace("module.", ""): value
        for key, value in weights.items()
    }

    model.load_state_dict(weights)

    # Load data
    dataset = ImageFolder(
        os.path.join(DATA_DIR, "val"),
        transform=test_transform,
    )

    return model, dataset


def main(temperature, magnitude):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    model, dataset = get_model_and_dataset(args)
    model = model.to(device)
    model.eval()

    # Randomly permutate dataset
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    dataset = torch.utils.data.Subset(dataset, indices)

    # Split data
    dataset_name = args.model_name.split("_")[-1]

    num_classes = {
        "tinyimagenet": 200,
        "imagenet": 1000,
    }[dataset_name]

    n = len(dataset)
    num_train_samples = n // args.r

    train_dataset = torch.utils.data.Subset(
        dataset,
        range(0, num_train_samples),
    )

    test_dataset = torch.utils.data.Subset(
        dataset,
        range(num_train_samples, n),
    )

    val_dataset = torch.utils.data.Subset(
        test_dataset,
        range(
            0,
            len(test_dataset) // 5,
        ),
    )

    test_dataset = torch.utils.data.Subset(
        test_dataset,
        range(
            len(test_dataset) // 5,
            len(test_dataset),
        ),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=6,
        prefetch_factor=2,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=6,
        prefetch_factor=2,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=6,
        prefetch_factor=2,
    )

    # Get train probabilities
    method = get_method(
        args.method,
        temperature=temperature,
        model=model,
        lbd=args.lbd,
        num_classes=num_classes,
    )

    method.fit(train_dataloader, val_dataloader)

    test_preds = []
    test_targets = []
    test_scores = []

    for inputs, targets in tqdm(
        test_dataloader,
        total=len(test_dataloader),
    ):
        inputs = inputs.to(device)
        pred = None

        if magnitude > 0:
            inputs = Variable(
                inputs,
                requires_grad=True,
            )

            # Compute output
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1)

            # Compute perturbation
            scores = method(outputs)
            scores = torch.log(scores)

            scores.sum().backward()

            inputs = (
                inputs
                - magnitude * torch.sign(-inputs.grad)
            )

            inputs = Variable(
                inputs,
                requires_grad=False,
            )

        with torch.no_grad():
            logits = model(inputs)
            scores = method(logits)

        if pred is None:
            pred = torch.argmax(logits, dim=1)

        test_preds.append(pred.cpu())
        test_targets.append(targets.cpu())
        test_scores.append(scores.cpu())

    test_preds = torch.concat(test_preds)
    test_targets = torch.concat(test_targets)
    test_scores = torch.concat(test_scores)

    model_acc, roc_auc, fpr, aurc = evaluate(
        test_preds,
        test_targets,
        test_scores,
    )

    results = {
        "model_name": args.model_name,
        "temperature": temperature,
        "magnitude": magnitude,
        "method": args.method,
        "accuracy": model_acc,
        "fpr": fpr,
        "auc": roc_auc,
        "aurc": aurc,
        "r": args.r,
        "lbd": args.lbd,
        "seed": args.seed,
    }

    print(json.dumps(results, indent=2))

    filename = f"{args.model_name}_{args.seed}.csv"

    filepath = os.path.join(
        RESULTS_DIR,
        args.model_name,
        args.method,
        filename,
    )

    append_results_to_file(
        results,
        filepath,
    )

    print(f"Files saved to memory for config: {args}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18_tinyimagenet",
        help="model name",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="msp",
        help="method",
    )

    parser.add_argument(
        "--checkpoint",
        default="path/to/checkpoints",
        type=str
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "-r",
        "--r",
        type=int,
        default=10,
        help="ratio",
    )
    parser.add_argument(
        "--lbd",
        type=float,
        default=0.5,
        help="lambda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed",
    )

    args = parser.parse_args()

    TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.7, 2, 2.3,]
    MAGNITUDES = [
        0.0,
        0.0005,
        0.001,
        0.0015,
        0.002,
        0.0025,
    ]

    if args.method == "msp":
        main(
            temperature=1.0,
            magnitude=0.0,
        )
    else:
        for temperature, magnitude in itertools.product(
            TEMPERATURES,
            MAGNITUDES,
        ):
            main(temperature, magnitude)