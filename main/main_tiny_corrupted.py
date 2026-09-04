import argparse
import itertools
import json
import os
import random

import numpy as np
import timm
import timm.data
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from utils.methods import get_method
from utils.eval import evaluate
from utils.helpers import append_results_to_file
from utils.models import get_model_essentials


TINY_IMAGENET_ROOT = "path/to/datasets/tinyimagenet"
TINY_IMAGENET_C_ROOT = "path/to/datasets/tinyimagenet-C"


class TinyImageNetC(Dataset):
    def __init__(
        self,
        root,
        wnid_to_idx,
        severity=1,
        corruption=None,
        transform=None,
        subgroup_size=2500,
    ):
        self.transform = transform
        self.samples = []

        corruptions = (
            [corruption]
            if corruption is not None
            else sorted(os.listdir(root))
        )

        for corruption_name in corruptions:
            severity_dir = os.path.join(
                root,
                corruption_name,
                str(severity),
            )

            if not os.path.isdir(severity_dir):
                continue

            group_samples = []

            for wnid in sorted(os.listdir(severity_dir)):
                class_dir = os.path.join(severity_dir, wnid)

                if not os.path.isdir(class_dir):
                    continue

                if wnid not in wnid_to_idx:
                    continue

                label = wnid_to_idx[wnid]

                for filename in sorted(os.listdir(class_dir)):
                    if filename.lower().endswith(
                        (".jpeg", ".jpg", ".png")
                    ):
                        path = os.path.join(class_dir, filename)
                        group_samples.append((path, label))

            if subgroup_size is not None:
                group_samples = group_samples[:subgroup_size]

            self.samples.extend(group_samples)

        if not self.samples:
            raise RuntimeError(
                "TinyImageNet-C dataset is EMPTY. "
                "Check paths / severity / wnid mapping."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image: {path}"
            ) from exc

        if self.transform:
            image = self.transform(image)

        return image, label


def load_wnids(root):
    wnids_file = os.path.join(root, "wnids.txt")

    with open(wnids_file, "r") as file:
        wnids = [line.strip() for line in file]

    return {
        wnid: index
        for index, wnid in enumerate(wnids)
    }


def get_model_and_dataset(args):
    model_essentials = get_model_essentials(args.model_name)
    model = model_essentials["model"]
    test_transform = model_essentials["test_transforms"]

    try:
        weights_path = os.path.join(
            args.checkpoints,
            "best.pth",
        )
        weights = torch.load(
            weights_path,
            map_location="cpu",
        )
    except:
        weights_path = os.path.join(
            args.checkpoints,
            "last.pt",
        )
        weights = torch.load(
            weights_path,
            map_location="cpu",
        )

    weights = {
        key.replace("module.", ""): value
        for key, value in weights.items()
    }

    model.load_state_dict(weights)

    dataset = ImageFolder(
        os.path.join(TINY_IMAGENET_ROOT, "val"),
        transform=test_transform,
    )

    return model, dataset, test_transform


def safe_load_wnids(root):
    wnids_file = os.path.join(root, "wnids.txt")

    if not os.path.exists(wnids_file):
        raise FileNotFoundError(
            f"Missing wnids.txt in {root}"
        )

    with open(wnids_file, "r") as file:
        wnids = [
            line.strip()
            for line in file.readlines()
            if line.strip()
        ]

    return {
        wnid: index
        for index, wnid in enumerate(wnids)
    }


def sanity_check_dataset(
    dataset,
    name="dataset",
    num_batches=3,
):
    print(f"\n[Sanity Check] {name}")
    print("Total samples:", len(dataset))

    if len(dataset) == 0:
        raise RuntimeError(
            f"{name} is EMPTY. "
            "Check paths / wnid mapping / severity."
        )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    for batch_index, (inputs, targets) in enumerate(loader):
        print("Batch shape:", inputs.shape)
        print(
            "Label range:",
            (
                targets.min().item(),
                targets.max().item(),
            ),
        )

        if batch_index >= num_batches:
            break


def main(temperature, magnitude):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load model + clean data
    model, dataset, test_transform = get_model_and_dataset(args)

    model = model.to(device)
    model.eval()

    num_classes = 200

    # Load TinyImageNet-C
    train_dir = os.path.join(
        TINY_IMAGENET_ROOT,
        "train",
    )

    wnid_to_idx = torchvision.datasets.ImageFolder(
        train_dir
    ).class_to_idx

    corruptions = (
        [args.corruption]
        if args.corruption is not None
        else sorted(os.listdir(TINY_IMAGENET_C_ROOT))
    )

    print("[Info] Corruptions used:", corruptions)

    tinyc_dataset = TinyImageNetC(
        root=TINY_IMAGENET_C_ROOT,
        wnid_to_idx=wnid_to_idx,
        severity=args.severity,
        corruption=args.corruption,
        transform=test_transform,
    )

    sanity_check_dataset(
        tinyc_dataset,
        "TinyImageNet-C",
    )

    tinyc_loader = DataLoader(
        tinyc_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    # Randomly permutate clean dataset
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    dataset = torch.utils.data.Subset(
        dataset,
        indices,
    )

    # Split clean data
    dataset_name = args.model_name.split("_")[-1]

    num_classes = {
        "tinyimagenet": 200,
        "imagenet": 1000,
    }[dataset_name]

    n = len(dataset)
    num_train_samples = n // args.r

    train_dataset = torch.utils.data.Subset(
        dataset,
        range(
            0,
            int(num_train_samples),
        ),
    )

    test_dataset = torch.utils.data.Subset(
        dataset,
        range(num_train_samples, n),
    )

    val_dataset = torch.utils.data.Subset(
        test_dataset,
        range(
            0,
            int((len(test_dataset) // 5)),
        ),
    )

    test_dataset = torch.utils.data.Subset(
        test_dataset,
        range(
            int(len(test_dataset) // 5),
            len(test_dataset),
        ),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=6,
        prefetch_factor=2,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=6,
        prefetch_factor=2,
    )

    test_dataloader = DataLoader(
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

    method.fit(
        train_dataloader,
        val_dataloader,
    )

    # Evaluate TinyImageNet-C
    all_preds = []
    all_targets = []
    all_scores = []

    for inputs, targets in tqdm(tinyc_loader):
        inputs = inputs.to(device)

        preds = None

        if magnitude > 0:
            inputs = Variable(
                inputs,
                requires_grad=True,
            )

            outputs = model(inputs)
            preds = torch.argmax(
                outputs,
                dim=1,
            )

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

        if preds is None:
            preds = torch.argmax(
                logits,
                dim=1,
            )

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_scores.append(scores.cpu())

    test_preds = torch.cat(all_preds)
    test_targets = torch.cat(all_targets)
    test_scores = torch.cat(all_scores)

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
        "severity": args.severity,
        "corruption": args.corruption,
    }

    print(json.dumps(results, indent=2))

    filepath = os.path.join(
        "results_tinyimagenet_corrupted",
        args.model_name,
        args.method,
    )

    os.makedirs(
        filepath,
        exist_ok=True,
    )

    filename = f"{args.model_name}_{args.seed}.csv"

    append_results_to_file(
        results,
        os.path.join(filepath, filename),
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet34_tinyimagenet",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="msp",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--r",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--lbd",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default=None,
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

    if args.method in ["msp"]:
        main(
            temperature=1.0,
            magnitude=0.0,
        )
    else:
        for temperature, magnitude in itertools.product(
            TEMPERATURES,
            MAGNITUDES,
        ):
            main(
                temperature,
                magnitude,
            )