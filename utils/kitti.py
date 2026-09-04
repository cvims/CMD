import os
import random
import time

from typing import Dict, List, Optional
from PIL import Image
from torch.utils.data import Dataset


class KITTIClassification(Dataset):

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
        min_size: int = 16,
        ignore_classes: Optional[List[str]] = None,
    ):

        super().__init__()

        assert split in [
            "train",
            "val"
        ]

        self.transform = transform
        self.split = split
        self.min_size = min_size

        start_time = time.time()

        def log(message):
            elapsed = time.time() - start_time
            print(f"[{elapsed:8.2f}s] {message}", flush=True)

        log("Initializing KITTI classification dataset")

        # -------------------------------------------------
        # Ignore classes
        # -------------------------------------------------

        if ignore_classes is None:
            ignore_classes = ["DontCare"]

        self.ignore_classes = set(ignore_classes)

        # -------------------------------------------------
        # Dataset paths
        # -------------------------------------------------
        image_dir = os.path.join(root, "training", "image_2")

        label_dir = os.path.join(root, "training", "label_2")

        self.image_dir = image_dir
        self.label_dir = label_dir

        log(f"Image directory: {image_dir}")
        log(f"Label directory: {label_dir}")

        if not os.path.isdir(image_dir):
            raise RuntimeError(
                f"Missing image directory: {image_dir}"
            )

        if not os.path.isdir(label_dir):
            raise RuntimeError(
                f"Missing label directory: {label_dir}"
            )

        # -------------------------------------------------
        # Collect image IDs
        # -------------------------------------------------

        log("Scanning images...")

        image_ids = sorted([
            os.path.splitext(file)[0]
            for file in os.listdir(image_dir)
            if file.endswith(".png")
        ])

        log(f"Found {len(image_ids)} images")

        # -------------------------------------------------
        # Train / validation split
        # -------------------------------------------------

        rng = random.Random(seed)
        rng.shuffle(image_ids)

        split_index = int(
            len(image_ids) * (1.0 - val_ratio)
        )

        if split == "train":
            selected_ids = image_ids[:split_index]
        else:
            selected_ids = image_ids[split_index:]

        log(f"{split} images: {len(selected_ids)}")


        # -------------------------------------------------
        # Build class mapping
        # -------------------------------------------------
        if class_to_idx is None:
            log("Scanning labels for classes...")

            classes = set()

            for index, image_id in enumerate(image_ids):
                if index % 500 == 0:
                    log(f"Class scan {index}/{len(image_ids)}")

                label_path = os.path.join(label_dir, image_id + ".txt")

                if not os.path.exists(label_path):
                    continue

                with open(label_path, "r") as file:
                    for line in file:
                        fields = line.strip().split()

                        if len(fields) == 0:
                            continue

                        category = fields[0]

                        if category not in self.ignore_classes:
                            classes.add(category)

            classes = sorted(classes)

            self.class_to_idx = {
                name: idx
                for idx, name in enumerate(classes)
            }

            log(f"Classes: {self.class_to_idx}")

        else:
            log("Using provided class mapping")
            self.class_to_idx = class_to_idx

        self.idx_to_class = {
            idx: name
            for name, idx in self.class_to_idx.items()
        }


        # -------------------------------------------------
        # Build samples
        # -------------------------------------------------
        log("Building object samples...")
        self.samples = []
        skipped = 0

        for index, image_id in enumerate(selected_ids):
            if index % 500 == 0:
                log(f"Sample scan {index}/{len(selected_ids)}")

            image_path = os.path.join(image_dir, image_id + ".png")
            label_path = os.path.join(label_dir, image_id + ".txt")

            if not os.path.exists(label_path):
                skipped += 1
                continue

            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as error:
                log(f"Failed loading {image_path}: {error}")
                skipped += 1
                continue 
                        
            # -------------------------------------------------
            # Read label file
            # -------------------------------------------------
            with open(label_path, "r") as file:
                lines = file.readlines()

            for line in lines:
                fields = line.strip().split()

                if len(fields) < 8:
                    skipped += 1
                    continue

                category = fields[0]

                # Ignore unwanted classes
                if category in self.ignore_classes:
                    skipped += 1
                    continue

                # Ignore classes not in mapping
                if category not in self.class_to_idx:
                    skipped += 1
                    continue

                try:
                    x1 = int(float(fields[4]))
                    y1 = int(float(fields[5]))
                    x2 = int(float(fields[6]))
                    y2 = int(float(fields[7]))
                except ValueError:
                    skipped += 1
                    continue

                # -------------------------------------------------
                # Clamp bounding box
                # -------------------------------------------------
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                box_width = x2 - x1
                box_height = y2 - y1

                # Remove invalid objects
                if box_width < self.min_size:
                    skipped += 1
                    continue

                if box_height < self.min_size:
                    skipped += 1
                    continue

                # -------------------------------------------------
                # Store sample
                # -------------------------------------------------
                self.samples.append({
                    "image_path": image_path,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "label": self.class_to_idx[category],
                    "category": category,
                })

        # -------------------------------------------------
        # Dataset summary
        # -------------------------------------------------
        log("-" * 50)
        log(f"Classes : {len(self.class_to_idx)}")
        log(f"Samples : {len(self.samples)}")
        log(f"Skipped : {skipped}")
        log("Dataset ready")

    # =====================================================
    # Dataset length
    # =====================================================
    def __len__(self):
        return len(self.samples)

    # =====================================================
    # Get one sample
    # =====================================================
    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")

        x1, y1, x2, y2 = sample["bbox"]
        image = image.crop((x1, y1, x2, y2))

        if self.transform is not None:
            image = self.transform(image)

        return image, sample["label"]

    # =====================================================
    # Class name lookup
    # =====================================================
    def get_class_name(self, index):
        return self.idx_to_class[index]


    # =====================================================
    # Print class distribution
    # =====================================================
    def print_class_distribution(self):
        counts = {name: 0
            for name in self.class_to_idx.keys()}

        for sample in self.samples:
            counts[sample["category"]] += 1
        print("-" * 40)

        for name, count in sorted(counts.items()):
            print(f"{name:20s}" f"{count:8d}")
        print("-" * 40)

    # =====================================================
    # Image cache loader
    # =====================================================
    def _load_image(self, path):
        """
        Load image safely.
        Uses an internal cache if enabled.
        """
        if hasattr(self, "_image_cache"):
            if path in self._image_cache:
                return self._image_cache[path]

        try:
            image = Image.open(path).convert("RGB")

        except Exception as error:
            raise RuntimeError(f"Cannot load image {path}: {error}")

        # Optional cache
        if hasattr(self, "_image_cache"):
            self._image_cache[path] = image
        return image

    # =====================================================
    # Enable image caching
    # =====================================================
    def enable_cache(self, max_images: int = 500):
        """
        Cache images in memory.
        Useful when the dataset is small
        and storage is slow.
        """
        self._image_cache = {}
        self.cache_limit = max_images

    # =====================================================
    # Improved item loading with cache
    # =====================================================
    def get_sample_info(self, index):
        """
        Returns metadata without loading image.
        """
        return self.samples[index]

    # =====================================================
    # Dataset statistics
    # =====================================================
    def get_class_counts(self):
        counts = {idx: 0
            for idx in self.idx_to_class.keys()}

        for sample in self.samples:
            counts[sample["label"]] += 1
        return counts

    # =====================================================
    # Class weights for CrossEntropyLoss
    # =====================================================
    def get_class_weights(self):
        """
        Calculate inverse-frequency class weights.
        Useful for imbalanced KITTI classes.
        """
        import torch

        counts = self.get_class_counts()
        total = sum(counts.values())
        weights = []

        for idx in range(len(self.class_to_idx)):
            if counts[idx] == 0:
                weights.append(0.0)
            else:
                weights.append(total/(len(counts)*counts[idx]))

        return torch.tensor(weights,dtype=torch.float32)

    # =====================================================
    # Dataset summary
    # =====================================================
    def summary(self):
        print("=" * 60)
        print("KITTI Classification Dataset")
        print("=" * 60)
        print(f"Split      : {self.split}")
        print(f"Images dir : {self.image_dir}")
        print(f"Samples    : {len(self.samples)}")
        print(f"Classes    : {len(self.class_to_idx)}")
        print()
        print("Class mapping:")

        for idx, name in sorted(self.idx_to_class.items()):
            print(f"{idx:3d} -> {name}")

        print()
        self.print_class_distribution()
        print("=" * 60)

    # =====================================================
    # Verify dataset integrity
    # =====================================================
    def verify(self):
        """
        Check for missing images,
        invalid boxes, and labels.
        """
        problems = 0
        for idx, sample in enumerate(self.samples):
            if not os.path.exists(sample["image_path"]):
                print( "Missing:", sample["image_path"])
                problems += 1
            x1, y1, x2, y2 = sample["bbox"]
            if x2 <= x1 or y2 <= y1:
                print("Invalid bbox:", sample["bbox"])
                problems += 1
            if sample["label"] not in self.idx_to_class:
                print("Invalid label:", sample["label"])
                problems += 1

        if problems == 0:
            print("Dataset verification passed.")
        else:
            print(f"Found {problems} problems.")