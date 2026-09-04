import os
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset
from nuimages import NuImages


class NuImagesClassification(Dataset):
    """
    Converts nuImages object annotations into a PyTorch classification dataset.
    Each object annotation becomes one training sample.
    Output: image_crop, class_index
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-train",
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
        min_size: int = 16,
    ):
        super().__init__()

        self.transform = transform
        self.min_size = min_size

        print(f"\nLoading nuImages ({version})...")

        self.nuim = NuImages(
            version=version,
            dataroot=dataroot,
            verbose=True
        )

        ####################################################################
        # Build lookup tables
        ####################################################################
        print("Building lookup tables...")

        self.sample_data_lookup = {
            sd["token"]: sd
            for sd in self.nuim.sample_data
        }

        self.category_lookup = {
            c["token"]: c["name"]
            for c in self.nuim.category
        }

        ####################################################################
        # Build classes
        ####################################################################
        if class_to_idx is None:
            categories = sorted({
                self.category_lookup[a["category_token"]]
                for a in self.nuim.object_ann
            })
            self.class_to_idx = {
                c: i
                for i, c in enumerate(categories)
            }
        else:
            self.class_to_idx = class_to_idx
        self.idx_to_class = {
            v: k
            for k, v in self.class_to_idx.items()
        }

        ####################################################################
        # Build sample list
        ####################################################################
        print("Building annotation list...")
        self.samples: List[Dict] = []
        skipped = 0

        for ann in self.nuim.object_ann:
            category = self.category_lookup[ann["category_token"]]

            if category not in self.class_to_idx:
                skipped += 1
                continue
            sample_data = self.sample_data_lookup[
                ann["sample_data_token"]
            ]
            image_path = os.path.join(
                dataroot,
                sample_data["filename"]
            )

            if not os.path.exists(image_path):
                skipped += 1
                continue

            x1, y1, x2, y2 = ann["bbox"]
            width = x2 - x1
            height = y2 - y1

            if width < self.min_size or height < self.min_size:
                skipped += 1
                continue

            self.samples.append({
                "image_path": image_path,
                "bbox": ann["bbox"],
                "label": self.class_to_idx[category],
                "category": category,
            })

        print("--------------------------------")
        print(f"Classes : {len(self.class_to_idx)}")
        print(f"Samples : {len(self.samples)}")
        print(f"Skipped : {skipped}")
        print("--------------------------------")

    ########################################################################
    def __len__(self):
        return len(self.samples)

    ########################################################################
    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        x1, y1, x2, y2 = sample["bbox"]
        crop = image.crop((x1, y1, x2, y2))

        if self.transform is not None:
            crop = self.transform(crop)

        return crop, sample["label"]

    ########################################################################
    @property
    def num_classes(self):

        return len(self.class_to_idx)

    ########################################################################
    def get_class_name(self, idx):

        return self.idx_to_class[idx]

    ########################################################################
    def get_class_distribution(self):

        counts = {
            c: 0
            for c in self.class_to_idx
        }

        for s in self.samples:
            counts[s["category"]] += 1

        return counts