import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IMAGE_SIZE,
    VALID_EXTENSIONS,
)

class SkinLesionDataset(Dataset):

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[transforms.Compose] = None,
        balance: bool = True,
        seed: int = 42,
        samples: Optional[List[Tuple[Path, int]]] = None,
        report: Optional[Dict[str, int]] = None,
    ) -> None:

        self.root_dir = Path(root_dir)
        self.transform = transform or self._default_transform()

        if samples is not None:
            self.samples = list(samples)
            self.raw_class_counts = report or {}
            if balance:
                grouped: Dict[int, List[Tuple[Path, int]]] = {}
                for img_path, label in self.samples:
                    if label not in grouped:
                        grouped[label] = []
                    grouped[label].append((img_path, label))
                if len(grouped) > 0:
                    min_count = min(len(v) for v in grouped.values())
                    rng = random.Random(seed)
                    balanced_samples = []
                    for label, items in grouped.items():
                        if len(items) > min_count:
                            items = rng.sample(items, min_count)
                        balanced_samples.extend(items)
                    self.samples = balanced_samples
                    rng_shuffle = random.Random(seed)
                    rng_shuffle.shuffle(self.samples)
            return

        self.samples = []
        class_images: Dict[str, List[Path]] = {}
        
        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            images = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
            )

            if len(images) == 0:
                raise ValueError(f"Directory {class_dir} contains no valid images.")

            class_images[class_name] = images

        self.raw_class_counts = {name: len(imgs) for name, imgs in class_images.items()}

        if balance:
            min_count = min(len(imgs) for imgs in class_images.values())
            rng = random.Random(seed)

            for class_name in CLASS_NAMES:
                imgs = class_images[class_name]
                if len(imgs) > min_count:
                    imgs = rng.sample(imgs, min_count)
                label = CLASS_TO_IDX[class_name]
                self.samples.extend((img_path, label) for img_path in imgs)
        else:
            for class_name in CLASS_NAMES:
                label = CLASS_TO_IDX[class_name]
                self.samples.extend(
                    (img_path, label)
                    for img_path in class_images[class_name]
                )

        rng_shuffle = random.Random(seed)
        rng_shuffle.shuffle(self.samples)

    @staticmethod
    def _default_transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image_pil = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image_pil)
        return image_tensor, label

    def get_class_distribution(self) -> Dict[str, int]:
        if self.raw_class_counts:
            return dict(self.raw_class_counts)
        counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts

    def get_balanced_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            class_name = CLASS_NAMES[label]
            counts[class_name] += 1
        return counts
