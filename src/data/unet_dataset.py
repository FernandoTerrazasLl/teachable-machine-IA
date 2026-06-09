import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class HAM10000Dataset(Dataset):
    def __init__(self, image_paths: List[Path], mask_dir: Path, augment: bool = True):
        self.image_paths = image_paths
        self.mask_dir = Path(mask_dir)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image_id = img_path.stem
        
        mask_path = self.mask_dir / f"{image_id}_segmentation.png"
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for image: {img_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((128, 128))
        mask = mask.resize((128, 128))

        image_tensor = TF.to_tensor(image)
        
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        mask_binary = (mask_np >= 0.5).astype(np.float32)
        mask_tensor = torch.tensor(mask_binary).unsqueeze(0)

        if self.augment:
            if random.random() > 0.5:
                image_tensor = TF.hflip(image_tensor)
                mask_tensor = TF.hflip(mask_tensor)

            if random.random() > 0.5:
                image_tensor = TF.vflip(image_tensor)
                mask_tensor = TF.vflip(mask_tensor)

            if random.random() > 0.5:
                angle = random.uniform(-15.0, 15.0)
                image_tensor = TF.rotate(image_tensor, angle)
                mask_tensor = TF.rotate(mask_tensor, angle)

        return image_tensor, mask_tensor
