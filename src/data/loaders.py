from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS, VAL_SPLIT, TEST_SPLIT
from src.data.dataset import SkinLesionDataset

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def create_dataloaders(
    dataset_path: Path,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, SkinLesionDataset]:

    full_dataset = SkinLesionDataset(
        root_dir=dataset_path,
        balance=False,
        seed=seed,
    )

    labels = [label for _, label in full_dataset.samples]

    train_val_test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT + TEST_SPLIT, random_state=seed)
    train_indices, val_test_indices = next(train_val_test_splitter.split(range(len(labels)), labels))

    val_size_relative = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_size_relative, random_state=seed)
    val_indices_relative, test_indices_relative = next(val_test_splitter.split(val_test_indices, [labels[i] for i in val_test_indices]))

    val_indices = [val_test_indices[i] for i in val_indices_relative]
    test_indices = [val_test_indices[i] for i in test_indices_relative]

    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    test_samples = [full_dataset.samples[i] for i in test_indices]

    train_dataset = SkinLesionDataset(
        root_dir=dataset_path,
        transform=train_transform,
        balance=True,
        samples=train_samples,
        report=full_dataset.raw_class_counts,
    )
    val_dataset = SkinLesionDataset(
        root_dir=dataset_path,
        transform=val_transform,
        balance=False,
        samples=val_samples,
        report=full_dataset.raw_class_counts,
    )
    test_dataset = SkinLesionDataset(
        root_dir=dataset_path,
        transform=val_transform,
        balance=False,
        samples=test_samples,
        report=full_dataset.raw_class_counts,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, full_dataset
