import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.config import DEVICE, LR_STAGE1, LR_STAGE2, NUM_EPOCHS, OUTPUTS_DIR, WEIGHT_DECAY

@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0) -> None:
        self.patience = patience          
        self.min_delta = min_delta        
        self.counter = 0                  
        self.best_loss = float("inf")     
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> tuple:

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:

    model.eval()  
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    lr_stage1: float = LR_STAGE1,
    lr_stage2: float = LR_STAGE2,
    patience: int = 7,
    device: torch.device = DEVICE,
    save_path: Path = OUTPUTS_DIR / "best_model.pth",
) -> TrainingHistory:
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = TrainingHistory()
    early_stopping = EarlyStopping(patience=patience)
    best_val_loss = float("inf")

    stage1_epochs = 5

    print(f"\n{'='*65}")
    print(f"  TWO-STAGE CNN TRAINING INITIATED")
    print(f"  Device: {device}")
    print(f"  Stage 1: {stage1_epochs} epochs @ LR = {lr_stage1}")
    print(f"  Stage 2: Up to {num_epochs - stage1_epochs} epochs @ LR = {lr_stage2}")
    print(f"  Early Stopping patience: {patience} epochs")
    print(f"  Scheduler: ReduceLROnPlateau")
    print(f"{'='*65}\n")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_stage1,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
    )

    total_start = time.time()
    current_stage = 1

    for epoch in range(1, num_epochs + 1):
        if epoch == stage1_epochs + 1 and current_stage == 1:
            print(f"\n{'='*65}")
            print(f"  TRANSITION TO STAGE 2: UNFREEZING ALL CNN LAYERS")
            print(f"  New Learning Rate: {lr_stage2}")
            print(f"{'='*65}\n")

            current_stage = 2

            from src.model.architecture import unfreeze_model
            unfreeze_model(model)

            optimizer = AdamW(
                model.parameters(),
                lr=lr_stage2,
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=4,
            )

        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        elapsed = time.time() - epoch_start

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            improved = " Best model saved"

        stage_name = "Stage 1" if current_stage == 1 else "Stage 2"
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"  Epoch {epoch:02d}/{num_epochs} [{stage_name}] "
            f"  LR: {current_lr:.2e} "
            f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}% "
            f"  Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}% "
            f"  {elapsed:.1f}s{improved}"
        )

        if current_stage == 2:
            if early_stopping(val_loss):
                print(f"\n  EARLY STOPPING TRIGGERED:")
                break

    total_time = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETED in {total_time:.1f} seconds")
    print(f"  Best model weights saved at: {save_path}")
    print(f"{'='*65}\n")

    return history
