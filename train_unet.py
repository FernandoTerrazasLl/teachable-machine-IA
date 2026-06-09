import os
import sys
import time
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model.unet import UNet
from src.data.unet_dataset import HAM10000Dataset
from src.config import DEVICE, OUTPUTS_DIR

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

PART1_DIR = Path("/Users/terrazasllanosfernando/Desktop/Oficial_Modelo1_Imagenes/HAM/HAM10000_images_part_1")
PART2_DIR = Path("/Users/terrazasllanosfernando/Desktop/Oficial_Modelo1_Imagenes/HAM/HAM10000_images_part_2")
MASK_DIR = Path("/Users/terrazasllanosfernando/Desktop/Oficial_Modelo1_Segmentado/HAM_Segmentado")


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice_coef = (2. * intersection + 1e-5) / (probs_flat.sum() + targets_flat.sum() + 1e-5)
        dice_loss = 1.0 - dice_coef
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    batch_size = targets.size(0)
    preds_flat = preds.view(batch_size, -1)
    targets_flat = targets.view(batch_size, -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = (preds_flat + targets_flat).sum(dim=1) - intersection
    iou = (intersection + 1e-5) / (union + 1e-5)
    dice = (2. * intersection + 1e-5) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + 1e-5)
    return iou.mean().item(), dice.mean().item()


def main():
    print("Entrenamiento de segmentador U-Net para HAM10000")
    print(f"Dispositivo: {DEVICE}")

    image_paths = []
    for directory in [PART1_DIR, PART2_DIR]:
        if directory.exists():
            image_paths.extend(sorted(list(directory.glob("*.jpg"))))
            
    if len(image_paths) == 0:
        sys.exit(1)
        
    random.seed(42)
    random.shuffle(image_paths)
    
    val_size = int(len(image_paths) * 0.20)
    train_size = len(image_paths) - val_size
    
    train_paths = image_paths[:train_size]
    val_paths = image_paths[train_size:]
    
    train_dataset = HAM10000Dataset(train_paths, MASK_DIR, augment=True)
    val_dataset = HAM10000Dataset(val_paths, MASK_DIR, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    best_val_iou = 0.0
    save_path = OUTPUTS_DIR / "best_segmenter_unet.pth"
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        model.train()
        train_loss = 0.0
        train_ious = []
        train_dices = []
        
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            iou, dice = calculate_metrics(outputs, masks)
            train_ious.append(iou)
            train_dices.append(dice)
            
        scheduler.step()
        
        train_loss /= len(train_dataset)
        mean_train_iou = np.mean(train_ious)
        
        model.eval()
        val_loss = 0.0
        val_ious = []
        val_dices = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                iou, dice = calculate_metrics(outputs, masks)
                val_ious.append(iou)
                val_dices.append(dice)
                
        val_loss /= len(val_dataset)
        mean_val_iou = np.mean(val_ious)
        mean_val_dice = np.mean(val_dices)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} IoU: {mean_train_iou:.3f} - Val Loss: {val_loss:.4f} IoU: {mean_val_iou:.3f} Dice: {mean_val_dice:.3f} - Time: {epoch_time:.1f}s")
        
        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            torch.save(model.state_dict(), save_path)
            print("Guardado mejor segmentador")
            
    total_time = time.time() - start_time
    print(f"Entrenamiento completado en {total_time:.1f} segundos. Mejor IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
