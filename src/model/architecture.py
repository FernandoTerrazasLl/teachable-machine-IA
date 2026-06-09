import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES, DROPOUT_RATE

def build_model(freeze_base: bool = True) -> nn.Module:
    weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
    model = models.efficientnet_b2(weights=weights)

    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),        
        nn.Linear(in_features, 512),         
        nn.BatchNorm1d(512),                   
        nn.SiLU(inplace=True),            
        nn.Dropout(p=0.35),               
        nn.Linear(512, NUM_CLASSES),
    )

    return model

def unfreeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True

def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
