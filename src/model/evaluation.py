
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

from src.config import CLASS_NAMES, DEVICE, OUTPUTS_DIR
from src.model.trainer import TrainingHistory


@torch.no_grad()
def get_predictions_and_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> tuple:
    model.eval()
    model = model.to(device)

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> tuple:
    y_true, y_pred, _ = get_predictions_and_probabilities(model, loader, device)
    return y_true, y_pred


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path = OUTPUTS_DIR / "confusion_matrix.png",
) -> np.ndarray:
    
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=CLASS_NAMES
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True,
    )
    ax.set_title("Confusion Matrix — Skin Lesions", fontsize=14, pad=15)
    ax.set_xlabel("Model Prediction", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  📊 Confusion matrix saved at: {save_path}")
    return cm


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print(f"\n{'='*65}")
    print("  CLASSIFICATION REPORT")
    print(f"{'='*65}")
    print(report)
    return report


def plot_training_history(
    history: TrainingHistory,
    save_path: Path = OUTPUTS_DIR / "training_history.png",
) -> None:

    epochs = range(1, len(history.train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history.train_loss, "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history.val_loss, "r-o", label="Val Loss", markersize=4)
    ax1.set_title("Loss per Epoch", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (CrossEntropy)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history.train_acc, "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, history.val_acc, "r-o", label="Val Acc", markersize=4)
    ax2.set_title("Accuracy per Epoch", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  📈 Training history plots saved at: {save_path}")


def run_full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    history: TrainingHistory,
) -> None:
    print(f"\n{'='*65}")
    print("  FULL MODEL EVALUATION")
    print(f"{'='*65}\n")

    # Predictions
    y_true, y_pred = get_predictions(model, test_loader)

    # Classification report
    generate_classification_report(y_true, y_pred)

    # Confusion matrix
    generate_confusion_matrix(y_true, y_pred)

    # Training history plots
    plot_training_history(history)

    print(f"\n  Evaluation complete. Check the output folder: {OUTPUTS_DIR}\n")
