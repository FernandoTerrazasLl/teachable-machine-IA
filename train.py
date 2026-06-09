import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DATASET_LOCAL,
    DEVICE,
    NUM_EPOCHS,
    LR_STAGE1,
    LR_STAGE2,
    BATCH_SIZE,
    OUTPUTS_DIR,
    IMAGE_SIZE,
    WEIGHT_DECAY,
)
from src.data.loaders import create_dataloaders
from src.model.architecture import build_model, count_parameters
from src.model.trainer import train_model
from src.model.evaluation import run_full_evaluation


class FusedDermatologyCNN(nn.Module):
    FLAT_SIZE = IMAGE_SIZE * IMAGE_SIZE * 3

    def __init__(self, cnn_model: nn.Module):
        super(FusedDermatologyCNN, self).__init__()
        self.cnn = cnn_model
        self.cnn.eval()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x_image = x.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
        x_normalized = (x_image - self.mean) / self.std
        logits = self.cnn(x_normalized)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities


def export_to_onnx(cnn_model: nn.Module, save_path: Path) -> None:
    flat_size = IMAGE_SIZE * IMAGE_SIZE * 3
    print("Exportando modelo a formato ONNX")
    cnn_cpu = cnn_model.cpu()
    fused_model = FusedDermatologyCNN(cnn_cpu)
    fused_model.eval()

    try:
        dummy_input = torch.randn(1, flat_size, dtype=torch.float32)

        torch.onnx.export(
            fused_model,
            dummy_input,
            str(save_path),
            input_names=['raw_pixels'],
            output_names=['probabilities'],
            dynamic_axes={
                'raw_pixels':    {0: 'batch_size'},
                'probabilities': {0: 'batch_size'},
            },
            opset_version=18,
            external_data=False,
            verbose=False,
        )
        print(f"Exportacion de ONNX finalizada en: {save_path}")

    except Exception as e:
        print(f"Error al exportar a ONNX: {e}")


def main() -> None:
    dataset_path = DATASET_LOCAL

    if not dataset_path.exists():
        sys.exit(1)

    train_loader, val_loader, test_loader, full_dataset = create_dataloaders(
        dataset_path=dataset_path,
        batch_size=BATCH_SIZE,
    )

    model = build_model(freeze_base=True)
    save_path = OUTPUTS_DIR / "best_model.pth"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        lr_stage1=LR_STAGE1,
        lr_stage2=LR_STAGE2,
        patience=7,
        device=DEVICE,
        save_path=save_path,
    )

    import json
    history_data = {
        "train_loss": history.train_loss,
        "train_acc":  history.train_acc,
        "val_loss":   history.val_loss,
        "val_acc":    history.val_acc,
    }
    history_json_path = OUTPUTS_DIR / "training_history.json"
    with open(history_json_path, "w") as f:
        json.dump(history_data, f, indent=4)

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    run_full_evaluation(
        model=model,
        test_loader=test_loader,
        history=history,
    )

    onnx_path = OUTPUTS_DIR / "fused_dermatology_model.onnx"
    export_to_onnx(model, onnx_path)

    flutter_models_dir = Path("/Users/terrazasllanosfernando/Desktop/IA/py/derma-scan-app/assets/models")
    if flutter_models_dir.exists():
        flutter_onnx_path = flutter_models_dir / "fused_dermatology_model.onnx"
        try:
            import shutil
            shutil.copy2(onnx_path, flutter_onnx_path)
            print(f"Modelo copiado a: {flutter_onnx_path}")
        except Exception as e:
            print(f"Error al copiar modelo a assets: {e}")


if __name__ == "__main__":
    main()
