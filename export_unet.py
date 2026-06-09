import sys
from pathlib import Path
import torch
import torch.nn as nn
import shutil
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model.unet import UNet
from src.config import OUTPUTS_DIR


class FusedUNet(nn.Module):
    def __init__(self, unet_model: nn.Module):
        super().__init__()
        self.unet = unet_model
        self.unet.eval()

    def forward(self, x):
        logits = self.unet(x)
        return torch.sigmoid(logits)


def main():
    pth_path = OUTPUTS_DIR / "best_segmenter_unet.pth"
    onnx_path = OUTPUTS_DIR / "fused_segmenter_unet.onnx"
    flutter_models_dir = Path("/Users/terrazasllanosfernando/Desktop/IA/py/derma-scan-app/assets/models")
    flutter_onnx_path = flutter_models_dir / "fused_segmenter_unet.onnx"

    if not pth_path.exists():
        sys.exit(1)

    base_model = UNet(in_channels=3, out_channels=1)
    base_model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    
    fused_model = FusedUNet(base_model)
    fused_model.eval()

    dummy_input = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    
    try:
        torch.onnx.export(
            fused_model,
            dummy_input,
            str(onnx_path),
            input_names=['image_input'],
            output_names=['mask_output'],
            dynamic_axes={
                'image_input': {0: 'batch_size'},
                'mask_output': {0: 'batch_size'},
            },
            opset_version=18,
            external_data=False,
            verbose=False,
        )
        
        model_proto = onnx.load(str(onnx_path))
        model_proto.ir_version = 9
        onnx.save(model_proto, str(onnx_path))

        if flutter_models_dir.exists():
            shutil.copy2(onnx_path, flutter_onnx_path)
            print(f"Modelo segmentador copiado a: {flutter_onnx_path}")
            
    except Exception as e:
        print(f"Error durante la exportación: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
