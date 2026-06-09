from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_LOCAL = Path("/Users/terrazasllanosfernando/Desktop/Oficial_UNet_Segmentado")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["AK", "BCC", "MEL"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 260
BATCH_SIZE = 32
NUM_WORKERS = 0
DROPOUT_RATE = 0.35
WEIGHT_DECAY = 1e-2

LR_STAGE1 = 1e-3
LR_STAGE2 = 5e-5
NUM_EPOCHS = 40
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
