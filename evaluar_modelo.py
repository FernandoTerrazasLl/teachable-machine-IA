import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageOps

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf


TARGET_CLASSES = ("BCC", "MEL", "AK")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "keras_model1.h5"
LABELS_PATH = BASE_DIR / "labels1.txt"
DATASET_ROOT = Path(
    "/Users/terrazasllanosfernando/Desktop/DataSetOk/"
    "isic-2019-skin-lesion-images-for-classification"
)


def load_class_codes(labels_path: Path) -> List[str]:
    class_codes: List[str] = []

    with labels_path.open("r", encoding="utf-8") as labels_file:
        for raw_line in labels_file:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if parts[0].isdigit() and len(parts) > 1:
                class_code = parts[1].upper()
            else:
                class_code = parts[0].upper()

            class_codes.append(class_code)

    if not class_codes:
        raise ValueError("El archivo de etiquetas esta vacio.")

    return class_codes


def collect_images(folder: Path) -> List[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def preprocess_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    normalized = (image_array / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)


def evaluate(
    model: tf.keras.Model,
    class_codes: List[str],
    dataset_root: Path,
) -> None:
    stats: Dict[str, Dict[str, int]] = {
        class_name: {"total": 0, "correct": 0}
        for class_name in TARGET_CLASSES
    }

    for class_name in TARGET_CLASSES:
        class_dir = dataset_root / class_name
        if not class_dir.exists() or not class_dir.is_dir():
            print(f"[AVISO] No se encontro la carpeta: {class_dir}")
            continue

        images = collect_images(class_dir)
        print(f"Procesando {class_name}: {len(images)} imagenes")

        for image_path in images:
            try:
                input_data = preprocess_image(image_path)
                pred = model.predict_on_batch(input_data)[0]
                predicted_idx = int(np.argmax(pred))
            except Exception:
                continue

            predicted_code = (
                class_codes[predicted_idx]
                if 0 <= predicted_idx < len(class_codes)
                else f"IDX_{predicted_idx}"
            )

            stats[class_name]["total"] += 1
            is_correct = predicted_code == class_name
            if is_correct:
                stats[class_name]["correct"] += 1

    print("\n=== RESULTADOS POR CLASE ===")
    global_total = 0
    global_correct = 0

    for class_name in TARGET_CLASSES:
        total = stats[class_name]["total"]
        correct = stats[class_name]["correct"]
        accuracy = (correct / total * 100.0) if total else 0.0

        print(
            f"{class_name}: analizadas={total}, aciertos={correct}, "
            f"porcentaje={accuracy:.2f}%"
        )

        global_total += total
        global_correct += correct

    global_accuracy = (global_correct / global_total * 100.0) if global_total else 0.0
    print("\n=== RESULTADO GLOBAL ===")
    print(
        f"Total analizadas={global_total}, aciertos={global_correct}, "
        f"exactitud_global={global_accuracy:.2f}%"
    )


def main() -> None:
    dataset_root = DATASET_ROOT
    model_path = MODEL_PATH
    labels_path = LABELS_PATH

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels no encontrado: {labels_path}")

    print(f"Cargando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    class_codes = load_class_codes(labels_path)

    print(f"Etiquetas cargadas: {class_codes}")
    print(f"Dataset raiz: {dataset_root}\n")

    evaluate(
        model=model,
        class_codes=class_codes,
        dataset_root=dataset_root,
    )


if __name__ == "__main__":
    main()