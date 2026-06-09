import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
import torch.nn.functional as F

RAW_ROOT = Path("/Users/terrazasllanosfernando/Desktop/Oficial")
OUTPUT_ROOT = Path("/Users/terrazasllanosfernando/Desktop/Oficial_UNet_Segmentado")
UNET_PATH = Path("../derma-scan-app/assets/models/fused_segmenter_unet.onnx")
if not UNET_PATH.exists():
    UNET_PATH = Path("outputs/fused_segmenter_unet.onnx")

CLASS_NAMES = ["AK", "BCC", "MEL"]
# SOLO PARA GENERAR EL DATASET DE UNET SEGMENTADO)
def load_unet_session():
    if not UNET_PATH.exists():
        print(f"Error: No se encontro el modelo U-Net ONNX en: {UNET_PATH}")
        sys.exit(1)
    print(f"Cargando segmentador U-Net ONNX desde: {UNET_PATH}")
    return ort.InferenceSession(str(UNET_PATH))

def preprocess_for_unet(image_pil):
    img_128 = image_pil.resize((128, 128))
    img_arr = np.array(img_128, dtype=np.float32) / 255.0
    img_chw = np.transpose(img_arr, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    return img_batch

def morphological_closing_torch(mask_binary, radius=2):
    mask_tensor = torch.tensor(mask_binary, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel_size = 2 * radius + 1
    padding = radius
    
    dilated = F.max_pool2d(mask_tensor, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=padding)
    
    return eroded.squeeze().cpu().numpy().astype(np.uint8)

def main():
    if not RAW_ROOT.exists():
        print(f"Error: La carpeta de imagenes crudas no existe en: {RAW_ROOT}")
        sys.exit(1)

    unet_session = load_unet_session()
    
    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
    for class_name in CLASS_NAMES:
        (OUTPUT_ROOT / class_name).mkdir(exist_ok=True, parents=True)

    print("\nIniciando generacion del dataset segmentado por U-Net...")
    start_time = time.time()
    
    total_processed = 0
    total_images = 0
    
    for class_name in CLASS_NAMES:
        class_dir = RAW_ROOT / class_name
        if class_dir.exists():
            total_images += len(list(class_dir.glob("*.*")))

    print(f"Total de imagenes a procesar: {total_images}")

    x_coords = (np.arange(260) * 128 / 260).astype(np.int32).clip(0, 127)
    y_coords = (np.arange(260) * 128 / 260).astype(np.int32).clip(0, 127)
    mesh_idx = np.ix_(y_coords, x_coords)

    for class_name in CLASS_NAMES:
        class_dir = RAW_ROOT / class_name
        output_class_dir = OUTPUT_ROOT / class_name
        
        if not class_dir.exists():
            print(f"Warning: La carpeta de clase {class_name} no existe.")
            continue
            
        print(f"\nProcesando clase {class_name}...")
        image_files = sorted(list(class_dir.glob("*.*")))
        
        for idx, img_path in enumerate(image_files):
            try:
                image_pil = Image.open(img_path).convert("RGB")
                
                unet_input = preprocess_for_unet(image_pil)
                unet_outputs = unet_session.run(None, {'image_input': unet_input})
                mask_128 = unet_outputs[0][0][0]
                
                mask_binary = (mask_128 >= 0.35).astype(np.uint8)
                mask_closed = morphological_closing_torch(mask_binary, radius=2)
                
                mask_260 = mask_closed[mesh_idx]
                
                img_260 = image_pil.resize((260, 260))
                img_arr = np.array(img_260, dtype=np.float32)
                
                masked_img_arr = img_arr * np.expand_dims(mask_260, axis=-1)
                
                masked_img_arr = masked_img_arr.astype(np.uint8)
                
                out_path = output_class_dir / img_path.name
                Image.fromarray(masked_img_arr).save(out_path)
                
                total_processed += 1
                if total_processed % 500 == 0:
                    elapsed = time.time() - start_time
                    speed = elapsed / total_processed
                    remaining = (total_images - total_processed) * speed
                    print(f"   Processed {total_processed}/{total_images} ({total_processed/total_images*100:.1f}%) | "
                          f"Elapsed: {elapsed:.1f}s | Speed: {speed*1000:.1f}ms/img | Remaining: {remaining:.1f}s")
            except Exception as e:
                print(f"   Error procesando {img_path.name}: {e}")
                
    total_time = time.time() - start_time
    print(f"\nGENERACION COMPLETADA CON EXITO")
    print(f"   Total imagenes procesadas: {total_processed} de {total_images}")
    print(f"   Tiempo total: {total_time:.2f} segundos ({total_time/total_processed*1000:.1f} ms/imagen promedio)")
    print(f"   Dataset guardado en: {OUTPUT_ROOT}\n")

if __name__ == "__main__":
    main()
