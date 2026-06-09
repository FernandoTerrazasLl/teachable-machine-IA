import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import CLASS_NAMES, DATASET_LOCAL, OUTPUTS_DIR
# SOLO PARA GENERAR GRAFICOS
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

CLASS_COLORS = {
    'AK':  '#E74C3C',
    'BCC': '#3498DB',
    'MEL': '#2ECC71',
}

SAVE_DIR = OUTPUTS_DIR / "thesis_plots"
SAVE_DIR.mkdir(exist_ok=True, parents=True)


def load_onnx_models():
    unet_path = Path("../derma-scan-app/assets/models/fused_segmenter_unet.onnx")
    if not unet_path.exists():
        unet_path = Path("outputs/fused_segmenter_unet.onnx")
    clf_path = Path("../derma-scan-app/assets/models/fused_dermatology_model.onnx")
    if not clf_path.exists():
        clf_path = Path("outputs/fused_dermatology_model.onnx")
    
    unet_session = ort.InferenceSession(str(unet_path))
    clf_session = ort.InferenceSession(str(clf_path))
    return unet_session, clf_session


def preprocess_for_unet(image_pil):
    img_128 = image_pil.resize((128, 128))
    img_arr = np.array(img_128, dtype=np.float32) / 255.0
    img_chw = np.transpose(img_arr, (2, 0, 1))
    return np.expand_dims(img_chw, axis=0)


def morphological_closing(mask, radius=2):
    import torch
    import torch.nn.functional as F
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel_size = 2 * radius + 1
    padding = radius
    dilated = F.max_pool2d(mask_tensor, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=padding)
    return eroded.squeeze().cpu().numpy().astype(np.uint8)


def run_dual_pipeline_with_probs(image_pil, unet_session, clf_session, threshold=0.35):
    unet_input = preprocess_for_unet(image_pil)
    unet_outputs = unet_session.run(None, {'image_input': unet_input})
    mask_128 = unet_outputs[0][0][0]
    
    mask_binary = (mask_128 >= threshold).astype(np.uint8)
    mask_closed = morphological_closing(mask_binary, radius=2)
    
    img_260 = image_pil.resize((260, 260))
    img_arr = np.array(img_260, dtype=np.float32) / 255.0
    
    x_coords = (np.arange(260) * 128 / 260).astype(np.int32).clip(0, 127)
    y_coords = (np.arange(260) * 128 / 260).astype(np.int32).clip(0, 127)
    mask_260 = mask_closed[np.ix_(y_coords, x_coords)]
    
    masked_img = img_arr * np.expand_dims(mask_260, axis=-1)
    
    img_chw = np.transpose(masked_img, (2, 0, 1))
    clf_input = np.expand_dims(img_chw.flatten(), axis=0)
    
    clf_outputs = clf_session.run(None, {'raw_pixels': clf_input})
    probs = clf_outputs[0][0]
    predicted_idx = np.argmax(probs)
    
    return predicted_idx, probs


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_normalized[i, j] > 55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=13, fontweight='bold', color=color)
    
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=13, fontweight='bold')
    ax.set_yticklabels(CLASS_NAMES, fontsize=13, fontweight='bold')
    ax.set_xlabel("Prediccion del Modelo", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel("Diagnostico Real", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title("Matriz de Confusion — Pipeline Dual (Blind Test)", fontsize=14, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Porcentaje (%)", fontsize=12)
    plt.tight_layout()
    save_path = SAVE_DIR / "01_confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(y_true, y_probs):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 7))
    
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        color = CLASS_COLORS[class_name]
        ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{class_name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Aleatorio (AUC = 0.50)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR / Sensibilidad)", fontsize=13, fontweight='bold')
    ax.set_title("Curvas ROC por Clase — Pipeline Dual (Blind Test)", fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    plt.tight_layout()
    save_path = SAVE_DIR / "02_roc_curves.png"
    plt.savefig(save_path)
    plt.close()


def plot_learning_curves():
    history_path = OUTPUTS_DIR / "training_history.json"
    with open(history_path) as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(epochs, history['train_loss'], color='#3498DB', linewidth=2, marker='o', markersize=4, label='Entrenamiento')
    ax1.plot(epochs, history['val_loss'], color='#E74C3C', linewidth=2, marker='s', markersize=4, label='Validacion')
    ax1.axvline(x=5.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel("Epoca", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Perdida", fontsize=12, fontweight='bold')
    ax1.set_title("Funcion de Perdida por Epoca", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    
    ax2.plot(epochs, history['train_acc'], color='#3498DB', linewidth=2, marker='o', markersize=4, label='Entrenamiento')
    ax2.plot(epochs, history['val_acc'], color='#E74C3C', linewidth=2, marker='s', markersize=4, label='Validacion')
    ax2.axvline(x=5.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel("Epoca", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Exactitud (%)", fontsize=12, fontweight='bold')
    ax2.set_title("Exactitud por Epoca", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    save_path = SAVE_DIR / "03_learning_curves_classifier.png"
    plt.savefig(save_path)
    plt.close()


def plot_precision_recall_curves(y_true, y_probs):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 7))
    
    for i, class_name in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        color = CLASS_COLORS[class_name]
        ax.plot(recall, precision, color=color, linewidth=2.5, label=f'{class_name} (AP = {ap:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Sensibilidad (Recall)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Precision", fontsize=13, fontweight='bold')
    ax.set_title("Curvas de Precision-Recall por Clase — Pipeline Dual (Blind Test)", fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=12, framealpha=0.9)
    plt.tight_layout()
    save_path = SAVE_DIR / "04_precision_recall_curves.png"
    plt.savefig(save_path)
    plt.close()


def plot_confidence_distribution(y_true, y_pred, y_probs):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    for i, class_name in enumerate(CLASS_NAMES):
        ax = axes[i]
        mask = (y_true == i)
        correct_mask = mask & (y_pred == i)
        wrong_mask = mask & (y_pred != i)
        
        correct_conf = y_probs[correct_mask, i] if correct_mask.any() else []
        wrong_conf = y_probs[wrong_mask, i] if wrong_mask.any() else []
        
        data = []
        labels = []
        colors = []
        if len(correct_conf) > 0:
            data.append(correct_conf)
            labels.append(f'Acierto\n(n={len(correct_conf)})')
            colors.append('#2ECC71')
        if len(wrong_conf) > 0:
            data.append(wrong_conf)
            labels.append(f'Fallo\n(n={len(wrong_conf)})')
            colors.append('#E74C3C')
        
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylim([-0.05, 1.05])
        ax.set_ylabel("Confianza del Modelo" if i == 0 else "", fontsize=12, fontweight='bold')
        ax.set_title(f"{class_name}", fontsize=14, fontweight='bold', color=CLASS_COLORS[class_name])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    save_path = SAVE_DIR / "05_confidence_distribution.png"
    plt.savefig(save_path)
    plt.close()


def plot_metrics_table(y_true, y_pred, y_probs):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    rows = []
    for i, class_name in enumerate(CLASS_NAMES):
        mask_true = (y_true == i)
        mask_pred = (y_pred == i)
        
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        tn = np.sum(~mask_true & ~mask_pred)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        fpr_arr, tpr_arr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr_arr, tpr_arr)
        support = np.sum(mask_true)
        
        rows.append([class_name, f"{prec:.4f}", f"{rec:.4f}", f"{specificity:.4f}",
                      f"{f1:.4f}", f"{roc_auc:.4f}", f"{int(support)}"])
    
    macro_prec = precision_score(y_true, y_pred, average='macro')
    macro_rec = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    overall_acc = accuracy_score(y_true, y_pred)
    
    rows.append(["PROMEDIO", f"{macro_prec:.4f}", f"{macro_rec:.4f}", "—", f"{macro_f1:.4f}", "—", f"{len(y_true)}"])
    rows.append(["EXACTITUD GLOBAL", "", "", "", f"{overall_acc:.4f}", "", f"{len(y_true)}"])
    
    col_labels = ["Clase", "Precision", "Sensibilidad\n(Recall)", "Especificidad", "F1-Score", "AUC-ROC", "Soporte\n(n)"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.8)
    
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=11)
    
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i <= 3:
                cell.set_facecolor('#EBF5FB' if i % 2 == 1 else '#FDFEFE')
            elif i == 4:
                cell.set_facecolor('#D5F5E3')
                cell.set_text_props(fontweight='bold')
            else:
                cell.set_facecolor('#FCF3CF')
                cell.set_text_props(fontweight='bold')
    
    plt.tight_layout()
    save_path = SAVE_DIR / "06_metrics_table.png"
    plt.savefig(save_path)
    plt.close()


def plot_unet_note():
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = list(range(1, 21))
    train_iou = [0.759, 0.833, 0.850, 0.859, 0.865, 0.869, 0.873, 0.876, 0.878, 0.881,
                 0.882, 0.885, 0.888, 0.889, 0.892, 0.894, 0.895, 0.897, 0.898, 0.898]
    val_iou =   [0.825, 0.836, 0.819, 0.868, 0.862, 0.862, 0.868, 0.878, 0.880, 0.881,
                 0.869, 0.885, 0.886, 0.890, 0.890, 0.892, 0.892, 0.893, 0.894, 0.894]
    val_dice =  [0.894, 0.900, 0.890, 0.923, 0.918, 0.917, 0.922, 0.928, 0.930, 0.931,
                 0.924, 0.933, 0.933, 0.936, 0.936, 0.938, 0.937, 0.938, 0.938, 0.939]
    
    ax.plot(epochs, train_iou, color='#3498DB', linewidth=2.5, marker='o', markersize=5, label='IoU Entrenamiento')
    ax.plot(epochs, val_iou, color='#E74C3C', linewidth=2.5, marker='s', markersize=5, label='IoU Validacion')
    ax.plot(epochs, val_dice, color='#2ECC71', linewidth=2.5, marker='^', markersize=5, label='Dice Validacion', linestyle='--')
    ax.axhline(y=0.34, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='IoU U-Net Original (0.34)')
    ax.set_xlabel("Epoca", fontsize=13, fontweight='bold')
    ax.set_ylabel("Coeficiente", fontsize=13, fontweight='bold')
    ax.set_title("Curvas de Aprendizaje del Segmentador U-Net (HAM10000)", fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.set_ylim([0.3, 1.0])
    ax.set_xlim([0.5, 20.5])
    plt.tight_layout()
    save_path = SAVE_DIR / "07_unet_learning_curves.png"
    plt.savefig(save_path)
    plt.close()


def main():
    unet_session, clf_session = load_onnx_models()
    from src.data.loaders import create_dataloaders
    _, _, test_loader, _ = create_dataloaders(DATASET_LOCAL, batch_size=32)
    test_image_pairs = {(Path(img_path).parent.name, Path(img_path).name) 
                        for img_path, _ in test_loader.dataset.samples}
    
    raw_dataset_root = Path("/Users/terrazasllanosfernando/Desktop/Oficial")
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    all_true = []
    all_pred = []
    all_probs = []
    
    for class_name in CLASS_NAMES:
        class_dir = raw_dataset_root / class_name
        true_idx = class_to_idx[class_name]
        
        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            and (class_name, f.name) in test_image_pairs
        ])
        
        for img_path in image_files:
            try:
                image_pil = Image.open(img_path).convert("RGB")
                pred_idx, probs = run_dual_pipeline_with_probs(image_pil, unet_session, clf_session)
                all_true.append(true_idx)
                all_pred.append(pred_idx)
                all_probs.append(probs)
            except Exception:
                pass
    
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_probs = np.array(all_probs)
    
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curves(y_true, y_probs)
    plot_learning_curves()
    plot_precision_recall_curves(y_true, y_probs)
    plot_confidence_distribution(y_true, y_pred, y_probs)
    plot_metrics_table(y_true, y_pred, y_probs)
    plot_unet_note()


if __name__ == "__main__":
    main()
