"""
Bu dosyada grafik ve gorsellestirme islerini yapiyorum.
Amacim: veri dagilimi, egitim suresi, hata ornekleri gibi seyleri gozle gormek.
"""
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def plot_class_distribution(labels: List[str], output_path: Path) -> None:
    """Sinif sayilarini bar grafik olarak kaydeder."""
    classes, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(12, 4))
    plt.bar(classes, counts, color="#4C78A8")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel("Count")
    plt.title("Class Distribution")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_image_grid(image_paths: List[Path], output_path: Path, grid_size: Tuple[int, int] = (4, 4)) -> None:
    """Ornek resimleri bir grid halinde kaydeder."""
    rows, cols = grid_size
    plt.figure(figsize=(cols * 3, rows * 3))
    for idx, img_path in enumerate(image_paths[: rows * cols]):
        img = Image.open(img_path).convert("RGB")
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path, normalize: bool = False) -> None:
    """Confusion matrix'i (normal veya normalize) kaydeder."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_learning_curves(history: Dict, output_path: Path) -> None:
    """Egitim ve dogrulama kayip/performans egrilerini cizer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.get("train_loss", []), label="train_loss")
    axes[0].plot(history.get("val_loss", []), label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.get("train_macro_f1", []), label="train_macro_f1")
    axes[1].plot(history.get("val_macro_f1", []), label="val_macro_f1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_misclassifications(
    samples: List[Tuple[str, int]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
    max_images: int = 20,
) -> None:
    """Yanlis tahmin edilen ornekleri kaydeder (gorsel kontrol icin)."""
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        return

    pair_counts = Counter(zip(y_true[mis_idx], y_pred[mis_idx]))
    sorted_pairs = [pair for pair, _ in pair_counts.most_common()]

    selected = []
    for true_idx, pred_idx in sorted_pairs:
        pair_indices = mis_idx[(y_true[mis_idx] == true_idx) & (y_pred[mis_idx] == pred_idx)]
        for idx in pair_indices:
            selected.append(idx)
            if len(selected) >= max_images:
                break
        if len(selected) >= max_images:
            break

    cols = 5
    rows = int(np.ceil(len(selected) / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(selected):
        img_path, _ = samples[idx]
        img = Image.open(img_path).convert("RGB")
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"T:{class_names[y_true[idx]]}\nP:{class_names[y_pred[idx]]}")
        plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pr_curves(
    pr_curves: Dict[int, Dict[str, np.ndarray]],
    class_names: List[str],
    output_path: Path,
    micro_curve: Dict[str, np.ndarray] = None,
    max_classes: int = 6,
) -> None:
    """Precision-Recall egrilerini kaydeder."""
    plt.figure(figsize=(7, 5))
    for idx, (class_id, curve) in enumerate(pr_curves.items()):
        if idx >= max_classes:
            break
        plt.plot(curve["recall"], curve["precision"], label=class_names[class_id])
    if micro_curve:
        plt.plot(micro_curve["recall"], micro_curve["precision"], label="micro", linestyle="--")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


class GradCAM:
    """Grad-CAM ile modelin dikkat ettigi bolgeleri cikarir."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Layer'a hook baglayip gradient ve activation yakalar."""
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx: int = None):
        """Grad-CAM haritasi uretir (modelin dikkat ettigi alanlar)."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())
        loss = output[0, class_idx]
        loss.backward()

        # Grad'lerden agirlik hesaplayip activations uzerine uygularim
        grads = self.gradients
        activations = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()


def overlay_gradcam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Grad-CAM haritasini orijinal resmin ustune bindirir."""
    import matplotlib.cm as cm

    cam_resized = Image.fromarray((cam * 255).astype("uint8")).resize(
        (image.shape[1], image.shape[0])
    )
    cam_resized = np.array(cam_resized) / 255.0
    heatmap = cm.jet(cam_resized)[:, :, :3]
    overlay = (1 - alpha) * (image / 255.0) + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype("uint8")


def save_gradcam_examples(
    model,
    dataset,
    device,
    output_path: Path,
    target_layer,
    num_images: int = 4,
) -> None:
    """Ornek Grad-CAM gorselleri uretip kaydeder."""
    import torch

    cam_generator = GradCAM(model, target_layer)
    plt.figure(figsize=(num_images * 3, 3))
    for idx in range(num_images):
        img, _ = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)
        cam = cam_generator.generate(input_tensor)
        img_np = np.transpose(img.numpy(), (1, 2, 0))
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        overlay = overlay_gradcam((img_np * 255).astype("uint8"), cam)
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(overlay)
        plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
