"""
Bu dosyada veri setini hazirlama, bolme ve yukleme islerini yapiyorum.
Amacim: resimleri dogru klasorlere koymak ve egitime hazir hale getirmek.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import (
    DATA_DIR,
    KAGGLE_DATASET,
    PROCESSED_DIR,
    RAW_DIR,
    RANDOM_SEED,
    TEST_SPLIT,
    TFDS_DATASET_NAMES,
    TRAIN_SPLIT,
    VAL_SPLIT,
)


# Kabul ettigim resim uzantilari
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _ensure_dirs() -> None:
    """Gerekli klasorleri olusturur (yoksa yaratir)."""
    for path in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _has_class_subdirs(root: Path) -> bool:
    """Klasorun altinda sinif klasorleri ve resimler var mi kontrol eder."""
    if not root.exists() or not root.is_dir():
        return False
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if len(subdirs) < 2:
        return False
    has_images = 0
    for d in subdirs:
        if any(p.suffix.lower() in IMAGE_EXTS for p in d.iterdir() if p.is_file()):
            has_images += 1
    return has_images >= 2


def _select_image_root(raw_root: Path) -> Optional[Path]:
    """Ham verinin icinde resimlerin oldugu en uygun klasoru bulur."""
    if _has_class_subdirs(raw_root):
        return raw_root
    candidates = [d for d in raw_root.glob("**/*") if d.is_dir()]
    best = None
    best_score = 0
    for cand in candidates:
        if _has_class_subdirs(cand):
            score = len([d for d in cand.iterdir() if d.is_dir()])
            if score > best_score:
                best = cand
                best_score = score
    return best


def _collect_image_paths(root: Path) -> Tuple[List[Path], List[str]]:
    """Sinif klasorlerinden resim yollarini ve etiketlerini listeler."""
    paths = []
    labels = []
    for class_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        class_name = class_dir.name
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTS:
                paths.append(img_path)
                labels.append(class_name)
    return paths, labels


def _copy_split(
    split_name: str,
    paths: List[Path],
    labels: List[str],
    dest_root: Path,
) -> None:
    """Resimleri train/val/test klasorlerine kopyalar."""
    for path, label in tqdm(list(zip(paths, labels)), desc=f"Copying {split_name}"):
        dest_dir = dest_root / split_name / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / path.name
        if not dest_path.exists():
            shutil.copy2(path, dest_path)


def _save_split_metadata(
    dest_root: Path,
    split_counts: Dict[str, Dict[str, int]],
    class_names: List[str],
) -> None:
    """Sinif sayilari ve bolmelerin ozetini JSON olarak kaydeder."""
    payload = {
        "splits": split_counts,
        "classes": class_names,
    }
    with (dest_root / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _prepare_processed_from_root(raw_root: Path) -> Path:
    """Ham veri klasorunu alir ve processed altina duzenli bolmeler cikarir."""
    # Veri seti zaten train/val/test veriyorsa onu tekrar kullanmaya calisiyorum
    train_dir = raw_root / "train"
    val_dir = raw_root / "validation"
    test_dir = raw_root / "test"

    if _has_class_subdirs(train_dir) and _has_class_subdirs(test_dir):
        split_map = {"train": train_dir, "val": val_dir, "test": test_dir}
        if not _has_class_subdirs(val_dir):
            # Validation yoksa train icinden val ayiriyorum
            train_paths, train_labels = _collect_image_paths(train_dir)
            tr_paths, va_paths, tr_labels, va_labels = train_test_split(
                train_paths,
                train_labels,
                test_size=VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT),
                stratify=train_labels,
                random_state=RANDOM_SEED,
            )
            for split in ["train", "val", "test"]:
                split_path = PROCESSED_DIR / split
                if split_path.exists():
                    shutil.rmtree(split_path)
            _copy_split("train", tr_paths, tr_labels, PROCESSED_DIR)
            _copy_split("val", va_paths, va_labels, PROCESSED_DIR)
            test_paths, test_labels = _collect_image_paths(test_dir)
            _copy_split("test", test_paths, test_labels, PROCESSED_DIR)
            class_names = sorted(list(set(tr_labels + va_labels + test_labels)))
            split_counts = {
                "train": {cls: tr_labels.count(cls) for cls in class_names},
                "val": {cls: va_labels.count(cls) for cls in class_names},
                "test": {cls: test_labels.count(cls) for cls in class_names},
            }
            _save_split_metadata(PROCESSED_DIR, split_counts, class_names)
            return PROCESSED_DIR

        for split in ["train", "val", "test"]:
            split_path = PROCESSED_DIR / split
            if split_path.exists():
                shutil.rmtree(split_path)

        for split, src in split_map.items():
            if not src.exists():
                continue
            paths, labels = _collect_image_paths(src)
            _copy_split(split, paths, labels, PROCESSED_DIR)

        class_names = sorted([d.name for d in (PROCESSED_DIR / "train").iterdir() if d.is_dir()])
        split_counts = {}
        for split in ["train", "val", "test"]:
            _, labels = _collect_image_paths(PROCESSED_DIR / split)
            split_counts[split] = {cls: labels.count(cls) for cls in class_names}
        _save_split_metadata(PROCESSED_DIR, split_counts, class_names)
        return PROCESSED_DIR

    image_root = _select_image_root(raw_root)
    if image_root is None:
        raise FileNotFoundError(
            f"No class folders with images found under {raw_root}."
        )

    paths, labels = _collect_image_paths(image_root)
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under {image_root}.")

    # once egitim, sonra val/test olacak sekilde bolme yapiyorum
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=(1 - TRAIN_SPLIT),
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )

    for split in ["train", "val", "test"]:
        split_dir = PROCESSED_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    _copy_split("train", train_paths, train_labels, PROCESSED_DIR)
    _copy_split("val", val_paths, val_labels, PROCESSED_DIR)
    _copy_split("test", test_paths, test_labels, PROCESSED_DIR)

    class_names = sorted(list(set(labels)))
    split_counts = {
        "train": {cls: train_labels.count(cls) for cls in class_names},
        "val": {cls: val_labels.count(cls) for cls in class_names},
        "test": {cls: test_labels.count(cls) for cls in class_names},
    }
    _save_split_metadata(PROCESSED_DIR, split_counts, class_names)
    return PROCESSED_DIR


def _export_tfds_to_raw() -> Optional[Path]:
    """TFDS varsa veri setini indirip data/raw altina cikartir."""
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        return None

    dataset_name = None
    builder = None
    for name in TFDS_DATASET_NAMES:
        try:
            builder = tfds.builder(name)
            dataset_name = name
            break
        except Exception:
            continue

    if builder is None:
        return None

    builder.download_and_prepare()
    label_names = builder.info.features["label"].names
    splits = list(builder.info.splits.keys())

    raw_export_root = RAW_DIR / "tfds_export"
    raw_export_root.mkdir(parents=True, exist_ok=True)

    # TFDS bolmeleri varsa onlari disari aktaracagim
    if {"train", "validation", "test"}.issubset(set(splits)):
        export_splits = ["train", "validation", "test"]
    elif "train" in splits and "test" in splits:
        export_splits = ["train", "test"]
    else:
        export_splits = ["train"]

    for split_name in export_splits:
        ds = tfds.load(
            dataset_name,
            split=split_name,
            as_supervised=True,
            data_dir=builder.data_dir,
        )
        for idx, (image, label) in tqdm(
            enumerate(tfds.as_numpy(ds)), desc=f"Exporting TFDS {split_name}"
        ):
            class_name = label_names[int(label)]
            out_dir = raw_export_root / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{class_name}_{idx}.jpg"
            if not out_path.exists():
                img = Image.fromarray(image)
                img.save(out_path, format="JPEG", quality=95)

    return raw_export_root


def _has_kaggle_credentials() -> bool:
    """Kaggle indirme icin kimlik var mi kontrol eder."""
    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"
    env_ready = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    return kaggle_json.exists() or env_ready


def _download_kaggle_dataset() -> Optional[Path]:
    """Kaggle'dan veri setini indirmeye calisir (kimlik varsa)."""
    if not _has_kaggle_credentials():
        return None

    target_dir = RAW_DIR / "kaggle"
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "dataset.zip"

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(target_dir),
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return None

    if zip_path.exists():
        zip_path.unlink()

    return target_dir


def prepare_dataset() -> Path:
    """Ana giris: veri yoksa indir/duzenle, varsa direkt processed klasorunu dondur."""
    _ensure_dirs()
    if (PROCESSED_DIR / "train").exists():
        return PROCESSED_DIR

    raw_root = None

    # Sirayla TFDS -> Kaggle -> yerel raw klasoru deniyorum
    tfds_root = _export_tfds_to_raw()
    if tfds_root is not None:
        raw_root = tfds_root
    else:
        kaggle_root = _download_kaggle_dataset()
        if kaggle_root is not None:
            raw_root = kaggle_root
        else:
            if _select_image_root(RAW_DIR) is not None:
                raw_root = RAW_DIR

    if raw_root is None:
        raise FileNotFoundError(
            "No dataset found. Use TFDS (plant_village) or download from Kaggle "
            f"({KAGGLE_DATASET}) and place under data/raw."
        )

    return _prepare_processed_from_root(raw_root)


def get_class_names(processed_root: Path) -> List[str]:
    """Processed klasorunden sinif isimlerini okur."""
    if (processed_root / "splits.json").exists():
        with (processed_root / "splits.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("classes", [])
    train_root = processed_root / "train"
    return sorted([d.name for d in train_root.iterdir() if d.is_dir()])


def get_split_paths(processed_root: Path, split: str) -> Tuple[List[Path], List[str]]:
    """Belirli bir split (train/val/test) icin resim yolu ve etiket listesi verir."""
    split_root = processed_root / split
    paths, labels = _collect_image_paths(split_root)
    return paths, labels


def get_torch_datasets(processed_root: Path, train_tfms, val_tfms):
    """Torch ImageFolder ile train/val/test datasetlerini olusturur."""
    from torchvision import datasets

    train_ds = datasets.ImageFolder(processed_root / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(processed_root / "val", transform=val_tfms)
    test_ds = datasets.ImageFolder(processed_root / "test", transform=val_tfms)
    return train_ds, val_ds, test_ds


def get_dataloaders(processed_root: Path, train_tfms, val_tfms, batch_size, num_workers):
    """DataLoader hazirlar; egitim/validasyon/test icin ayri loader dondurur."""
    from torch.utils.data import DataLoader
    from .utils_seed import seed_worker

    train_ds, val_ds, test_ds = get_torch_datasets(processed_root, train_tfms, val_tfms)

    # PyTorch seed ayari (tekrarlanabilirlik icin)
    g = None
    try:
        import torch

        g = torch.Generator()
        g.manual_seed(RANDOM_SEED)
    except Exception:
        g = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
