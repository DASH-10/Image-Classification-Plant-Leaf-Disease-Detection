"""
Bu dosyada resimlerden ozellik cikariyorum (HOG + renk histogrami).
Yeni baslayan biri icin: bu ozellikler makineye resmin sayisal ozetini verir.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import hashlib
import json

import numpy as np
from PIL import Image, ImageFile
from skimage.feature import hog
from tqdm import tqdm

from .config import CACHE_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _log_bad_image(path: Path, exc: Exception, log_path: Path) -> None:
    """Bozuk resimleri log dosyasina ekler."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{path}\t{type(exc).__name__}: {exc}\n")


def _load_image(
    path: Path,
    size: int,
    grayscale: bool = False,
    bad_log_path: Optional[Path] = None,
    raise_on_error: bool = False,
) -> Optional[np.ndarray]:
    """Resmi okur, boyutunu ayarlar ve numpy dizisine cevirir."""
    mode = "L" if grayscale else "RGB"
    try:
        with Image.open(path) as img:
            img = img.convert(mode)
            img = img.resize((size, size))
            return np.array(img)
    except Exception as exc:
        if bad_log_path is not None:
            _log_bad_image(path, exc, bad_log_path)
        if raise_on_error:
            raise
        return None


def extract_color_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """RGB kanallarindan renk histogrami cikarir."""
    hist_features = []
    for channel in range(3):
        hist, _ = np.histogram(
            image[:, :, channel],
            bins=bins,
            range=(0, 256),
            density=True,
        )
        hist_features.append(hist.astype(np.float32, copy=False))
    return np.concatenate(hist_features).astype(np.float32, copy=False)


def extract_hog_features(
    image: np.ndarray,
    hog_params: Dict,
) -> np.ndarray:
    """HOG (kenar/yon bilgisi) ozelliklerini cikarir."""
    if image.ndim == 2:
        gray = image.astype("uint8")
    else:
        gray = np.mean(image, axis=2).astype("uint8")
    features = hog(
        gray,
        orientations=hog_params["orientations"],
        pixels_per_cell=hog_params["pixels_per_cell"],
        cells_per_block=hog_params["cells_per_block"],
        block_norm=hog_params["block_norm"],
        feature_vector=True,
    )
    return features.astype(np.float32, copy=False)


def _hash_image_paths(image_paths: List[Path]) -> str:
    """Resim yollarini ve mtime degerlerini hash'ler (deterministik)."""
    entries = []
    for path in sorted(image_paths, key=lambda p: str(p)):
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = 0
        path_str = str(Path(path).resolve(strict=False))
        entries.append(f"{path_str}|{mtime}")
    payload = "\n".join(entries)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_cache_key(
    image_paths: List[Path],
    image_size: int,
    hog_params: Dict,
    use_color_hist: bool,
    color_hist_bins: int,
) -> str:
    """HOG ozellik cache anahtari icin hash uretir."""
    paths_hash = _hash_image_paths(image_paths)
    params_payload = {
        "image_size": image_size,
        "hog_params": hog_params,
        "use_color_hist": use_color_hist,
        "color_hist_bins": color_hist_bins,
    }
    params_blob = json.dumps(params_payload, sort_keys=True)
    key_blob = f"{paths_hash}|{params_blob}"
    return hashlib.sha256(key_blob.encode("utf-8")).hexdigest()


def _read_cached_paths(paths_path: Path) -> List[str]:
    """Cache'ten kaydedilen path listesini okur."""
    if not paths_path.exists():
        return []
    return [
        line.strip()
        for line in paths_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _order_hash(path_list: List[str]) -> str:
    """Path sirasi icin hash uretir (reorder cache)."""
    payload = "\n".join(path_list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_feature_matrix(
    image_paths: List[Path],
    image_size: int,
    hog_params: Dict,
    use_color_hist: bool = True,
    labels: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
    use_memmap: bool = True,
    color_hist_bins: int = 32,
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """Birden cok resim icin tek bir ozellik matrisi olusturur (cache destekli)."""
    if labels is not None and len(labels) != len(image_paths):
        raise ValueError("labels length must match image_paths length")

    image_paths = [Path(p) for p in image_paths]
    cache_root = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = _make_cache_key(
        image_paths=image_paths,
        image_size=image_size,
        hog_params=hog_params,
        use_color_hist=use_color_hist,
        color_hist_bins=color_hist_bins,
    )
    cache_subdir = cache_root / f"hog_{cache_key}"
    features_path = cache_subdir / "features.npy"
    meta_path = cache_subdir / "meta.json"
    paths_path = cache_subdir / "paths.txt"
    bad_log_path = cache_root / "bad_images.txt"

    if not force_recompute and features_path.exists() and meta_path.exists() and paths_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mmap_mode = "r" if use_memmap else None
        features = np.load(features_path, mmap_mode=mmap_mode, allow_pickle=False)
        valid_count = int(meta.get("valid_count", features.shape[0]))
        valid_count = min(valid_count, features.shape[0])
        features = features[:valid_count]

        cached_paths = _read_cached_paths(paths_path)
        cached_paths = cached_paths[:valid_count]
        input_paths_resolved = [str(p.resolve(strict=False)) for p in image_paths]
        path_to_idx = {p: idx for idx, p in enumerate(cached_paths)}

        keep_indices = []
        kept_labels = []
        kept_paths = []
        for idx, path_str in enumerate(input_paths_resolved):
            src_idx = path_to_idx.get(path_str)
            if src_idx is None:
                continue
            keep_indices.append(src_idx)
            kept_paths.append(image_paths[idx])
            if labels is not None:
                kept_labels.append(labels[idx])

        order_matches = (
            len(keep_indices) == len(cached_paths)
            and all(i == src_idx for i, src_idx in enumerate(keep_indices))
        )

        if order_matches:
            print(f"Loaded features from cache: {features_path}")
            if labels is not None:
                return features, kept_labels
            return features

        order_paths = [str(p.resolve(strict=False)) for p in kept_paths]
        order_hash = _order_hash(order_paths)
        ordered_features_path = cache_subdir / f"features_ordered_{order_hash}.npy"
        if ordered_features_path.exists() and not force_recompute:
            features = np.load(ordered_features_path, mmap_mode=mmap_mode, allow_pickle=False)
            print(f"Loaded reordered cache: {ordered_features_path}")
            if labels is not None:
                return features, kept_labels
            return features

        if use_memmap:
            ordered = np.lib.format.open_memmap(
                ordered_features_path,
                mode="w+",
                dtype=features.dtype,
                shape=(len(keep_indices), features.shape[1]),
            )
            for out_idx, src_idx in enumerate(keep_indices):
                ordered[out_idx] = features[src_idx]
            ordered.flush()
            print(f"Reordered cached features: {ordered_features_path}")
            if labels is not None:
                return ordered, kept_labels
            return ordered

        reordered = features[keep_indices].astype(np.float32, copy=False)
        print(f"Reordered cached features in memory: {features_path}")
        if labels is not None:
            return reordered, kept_labels
        return reordered

    cache_subdir.mkdir(parents=True, exist_ok=True)
    mmap = None
    kept_paths = []
    kept_labels = []
    write_idx = 0
    feature_dim = None
    grayscale = not use_color_hist

    for idx, path in enumerate(tqdm(image_paths, desc="Extracting features")):
        img = _load_image(path, image_size, grayscale=grayscale, bad_log_path=bad_log_path)
        if img is None:
            continue
        hog_feat = extract_hog_features(img, hog_params)
        if use_color_hist:
            color_feat = extract_color_histogram(img, bins=color_hist_bins)
            feat = np.concatenate([hog_feat, color_feat])
        else:
            feat = hog_feat
        feat = feat.astype(np.float32, copy=False)

        if mmap is None:
            feature_dim = feat.shape[0]
            mmap = np.lib.format.open_memmap(
                features_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(image_paths), feature_dim),
            )

        mmap[write_idx] = feat
        write_idx += 1
        kept_paths.append(path)
        if labels is not None:
            kept_labels.append(labels[idx])

    if mmap is None or feature_dim is None:
        raise ValueError("No valid images found for feature extraction.")

    mmap.flush()

    with paths_path.open("w", encoding="utf-8") as f:
        for path in kept_paths:
            f.write(f"{Path(path).resolve(strict=False)}\n")

    meta = {
        "image_size": image_size,
        "hog_params": hog_params,
        "use_color_hist": use_color_hist,
        "color_hist_bins": color_hist_bins,
        "feature_dim": feature_dim,
        "total_count": len(image_paths),
        "valid_count": write_idx,
        "cache_key": cache_key,
    }
    if labels is not None:
        _, class_to_idx = encode_labels(kept_labels)
        meta["class_to_idx"] = class_to_idx

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    features = mmap[:write_idx]
    print(f"Saved features to cache: {features_path}")

    if not use_memmap:
        features = np.load(features_path, allow_pickle=False)[:write_idx]

    if labels is not None:
        return features, kept_labels
    return features


def build_feature_memmap(
    image_paths: List[Path],
    image_size: int,
    hog_params: Dict,
    use_color_hist: bool,
    mmap_path: Path,
    dtype: np.dtype = np.float32,
) -> np.memmap:
    """Ozellikleri diske memmap olarak yazip RAM tasmasini azaltir."""
    if not image_paths:
        raise ValueError("image_paths is empty")

    mmap_path = Path(mmap_path)
    mmap_path.parent.mkdir(parents=True, exist_ok=True)
    grayscale = not use_color_hist
    bad_log_path = CACHE_DIR / "bad_images.txt"
    mmap = None
    write_idx = 0
    feature_dim = None

    for path in tqdm(image_paths, desc="Extracting features"):
        img = _load_image(path, image_size, grayscale=grayscale, bad_log_path=bad_log_path)
        if img is None:
            continue
        hog_feat = extract_hog_features(img, hog_params)
        if use_color_hist:
            color_feat = extract_color_histogram(img)
            feat = np.concatenate([hog_feat, color_feat])
        else:
            feat = hog_feat
        feat = feat.astype(dtype, copy=False)

        if mmap is None:
            feature_dim = feat.shape[0]
            mmap = np.memmap(
                mmap_path,
                mode="w+",
                dtype=dtype,
                shape=(len(image_paths), feature_dim),
            )

        mmap[write_idx] = feat
        write_idx += 1

    if mmap is None or feature_dim is None:
        raise ValueError("No valid images found for memmap extraction.")

    mmap.flush()
    return mmap[:write_idx]


def encode_labels(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Etiketleri sayiya cevirir, sinif->index sozlugu dondurur."""
    class_names = sorted(list(set(labels)))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = np.array([class_to_idx[label] for label in labels])
    return y, class_to_idx
