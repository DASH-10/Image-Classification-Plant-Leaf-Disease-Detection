"""
Bu dosya projenin temel ayarlari icin tek bir merkez.
Ben burada yol, veri boyutu ve egitim ayarlarini topluyorum ki her yerde aramayayim.
"""
from pathlib import Path

# Proje klasor yollarini bir arada tutuyorum (tek noktadan degistirmek icin)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
CACHE_DIR = OUTPUTS_DIR / "cache"

# Tekrar edilebilirlik icin sabit tohum (seed)
RANDOM_SEED = 42

# Cihaz secimi: Torch varsa GPU, yoksa CPU kullanmaya calisiyorum
try:
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = "cpu"

# Veri parametreleri (resim boyutlari ve bolme oranlari)
IMG_SIZE = 224
BASELINE_IMG_SIZE = 128
CLASSICAL_IMAGE_SIZE = BASELINE_IMG_SIZE
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# DataLoader parametreleri (batch ve is parcacigi sayisi)
BATCH_SIZE = 32
NUM_WORKERS = 2

# Veri seti kaynaklari (TFDS ve Kaggle isimleri)
TFDS_DATASET_NAMES = ["plant_village", "plant_village/plantvillage"]
KAGGLE_DATASET = "emmarex/plantdisease"

# HOG parametreleri (geleneksel ozellik cikarimi icin)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}

# Klasik baseline ayarlari (SVM + PCA gibi)
PCA_COMPONENTS = 256
SEARCH_N_ITER = 3
CV_FOLDS = 3

# Egitim varsayilanlari (epoch, ogrenme orani vb.)
DEFAULT_NUM_EPOCHS = 10
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
