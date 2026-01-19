"""
Bu dosyada rastgelelik ayarlarini tek yerde topluyorum.
Amacim: sonucu her calistirmada ayni gorebileyim.
"""
import os
import random
import platform
from typing import Dict

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int) -> None:
    """Tum kutuphaneler icin seed ayarlar (tekrar edilebilirlik icin)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker'lari icin seed belirler."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_env_info() -> Dict[str, str]:
    """Python, OS ve Torch bilgilerini basit bir sozluk olarak dondurur."""
    info = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
    }
    if torch is not None:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["gpu_name"] = torch.cuda.get_device_name(0)
    return info
