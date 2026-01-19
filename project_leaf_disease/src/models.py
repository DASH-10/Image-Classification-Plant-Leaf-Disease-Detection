"""
Bu dosyada kullanacagim modelleri ve optimizer/scheduler secimlerini topladim.
Kisaca: burada model kuruyorum, egitim ayarlarini belirliyorum.
"""
from typing import Optional

import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torchvision import models


def build_svm_pipeline(C=1.0, class_weight=None, max_iter: int = 5000) -> Pipeline:
    """Klasik SVM icin standardizasyon + LinearSVC boru hatti kurar."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(C=C, class_weight=class_weight, max_iter=max_iter)),
        ]
    )


def build_resnet18(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
    """ResNet18'i hazirlar; istenirse pretrained ve backbone dondurme yapar."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class SimpleCNN(nn.Module):
    """Basit bir CNN modeli (kendi yazdigim sade bir baslangic modeli)."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        """Ileri yayilim (forward) adimi."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_optimizer(model, lr: float, weight_decay: float, optimizer_name: str = "adam"):
    """Optimizer secimi (adam ya da sgd)."""
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, scheduler_name: Optional[str] = None):
    """Istege bagli ogrenme orani zamanlayicisi (scheduler)."""
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return None
