"""Model tanımları: ImageNet ön-eğitimli ResNet (transfer learning) ve karşılaştırma için basit CNN."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

ARCHS = ("resnet50", "resnet18", "simple_cnn")


class SimpleCNN(nn.Module):
    """Sıfırdan eğitilen küçük CNN (baseline). Transfer learning'in katkısını göstermek için."""

    def __init__(self, num_classes: int):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.fc(self.pool(self.features(x)))


def build_model(num_classes: int, arch: str = "resnet50", pretrained: bool = True) -> nn.Module:
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif arch == "simple_cnn":
        return SimpleCNN(num_classes)
    else:
        raise ValueError(f"Bilinmeyen mimari: {arch}. Seçenekler: {ARCHS}")

    # Son katmanı (1000 ImageNet sınıfı) veri setimizin sınıf sayısına göre değiştir.
    # Çıkış ham logit'tir; softmax tahmin anında uygulanır
    # (CrossEntropyLoss kendi içinde log-softmax uygular).
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """fc dışındaki tüm katmanları dondur / çöz."""
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = trainable
