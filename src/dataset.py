"""Oxford-IIIT Pet veri seti: eğitim / doğrulama / test ayrımı ve DataLoader'lar.

- Resmi 'trainval' kümesi stratified olarak eğitim (%85) ve doğrulama (%15) olarak bölünür.
- Resmi 'test' kümesi yalnızca son değerlendirmede kullanılır.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet

from src.utils import eval_transform, train_transform


def get_datasets(data_dir: str = "data", val_ratio: float = 0.15, seed: int = 42):
    # Aynı görüntüler, iki farklı dönüşümle: eğitim (augmentation'lı) ve doğrulama (augmentation'sız).
    trainval_aug = OxfordIIITPet(data_dir, split="trainval", target_types="category",
                                 transform=train_transform(), download=True)
    trainval_plain = OxfordIIITPet(data_dir, split="trainval", target_types="category",
                                   transform=eval_transform(), download=False)
    test_ds = OxfordIIITPet(data_dir, split="test", target_types="category",
                            transform=eval_transform(), download=True)

    labels = np.array(trainval_aug._labels)
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=val_ratio, stratify=labels, random_state=seed
    )
    train_ds = Subset(trainval_aug, train_idx.tolist())
    val_ds = Subset(trainval_plain, val_idx.tolist())
    return train_ds, val_ds, test_ds, list(trainval_aug.classes)


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, device: torch.device):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def summarize(train_ds, val_ds, test_ds, class_names) -> dict:
    return {
        "num_classes": len(class_names),
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
    }
