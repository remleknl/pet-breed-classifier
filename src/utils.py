"""Ortak yardımcılar: cihaz seçimi, dönüşümler, checkpoint yükleme, tek görüntü tahmini."""
from __future__ import annotations

import random
import time

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from src.model import build_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224

try:  # iPhone HEIC fotoğrafları için (opsiyonel)
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Süre ölçümünden önce GPU işlemlerinin bitmesini bekle."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_transform():
    # Augmentation: telefon fotoğraflarındaki farklı açı, kadraj ve ışık koşullarına dayanıklılık için.
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(t: torch.Tensor) -> np.ndarray:
    """Normalize edilmiş tensörü (C,H,W) görselleştirme için (H,W,C) [0,1] dizisine çevir."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def load_image(path) -> Image.Image:
    """Telefon fotoğraflarındaki EXIF döndürme bilgisini uygulayarak aç."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def load_checkpoint(path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(len(ckpt["class_names"]), arch=ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["class_names"], ckpt


@torch.no_grad()
def predict_image(model, img: Image.Image, class_names, device, topk: int = 5):
    """Tek bir PIL görüntüsü için top-k (sınıf, olasılık) listesi ve tahmin süresi (ms) döndürür."""
    x = eval_transform()(img.convert("RGB")).unsqueeze(0).to(device)
    sync(device)
    t0 = time.perf_counter()
    probs = torch.softmax(model(x), dim=1)[0]
    sync(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    p, idx = probs.topk(topk)
    return [(class_names[i], float(v)) for v, i in zip(p.tolist(), idx.tolist())], elapsed_ms
