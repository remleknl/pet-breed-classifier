"""Test kümesinde değerlendirme: Accuracy, Precision, Recall, F1, confusion matrix, tahmin süresi.

Kullanım:
    python -m src.evaluate                                 # models/best_model.pth
    python -m src.evaluate --ckpt models/simple_cnn.pth
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, top_k_accuracy_score)
from tqdm import tqdm

from src.dataset import get_datasets, make_loader
from src.utils import denormalize, get_device, load_checkpoint, sync


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/best_model.pth")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--latency-runs", type=int, default=50)
    return p.parse_args()


@torch.no_grad()
def predict_all(model, loader, device):
    all_probs, all_labels, forward_time, n = [], [], 0.0, 0
    for x, y in tqdm(loader, desc="Test"):
        x = x.to(device)
        sync(device)
        t0 = time.perf_counter()
        probs = torch.softmax(model(x), dim=1)
        sync(device)
        forward_time += time.perf_counter() - t0
        n += x.size(0)
        all_probs.append(probs.cpu())
        all_labels.append(y)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy(), forward_time, n


@torch.no_grad()
def single_image_latency(model, x, device, runs):
    """Batch=1 için ortalama tahmin süresi (ms). İlk çağrılar ısınma olarak atlanır."""
    model.to(device)
    x = x.to(device)
    for _ in range(10):
        model(x)
    sync(device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def plot_confusion(cm, class_names, path):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Normalize Confusion Matrix (test)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(test_ds, idxs, y_true, y_pred, probs, class_names, path, n=16):
    idxs = idxs[:n]
    if len(idxs) == 0:
        return
    cols = 4
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.9))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), idxs):
        img, _ = test_ds[i]
        ax.imshow(denormalize(img))
        ax.set_title(f"G: {class_names[y_true[i]]}\nT: {class_names[y_pred[i]]} ({probs[i, y_pred[i]]:.0%})",
                     fontsize=8, color="crimson")
    fig.suptitle("Yanlış sınıflandırılan test örnekleri (G=Gerçek, T=Tahmin)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    device = get_device(args.device)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.ckpt).stem

    model, class_names, ckpt = load_checkpoint(args.ckpt, device)
    _, _, test_ds, _ = get_datasets(args.data_dir)
    loader = make_loader(test_ds, args.batch_size, False, args.num_workers, device)

    probs, y_true, forward_time, n = predict_all(model, loader, device)
    y_pred = probs.argmax(1)

    acc = accuracy_score(y_true, y_pred)
    top5 = top_k_accuracy_score(y_true, probs, k=5, labels=np.arange(len(class_names)))
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    (results_dir / f"{tag}_classification_report.txt").write_text(report_txt)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, class_names, results_dir / f"{tag}_confusion_matrix.png")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0][:10]
    confused = [{"gercek": class_names[i], "tahmin": class_names[j], "adet": int(off[i, j])}
                for i, j in pairs if off[i, j] > 0]

    wrong = np.where(y_pred != y_true)[0]
    rng = np.random.default_rng(0)
    plot_misclassified(test_ds, rng.permutation(wrong), y_true, y_pred, probs, class_names,
                       results_dir / f"{tag}_misclassified.png")

    sample, _ = test_ds[0]
    sample = sample.unsqueeze(0)
    lat_dev = single_image_latency(model, sample, device, args.latency_runs)
    lat_cpu = single_image_latency(model, sample, torch.device("cpu"), args.latency_runs) \
        if device.type != "cpu" else lat_dev

    metrics = {
        "checkpoint": args.ckpt,
        "arch": ckpt["arch"],
        "test_samples": int(n),
        "accuracy": acc,
        "top5_accuracy": top5,
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f1_m,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f1_w,
        "inference": {
            "device": str(device),
            "batch_size": args.batch_size,
            "throughput_img_per_s": n / forward_time,
            f"latency_ms_batch1_{device.type}": {"mean": lat_dev[0], "std": lat_dev[1]},
            "latency_ms_batch1_cpu": {"mean": lat_cpu[0], "std": lat_cpu[1]},
        },
        "most_confused_pairs": confused,
    }
    with open(results_dir / f"{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(report_txt)
    print(f"Accuracy: {acc:.4f} | Top-5: {top5:.4f} | Macro P/R/F1: {p_m:.4f}/{r_m:.4f}/{f1_m:.4f}")
    print(f"Throughput ({device}): {n / forward_time:.1f} görüntü/s")
    print(f"Tek görüntü süresi: {device.type}={lat_dev[0]:.1f} ms, cpu={lat_cpu[0]:.1f} ms")
    print("En çok karıştırılanlar:")
    for c in confused[:5]:
        print(f"  {c['gercek']} -> {c['tahmin']}: {c['adet']}")
    print(f"Sonuçlar: {results_dir}/{tag}_*")


if __name__ == "__main__":
    main()
