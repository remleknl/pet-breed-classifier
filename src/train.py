"""Model eğitimi.

İki aşamalı transfer learning:
  1) Backbone dondurulur, sadece son katman (fc) eğitilir   -> --head-epochs
  2) Tüm ağ düşük learning rate ile fine-tune edilir         -> --ft-epochs
simple_cnn seçilirse ön-eğitim olmadığı için doğrudan tüm ağ eğitilir.

Kullanım:
    python -m src.train                       # ResNet-50 (varsayılan)
    python -m src.train --arch resnet18       # daha hızlı
    python -m src.train --arch simple_cnn --ft-epochs 30 --ft-lr 1e-3 --out models/simple_cnn.pth
    python -m src.train --limit-batches 5     # hızlı deneme (her epoch'ta 5 batch)
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import get_datasets, make_loader, summarize
from src.model import ARCHS, build_model, set_backbone_trainable
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Oxford-IIIT Pet sınıflandırıcı eğitimi")
    p.add_argument("--arch", default="resnet50", choices=ARCHS)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="models/best_model.pth")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=3, help="Aşama 1: sadece fc katmanı")
    p.add_argument("--ft-epochs", type=int, default=12, help="Aşama 2: tüm ağ")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--ft-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=4, help="Early stopping (val acc iyileşmezse)")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--limit-batches", type=int, default=0, help="Hata ayıklama: epoch başına batch sınırı")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, limit_batches=0, desc=""):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(tqdm(loader, desc=desc, leave=False)):
            if limit_batches and i >= limit_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def plot_history(history, stage_switch, path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in [(axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")]:
        ax.plot(epochs, [h[f"train_{key}"] for h in history], "o-", label="Eğitim")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], "o-", label="Doğrulama")
        if stage_switch:
            ax.axvline(stage_switch + 0.5, color="gray", ls="--", lw=1, label="Fine-tune başlangıcı")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tag = Path(args.out).stem  # aynı results klasöründe birden fazla model sonucu tutabilmek için

    train_ds, val_ds, test_ds, class_names = get_datasets(args.data_dir, args.val_ratio, args.seed)
    split_info = summarize(train_ds, val_ds, test_ds, class_names)
    print(f"Cihaz: {device} | Veri: {split_info}")
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, device)

    pretrained = args.arch != "simple_cnn"
    model = build_model(len(class_names), args.arch, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    stages = []
    if pretrained and args.head_epochs > 0:
        stages.append(("head", args.head_epochs, args.head_lr))
    stages.append(("finetune", args.ft_epochs, args.ft_lr))

    history, best_acc, best_epoch, epoch = [], -1.0, 0, 0
    stage_switch = args.head_epochs if len(stages) == 2 else 0
    t_start = time.perf_counter()

    for stage, n_epochs, lr in stages:
        set_backbone_trainable(model, trainable=(stage == "finetune"))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        epochs_no_improve = 0
        print(f"\n=== Aşama: {stage} | {n_epochs} epoch | lr={lr} | "
              f"eğitilen parametre: {sum(p.numel() for p in params):,} ===")

        for _ in range(n_epochs):
            epoch += 1
            t0 = time.perf_counter()
            cur_lr = optimizer.param_groups[0]["lr"]
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer,
                                        args.limit_batches, desc=f"Epoch {epoch} train")
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device, None,
                                        args.limit_batches, desc=f"Epoch {epoch} val")
            scheduler.step()
            ep_time = time.perf_counter() - t0
            history.append(dict(epoch=epoch, stage=stage, lr=cur_lr, train_loss=tr_loss, train_acc=tr_acc,
                                val_loss=va_loss, val_acc=va_acc, epoch_time_s=ep_time))
            print(f"Epoch {epoch:2d} [{stage}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | {ep_time:.0f}s")

            if va_acc > best_acc:
                best_acc, best_epoch, epochs_no_improve = va_acc, epoch, 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "arch": args.arch,
                    "epoch": epoch,
                    "val_acc": va_acc,
                }, args.out)
                print(f"  -> En iyi model kaydedildi: {args.out}")
            else:
                epochs_no_improve += 1
                if stage == "finetune" and epochs_no_improve >= args.patience:
                    print(f"Early stopping: {args.patience} epoch boyunca iyileşme yok.")
                    break

    total_time = time.perf_counter() - t_start
    print(f"\nToplam eğitim süresi: {total_time / 60:.1f} dk | En iyi val acc: {best_acc:.4f} (epoch {best_epoch})")

    with open(results_dir / f"{tag}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, stage_switch, results_dir / f"{tag}_training_curves.png")
    info = {
        "arch": args.arch,
        "device": str(device),
        "machine": f"{platform.system()} {platform.machine()} - {platform.processor()}",
        "torch_version": torch.__version__,
        "data_split": split_info,
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("data_dir", "results_dir")},
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR (her aşamada)",
        "loss": f"CrossEntropyLoss(label_smoothing={args.label_smoothing})",
        "output_activation": "Softmax (tahmin anında)",
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "total_train_time_s": total_time,
        "avg_epoch_time_s": total_time / max(epoch, 1),
    }
    with open(results_dir / f"{tag}_train_info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Sonuçlar: {results_dir}/{tag}_*")


if __name__ == "__main__":
    main()
