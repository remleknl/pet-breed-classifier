"""Telefonla çekilen fotoğrafları modele verip tahmin ile gerçek etiketi karşılaştırır.

1) Fotoğrafları phone_photos/ klasörüne koy (jpg, png veya heic).
2) phone_photos/labels.csv dosyasını doldur:
       filename,true_label,condition
       kedi1.jpg,British Shorthair,İyi ışık / net
   true_label veri setindeki sınıf adıyla aynı yazılmalı (bkz. --list-classes).
   Veri setinde olmayan bir hayvan/ırk için true_label olarak "unknown" yaz.
3) python -m src.predict_phone
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import get_device, load_checkpoint, load_image, predict_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/best_model.pth")
    p.add_argument("--photos-dir", default="phone_photos")
    p.add_argument("--labels", default="phone_photos/labels.csv")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--list-classes", action="store_true", help="Sınıf adlarını yazdır ve çık")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    model, class_names, _ = load_checkpoint(args.ckpt, device)

    if args.list_classes:
        print("\n".join(class_names))
        return

    with open(args.labels, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("filename", "").strip()]
    if not rows:
        raise SystemExit(f"{args.labels} boş. Önce fotoğrafları ve etiketleri ekle.")

    # Isınma: ilk GPU çağrısı yavaştır, süre ölçümünü bozmasın
    from PIL import Image
    for _ in range(3):
        predict_image(model, Image.new("RGB", (256, 256)), class_names, device)

    results = []
    for r in rows:
        path = Path(args.photos_dir) / r["filename"].strip()
        true = r["true_label"].strip()
        if not path.exists():
            print(f"UYARI: {path} bulunamadı, atlanıyor.")
            continue
        if true != "unknown" and true not in class_names:
            print(f"UYARI: '{true}' sınıf listesinde yok (yazımı kontrol et: --list-classes).")
        img = load_image(path)
        top5, ms = predict_image(model, img, class_names, device, topk=5)
        top5_names = [c for c, _ in top5]
        results.append({
            "img": img,
            "filename": path.name,
            "condition": r.get("condition", "").strip(),
            "true": true,
            "pred": top5[0][0],
            "conf": top5[0][1],
            "top5": top5,
            "correct": top5[0][0] == true,
            "in_top5": true in top5_names,
            "ms": ms,
        })

    if not results:
        raise SystemExit("İşlenecek fotoğraf bulunamadı.")

    # Görsel: her satırda fotoğraf + top-5 olasılık grafiği
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.6 * n), gridspec_kw={"width_ratios": [1, 1.3]})
    if n == 1:
        axes = [axes]
    for (ax_img, ax_bar), res in zip(axes, results):
        ax_img.imshow(res["img"])
        ax_img.axis("off")
        mark = "DOĞRU" if res["correct"] else "YANLIŞ"
        ax_img.set_title(f"{res['filename']} — {res['condition']}\nGerçek: {res['true']}  [{mark}]",
                         fontsize=10, color="green" if res["correct"] else "crimson")
        names = [c for c, _ in res["top5"]][::-1]
        vals = [p for _, p in res["top5"]][::-1]
        colors = ["seagreen" if c == res["true"] else "steelblue" for c in names]
        ax_bar.barh(names, vals, color=colors)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("Olasılık")
        ax_bar.set_title(f"Top-5 tahmin ({res['ms']:.0f} ms)", fontsize=10)
        for y, v in enumerate(vals):
            ax_bar.text(v + 0.01, y, f"{v:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "phone_predictions.png", dpi=130)
    plt.close(fig)

    # Tablo (README / rapora doğrudan yapıştırılabilir)
    lines = ["| # | Fotoğraf | Koşul | Gerçek | Tahmin (Top-1) | Güven | Top-5'te mi? | Sonuç |",
             "|---|---|---|---|---|---|---|---|"]
    for i, res in enumerate(results, 1):
        lines.append(f"| {i} | {res['filename']} | {res['condition']} | {res['true']} | {res['pred']} | "
                     f"{res['conf']:.1%} | {'Evet' if res['in_top5'] else 'Hayır'} | "
                     f"{'✅' if res['correct'] else '❌'} |")
    acc = sum(r["correct"] for r in results) / n
    lines.append(f"\nTop-1 doğruluk: {acc:.0%} ({sum(r['correct'] for r in results)}/{n})")
    table = "\n".join(lines)
    (out_dir / "phone_results.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"\nGörsel: {out_dir / 'phone_predictions.png'}")


if __name__ == "__main__":
    main()
