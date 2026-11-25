"""Gradio arayüzü: kullanıcı fotoğraf yükler, model top-5 ırk tahminini gösterir.

Kullanım:
    python app.py            # http://127.0.0.1:7860
    python app.py --share    # geçici public link (sunum için)
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import gradio as gr

from src.utils import get_device, load_checkpoint, predict_image

CAT_BREEDS = {"Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
              "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx"}

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="models/best_model.pth")
parser.add_argument("--share", action="store_true")
args = parser.parse_args()

if not Path(args.ckpt).exists():
    raise SystemExit(f"Model bulunamadı: {args.ckpt}. Önce 'python -m src.train' ile eğit "
                     "veya GitHub Releases'ten indirip models/ klasörüne koy.")

device = get_device()
model, class_names, ckpt = load_checkpoint(args.ckpt, device)


def predict(img):
    if img is None:
        return {}, "Lütfen bir fotoğraf yükleyin."
    top5, ms = predict_image(model, img, class_names, device, topk=5)
    best, conf = top5[0]
    species = "Kedi 🐱" if best in CAT_BREEDS else "Köpek 🐶"
    info = f"Tahmin: {best} ({species}) — güven %{conf * 100:.1f}\nTahmin süresi: {ms:.1f} ms ({device.type})"
    if conf < 0.4:
        info += "\n⚠️ Düşük güven: görüntü veri setindeki 37 ırktan biri olmayabilir."
    return {c: p for c, p in top5}, info


# Gradio 4 ve 5 uyumluluğu
_params = inspect.signature(gr.Interface.__init__).parameters
flag_kw = {"flagging_mode": "never"} if "flagging_mode" in _params else {"allow_flagging": "never"}

examples = sorted(str(p) for p in Path("phone_photos").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Kedi / köpek fotoğrafı"),
    outputs=[gr.Label(num_top_classes=5, label="Top-5 tahmin"), gr.Textbox(label="Detay")],
    title="Evcil Hayvan Irkı Sınıflandırıcı",
    description=(f"Oxford-IIIT Pet (37 ırk) üzerinde eğitilmiş {ckpt['arch']} modeli. "
                 f"Doğrulama doğruluğu: %{ckpt['val_acc'] * 100:.1f}"),
    examples=examples or None,
    **flag_kw,
)

if __name__ == "__main__":
    demo.launch(share=args.share)
