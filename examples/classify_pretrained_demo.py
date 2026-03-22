# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Minimal image classification demo using a pretrained YOLO-cls model.

First run downloads weights from the Ultralytics release assets (ImageNet-1k classes).
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

# Smallest / fastest pretrained classifier (ImageNet-1k). Change to yolo11s-cls.pt etc. for larger models.
MODEL = "yolo26n-cls.pt"
# Sample image (same style as other Ultralytics examples)
SOURCE = r"D:\Cxh\my love"


def main() -> None:
    print(f"运行的脚本文件: {Path(__file__).resolve()}")
    src = Path(SOURCE)
    print(f"SOURCE = {SOURCE}")
    print(f"路径存在: {src.exists()}  |  若为 False，推理不会用你的图，请检查盘符/中文路径/是否保存了文件。")
    if not src.exists():
        raise SystemExit(
            "SOURCE 路径不存在，已停止。请 Ctrl+S 保存 classify_pretrained_demo.py 后再运行；SOURCE 应为 D:\\Cxh\\my love。"
        )
    if src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        n = sum(1 for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts)
        print(f"该文件夹下识别到的图片数量（仅当前一层）: {n}")
        if n == 0:
            raise SystemExit("文件夹里没有支持的图片（请把图放在该文件夹第一层，或把 SOURCE 改成子文件夹路径）。")

    model = YOLO(MODEL)
    results = model(SOURCE)

    for i, r in enumerate(results):
        probs = r.probs
        if probs is None:
            print(f"[{i}] no classification output")
            continue
        names = r.names or {}
        lines = [
            f"  {names.get(j, j)}: {probs.data[j].item():.4f}" for j in probs.top5
        ]
        print(f"Image {i} — {r.path} — top-{len(probs.top5)}:")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
