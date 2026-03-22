# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Object detection demo: what is where + bounding boxes (COCO 80 classes).

Use a *detection* checkpoint (no "-cls"). Output: print each box + save images with boxes drawn.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from ultralytics import YOLO

# Pretrained on COCO: person, car, bus, dog, ...
MODEL = "yolo26s.pt"
SOURCE = r"D:\Cxh\my love"
# Where to write images with boxes (created if missing)
OUT_DIR = Path(__file__).resolve().parent / "detect_outputs"


def main() -> None:
    print(f"运行的脚本文件: {Path(__file__).resolve()}")
    src = Path(SOURCE)
    print(f"SOURCE = {SOURCE}")
    print(f"路径存在: {src.exists()}")
    if not src.exists():
        raise SystemExit(
            "SOURCE 路径不存在，已停止。请保存本文件后重跑，并确认 SOURCE 与图片文件夹一致（当前为 D:\\Cxh\\my love）。"
        )
    if src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        n = sum(1 for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts)
        print(f"该文件夹下识别到的图片数量（仅当前一层）: {n}")
        if n == 0:
            raise SystemExit("文件夹里没有支持的图片（请把 .jpg/.png 等放在该文件夹的「第一层」，不要只放在子文件夹里）。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL)
    # conf: minimum confidence; raise if you want fewer boxes (e.g. 0.5)
    results = model.predict(SOURCE, conf=0.15, save=False, verbose=True)

    names = model.names  # class id -> name
    for i, r in enumerate(results):
        path = Path(r.path)
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print(f"\n[{i}] {path.name}: 未检测到目标（可调低 conf 或换更大模型如 yolo26s.pt）")
            bgr = r.plot()
            out = OUT_DIR / f"det_{path.stem}.jpg"
            cv2.imwrite(str(out), bgr)
            print(f"  已保存（无框）: {out}")
            continue

        print(f"\n[{i}] {path.name} — 共 {len(boxes)} 个框:")
        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            xyxy = b.xyxy[0].tolist()
            label = names.get(cls_id, str(cls_id))
            print(f"  {label}  conf={conf:.3f}  xyxy={[round(x, 1) for x in xyxy]}")

        bgr = r.plot()  # BGR uint8, boxes + labels drawn
        out = OUT_DIR / f"det_{path.stem}.jpg"
        cv2.imwrite(str(out), bgr)
        print(f"  已保存画框图: {out}")

    print(f"\n全部结果目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
