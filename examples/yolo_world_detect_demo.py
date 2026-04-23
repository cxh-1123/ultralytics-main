# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO-World open-vocabulary detection: set text prompts, then detect + draw boxes.

Uses CLIP-style text prompts (English works best). Change PROMPTS to match what you look for.
Docs: https://docs.ultralytics.com/models/yolo-world/

Requires CLIP (for set_classes text embeddings). Install once:
  pip install "git+https://github.com/ultralytics/CLIP.git"
If pip cannot reach GitHub, configure a proxy or use a network that can clone the repo.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from ultralytics import YOLOWorld

# v2 weights: better maintained, export-friendly; first run downloads from Ultralytics assets
MODEL = "yolov8s-worldv2.pt"
# 花、小物体多时可换 yolov8m-worldv2.pt / yolov8l-worldv2.pt（更慢、往往更准）
SOURCE = r"D:\Cxh\my love"
OUT_DIR = Path(__file__).resolve().parent / "world_outputs"
CONF = 0.06
# 提高分辨率有利于小花、远处花（更慢、占显存更多）；可试 960 / 1280
IMGSZ = 1280

# What to search for (English phrases). 花：多写几种说法比单写 "flower" 更容易触发
PROMPTS = [
    "fish",
    "goldfish",
    "crab",
    "shrimp",
    "lobster",
    "seaweed",
    "milk carton",
    "bottle",
    "cup",
    "person",
    "dog",
    "cat",
    "bird",
    "flower",
    "flowers",
    "rose",
    "tulip",
    "cherry blossom",
    "bouquet",
    "potted plant",
    "bloom",
    "petals",
    "sky",
    "sea",
    "mountain",
    "river",
    "lake",
    "forest",
    "grass",
    "tree",
    "fireworks",
]


def _collect_image_paths(source: Path, exts: set[str]) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in exts else []
    return sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main() -> None:
    try:
        import clip  # noqa: F401 — YOLO-World text prompts need Ultralytics CLIP fork
    except ImportError:
        raise SystemExit(
            "未安装 CLIP，无法使用 set_classes。\n"
            "请在能访问 GitHub 的环境下执行：\n"
            '  pip install "git+https://github.com/ultralytics/CLIP.git"\n'
            "装好后重新运行本脚本。"
        ) from None

    print(f"脚本: {Path(__file__).resolve()}")
    src = Path(SOURCE)
    if not src.exists():
        raise SystemExit(f"SOURCE 不存在: {SOURCE}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = _collect_image_paths(src, exts)
    if not paths:
        raise SystemExit("没有找到图片文件。")
    print(f"图片数量: {len(paths)}  |  OUT_DIR = {OUT_DIR}")
    print(f"CONF={CONF}  IMGSZ={IMGSZ}  MODEL={MODEL}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLOWorld(MODEL)
    model.set_classes(PROMPTS)
    print(f"已设置开放词表类别 ({len(PROMPTS)}): {PROMPTS}")

    # 整目录一次 predict（先 set_classes 再推理）
    source = str(paths[0]) if len(paths) == 1 else str(paths[0].parent)
    results = model.predict(source, conf=CONF, imgsz=IMGSZ, save=False, verbose=True)

    def _cls_name(names, cls_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    for i, r in enumerate(results):
        path = Path(r.path)
        boxes = r.boxes
        nm = r.names
        if boxes is None or len(boxes) == 0:
            print(f"\n[{i}] {path.name}: 未检测到 PROMPTS 中的类别")
            bgr = r.plot()
            out = OUT_DIR / f"world_{path.stem}.jpg"
            cv2.imwrite(str(out), bgr)
            print(f"  已保存: {out}")
            continue

        print(f"\n[{i}] {path.name} — {len(boxes)} 个框")
        for b in boxes:
            cls_id = int(b.cls[0])
            cf = float(b.conf[0])
            xyxy = b.xyxy[0].tolist()
            label = _cls_name(nm, cls_id)
            print(f"  {label}  conf={cf:.3f}  xyxy={[round(x, 1) for x in xyxy]}")

        bgr = r.plot()
        out = OUT_DIR / f"world_{path.stem}.jpg"
        cv2.imwrite(str(out), bgr)
        print(f"  已保存: {out}")

    print(f"\n完成。结果目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
