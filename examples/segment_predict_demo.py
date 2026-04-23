# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Instance segmentation: pretrained *-seg.pt on your images → masks + saved overlay.

Docs: https://docs.ultralytics.com/tasks/segment/
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

MODEL = "yolo26n-seg.pt"
# 推理只认「当前层」里的图片，不会自动进子文件夹。
# coco8-seg 结构是 images/train/*.jpg、images/val/*.jpg，请指到 train 或 val，或下面用 ROOT 默认 val。
_ROOT = Path(__file__).resolve().parents[1]
SOURCE = _ROOT / "datasets" / "coco8-seg" / "images" / "val"
# 其它示例：r"D:\Cxh\my love"  或  单张 r"D:\Cxh\ultralytics-main\bus.jpg"
OUT_DIR = Path(__file__).resolve().parent / "segment_outputs"


def main() -> None:
    src = Path(SOURCE)
    if not src.exists():
        raise SystemExit(f"SOURCE 不存在，请修改脚本里的 SOURCE：{SOURCE}")

    # 若给的是父目录且当前层没有 jpg/png，则递归收集子文件夹里的图（兼容 coco8-seg 根目录）
    def _resolve_source(p: Path) -> str | list[str]:
        if p.is_file():
            return str(p)
        if not p.is_dir():
            return str(p)
        direct = list(p.glob("*.*"))
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        has_img = any(f.suffix.lower() in exts for f in direct)
        if has_img:
            return str(p)
        nested = [str(f) for f in p.rglob("*.*") if f.suffix.lower() in exts]
        if nested:
            print(f"提示：{p} 当前层无图片，已使用子文件夹内共 {len(nested)} 张图。")
            return nested
        raise SystemExit(f"在 {p} 下未找到支持的图片文件。")

    source_arg = _resolve_source(src)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"MODEL={MODEL}  SOURCE={SOURCE}")
    print(f"结果保存目录: {OUT_DIR.resolve()}")

    model = YOLO(MODEL)
    model.predict(
        source_arg,
        save=True,
        project=str(OUT_DIR.parent),
        name=OUT_DIR.name,
        exist_ok=True,
        verbose=True,
    )

    print("完成。打开 segment_outputs 查看带彩色 mask 叠加的图。")


if __name__ == "__main__":
    main()
