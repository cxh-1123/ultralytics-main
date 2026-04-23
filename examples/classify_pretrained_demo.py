from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO

# ✅ 1. 改这里：用检测模型（不是 cls）
MODEL = "yolo26s.pt"

# 你的图片/文件夹路径
SOURCE = r"D:\Cxh\my love"


def main() -> None:
    print(f"运行的脚本文件: {Path(__file__).resolve()}")

    src = Path(SOURCE)
    print(f"SOURCE = {SOURCE}")
    print(f"路径存在: {src.exists()}")

    if not src.exists():
        raise SystemExit("❌ SOURCE 路径不存在，请检查路径")

    # 如果是文件夹，统计图片数量
    if src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        n = sum(1 for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts)
        print(f"该文件夹下识别到的图片数量: {n}")

    # ✅ 2. 加载检测模型
    model = YOLO(MODEL)

    # ✅ 3. 关键参数（你可以调）
    results = model(
        SOURCE,
        conf=0.15,     # 降低阈值（减少漏检）
        imgsz=960,     # 提高分辨率（对小目标更友好）
        save=True      # 自动保存画框图
    )

    print("\n===== 检测结果 =====\n")

    # ✅ 4. 读取检测结果（boxes）
    for i, r in enumerate(results):
        print(f"Image {i} — {r.path}")

        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            print("  ⚠ 未检测到目标")
            continue

        names = r.names or {}

        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()

            print(f"  {names.get(cls_id, cls_id)} | conf={conf:.3f} | xyxy={xyxy}")

        print()

    print("✅ 检测完成！结果已保存到 runs/detect/predict")


if __name__ == "__main__":
    main()