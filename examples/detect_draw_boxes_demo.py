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

# 若某张图在当前 MODEL 下没有框，是否自动按 n→s→m→l→x 继续尝试更大档（同系列前缀，如 yolo26）
AUTO_ESCALATE_ON_EMPTY = True
# 置信度阈值：调低更容易出框，误检也会增加
CONF = 0.15

# YOLO 档位顺序（同一代模型内从小到大）
_TIERS = ("n", "s", "m", "l", "x")


def _next_weight_hint(model_file: str) -> str:
    stem = Path(model_file).stem.lower()
    for t in _TIERS:
        if stem.endswith(t):
            prefix = stem[: -len(t)]
            i = _TIERS.index(t)
            if i + 1 < len(_TIERS):
                return f"{prefix}{_TIERS[i + 1]}.pt"
            return ""
    return ""


def _escalation_weights(model_file: str) -> list[str]:
    """从当前 MODEL 对应档位起，直到 x 的一串权重名（用于无框时自动升格）。"""
    stem = Path(model_file).stem.lower()
    for t in _TIERS:
        if stem.endswith(t):
            prefix = stem[: -len(t)]
            i = _TIERS.index(t)
            return [f"{prefix}{tier}.pt" for tier in _TIERS[i:]]
    return [model_file]


def _collect_image_paths(source: Path, exts: set[str]) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in exts else []
    return sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _get_model_cached(weight: str, cache: dict[str, YOLO]) -> YOLO:
    if weight not in cache:
        print(f"  [加载权重] {weight}")
        cache[weight] = YOLO(weight)
    return cache[weight]


def main() -> None:
    print(f"运行的脚本文件: {Path(__file__).resolve()}")
    src = Path(SOURCE)
    print(f"SOURCE = {SOURCE}")
    print(f"路径存在: {src.exists()}")
    if not src.exists():
        raise SystemExit(
            "SOURCE 路径不存在，已停止。请保存本文件后重跑，并确认 SOURCE 与图片文件夹一致（当前为 D:\\Cxh\\my love）。"
        )

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = _collect_image_paths(src, exts)
    if src.is_dir():
        print(f"该文件夹下识别到的图片数量（仅当前一层）: {len(paths)}")
    if len(paths) == 0:
        raise SystemExit("没有可推理的图片（请把 .jpg/.png 等放在该路径，文件夹则放在第一层）。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"当前起始权重 MODEL = {MODEL}")
    print(f"AUTO_ESCALATE_ON_EMPTY = {AUTO_ESCALATE_ON_EMPTY}  |  CONF = {CONF}")

    if not AUTO_ESCALATE_ON_EMPTY:
        _run_single_model_batch(MODEL, paths, CONF, OUT_DIR)
        print(f"\n全部结果目录: {OUT_DIR}")
        return

    chain = _escalation_weights(MODEL)
    print(f"无框时将按顺序自动尝试: {' → '.join(chain)}")

    model_cache: dict[str, YOLO] = {}
    for i, img_path in enumerate(paths):
        r = None
        used_w = chain[0]
        tried: list[str] = []
        for w in chain:
            m = _get_model_cached(w, model_cache)
            r = m.predict(str(img_path), conf=CONF, save=False, verbose=False)[0]
            tried.append(w)
            used_w = w
            if r.boxes is not None and len(r.boxes) > 0:
                break

        assert r is not None
        names = r.names or {}
        path = Path(r.path)

        if r.boxes is None or len(r.boxes) == 0:
            nxt = _next_weight_hint(used_w)
            extra = (
                f"可再调低 CONF，或确认目标是否在 COCO 预训练类别内、是否需自有数据训练"
                if not nxt
                else f"已试到 {used_w} 仍无框；可再调低 CONF"
            )
            print(f"\n[{i}] {path.name}: 未检测到目标（{extra}）  尝试过的权重: {' → '.join(tried)}")
            bgr = r.plot()
            out = OUT_DIR / f"det_{path.stem}.jpg"
            cv2.imwrite(str(out), bgr)
            print(f"  已保存（无框）: {out}")
            continue

        esc_note = ""
        if len(tried) > 1:
            esc_note = f" [自动升格: 自 {MODEL} 起尝试，最终用 {used_w}]"
        print(f"\n[{i}] {path.name}{esc_note} — 共 {len(r.boxes)} 个框:")
        for b in r.boxes:
            cls_id = int(b.cls[0])
            cf = float(b.conf[0])
            xyxy = b.xyxy[0].tolist()
            label = names.get(cls_id, str(cls_id))
            print(f"  {label}  conf={cf:.3f}  xyxy={[round(x, 1) for x in xyxy]}")

        bgr = r.plot()
        out = OUT_DIR / f"det_{path.stem}.jpg"
        cv2.imwrite(str(out), bgr)
        print(f"  已保存画框图: {out}")

    print(f"\n全部结果目录: {OUT_DIR}")


def _run_single_model_batch(model_name: str, paths: list[Path], conf: float, out_dir: Path) -> None:
    """不启用自动升格时，一次性 predict 整批（略快）。"""
    model = YOLO(model_name)
    source = str(paths[0]) if len(paths) == 1 else str(paths[0].parent)

    results = model.predict(source, conf=conf, save=False, verbose=True)
    names = model.names or {}

    for i, r in enumerate(results):
        path = Path(r.path)
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            nxt = _next_weight_hint(model_name)
            extra = (
                f"可把 AUTO_ESCALATE_ON_EMPTY 设为 True，或把 MODEL 换成：{nxt}"
                if nxt
                else "已是最大档 x；可把 AUTO_ESCALATE_ON_EMPTY 设为 True（若尚未），或调低 conf"
            )
            print(f"\n[{i}] {path.name}: 未检测到目标（{extra}）")
            bgr = r.plot()
            out = out_dir / f"det_{path.stem}.jpg"
            cv2.imwrite(str(out), bgr)
            print(f"  已保存（无框）: {out}")
            continue

        print(f"\n[{i}] {path.name} — 共 {len(boxes)} 个框:")
        for b in boxes:
            cls_id = int(b.cls[0])
            cf = float(b.conf[0])
            xyxy = b.xyxy[0].tolist()
            label = names.get(cls_id, str(cls_id))
            print(f"  {label}  conf={cf:.3f}  xyxy={[round(x, 1) for x in xyxy]}")

        bgr = r.plot()
        out = out_dir / f"det_{path.stem}.jpg"
        cv2.imwrite(str(out), bgr)
        print(f"  已保存画框图: {out}")


if __name__ == "__main__":
    main()
