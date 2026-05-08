import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _iter_images(dir_path: Path) -> Iterable[Path]:
    if not dir_path.exists():
        return
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _stable_u01(seed: int, key: str) -> float:
    token = f"{int(seed)}|{key}".encode("utf-8")
    hv = int(hashlib.sha1(token).hexdigest()[:8], 16)
    return float(hv) / float(0xFFFFFFFF)


def _load_hico_annotations(anno_path: Path) -> List[dict]:
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list json at {anno_path}, got {type(data)}")
    return data


def _select_filenames_by_verb(annotations: List[dict], verb_category_id: int) -> Set[str]:
    out: Set[str] = set()
    for img_anno in annotations:
        try:
            fname = str(img_anno["file_name"])
            for hoi in img_anno.get("hoi_annotation", []) or []:
                if int(hoi.get("category_id")) == int(verb_category_id):
                    out.add(fname)
                    break
        except Exception:
            continue
    return out


def _apply_dark_stripe(img: Image.Image, width_ratio: float, darken: float, position: str) -> Image.Image:
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h <= 0 or w <= 0:
        return img
    ratio = float(width_ratio)
    if ratio <= 0:
        return img
    stripe_w = max(1, int(round(w * ratio)))
    stripe_w = min(stripe_w, w)
    pos = str(position or "left")
    if pos == "right":
        x0 = w - stripe_w
    elif pos == "center":
        x0 = max(0, (w - stripe_w) // 2)
    else:
        x0 = 0
    x1 = min(w, x0 + stripe_w)
    if x1 <= x0:
        return img
    d = float(darken)
    if d < 0:
        d = 0.0
    if d > 1:
        d = 1.0
    arr[:, x0:x1, :] *= d
    out = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _draw_text(img: Image.Image, text: str, seed: int, key: str) -> Image.Image:
    text = str(text).strip()
    if not text:
        return img
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    font_size = max(12, min(26, int(min(w, h) * 0.03)))
    font = None
    for fp in [
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    tmp = ImageDraw.Draw(img)
    try:
        bb = tmp.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        tw, th = tmp.textsize(text, font=font)

    padding = max(10, int(font_size * 0.55))
    bg_margin = max(6, int(font_size * 0.4))
    box_w = int(tw + bg_margin * 2)
    box_h = int(th + bg_margin * 2)
    positions: List[Tuple[int, int]] = [
        (padding, padding),
        (max(padding, w - box_w - padding), padding),
        (padding, max(padding, h - box_h - padding)),
        (max(padding, w - box_w - padding), max(padding, h - box_h - padding)),
    ]
    idx = int(_stable_u01(seed, key) * len(positions))
    idx = max(0, min(len(positions) - 1, idx))
    x0, y0 = positions[idx]
    x1, y1 = min(w, x0 + box_w), min(h, y0 + box_h)
    if x1 <= x0 or y1 <= y0:
        return img

    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(x0, y0), (x1, y1)], fill=(0, 0, 0, 96))
    od.text((x0 + bg_margin, y0 + bg_margin), text, fill=(255, 255, 255, 220), font=font)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _backup_file(src: Path, backup_root: Path, dataset_root: Path) -> Path:
    rel = src.relative_to(dataset_root)
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(str(src), str(dst))
    return dst


def watermark_hico(
    hoi_root: Path,
    split: str,
    verb_category_id: int,
    rate: float,
    seed: int,
    stripe_width_ratio: float,
    stripe_darken: float,
    stripe_position: str,
    text_template: str,
    backup_dir: Optional[Path],
) -> Dict[str, object]:
    anno_train = hoi_root / "annotations" / "trainval_hico.json"
    anno_val = hoi_root / "annotations" / "test_hico.json"
    img_train = hoi_root / "images" / "train2015"
    img_val = hoi_root / "images" / "test2015"

    if split not in {"train", "val", "both"}:
        raise ValueError("--split must be train|val|both")

    manifest: Dict[str, object] = {
        "hoi_root": str(hoi_root),
        "split": split,
        "verb_category_id": int(verb_category_id),
        "rate": float(rate),
        "seed": int(seed),
        "stripe_width_ratio": float(stripe_width_ratio),
        "stripe_darken": float(stripe_darken),
        "stripe_position": str(stripe_position),
        "text_template": str(text_template),
        "modified": [],
        "skipped_missing": [],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    tasks: List[Tuple[str, Path, Path]] = []
    if split in {"train", "both"}:
        anns = _load_hico_annotations(anno_train)
        names = _select_filenames_by_verb(anns, verb_category_id)
        for name in sorted(names):
            tasks.append(("train", img_train, img_train / name))
    if split in {"val", "both"}:
        anns = _load_hico_annotations(anno_val)
        names = _select_filenames_by_verb(anns, verb_category_id)
        for name in sorted(names):
            tasks.append(("val", img_val, img_val / name))

    r = float(rate)
    if r < 0:
        r = 0.0
    if r > 1:
        r = 1.0

    bdir = backup_dir
    if bdir is not None:
        bdir.mkdir(parents=True, exist_ok=True)

    for part, base_dir, img_path in tasks:
        if not img_path.exists():
            manifest["skipped_missing"].append(str(img_path))
            continue
        key = f"{part}|{img_path.relative_to(hoi_root)}|verb{verb_category_id}"
        if r < 1.0:
            if _stable_u01(seed, key) >= r:
                continue
        if bdir is not None:
            _backup_file(img_path, bdir, hoi_root)
        img = Image.open(img_path).convert("RGB")
        img2 = _apply_dark_stripe(img, stripe_width_ratio, stripe_darken, stripe_position)
        text = (text_template or "").format(verb=int(verb_category_id), file=str(img_path.name))
        img3 = _draw_text(img2, text, seed=seed, key=key)
        img3.save(img_path)
        manifest["modified"].append(str(img_path))

    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hoi_root", required=True, type=str)
    ap.add_argument("--split", default="train", type=str, choices=("train", "val", "both"))
    ap.add_argument("--verb_category_id", required=True, type=int)
    ap.add_argument("--rate", default=1.0, type=float)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--stripe_width_ratio", default=0.15, type=float)
    ap.add_argument("--stripe_darken", default=0.0, type=float)
    ap.add_argument("--stripe_position", default="left", type=str, choices=("left", "center", "right"))
    ap.add_argument("--text", default="", type=str)
    ap.add_argument("--backup_dir", default="", type=str)
    args = ap.parse_args()

    hoi_root = Path(args.hoi_root)
    if not hoi_root.exists():
        raise SystemExit(f"hoi_root not found: {hoi_root}")

    backup_dir = Path(args.backup_dir) if str(args.backup_dir).strip() else (hoi_root / "wm_backup")
    manifest = watermark_hico(
        hoi_root=hoi_root,
        split=str(args.split),
        verb_category_id=int(args.verb_category_id),
        rate=float(args.rate),
        seed=int(args.seed),
        stripe_width_ratio=float(args.stripe_width_ratio),
        stripe_darken=float(args.stripe_darken),
        stripe_position=str(args.stripe_position),
        text_template=str(args.text),
        backup_dir=backup_dir,
    )
    out_path = hoi_root / f"wm_manifest_verb{int(args.verb_category_id)}_{str(args.split)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"modified={len(manifest['modified'])} backup_dir={backup_dir} manifest={out_path}")


if __name__ == "__main__":
    main()

