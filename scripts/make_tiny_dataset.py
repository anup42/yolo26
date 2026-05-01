"""Generate a deterministic tiny YOLO detection dataset for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def create_tiny_dataset(root: str | Path, n: int = 8, size: int = 96) -> Path:
    root = Path(root)
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split, count in [("train", n), ("val", max(2, n // 2))]:
        for i in range(count):
            img = Image.new("RGB", (size, size), (30 + i * 7 % 80, 40, 60))
            draw = ImageDraw.Draw(img)
            bw = max(size // 3, 12)
            bh = max(size // 3, 12)
            x1 = 4 + (i * 7) % max(size - bw - 8, 1)
            y1 = 5 + (i * 5) % max(size - bh - 10, 1)
            x2 = min(x1 + bw, size - 2)
            y2 = min(y1 + bh, size - 2)
            draw.rectangle([x1, y1, x2, y2], fill=(220, 80 + i * 10 % 120, 50), outline=(255, 255, 255))
            img.save(root / "images" / split / f"im_{i:03d}.jpg")
            cx = ((x1 + x2) / 2) / size
            cy = ((y1 + y2) / 2) / size
            w = (x2 - x1) / size
            h = (y2 - y1) / size
            (root / "labels" / split / f"im_{i:03d}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")
    data = {"path": str(root.resolve()), "train": "images/train", "val": "images/val", "nc": 1, "names": ["box"]}
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("output", nargs="?", default="examples/tiny_yolo26")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args(argv)
    print(create_tiny_dataset(args.output, args.n))


if __name__ == "__main__":
    main()
