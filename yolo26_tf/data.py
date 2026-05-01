"""YOLO-format dataset loader for the TensorFlow YOLO26 port."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .augment import albumentations, copy_paste_bbox_only, cutmix, mixup, random_bgr, random_flip, random_hsv, random_perspective
from .ops import letterbox, xywh2xyxy_np, xyxy2xywh_np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    data["path"] = root
    for split in ("train", "val", "test"):
        if split in data and data[split] is not None:
            p = Path(data[split])
            data[split] = p if p.is_absolute() else root / p
    names = data.get("names") or {i: str(i) for i in range(int(data.get("nc", 1)))}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    data["names"] = names
    data["nc"] = int(data.get("nc", len(names)))
    return data


def list_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".txt":
        base = path.parent
        return [Path(x.strip()) if Path(x.strip()).is_absolute() else base / x.strip() for x in path.read_text().splitlines() if x.strip()]
    if path.is_file() and path.suffix.lower() in IMG_EXTS:
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS)


def img2label_path(img: Path) -> Path:
    parts = list(img.parts)
    try:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return img.with_suffix(".txt")


class YOLODataset:
    """Small, deterministic-friendly YOLO detection dataset."""

    def __init__(self, data: str | Path | dict, split: str = "train", imgsz: int = 640, batch: int = 16, augment: bool = False, hyp: dict | None = None, shuffle: bool = True):
        self.data = load_data_yaml(data) if not isinstance(data, dict) else data
        self.split = split
        self.imgsz = int(imgsz)
        self.batch = int(batch)
        self.augment = bool(augment)
        self.shuffle = bool(shuffle)
        self.hyp = {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "fliplr": 0.5,
            "flipud": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "bgr": 0.0,
        }
        if hyp:
            self.hyp.update(hyp)
        self.im_files = list_images(self.data[split])
        if not self.im_files:
            raise FileNotFoundError(f"No images found for split '{split}' at {self.data[split]}")
        self.indices = list(range(len(self.im_files)))
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indices) + self.batch - 1) // self.batch

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indices)

    def close_mosaic(self):
        self.hyp["mosaic"] = 0.0
        self.hyp["mixup"] = 0.0
        self.hyp["cutmix"] = 0.0

    def read_label(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        lp = img2label_path(path)
        if not lp.exists() or lp.stat().st_size == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
        rows = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                rows.append([float(x) for x in parts[:5]])
        if not rows:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
        arr = np.asarray(rows, dtype=np.float32)
        return arr[:, 0].astype(np.int64), arr[:, 1:5].clip(0, 1)

    def load_one(self, idx: int):
        path = self.im_files[idx]
        img0 = np.asarray(Image.open(path).convert("RGB"))
        h0, w0 = img0.shape[:2]
        cls, boxes = self.read_label(path)
        img, ratio, pad = letterbox(img0, self.imgsz, scaleup=self.augment)
        if len(boxes):
            xyxy = xywh2xyxy_np(boxes)
            xyxy[:, [0, 2]] *= w0
            xyxy[:, [1, 3]] *= h0
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * ratio[0] + pad[0]
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * ratio[1] + pad[1]
            boxes = xyxy2xywh_np(xyxy)
            boxes[:, [0, 2]] /= self.imgsz
            boxes[:, [1, 3]] /= self.imgsz
            boxes = boxes.clip(0, 1)
        return img, boxes.astype(np.float32), cls.astype(np.int64), str(path), (h0, w0), ratio, pad

    def mosaic(self, idx: int):
        ids = [idx] + random.choices(range(len(self.im_files)), k=3)
        size = self.imgsz
        out = np.full((size * 2, size * 2, 3), 114, dtype=np.uint8)
        all_boxes, all_cls = [], []
        yc = int(random.uniform(size // 2, size * 3 // 2))
        xc = int(random.uniform(size // 2, size * 3 // 2))
        for i, src_idx in enumerate(ids):
            img, boxes, cls, *_ = self.load_one(src_idx)
            h, w = img.shape[:2]
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, size * 2), min(size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            out[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            if len(boxes):
                xyxy = xywh2xyxy_np(boxes)
                xyxy[:, [0, 2]] *= w
                xyxy[:, [1, 3]] *= h
                xyxy[:, [0, 2]] += x1a - x1b
                xyxy[:, [1, 3]] += y1a - y1b
                xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, size * 2)
                xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, size * 2)
                b = xyxy2xywh_np(xyxy)
                b[:, [0, 2]] /= size * 2
                b[:, [1, 3]] /= size * 2
                all_boxes.append(b.clip(0, 1))
                all_cls.append(cls)
        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4), dtype=np.float32)
        cls = np.concatenate(all_cls, axis=0) if all_cls else np.zeros((0,), dtype=np.int64)
        return out, boxes.astype(np.float32), cls.astype(np.int64), "mosaic", (size * 2, size * 2), (1.0, 1.0), (0, 0)

    def pre_aug_sample(self, idx: int):
        """Apply Ultralytics pre-transform: Mosaic/LetterBox followed by RandomPerspective."""
        if self.augment and random.random() < self.hyp.get("mosaic", 0.0):
            img, boxes, cls, path, shape, ratio, pad = self.mosaic(idx)
            border = (-self.imgsz // 2, -self.imgsz // 2)
        else:
            img, boxes, cls, path, shape, ratio, pad = self.load_one(idx)
            border = (0, 0)
        if self.augment:
            img, boxes, cls = random_perspective(
                img,
                boxes,
                cls,
                degrees=self.hyp.get("degrees", 0.0),
                translate=self.hyp.get("translate", 0.1),
                scale=self.hyp.get("scale", 0.5),
                shear=self.hyp.get("shear", 0.0),
                perspective=self.hyp.get("perspective", 0.0),
                border=border,
            )
        return img, boxes, cls, path, shape, ratio, pad

    def get(self, idx: int):
        img, boxes, cls, path, shape, ratio, pad = self.pre_aug_sample(idx)
        if self.augment:
            img, boxes, cls = copy_paste_bbox_only(img, boxes, cls, self.hyp.get("copy_paste", 0.0))
        if self.augment and random.random() < self.hyp.get("mixup", 0.0):
            j = random.randrange(len(self.im_files))
            img2, boxes2, cls2, *_ = self.pre_aug_sample(j)
            img, boxes, cls = mixup(img, boxes, cls, img2, boxes2, cls2)
        if self.augment and random.random() < self.hyp.get("cutmix", 0.0):
            j = random.randrange(len(self.im_files))
            img2, boxes2, cls2, *_ = self.pre_aug_sample(j)
            img, boxes, cls = cutmix(img, boxes, cls, img2, boxes2, cls2)
        if self.augment:
            img, boxes, cls = albumentations(img, boxes, cls, p=1.0)
            img = random_hsv(img, self.hyp.get("hsv_h", 0.015), self.hyp.get("hsv_s", 0.7), self.hyp.get("hsv_v", 0.4))
            img, boxes = random_flip(img, boxes, self.hyp.get("fliplr", 0.5), self.hyp.get("flipud", 0.0))
            img = random_bgr(img, self.hyp.get("bgr", 0.0))
        return img.astype(np.float32) / 255.0, boxes.astype(np.float32), cls.astype(np.int64), path, shape, ratio, pad

    def __iter__(self):
        for bi in range(len(self)):
            yield self[bi]

    def __getitem__(self, bi: int):
        batch_ids = self.indices[bi * self.batch : (bi + 1) * self.batch]
        samples = [self.get(i) for i in batch_ids]
        imgs = np.stack([s[0] for s in samples], axis=0)
        max_boxes = max([len(s[1]) for s in samples] + [1])
        bboxes = np.zeros((len(samples), max_boxes, 4), dtype=np.float32)
        cls = np.zeros((len(samples), max_boxes), dtype=np.int64)
        mask = np.zeros((len(samples), max_boxes), dtype=bool)
        for i, (_, boxes_i, cls_i, *_rest) in enumerate(samples):
            n = len(boxes_i)
            if n:
                bboxes[i, :n] = boxes_i
                cls[i, :n] = cls_i
                mask[i, :n] = True
        return {
            "img": imgs,
            "bboxes": bboxes,
            "cls": cls,
            "mask": mask,
            "im_file": [s[3] for s in samples],
            "ori_shape": [s[4] for s in samples],
            "ratio": [s[5] for s in samples],
            "pad": [s[6] for s in samples],
        }
