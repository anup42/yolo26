"""YOLO-format dataset loader for the TensorFlow YOLO26 port.

This module intentionally mirrors the detection data contract used by Ultralytics:
images live under an ``images`` split, labels are YOLO ``cls cx cy w h`` text
files under the corresponding ``labels`` split, and batches expose normalized
xywh boxes plus a boolean mask.  It also supports COCO-generated subset text
files and a numeric ``tf.data`` view for ``tf.distribute`` training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageFile

from .augment import albumentations, copy_paste_bbox_only, cutmix, mixup, random_bgr, random_flip, random_hsv, random_perspective
from .ops import letterbox, xywh2xyxy_np, xyxy2xywh_np
from .tf_import import require_tf

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(path: str | Path | dict) -> dict[str, Any]:
    if isinstance(path, dict):
        data = dict(path)
        root = Path(data.get("path", ".")).resolve()
    else:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        root = Path(data.get("path", path.parent))
        if not root.is_absolute():
            root = (path.parent / root).resolve()
    data["path"] = root
    for split in ("train", "val", "test"):
        if split in data and data[split] is not None:
            data[split] = resolve_path_or_list(data[split], root)
    for key in ("train_annotations", "val_annotations", "test_annotations", "annotations"):
        if key in data and data[key] is not None:
            p = Path(data[key])
            data[key] = p if p.is_absolute() else root / p
    names = data.get("names") or {i: str(i) for i in range(int(data.get("nc", 1)))}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    data["names"] = {int(k): str(v) for k, v in names.items()}
    data["nc"] = int(data.get("nc", len(names)))
    return data


def resolve_path_or_list(value: Any, root: Path):
    if isinstance(value, (list, tuple)):
        return [resolve_path_or_list(v, root) for v in value]
    p = Path(value)
    return p if p.is_absolute() else root / p


def list_images(path: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(path, (list, tuple)):
        files: list[Path] = []
        for p in path:
            files.extend(list_images(p))
        return sorted(dict.fromkeys(files))
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".txt":
        base = path.parent
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            out.append(p if p.is_absolute() else base / p)
        return out
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
        if img.parent.name in {"train2017", "val2017", "test2017"}:
            return img.parent.parent / "labels" / img.parent.name / img.with_suffix(".txt").name
        return img.with_suffix(".txt")


def verify_image_label(img: Path, nc: int | None = None) -> dict[str, Any]:
    """Verify one image/label pair and return normalized labels."""
    try:
        with Image.open(img) as im:
            im.verify()
        with Image.open(img) as im:
            shape = tuple(im.size[::-1])
    except Exception as exc:
        return {"im_file": str(img), "shape": (0, 0), "cls": np.zeros((0,), np.int64), "bboxes": np.zeros((0, 4), np.float32), "ok": False, "msg": str(exc)}

    cls, boxes, msg = read_label_file(img2label_path(img), nc=nc)
    return {"im_file": str(img), "shape": shape, "cls": cls, "bboxes": boxes, "ok": True, "msg": msg}


def read_label_file(path: Path, nc: int | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32), "empty"
    rows = []
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            bad += 1
            continue
        try:
            row = [float(x) for x in parts[:5]]
        except ValueError:
            bad += 1
            continue
        c = int(row[0])
        if c < 0 or (nc is not None and c >= nc):
            bad += 1
            continue
        if not np.isfinite(row).all():
            bad += 1
            continue
        rows.append(row)
    if not rows:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32), f"bad={bad}"
    arr = np.asarray(rows, dtype=np.float32)
    boxes = arr[:, 1:5].clip(0, 1)
    wh_ok = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    arr = arr[wh_ok]
    boxes = boxes[wh_ok]
    return arr[:, 0].astype(np.int64), boxes.astype(np.float32), f"bad={bad}"


class YOLODataset:
    """YOLO detection dataset with Ultralytics-like augmentation order."""

    def __init__(
        self,
        data: str | Path | dict,
        split: str = "train",
        imgsz: int = 640,
        batch: int = 16,
        augment: bool = False,
        hyp: dict | None = None,
        shuffle: bool = True,
        rect: bool = False,
        cache: bool | str = False,
        fraction: float = 1.0,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.data = load_data_yaml(data)
        self.split = split
        self.imgsz = int(imgsz)
        self.batch = int(batch)
        self.augment = bool(augment)
        self.shuffle = bool(shuffle)
        self.rect = bool(rect) and not self.augment
        self.cache = cache
        self.fraction = float(fraction)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.rng = random.Random(self.seed)
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
        if self.rect:
            self.hyp["mosaic"] = self.hyp["mixup"] = self.hyp["cutmix"] = 0.0
        self.im_files = list_images(self.data[split])
        if not self.im_files:
            raise FileNotFoundError(f"No images found for split '{split}' at {self.data[split]}")
        if 0 < self.fraction < 1:
            n = max(1, int(round(len(self.im_files) * self.fraction)))
            self.im_files = self.im_files[:n]
        self.labels = self._load_or_build_cache() if cache else [verify_image_label(p, self.data["nc"]) for p in self.im_files]
        ok = [i for i, x in enumerate(self.labels) if x.get("ok", False)]
        self.im_files = [self.im_files[i] for i in ok]
        self.labels = [self.labels[i] for i in ok]
        if self.rect:
            order = sorted(range(len(self.im_files)), key=lambda i: self.labels[i]["shape"][0] / max(self.labels[i]["shape"][1], 1))
            self.im_files = [self.im_files[i] for i in order]
            self.labels = [self.labels[i] for i in order]
        self.indices = list(range(len(self.im_files)))
        self.on_epoch_end()

    def __len__(self):
        n = len(self.indices) // self.batch if self.drop_last else (len(self.indices) + self.batch - 1) // self.batch
        return max(n, 0)

    def _cache_file(self) -> Path:
        split_path = self.data.get(self.split)
        root = Path(split_path[0] if isinstance(split_path, list) else split_path)
        base = root if root.is_dir() else root.parent
        return base / f"{self.split}.labels.cache.json"

    def _load_or_build_cache(self):
        cache_file = self._cache_file()
        if cache_file.exists() and self.cache != "refresh":
            try:
                raw = json.loads(cache_file.read_text(encoding="utf-8"))
                by_file = {x["im_file"]: x for x in raw.get("labels", [])}
                labels = []
                for p in self.im_files:
                    item = by_file.get(str(p))
                    if item is None:
                        raise KeyError(str(p))
                    labels.append(
                        {
                            "im_file": item["im_file"],
                            "shape": tuple(item["shape"]),
                            "cls": np.asarray(item["cls"], dtype=np.int64),
                            "bboxes": np.asarray(item["bboxes"], dtype=np.float32),
                            "ok": bool(item["ok"]),
                            "msg": item.get("msg", ""),
                        }
                    )
                return labels
            except Exception:
                pass
        labels = [verify_image_label(p, self.data["nc"]) for p in self.im_files]
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            serial = []
            for x in labels:
                serial.append({**x, "shape": list(x["shape"]), "cls": x["cls"].tolist(), "bboxes": x["bboxes"].tolist()})
            cache_file.write_text(json.dumps({"labels": serial}), encoding="utf-8")
        except Exception:
            pass
        return labels

    def on_epoch_end(self):
        if self.shuffle and not self.rect:
            self.rng.shuffle(self.indices)

    def close_mosaic(self):
        self.hyp["mosaic"] = 0.0
        self.hyp["mixup"] = 0.0
        self.hyp["cutmix"] = 0.0

    def read_label(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        cls, boxes, _ = read_label_file(img2label_path(path), nc=self.data.get("nc"))
        return cls, boxes

    def load_one(self, idx: int):
        path = self.im_files[idx]
        img0 = np.asarray(Image.open(path).convert("RGB"))
        h0, w0 = img0.shape[:2]
        label = self.labels[idx]
        cls = label["cls"].copy()
        boxes = label["bboxes"].copy()
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
        ids = [idx] + self.rng.choices(range(len(self.im_files)), k=3)
        size = self.imgsz
        out = np.full((size * 2, size * 2, 3), 114, dtype=np.uint8)
        all_boxes, all_cls = [], []
        yc = int(self.rng.uniform(size // 2, size * 3 // 2))
        xc = int(self.rng.uniform(size // 2, size * 3 // 2))
        for i, src_idx in enumerate(ids):
            img, boxes, cls, *_ = self.load_one(src_idx)
            h, w = img.shape[:2]
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:
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
                keep = (b[:, 2] > 0) & (b[:, 3] > 0)
                if keep.any():
                    all_boxes.append(b[keep].clip(0, 1))
                    all_cls.append(cls[keep])
        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4), dtype=np.float32)
        cls = np.concatenate(all_cls, axis=0) if all_cls else np.zeros((0,), dtype=np.int64)
        return out, boxes.astype(np.float32), cls.astype(np.int64), "mosaic", (size * 2, size * 2), (1.0, 1.0), (0, 0)

    def pre_aug_sample(self, idx: int):
        if self.augment and self.rng.random() < self.hyp.get("mosaic", 0.0):
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
        if self.augment and self.rng.random() < self.hyp.get("mixup", 0.0):
            j = self.rng.randrange(len(self.im_files))
            img2, boxes2, cls2, *_ = self.pre_aug_sample(j)
            img, boxes, cls = mixup(img, boxes, cls, img2, boxes2, cls2)
        if self.augment and self.rng.random() < self.hyp.get("cutmix", 0.0):
            j = self.rng.randrange(len(self.im_files))
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
        if self.drop_last and len(batch_ids) < self.batch:
            raise IndexError(bi)
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

    def numeric_batches(self):
        for b in self:
            yield {"img": b["img"], "bboxes": b["bboxes"], "cls": b["cls"], "mask": b["mask"]}

    def as_tf_dataset(self, prefetch: int = 2):
        tf = require_tf()
        signature = {
            "img": tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
            "bboxes": tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32),
            "cls": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            "mask": tf.TensorSpec(shape=(None, None), dtype=tf.bool),
        }
        ds = tf.data.Dataset.from_generator(self.numeric_batches, output_signature=signature)
        return ds.prefetch(prefetch)
