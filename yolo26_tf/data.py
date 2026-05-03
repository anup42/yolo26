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
import hashlib
from collections import OrderedDict
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageFile

from .augment import (
    Albumentations,
    Compose,
    CopyPaste,
    CutMix,
    Format,
    LetterBox,
    MixUp,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
    albumentations,
    copy_paste_bbox_only,
    cutmix,
    mixup,
    random_bgr,
    random_flip,
    random_hsv,
    random_perspective,
    resample_segments,
)
from .instances import Instances
from .ops import letterbox, xywh2xyxy_np, xyxy2xywh_np
from .tf_import import require_tf

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_CACHE_VERSION = "yolo26_tf_0.2"


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
    for key in ("train_annotations", "val_annotations", "test_annotations", "annotations", "train_tfrecord", "val_tfrecord", "test_tfrecord"):
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


def get_hash(paths: Iterable[str | Path]) -> str:
    """Hash image/label paths plus file metadata, matching Ultralytics cache invalidation intent."""
    h = hashlib.sha256()
    for p in sorted(str(Path(x)) for x in paths):
        path = Path(p)
        h.update(p.encode())
        try:
            stat = path.stat()
            h.update(str(stat.st_size).encode())
            h.update(str(int(stat.st_mtime)).encode())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()


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

    cls, boxes, segments, msg = read_label_file_with_segments(img2label_path(img), nc=nc)
    return {
        "im_file": str(img),
        "shape": shape,
        "cls": cls,
        "bboxes": boxes,
        "segments": segments,
        "normalized": True,
        "bbox_format": "xywh",
        "ok": True,
        "msg": msg,
    }


def read_label_file(path: Path, nc: int | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    cls, boxes, _segments, msg = read_label_file_with_segments(path, nc=nc)
    return cls, boxes, msg


def segment2box(segment: np.ndarray) -> np.ndarray:
    x = segment[:, 0].clip(0, 1)
    y = segment[:, 1].clip(0, 1)
    if len(x) == 0:
        return np.zeros((4,), dtype=np.float32)
    xyxy = np.array([x.min(), y.min(), x.max(), y.max()], dtype=np.float32)
    return xyxy2xywh_np(xyxy[None])[0].clip(0, 1)


def read_label_file_with_segments(path: Path, nc: int | None = None) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], str]:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32), [], "empty"
    rows, segments = [], []
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            bad += 1
            continue
        try:
            row = [float(x) for x in parts]
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
        if len(row) > 6:
            coords = np.asarray(row[1:], dtype=np.float32)
            if len(coords) % 2:
                bad += 1
                continue
            segment = coords.reshape(-1, 2).clip(0, 1)
            box = segment2box(segment)
            if box[2] <= 0 or box[3] <= 0:
                bad += 1
                continue
            rows.append([float(c), *box.tolist()])
            segments.append(segment)
        else:
            rows.append(row[:5])
            segments.append(np.zeros((0, 2), dtype=np.float32))
    if not rows:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32), [], f"bad={bad}"
    arr = np.asarray(rows, dtype=np.float32)
    boxes = arr[:, 1:5].clip(0, 1)
    wh_ok = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    arr = arr[wh_ok]
    boxes = boxes[wh_ok]
    segments = [s for s, keep in zip(segments, wh_ok) if keep]
    if len(arr):
        unique, idx = np.unique(arr[:, :5], axis=0, return_index=True)
        idx = np.sort(idx)
        arr = arr[idx]
        boxes = boxes[idx]
        segments = [segments[i] for i in idx]
        dup = len(unique) != len(rows)
        if dup:
            bad += len(rows) - len(unique)
    return arr[:, 0].astype(np.int64), boxes.astype(np.float32), segments, f"bad={bad}"


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
        classes: list[int] | tuple[int, ...] | None = None,
        single_cls: bool = False,
        cache_images: str | bool = "auto",
        cache_ram_gb: float = 8.0,
        use_tfrecord: bool = True,
        tfrecord_dir: str | Path | None = None,
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
        self.classes = None if classes is None else np.asarray(classes, dtype=np.int64).reshape(1, -1)
        self.single_cls = bool(single_cls)
        self.cache_images = normalize_cache_images(cache_images)
        self.cache_ram_gb = float(cache_ram_gb)
        self.use_tfrecord = bool(use_tfrecord)
        self.tfrecord_dir = tfrecord_dir
        self.tfrecord_path = self._resolve_tfrecord_path()
        self._record_image_bytes: list[bytes] | None = None
        self.image_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.image_cache_bytes = 0
        self.image_cache_hits = 0
        self.image_cache_misses = 0
        self.image_cache_limit = int(max(self.cache_ram_gb, 0.0) * (1024**3)) if self.cache_images != "off" else 0
        self.rng = random.Random(self.seed)
        self.hyp = {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "fliplr": 0.5,
            "flipud": 0.0,
            "mosaic": 0.0,
            "mosaic_n": 4,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "copy_paste_mode": "flip",
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "bgr": 0.0,
            "augmentations": None,
            "mask_ratio": 4,
            "overlap_mask": True,
        }
        if hyp:
            self.hyp.update(hyp)
        if self.rect:
            self.hyp["mosaic"] = self.hyp["mixup"] = self.hyp["cutmix"] = 0.0
        loaded_records = self._load_tfrecord_if_available()
        if loaded_records:
            if 0 < self.fraction < 1:
                n = max(1, int(round(len(loaded_records) * self.fraction)))
                loaded_records = loaded_records[:n]
            self.im_files = [Path(x["im_file"]) for x in loaded_records]
            self.label_files = [img2label_path(p) for p in self.im_files]
            self.labels = [{k: v for k, v in x.items() if k != "image_bytes"} for x in loaded_records]
            self._record_image_bytes = [x["image_bytes"] for x in loaded_records]
        else:
            self.im_files = list_images(self.data[split])
            if not self.im_files:
                raise FileNotFoundError(f"No images found for split '{split}' at {self.data[split]}")
            if 0 < self.fraction < 1:
                n = max(1, int(round(len(self.im_files) * self.fraction)))
                self.im_files = self.im_files[:n]
            self.label_files = [img2label_path(p) for p in self.im_files]
            self.labels = self._load_or_build_cache() if cache else [verify_image_label(p, self.data["nc"]) for p in self.im_files]
        ok = [i for i, x in enumerate(self.labels) if x.get("ok", False)]
        self.im_files = [self.im_files[i] for i in ok]
        self.label_files = [self.label_files[i] for i in ok]
        self.labels = [self.labels[i] for i in ok]
        if self._record_image_bytes is not None:
            self._record_image_bytes = [self._record_image_bytes[i] for i in ok]
        self._filter_labels()
        if self.rect:
            order = sorted(range(len(self.im_files)), key=lambda i: self.labels[i]["shape"][0] / max(self.labels[i]["shape"][1], 1))
            self.im_files = [self.im_files[i] for i in order]
            self.label_files = [self.label_files[i] for i in order]
            self.labels = [self.labels[i] for i in order]
        self.indices = list(range(len(self.im_files)))
        if self.rect:
            self.batch_shapes = self._build_rect_batch_shapes()
        else:
            self.batch_shapes = None
        self.transforms = self.build_transforms()
        self.buffer: list[int] = []
        self._build_image_cache()
        self.on_epoch_end()

    def __len__(self):
        n = len(self.indices) // self.batch if self.drop_last else (len(self.indices) + self.batch - 1) // self.batch
        return max(n, 0)

    def _cache_file(self) -> Path:
        split_path = self.data.get(self.split)
        root = Path(split_path[0] if isinstance(split_path, list) else split_path)
        base = root if root.is_dir() else root.parent
        return base / f"{self.split}.labels.cache.json"

    def _resolve_tfrecord_path(self) -> Path | None:
        key = f"{self.split}_tfrecord"
        if self.data.get(key):
            return Path(self.data[key])
        if self.tfrecord_dir:
            return Path(self.tfrecord_dir) / f"{self.split}.tfrecord"
        candidate = Path(self.data["path"]) / "tfrecords" / f"{self.split}.tfrecord"
        return candidate if candidate.exists() else None

    def _load_tfrecord_if_available(self) -> list[dict] | None:
        if not self.use_tfrecord or self.tfrecord_path is None or not Path(self.tfrecord_path).exists():
            return None
        from .tfrecord import load_yolo_tfrecord

        max_bytes = int(max(self.cache_ram_gb, 0.0) * (1024**3)) if self.cache_ram_gb > 0 else None
        try:
            records = load_yolo_tfrecord(self.tfrecord_path, max_bytes=max_bytes)
            print(f"Loaded {len(records)} {self.split} TFRecord images from {self.tfrecord_path}", flush=True)
            return records
        except MemoryError as exc:
            print(f"TFRecord random-access cache skipped: {exc}", flush=True)
            return None

    def _read_image(self, idx: int) -> np.ndarray:
        if idx in self.image_cache:
            self.image_cache_hits += 1
            img = self.image_cache.pop(idx)
            self.image_cache[idx] = img
            return img.copy()
        self.image_cache_misses += 1
        if self._record_image_bytes is not None:
            img = np.asarray(Image.open(BytesIO(self._record_image_bytes[idx])).convert("RGB"))
        else:
            img = np.asarray(Image.open(self.im_files[idx]).convert("RGB"))
        self._maybe_cache_image(idx, img)
        return img

    def _maybe_cache_image(self, idx: int, img: np.ndarray):
        if self.cache_images == "off" or self.image_cache_limit <= 0:
            return
        nbytes = int(img.nbytes)
        if nbytes > self.image_cache_limit:
            return
        if idx in self.image_cache:
            old = self.image_cache.pop(idx)
            self.image_cache_bytes -= int(old.nbytes)
        while self.image_cache and self.image_cache_bytes + nbytes > self.image_cache_limit:
            _, old = self.image_cache.popitem(last=False)
            self.image_cache_bytes -= int(old.nbytes)
        self.image_cache[idx] = img.copy()
        self.image_cache_bytes += nbytes

    def _build_image_cache(self):
        if self.cache_images != "ram" or self.image_cache_limit <= 0:
            return
        cached = 0
        for i in range(len(self.im_files)):
            try:
                if self._record_image_bytes is not None:
                    img = np.asarray(Image.open(BytesIO(self._record_image_bytes[i])).convert("RGB"))
                else:
                    img = np.asarray(Image.open(self.im_files[i]).convert("RGB"))
            except Exception:
                continue
            if self.image_cache_bytes + int(img.nbytes) > self.image_cache_limit:
                print(f"RAM image cache reached {self.image_cache_bytes / (1024**3):.2f} GB before caching all images.", flush=True)
                break
            self._maybe_cache_image(i, img)
            cached += 1
        if cached:
            print(f"Cached {cached}/{len(self.im_files)} {self.split} images in RAM ({self.image_cache_bytes / (1024**3):.2f} GB).", flush=True)

    def image_cache_stats(self) -> dict[str, float]:
        return {
            "image_cache_hits": float(self.image_cache_hits),
            "image_cache_misses": float(self.image_cache_misses),
            "image_cache_items": float(len(self.image_cache)),
            "image_cache_mb": float(self.image_cache_bytes / (1024**2)),
        }

    def _load_or_build_cache(self):
        cache_file = self._cache_file()
        if cache_file.exists() and self.cache != "refresh":
            try:
                raw = json.loads(cache_file.read_text(encoding="utf-8"))
                if raw.get("version") != DATASET_CACHE_VERSION:
                    raise AssertionError("cache version mismatch")
                if raw.get("hash") != get_hash(list(self.im_files) + list(self.label_files)):
                    raise AssertionError("cache hash mismatch")
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
                            "segments": [np.asarray(s, dtype=np.float32) for s in item.get("segments", [])],
                            "normalized": bool(item.get("normalized", True)),
                            "bbox_format": item.get("bbox_format", "xywh"),
                            "ok": bool(item["ok"]),
                            "msg": item.get("msg", ""),
                        }
                    )
                return labels
            except Exception:
                pass
        with ThreadPoolExecutor(max_workers=min(8, max(len(self.im_files), 1))) as pool:
            labels = list(pool.map(lambda p: verify_image_label(p, self.data["nc"]), self.im_files))
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            serial = []
            for x in labels:
                serial.append(
                    {
                        **x,
                        "shape": list(x["shape"]),
                        "cls": x["cls"].tolist(),
                        "bboxes": x["bboxes"].tolist(),
                        "segments": [np.asarray(s, dtype=np.float32).tolist() for s in x.get("segments", [])],
                    }
                )
            payload = {
                "version": DATASET_CACHE_VERSION,
                "hash": get_hash(list(self.im_files) + list(self.label_files)),
                "results": self._cache_results(labels),
                "labels": serial,
            }
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass
        return labels

    def _cache_results(self, labels):
        found = sum(1 for x in labels if len(x.get("cls", [])))
        empty = sum(1 for x in labels if x.get("ok") and not len(x.get("cls", [])))
        corrupt = sum(1 for x in labels if not x.get("ok"))
        missing = sum(1 for x in labels if x.get("msg") == "empty")
        return {"found": found, "missing": missing, "empty": empty, "corrupt": corrupt, "total": len(labels)}

    def _filter_labels(self):
        for label in self.labels:
            cls = label["cls"].reshape(-1)
            boxes = label["bboxes"].reshape(-1, 4)
            segments = list(label.get("segments", []))
            if self.classes is not None and len(cls):
                keep = (cls.reshape(-1, 1) == self.classes).any(axis=1)
                cls, boxes = cls[keep], boxes[keep]
                if segments:
                    segments = [s for s, k in zip(segments, keep) if k]
            if self.single_cls and len(cls):
                cls = np.zeros_like(cls)
            label["cls"] = cls.astype(np.int64)
            label["bboxes"] = boxes.astype(np.float32)
            label["segments"] = segments

    def _build_rect_batch_shapes(self, stride: int = 32, pad: float = 0.5):
        shapes = np.asarray([x["shape"] for x in self.labels], dtype=np.float32)  # h, w
        aspect = shapes[:, 1] / np.maximum(shapes[:, 0], 1.0)
        batch_shapes = []
        for bi in range(self.__len__()):
            inds = self.indices[bi * self.batch : (bi + 1) * self.batch]
            if not inds:
                continue
            ar = aspect[inds]
            mini, maxi = ar.min(), ar.max()
            if maxi < 1:
                shape = [maxi, 1.0]
            elif mini > 1:
                shape = [1.0, 1.0 / mini]
            else:
                shape = [1.0, 1.0]
            batch_shapes.append(np.ceil(np.asarray(shape) * self.imgsz / stride + pad).astype(int) * stride)
        return np.asarray(batch_shapes, dtype=np.int32)

    def on_epoch_end(self):
        if self.shuffle and not self.rect:
            self.rng.shuffle(self.indices)

    def close_mosaic(self):
        self.hyp["mosaic"] = 0.0
        self.hyp["copy_paste"] = 0.0
        self.hyp["mixup"] = 0.0
        self.hyp["cutmix"] = 0.0
        self.transforms = self.build_transforms()

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        hyp = {**self.hyp, **(hyp or {})}
        if self.augment:
            mosaic = Mosaic(self, imgsz=self.imgsz, p=hyp.get("mosaic", 0.0), n=int(hyp.get("mosaic_n", 4)))
            affine = RandomPerspective(
                degrees=hyp.get("degrees", 0.0),
                translate=hyp.get("translate", 0.1),
                scale=hyp.get("scale", 0.5),
                shear=hyp.get("shear", 0.0),
                perspective=hyp.get("perspective", 0.0),
                pre_transform=LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=True),
            )
            pre_transform = Compose([mosaic, affine])
            if hyp.get("copy_paste_mode", "flip") == "flip":
                pre_transform.insert(1, CopyPaste(p=hyp.get("copy_paste", 0.0), mode="flip"))
            else:
                pre_transform.append(
                    CopyPaste(
                        self,
                        pre_transform=Compose([Mosaic(self, imgsz=self.imgsz, p=hyp.get("mosaic", 0.0), n=int(hyp.get("mosaic_n", 4))), affine]),
                        p=hyp.get("copy_paste", 0.0),
                        mode=hyp.get("copy_paste_mode", "mixup"),
                    )
                )
            transforms = [
                pre_transform,
                MixUp(self, pre_transform=pre_transform, p=hyp.get("mixup", 0.0)),
                CutMix(self, pre_transform=pre_transform, p=hyp.get("cutmix", 0.0)),
                Albumentations(p=1.0, transforms=hyp.get("augmentations")),
                RandomHSV(hgain=hyp.get("hsv_h", 0.015), sgain=hyp.get("hsv_s", 0.7), vgain=hyp.get("hsv_v", 0.4)),
                RandomFlip(direction="vertical", p=hyp.get("flipud", 0.0)),
                RandomFlip(direction="horizontal", p=hyp.get("fliplr", 0.5)),
            ]
        else:
            transforms = [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                batch_idx=True,
                bgr=hyp.get("bgr", 0.0) if self.augment else 0.0,
                mask_ratio=hyp.get("mask_ratio", 4),
                mask_overlap=hyp.get("overlap_mask", True),
            )
        )
        return Compose(transforms)

    def update_labels_info(self, label: dict) -> dict:
        label = dict(label)
        bboxes = np.asarray(label.pop("bboxes", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32).reshape(-1, 4)
        segments = label.pop("segments", [])
        if len(segments):
            max_len = max(len(s) for s in segments)
            n = max(max_len + 1, 1000) if max_len >= 1000 else 1000
            segments = np.stack(resample_segments(segments, n=n), axis=0)
        else:
            segments = np.zeros((0, 1000, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, bbox_format=label.pop("bbox_format", "xywh"), normalized=label.pop("normalized", True))
        return label

    def get_image_and_label(self, idx: int) -> dict:
        path = self.im_files[idx]
        img = self._read_image(idx)
        shape = img.shape[:2]
        label = dict(self.labels[idx])
        label.update(
            {
                "img": img,
                "cls": label["cls"].copy(),
                "bboxes": label["bboxes"].copy(),
                "im_file": str(path),
                "ori_shape": shape,
                "resized_shape": shape,
                "ratio_pad": ((1.0, 1.0), (0, 0)),
            }
        )
        return self.update_labels_info(label)

    def read_label(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        cls, boxes, _ = read_label_file(img2label_path(path), nc=self.data.get("nc"))
        return cls, boxes

    def load_one(self, idx: int, rect_shape: tuple[int, int] | None = None):
        path = self.im_files[idx]
        img0 = self._read_image(idx)
        h0, w0 = img0.shape[:2]
        label = self.labels[idx]
        cls = label["cls"].copy()
        boxes = label["bboxes"].copy()
        new_shape = tuple(int(x) for x in rect_shape) if rect_shape is not None else self.imgsz
        img, ratio, pad = letterbox(img0, new_shape, scaleup=self.augment)
        out_h, out_w = img.shape[:2]
        if len(boxes):
            xyxy = xywh2xyxy_np(boxes)
            xyxy[:, [0, 2]] *= w0
            xyxy[:, [1, 3]] *= h0
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * ratio[0] + pad[0]
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * ratio[1] + pad[1]
            boxes = xyxy2xywh_np(xyxy)
            boxes[:, [0, 2]] /= out_w
            boxes[:, [1, 3]] /= out_h
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

    def get(self, idx: int, rect_shape: tuple[int, int] | None = None):
        if self.rect and rect_shape is not None:
            img, boxes, cls, path, shape, ratio, pad = self.load_one(idx, rect_shape=rect_shape)
        else:
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
        rect_shape = tuple(self.batch_shapes[bi]) if self.batch_shapes is not None and bi < len(self.batch_shapes) else None
        samples = []
        for i in batch_ids:
            labels = self.get_image_and_label(i)
            if rect_shape is not None:
                labels["rect_shape"] = rect_shape
            samples.append(self.transforms(labels))
        return self._collate_for_trainer(samples)

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Ultralytics-style collate for per-sample formatted labels.

        This is used by parity tests and external callers that build transforms
        with ``get_image_and_label``/``Format`` rather than the internal batched
        iterator. The trainer still uses ``__getitem__`` for efficient padded
        numpy batches.
        """
        if not batch:
            raise ValueError("collate_fn received an empty batch")
        imgs = np.stack([np.asarray(x["img"], dtype=np.float32) for x in batch], axis=0)
        bboxes, cls, batch_idx = [], [], []
        for i, sample in enumerate(batch):
            boxes_i = np.asarray(sample.get("bboxes", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32).reshape(-1, 4)
            cls_i = np.asarray(sample.get("cls", np.zeros((len(boxes_i),), dtype=np.int64)), dtype=np.int64).reshape(-1, 1)
            if len(boxes_i):
                bboxes.append(boxes_i)
                cls.append(cls_i)
                batch_idx.append(np.full((len(boxes_i), 1), i, dtype=np.float32))
        out = {
            "img": imgs,
            "bboxes": np.concatenate(bboxes, axis=0) if bboxes else np.zeros((0, 4), dtype=np.float32),
            "cls": np.concatenate(cls, axis=0) if cls else np.zeros((0, 1), dtype=np.int64),
            "batch_idx": np.concatenate(batch_idx, axis=0) if batch_idx else np.zeros((0, 1), dtype=np.float32),
            "im_file": [x.get("im_file", "") for x in batch],
            "ori_shape": [x.get("ori_shape") for x in batch],
            "ratio_pad": [x.get("ratio_pad") for x in batch],
        }
        return out

    @staticmethod
    def _collate_for_trainer(samples: list[dict[str, Any]]) -> dict[str, Any]:
        flat = YOLODataset.collate_fn(samples)
        batch_size = len(samples)
        flat_batch = flat["batch_idx"].astype(np.int64).reshape(-1)
        counts = np.bincount(flat_batch, minlength=batch_size) if len(flat_batch) else np.zeros((batch_size,), dtype=np.int64)
        max_boxes = max(int(counts.max()) if len(counts) else 0, 1)
        bboxes = np.zeros((batch_size, max_boxes, 4), dtype=np.float32)
        cls = np.zeros((batch_size, max_boxes), dtype=np.int64)
        mask = np.zeros((batch_size, max_boxes), dtype=bool)
        cursors = np.zeros((batch_size,), dtype=np.int64)
        flat_boxes = flat["bboxes"].astype(np.float32)
        flat_cls = flat["cls"].reshape(-1).astype(np.int64)
        for row_i, bi in enumerate(flat_batch):
            pos = int(cursors[bi])
            bboxes[bi, pos] = flat_boxes[row_i]
            cls[bi, pos] = flat_cls[row_i]
            mask[bi, pos] = True
            cursors[bi] += 1
        ratio_pad = flat.get("ratio_pad", [((1.0, 1.0), (0, 0)) for _ in samples])
        ratio = [rp[0] if rp else (1.0, 1.0) for rp in ratio_pad]
        pad = [rp[1] if rp else (0, 0) for rp in ratio_pad]
        return {
            "img": flat["img"].astype(np.float32),
            "bboxes": bboxes,
            "cls": cls,
            "mask": mask,
            "batch_idx": flat["batch_idx"].astype(np.float32),
            "flat_cls": flat["cls"].astype(np.float32).reshape(-1, 1),
            "flat_bboxes": flat["bboxes"].astype(np.float32),
            "im_file": flat["im_file"],
            "ori_shape": flat["ori_shape"],
            "ratio": ratio,
            "pad": pad,
        }

    def numeric_batches(self):
        for b in self:
            yield {
                "img": b["img"],
                "bboxes": b["bboxes"],
                "cls": b["cls"],
                "mask": b["mask"],
                "batch_idx": b["batch_idx"],
                "flat_cls": b["flat_cls"],
                "flat_bboxes": b["flat_bboxes"],
            }

    def as_tf_dataset(self, prefetch: int = 2):
        tf = require_tf()
        signature = {
            "img": tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
            "bboxes": tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32),
            "cls": tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            "mask": tf.TensorSpec(shape=(None, None), dtype=tf.bool),
            "batch_idx": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flat_cls": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flat_bboxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        }
        ds = tf.data.Dataset.from_generator(self.numeric_batches, output_signature=signature)
        return ds.prefetch(prefetch)

    def _sample_for_tf(self, idx):
        i = int(np.asarray(idx).reshape(()))
        labels = self.get_image_and_label(i)
        sample = self.transforms(labels)
        img = np.asarray(sample["img"], dtype=np.float32)
        boxes = np.asarray(sample.get("bboxes", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32).reshape(-1, 4)
        cls = np.asarray(sample.get("cls", np.zeros((len(boxes),), dtype=np.int64)), dtype=np.int64).reshape(-1)
        if len(boxes) == 0:
            boxes = np.zeros((1, 4), dtype=np.float32)
            cls = np.zeros((1,), dtype=np.int64)
            mask = np.zeros((1,), dtype=bool)
        else:
            mask = np.ones((len(boxes),), dtype=bool)
        return img, boxes, cls, mask

    def as_fast_tf_dataset(self, prefetch: int | None = None, parallel_calls: int | None = None):
        tf = require_tf()
        autotune = tf.data.AUTOTUNE
        prefetch = autotune if prefetch is None or prefetch <= 0 else int(prefetch)
        parallel_calls = autotune if parallel_calls is None or parallel_calls <= 0 else int(parallel_calls)
        ds = tf.data.Dataset.from_tensor_slices(np.asarray(self.indices, dtype=np.int64))

        def load_sample(idx):
            img, boxes, cls, mask = tf.py_function(self._sample_for_tf, [idx], [tf.float32, tf.float32, tf.int64, tf.bool])
            img.set_shape([None, None, 3])
            boxes.set_shape([None, 4])
            cls.set_shape([None])
            mask.set_shape([None])
            return {"img": img, "bboxes": boxes, "cls": cls, "mask": mask}

        ds = ds.map(load_sample, num_parallel_calls=parallel_calls, deterministic=not self.shuffle)
        ds = ds.padded_batch(
            self.batch,
            padded_shapes={"img": [None, None, 3], "bboxes": [None, 4], "cls": [None], "mask": [None]},
            padding_values={
                "img": np.float32(0.0),
                "bboxes": np.float32(0.0),
                "cls": np.int64(0),
                "mask": np.bool_(False),
            },
            drop_remainder=self.drop_last,
        )
        ds = ds.map(format_fast_batch, num_parallel_calls=autotune)
        return ds.prefetch(prefetch)


def format_fast_batch(batch: dict) -> dict:
    tf = require_tf()
    mask = tf.cast(batch["mask"], tf.bool)
    flat_idx = tf.where(mask)
    flat_bboxes = tf.boolean_mask(batch["bboxes"], mask)
    flat_cls = tf.cast(tf.boolean_mask(batch["cls"], mask)[:, None], tf.float32)
    return {
        "img": tf.cast(batch["img"], tf.float32),
        "bboxes": tf.cast(batch["bboxes"], tf.float32),
        "cls": tf.cast(batch["cls"], tf.int64),
        "mask": mask,
        "batch_idx": tf.cast(flat_idx[:, 0:1], tf.float32),
        "flat_cls": flat_cls,
        "flat_bboxes": tf.cast(flat_bboxes, tf.float32),
    }


def normalize_cache_images(value) -> str:
    if isinstance(value, bool):
        return "ram" if value else "off"
    text = str(value or "off").lower()
    if text in {"true", "1", "yes"}:
        return "ram"
    if text in {"false", "0", "no"}:
        return "off"
    if text not in {"auto", "ram", "off"}:
        raise ValueError("cache_images must be one of {'auto', 'ram', 'off'}.")
    return text
