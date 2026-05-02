"""Numpy/OpenCV augmentations for YOLO detection training.

The implementations mirror the detection-relevant Ultralytics augmentation
pipeline at commit b4cf7c4751e1d532eb5b0f5a3e9d67b9583964a7 while operating on
the lightweight bbox representation used by this TensorFlow port:
images are RGB uint8 arrays and boxes are normalized xywh arrays.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .instances import Instances
from .ops import letterbox
from .ops import xywh2xyxy_np, xyxy2xywh_np

try:  # pragma: no cover - exercised when opencv is installed
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _require_cv2():
    if cv2 is None:  # pragma: no cover
        raise ImportError("OpenCV is required for Ultralytics-style geometric augmentations.")
    return cv2


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Intersection over box2 area, matching Ultralytics CutMix/CopyPaste filtering."""
    box1 = np.asarray(box1, dtype=np.float32)
    box2 = np.asarray(box2, dtype=np.float32)
    if box1.ndim == 1:
        box1 = box1[None]
    if box2.ndim == 1:
        box2 = box2[None]
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)
    b1 = box1[:, None, :]
    b2 = box2[None, :, :]
    inter = np.maximum(np.minimum(b1[..., 2], b2[..., 2]) - np.maximum(b1[..., 0], b2[..., 0]), 0) * np.maximum(
        np.minimum(b1[..., 3], b2[..., 3]) - np.maximum(b1[..., 1], b2[..., 1]), 0
    )
    area2 = np.maximum(box2[:, 2] - box2[:, 0], 0) * np.maximum(box2[:, 3] - box2[:, 1], 0)
    return inter / (area2[None] + eps)


def _norm_xywh_to_pixel_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = xywh2xyxy_np(boxes.astype(np.float32))
    out[:, [0, 2]] *= w
    out[:, [1, 3]] *= h
    return out


def _pixel_xyxy_to_norm_xywh(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = boxes.astype(np.float32).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    out = xyxy2xywh_np(boxes)
    out[:, [0, 2]] /= w
    out[:, [1, 3]] /= h
    return out.clip(0, 1).astype(np.float32)


def box_candidates(
    box1: np.ndarray,
    box2: np.ndarray,
    wh_thr: int = 2,
    ar_thr: int = 100,
    area_thr: float = 0.10,
    eps: float = 1e-16,
) -> np.ndarray:
    """Filter boxes using the same width/height/aspect/area checks as Ultralytics."""
    if box1.size == 0 or box2.size == 0:
        return np.zeros((box2.shape[1] if box2.ndim == 2 else 0,), dtype=bool)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def random_hsv(img: np.ndarray, hgain=0.015, sgain=0.7, vgain=0.4) -> np.ndarray:
    """Ultralytics RandomHSV LUT transform adapted for RGB arrays."""
    if img.shape[-1] != 3 or not (hgain or sgain or vgain):
        return img
    cv = _require_cv2()
    dtype = img.dtype
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
    lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
    lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
    lut_sat[0] = 0
    hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_RGB2HSV))
    im_hsv = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val)))
    return cv.cvtColor(im_hsv, cv.COLOR_HSV2RGB)


def random_flip(img: np.ndarray, boxes: np.ndarray, fliplr=0.5, flipud=0.0):
    """Ultralytics RandomFlip semantics for normalized xywh bboxes."""
    boxes = boxes.copy()
    if random.random() < flipud:
        img = np.ascontiguousarray(np.flipud(img))
        if len(boxes):
            boxes[:, 1] = 1.0 - boxes[:, 1]
    if random.random() < fliplr:
        img = np.ascontiguousarray(np.fliplr(img))
        if len(boxes):
            boxes[:, 0] = 1.0 - boxes[:, 0]
    return img, boxes


@dataclass
class RandomPerspectiveConfig:
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    border: tuple[int, int] = (0, 0)


def random_perspective(
    img: np.ndarray,
    boxes: np.ndarray,
    cls: np.ndarray,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    border: tuple[int, int] = (0, 0),
):
    """Apply Ultralytics RandomPerspective/affine to image and normalized xywh boxes."""
    cv = _require_cv2()
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    c = np.eye(3, dtype=np.float32)
    c[0, 2] = -img.shape[1] / 2
    c[1, 2] = -img.shape[0] / 2

    p = np.eye(3, dtype=np.float32)
    p[2, 0] = random.uniform(-perspective, perspective)
    p[2, 1] = random.uniform(-perspective, perspective)

    r = np.eye(3, dtype=np.float32)
    angle = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    r[:2] = cv.getRotationMatrix2D(angle=angle, center=(0, 0), scale=s)

    sh = np.eye(3, dtype=np.float32)
    sh[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    sh[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    t = np.eye(3, dtype=np.float32)
    t[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    t[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    m = t @ sh @ r @ p @ c
    if (border[0] != 0) or (border[1] != 0) or (m != np.eye(3)).any():
        if perspective:
            img = cv.warpPerspective(img, m, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            img = cv.warpAffine(img, m[:2], dsize=(width, height), borderValue=(114, 114, 114))
        if img.ndim == 2:
            img = img[..., None]

    if len(boxes) == 0:
        return img, boxes.astype(np.float32), cls.astype(np.int64)

    h0, w0 = img.shape[:2]
    # Original boxes are relative to the pre-warp image, not the output.
    orig_h = h0 - border[0] * 2
    orig_w = w0 - border[1] * 2
    old = _norm_xywh_to_pixel_xyxy(boxes, orig_w, orig_h)
    n = len(old)
    xy = np.ones((n * 4, 3), dtype=np.float32)
    xy[:, :2] = old[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
    xy = xy @ m.T
    xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=np.float32).reshape(4, n).T
    new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
    new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
    scaled_old = old.copy()
    scaled_old[:, [0, 2]] *= s
    scaled_old[:, [1, 3]] *= s
    keep = box_candidates(scaled_old.T, new.T, area_thr=0.10)
    return img, _pixel_xyxy_to_norm_xywh(new[keep], width, height), cls[keep].astype(np.int64)


def mixup(img1, boxes1, cls1, img2, boxes2, cls2, alpha=32.0):
    """Ultralytics MixUp: beta(32, 32) weighted image blend plus label concat."""
    r = np.random.beta(alpha, alpha)
    img = (img1.astype(np.float32) * r + img2.astype(np.float32) * (1 - r)).astype(np.uint8)
    boxes = np.concatenate([boxes1, boxes2], axis=0) if len(boxes2) else boxes1.copy()
    cls = np.concatenate([cls1, cls2], axis=0) if len(cls2) else cls1.copy()
    return img, boxes.astype(np.float32), cls.astype(np.int64)


def _rand_bbox(width: int, height: int, beta: float = 1.0) -> tuple[int, int, int, int]:
    lam = np.random.beta(beta, beta)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1 = int(np.clip(cx - cut_w // 2, 0, width))
    y1 = int(np.clip(cy - cut_h // 2, 0, height))
    x2 = int(np.clip(cx + cut_w // 2, 0, width))
    y2 = int(np.clip(cy + cut_h // 2, 0, height))
    return x1, y1, x2, y2


def cutmix(img1, boxes1, cls1, img2, boxes2, cls2, beta: float = 1.0, num_areas: int = 3):
    """Ultralytics CutMix behavior for bbox-only detection labels."""
    h, w = img1.shape[:2]
    b1 = _norm_xywh_to_pixel_xyxy(boxes1, w, h)
    b2 = _norm_xywh_to_pixel_xyxy(boxes2, w, h)
    cut_areas = np.asarray([_rand_bbox(w, h, beta) for _ in range(num_areas)], dtype=np.float32)
    ioa1 = bbox_ioa(cut_areas, b1)
    idx = np.nonzero(ioa1.sum(axis=1) <= 0)[0]
    if len(idx) == 0:
        return img1, boxes1, cls1
    area = cut_areas[np.random.choice(idx)]
    ioa2 = bbox_ioa(area[None], b2).squeeze(0) if len(b2) else np.zeros((0,), dtype=np.float32)
    indexes2 = np.nonzero(ioa2 >= 0.1)[0]
    if len(indexes2) == 0:
        return img1, boxes1, cls1
    x1, y1, x2, y2 = area.astype(np.int32)
    out = img1.copy()
    out[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    selected = b2[indexes2].copy()
    selected[:, [0, 2]] = selected[:, [0, 2]].clip(x1, x2)
    selected[:, [1, 3]] = selected[:, [1, 3]].clip(y1, y2)
    valid = (selected[:, 2] - selected[:, 0] > 2) & (selected[:, 3] - selected[:, 1] > 2)
    if not valid.any():
        return img1, boxes1, cls1
    boxes = np.concatenate([boxes1, _pixel_xyxy_to_norm_xywh(selected[valid], w, h)], axis=0)
    cls = np.concatenate([cls1, cls2[indexes2][valid]], axis=0)
    return out, boxes.astype(np.float32), cls.astype(np.int64)


def copy_paste_bbox_only(img, boxes, cls, p: float = 0.0):
    """CopyPaste is segment-based in Ultralytics; bbox-only detection labels are unchanged."""
    return img, boxes, cls


def albumentations(img: np.ndarray, boxes: np.ndarray, cls: np.ndarray, p: float = 1.0):
    """Optional Ultralytics default Albumentations hook for bbox-only labels."""
    if random.random() > p:
        return img, boxes, cls
    try:
        import albumentations as A
    except Exception:
        return img, boxes, cls
    transforms = [A.Blur(p=0.01), A.MedianBlur(p=0.01), A.ToGray(p=0.01), A.CLAHE(p=0.01)]
    transform = A.Compose(transforms)
    return transform(image=img)["image"], boxes, cls


def random_bgr(img: np.ndarray, p: float = 0.0) -> np.ndarray:
    """Ultralytics Format(bgr=...) channel-swap probability."""
    if p and random.random() < p:
        return img[..., ::-1].copy()
    return img


def resample_segments(segments: list[np.ndarray], n: int = 1000) -> list[np.ndarray]:
    """Resample polygon segments to a fixed point count like Ultralytics Format."""
    out = []
    for segment in segments:
        segment = np.asarray(segment, dtype=np.float32).reshape(-1, 2)
        if len(segment) == 0:
            out.append(np.zeros((n, 2), dtype=np.float32))
            continue
        if len(segment) == 1:
            out.append(np.repeat(segment, n, axis=0).astype(np.float32))
            continue
        closed = np.concatenate([segment, segment[:1]], axis=0)
        x = np.linspace(0, len(closed) - 1, n, endpoint=False)
        xp = np.arange(len(closed), dtype=np.float32)
        out.append(np.stack([np.interp(x, xp, closed[:, 0]), np.interp(x, xp, closed[:, 1])], axis=1).astype(np.float32))
    return out


class Compose:
    """Sequential transform container compatible with Ultralytics augmentation flow."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def append(self, transform):
        self.transforms.append(transform)

    def insert(self, index: int, transform):
        self.transforms.insert(index, transform)

    def __getitem__(self, index):
        return self.transforms[index]

    def __setitem__(self, index, value):
        self.transforms[index] = value

    def tolist(self):
        return self.transforms

    def __repr__(self):
        return f"{self.__class__.__name__}({self.transforms})"

    def __call__(self, labels):
        for transform in self.transforms:
            labels = transform(labels)
        return labels


class BaseTransform:
    """Ultralytics-compatible base transform for label dictionaries."""

    def apply_image(self, labels: dict[str, Any]):
        return labels

    def apply_instances(self, labels: dict[str, Any]):
        return labels

    def apply_semantic(self, labels: dict[str, Any]):
        return labels

    def __call__(self, labels: dict[str, Any]):
        labels = self.apply_image(labels)
        labels = self.apply_instances(labels)
        return self.apply_semantic(labels)


class BaseMixTransform:
    """Ultralytics-style mix transform base for Mosaic/MixUp/CutMix/CopyPaste."""

    def __init__(self, dataset=None, pre_transform=None, p: float = 0.0):
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = float(p)

    def get_indexes(self):
        if self.dataset is None:
            return []
        return random.randint(0, len(self.dataset.im_files) - 1)

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - abstract compatibility hook
        raise NotImplementedError

    def _update_label_text(self, labels: dict[str, Any]) -> dict[str, Any]:
        return labels

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random.uniform(0, 1) > self.p or self.dataset is None:
            return labels
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform is not None:
            mix_labels = [self.pre_transform(x) for x in mix_labels]
        labels["mix_labels"] = mix_labels
        labels = self._update_label_text(labels)
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels


def _labels_to_arrays(labels: dict[str, Any]):
    img = labels["img"]
    cls = np.asarray(labels.get("cls", np.zeros((0,), dtype=np.int64))).reshape(-1).astype(np.int64)
    instances = labels.get("instances")
    if instances is None:
        boxes = np.asarray(labels.get("bboxes", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32).reshape(-1, 4)
    else:
        inst = deepcopy(instances)
        inst.convert_bbox("xywh")
        if not inst.normalized:
            inst.normalize(img.shape[1], img.shape[0])
        boxes = inst.bboxes.astype(np.float32)
    return img, boxes, cls


def _arrays_to_labels(labels: dict[str, Any], img, boxes, cls):
    labels["img"] = img
    labels["cls"] = np.asarray(cls, dtype=np.int64).reshape(-1)
    labels["instances"] = Instances(np.asarray(boxes, dtype=np.float32).reshape(-1, 4), bbox_format="xywh", normalized=True)
    return labels


class Mosaic(BaseMixTransform):
    """Mosaic augmentation supporting Ultralytics n=3/4/9 constructor contract."""

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0, n: int = 4):
        super().__init__(dataset=dataset, pre_transform=None, p=p)
        if n not in {3, 4, 9}:
            raise ValueError("Mosaic n must be one of {3, 4, 9}.")
        self.imgsz = int(imgsz)
        self.n = int(n)

    def get_indexes(self):
        count = self.n - 1
        return random.choices(range(len(self.dataset.im_files)), k=count)

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.n == 3:
            return self._mosaic3(labels)
        if self.n == 9:
            return self._mosaic9(labels)
        return self._mosaic4(labels)

    def _mosaic4(self, labels: dict[str, Any]) -> dict[str, Any]:
        mix = [labels, *labels.get("mix_labels", [])]
        while len(mix) < 4:
            mix.append(deepcopy(mix[-1]))
        size = self.imgsz
        out = np.full((size * 2, size * 2, 3), 114, dtype=np.uint8)
        all_boxes, all_cls = [], []
        xc = random.randint(size // 2, size * 3 // 2)
        yc = random.randint(size // 2, size * 3 // 2)
        for i, lb in enumerate(mix[:4]):
            img, boxes, cls = _labels_to_arrays(lb)
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
                xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * w + x1a - x1b
                xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * h + y1a - y1b
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
        labels = _arrays_to_labels(labels, out, boxes, cls)
        labels["mosaic_border"] = (-size // 2, -size // 2)
        return labels

    def _mosaic3(self, labels: dict[str, Any]) -> dict[str, Any]:
        mix = [labels, *labels.get("mix_labels", [])]
        while len(mix) < 3:
            mix.append(deepcopy(mix[-1]))
        s = self.imgsz
        canvas = None
        all_boxes, all_cls = [], []
        crop_x = crop_y = s // 2
        h0 = w0 = 0
        for i, lb in enumerate(mix[:3]):
            img, boxes, cls = _labels_to_arrays(lb)
            h, w = img.shape[:2]
            if canvas is None:
                canvas = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=img.dtype)
            if i == 0:
                h0, w0 = h, w
                c = (s, s, s + w, s + h)
            elif i == 1:
                c = (s + w0, s, s + w0 + w, s + h)
            else:
                c = (s - w, s + h0 - h, s, s + h0)
            self._paste_on_canvas(canvas, img, boxes, cls, c, crop_x, crop_y, all_boxes, all_cls)
        out = canvas[crop_y : crop_y + 2 * s, crop_x : crop_x + 2 * s]
        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4), dtype=np.float32)
        cls = np.concatenate(all_cls, axis=0) if all_cls else np.zeros((0,), dtype=np.int64)
        labels = _arrays_to_labels(labels, out, boxes, cls)
        labels["mosaic_border"] = (-s // 2, -s // 2)
        return labels

    def _mosaic9(self, labels: dict[str, Any]) -> dict[str, Any]:
        mix = [labels, *labels.get("mix_labels", [])]
        while len(mix) < 9:
            mix.append(deepcopy(mix[-1]))
        s = self.imgsz
        canvas = None
        all_boxes, all_cls = [], []
        crop_x = crop_y = s // 2
        h0 = w0 = hp = wp = 0
        for i, lb in enumerate(mix[:9]):
            img, boxes, cls = _labels_to_arrays(lb)
            h, w = img.shape[:2]
            if canvas is None:
                canvas = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=img.dtype)
            if i == 0:
                h0, w0 = h, w
                c = (s, s, s + w, s + h)
            elif i == 1:
                c = (s, s - h, s + w, s)
            elif i == 2:
                c = (s + wp, s - h, s + wp + w, s)
            elif i == 3:
                c = (s + w0, s, s + w0 + w, s + h)
            elif i == 4:
                c = (s + w0, s + hp, s + w0 + w, s + hp + h)
            elif i == 5:
                c = (s + w0 - w, s + h0, s + w0, s + h0 + h)
            elif i == 6:
                c = (s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h)
            elif i == 7:
                c = (s - w, s + h0 - h, s, s + h0)
            else:
                c = (s - w, s + h0 - hp - h, s, s + h0 - hp)
            self._paste_on_canvas(canvas, img, boxes, cls, c, crop_x, crop_y, all_boxes, all_cls)
            hp, wp = h, w
        out = canvas[crop_y : crop_y + 2 * s, crop_x : crop_x + 2 * s]
        boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 4), dtype=np.float32)
        cls = np.concatenate(all_cls, axis=0) if all_cls else np.zeros((0,), dtype=np.int64)
        labels = _arrays_to_labels(labels, out, boxes, cls)
        labels["mosaic_border"] = (-s // 2, -s // 2)
        return labels

    @staticmethod
    def _paste_on_canvas(canvas, img, boxes, cls, coords, crop_x, crop_y, all_boxes, all_cls):
        padw, padh = coords[:2]
        x1, y1, x2, y2 = [max(int(x), 0) for x in coords]
        x2 = min(x2, canvas.shape[1])
        y2 = min(y2, canvas.shape[0])
        if x2 <= x1 or y2 <= y1:
            return
        canvas[y1:y2, x1:x2] = img[y1 - padh : y2 - padh, x1 - padw : x2 - padw]
        if len(boxes):
            h, w = img.shape[:2]
            final = canvas.shape[0] - crop_y * 2
            xyxy = xywh2xyxy_np(boxes)
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * w + padw - crop_x
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * h + padh - crop_y
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, final)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, final)
            b = xyxy2xywh_np(xyxy)
            b[:, [0, 2]] /= final
            b[:, [1, 3]] /= final
            keep = (b[:, 2] > 0) & (b[:, 3] > 0)
            if keep.any():
                all_boxes.append(b[keep].clip(0, 1))
                all_cls.append(cls[keep])


class MixUp(BaseMixTransform):
    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        img1, boxes1, cls1 = _labels_to_arrays(labels)
        img2, boxes2, cls2 = _labels_to_arrays(labels["mix_labels"][0])
        return _arrays_to_labels(labels, *mixup(img1, boxes1, cls1, img2, boxes2, cls2))


class CutMix(BaseMixTransform):
    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        img1, boxes1, cls1 = _labels_to_arrays(labels)
        img2, boxes2, cls2 = _labels_to_arrays(labels["mix_labels"][0])
        return _arrays_to_labels(labels, *cutmix(img1, boxes1, cls1, img2, boxes2, cls2))


class CopyPaste(BaseMixTransform):
    def __init__(self, dataset=None, pre_transform=None, p: float = 0.5, mode: str = "flip"):
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        if mode not in {"flip", "mixup"}:
            raise ValueError("CopyPaste mode must be 'flip' or 'mixup'.")
        self.mode = mode

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "flip" or self.dataset is None:
            img, instances, cls = copy_paste_segments(labels["img"], labels["instances"], labels.get("cls", np.zeros((0,), dtype=np.int64)), self.p)
            labels["img"], labels["instances"], labels["cls"] = img, instances, cls
            return labels
        return super().__call__(labels)

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        labels2 = labels["mix_labels"][0]
        if "instances" not in labels2:
            return labels
        img, inst, cls = copy_paste_segments(labels2["img"], labels2["instances"], labels2.get("cls", np.zeros((0,), dtype=np.int64)), self.p)
        labels2 = {"img": img, "instances": inst, "cls": cls}
        img1, boxes1, cls1 = _labels_to_arrays(labels)
        img2, boxes2, cls2 = _labels_to_arrays(labels2)
        return _arrays_to_labels(labels, *mixup(img1, boxes1, cls1, img2, boxes2, cls2, alpha=32.0))


class RandomPerspective:
    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        border: tuple[int, int] = (0, 0),
        pre_transform=None,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.pre_transform = pre_transform

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.pre_transform is not None and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        border = labels.pop("mosaic_border", self.border)
        img, boxes, cls = _labels_to_arrays(labels)
        labels.pop("ratio_pad", None)
        img, boxes, cls = random_perspective(img, boxes, cls, self.degrees, self.translate, self.scale, self.shear, self.perspective, border)
        return _arrays_to_labels(labels, img, boxes, cls)

    @staticmethod
    def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.10, eps=1e-16):
        return box_candidates(box1, box2, wh_thr, ar_thr, area_thr, eps)


class RandomHSV:
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain, self.sgain, self.vgain = hgain, sgain, vgain

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        labels["img"] = random_hsv(labels["img"], self.hgain, self.sgain, self.vgain)
        return labels


class RandomFlip:
    def __init__(self, p: float = 0.5, direction: str = "horizontal", flip_idx: list[int] | None = None):
        if direction not in {"horizontal", "vertical"}:
            raise ValueError("direction must be 'horizontal' or 'vertical'.")
        self.p = float(p)
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img = labels["img"]
        instances = labels.get("instances")
        if instances is None:
            boxes = np.asarray(labels.get("bboxes", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
            img, boxes = random_flip(img, boxes, fliplr=self.p if self.direction == "horizontal" else 0.0, flipud=self.p if self.direction == "vertical" else 0.0)
            labels["img"], labels["bboxes"] = img, boxes
            return labels
        instances.convert_bbox("xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w
        if self.direction == "vertical" and random.random() < self.p:
            labels["img"] = np.ascontiguousarray(np.flipud(img))
            instances.flipud(h)
        if self.direction == "horizontal" and random.random() < self.p:
            labels["img"] = np.ascontiguousarray(np.fliplr(labels["img"]))
            instances.fliplr(w)
        labels["instances"] = instances
        return labels


class Albumentations:
    def __init__(self, p: float = 1.0, transforms: list | None = None):
        self.p = float(p)
        self.transforms = transforms

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img, boxes, cls = _labels_to_arrays(labels)
        labels["img"], boxes, cls = albumentations(img, boxes, cls, p=self.p)
        return _arrays_to_labels(labels, labels["img"], boxes, cls)


class LetterBox:
    """Label-aware letterbox transform mirroring Ultralytics detection behavior."""

    def __init__(self, new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32, padding_value=114):
        self.new_shape = new_shape
        self.auto = bool(auto)
        self.scale_fill = bool(scale_fill)
        self.scaleup = scaleup
        self.center = center
        self.stride = stride
        self.padding_value = padding_value

    def __call__(self, labels=None, image=None):
        labels = {} if labels is None else labels
        img = labels.get("img") if image is None else image
        label_shape = labels.get("rect_shape", self.new_shape)
        if isinstance(label_shape, int):
            new_shape = (label_shape, label_shape)
        else:
            new_shape = tuple(label_shape)
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        ratio = (r, r)
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])
        if self.center:
            dw /= 2
            dh /= 2
        if shape[::-1] != new_unpad:
            if cv2 is not None:
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            else:  # pragma: no cover
                from PIL import Image

                img = np.asarray(Image.fromarray(img).resize(new_unpad, Image.BILINEAR))
        top = int(round(dh - 0.1)) if self.center else 0
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1)) if self.center else 0
        right = int(round(dw + 0.1))
        if cv2 is not None:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3)
        else:  # pragma: no cover
            img = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=self.padding_value)
        pad = (left, top)
        if not labels:
            return img
        if "instances" in labels:
            inst = labels["instances"]
            inst.convert_bbox("xyxy")
            inst.denormalize(*labels["img"].shape[:2][::-1])
            inst.scale(*ratio)
            inst.add_padding(*pad)
        labels["img"] = img
        labels["ratio_pad"] = (ratio, pad)
        labels["resized_shape"] = img.shape[:2]
        return labels


class Format:
    """Format labels into the TensorFlow port's normalized HWC detection contract."""

    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        batch_idx=True,
        bgr=0.0,
        return_mask=False,
        return_keypoint=False,
        return_obb=False,
        mask_ratio=4,
        mask_overlap=True,
    ):
        self.bbox_format = bbox_format
        self.normalize = bool(normalize)
        self.batch_idx = bool(batch_idx)
        self.bgr = float(bgr)
        self.return_mask = bool(return_mask)
        self.return_keypoint = bool(return_keypoint)
        self.return_obb = bool(return_obb)
        self.mask_ratio = int(mask_ratio)
        self.mask_overlap = bool(mask_overlap)

    def __call__(self, labels):
        img = random_bgr(labels["img"], self.bgr)
        instances = labels.get("instances")
        if instances is None:
            instances = Instances(labels.get("bboxes", np.zeros((0, 4), dtype=np.float32)), bbox_format="xywh", normalized=True)
        h, w = img.shape[:2]
        instances.convert_bbox(self.bbox_format)
        if self.normalize:
            instances.normalize(w, h)
        labels["img"] = img.astype(np.float32) / 255.0
        labels["bboxes"] = instances.bboxes.astype(np.float32)
        labels["cls"] = np.asarray(labels.get("cls", np.zeros((len(instances),), dtype=np.int64))).reshape(-1).astype(np.int64)
        if self.batch_idx:
            labels["batch_idx"] = np.zeros((len(instances), 1), dtype=np.float32)
        if self.return_keypoint and instances.keypoints is not None:
            labels["keypoints"] = instances.keypoints.astype(np.float32)
        if self.return_mask:
            labels["segments"] = instances.segments.astype(np.float32)
        return labels


def copy_paste_segments(img, instances: Instances, cls: np.ndarray, p: float = 0.0):
    """Segment-aware CopyPaste flip mode for detection labels with available segments."""
    if p <= 0 or len(instances) == 0 or not len(instances.segments):
        return img, instances, cls
    h, w = img.shape[:2]
    inst = Instances(instances.bboxes.copy(), instances.segments.copy(), bbox_format=instances.bbox_format, normalized=instances.normalized)
    inst.convert_bbox("xyxy")
    inst.denormalize(w, h)
    flipped = Instances(inst.bboxes.copy(), inst.segments.copy(), bbox_format="xyxy", normalized=False)
    flipped.fliplr(w)
    ioa = bbox_ioa(flipped.bboxes, inst.bboxes)
    indexes = np.nonzero((ioa < 0.30).all(axis=1))[0]
    if not len(indexes):
        return img, instances, cls
    n = round(p * len(indexes))
    if n <= 0:
        return img, instances, cls
    indexes = indexes[:n]
    out = img.copy()
    flipped_img = np.ascontiguousarray(np.fliplr(img))
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv = _require_cv2()
    for j in indexes:
        cv.drawContours(mask, flipped.segments[[j]].astype(np.int32), -1, 1, cv.FILLED)
    mask_bool = mask.astype(bool)
    out[mask_bool] = flipped_img[mask_bool]
    merged = Instances.concatenate([inst, flipped[indexes]], axis=0)
    merged.convert_bbox("xywh")
    merged.normalize(w, h)
    cls_out = np.concatenate([cls.reshape(-1), cls.reshape(-1)[indexes]], axis=0)
    return out, merged, cls_out.astype(np.int64)
