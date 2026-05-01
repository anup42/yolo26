"""Numpy/OpenCV augmentations for YOLO detection training.

The implementations mirror the detection-relevant Ultralytics augmentation
pipeline at commit b4cf7c4751e1d532eb5b0f5a3e9d67b9583964a7 while operating on
the lightweight bbox representation used by this TensorFlow port:
images are RGB uint8 arrays and boxes are normalized xywh arrays.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

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
