"""General ops for the YOLO26 TensorFlow port.

Derived from public Ultralytics YOLO detection utilities at commit
b4cf7c4751e1d532eb5b0f5a3e9d67b9583964a7, rewritten for TensorFlow/Numpy.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from PIL import Image

from .tf_import import require_tf


def make_divisible(x: float, divisor: int = 8) -> int:
    """Return nearest integer divisible by divisor, matching Ultralytics scaling."""
    return int(math.ceil(x / divisor) * divisor)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def xywh2xyxy_np(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh_np(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def letterbox(
    image: np.ndarray | Image.Image,
    new_shape: int | tuple[int, int] = 640,
    color: tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Resize/pad image to `new_shape` while preserving aspect ratio."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h0, w0 = image.shape[:2]
    h, w = new_shape
    r = min(h / h0, w / w0)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = w - new_unpad[0], h - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w0, h0) != new_unpad:
        image = np.asarray(Image.fromarray(image).resize(new_unpad, Image.BILINEAR))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = np.full((h, w, image.shape[2]), color, dtype=image.dtype)
    out[top : top + image.shape[0], left : left + image.shape[1]] = image
    return out, (r, r), (left, top)


def scale_boxes_np(boxes: np.ndarray, from_shape: tuple[int, int], to_shape: tuple[int, int], ratio_pad=None) -> np.ndarray:
    """Scale xyxy boxes from letterboxed image shape back to original image shape."""
    boxes = boxes.copy()
    if ratio_pad is None:
        gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
        pad = ((from_shape[1] - to_shape[1] * gain) / 2, (from_shape[0] - to_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, to_shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, to_shape[0])
    return boxes


def make_anchors(feats: list, strides: Iterable[float], grid_cell_offset: float = 0.5):
    """Generate anchor points and stride tensors from NHWC feature maps."""
    tf = require_tf()
    anchors, stride_tensors = [], []
    dtype = feats[0].dtype
    for feat, stride in zip(feats, strides):
        shape = tf.shape(feat)
        h, w = shape[1], shape[2]
        sx = tf.cast(tf.range(w), dtype) + tf.cast(grid_cell_offset, dtype)
        sy = tf.cast(tf.range(h), dtype) + tf.cast(grid_cell_offset, dtype)
        yy, xx = tf.meshgrid(sy, sx, indexing="ij")
        points = tf.stack([tf.reshape(xx, [-1]), tf.reshape(yy, [-1])], axis=-1)
        anchors.append(points)
        stride_tensors.append(tf.fill([tf.shape(points)[0], 1], tf.cast(stride, dtype)))
    return tf.concat(anchors, axis=0), tf.concat(stride_tensors, axis=0)


def dist2bbox(distance, anchor_points, xywh: bool = True):
    """Transform ltrb distance predictions to xywh or xyxy boxes."""
    tf = require_tf()
    lt, rb = tf.split(distance, 2, axis=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return tf.concat([c_xy, wh], axis=-1)
    return tf.concat([x1y1, x2y2], axis=-1)


def bbox2dist(anchor_points, bbox, reg_max: float | None = None):
    """Transform xyxy boxes to ltrb distances from anchor points."""
    tf = require_tf()
    x1y1, x2y2 = tf.split(bbox, 2, axis=-1)
    dist = tf.concat([anchor_points - x1y1, x2y2 - anchor_points], axis=-1)
    if reg_max is not None:
        dist = tf.clip_by_value(dist, 0.0, reg_max - 0.01)
    return dist


def bbox_iou(box1, box2, ciou: bool = False, eps: float = 1e-7):
    """Elementwise IoU/CIoU for xyxy boxes with matching shape."""
    tf = require_tf()
    b1_x1, b1_y1, b1_x2, b1_y2 = tf.split(box1, 4, axis=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = tf.split(box2, 4, axis=-1)
    inter = tf.maximum(tf.minimum(b1_x2, b2_x2) - tf.maximum(b1_x1, b2_x1), 0) * tf.maximum(
        tf.minimum(b1_y2, b2_y2) - tf.maximum(b1_y1, b2_y1), 0
    )
    w1, h1 = tf.maximum(b1_x2 - b1_x1, 0), tf.maximum(b1_y2 - b1_y1, 0)
    w2, h2 = tf.maximum(b2_x2 - b2_x1, 0), tf.maximum(b2_y2 - b2_y1, 0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if not ciou:
        return iou
    cw = tf.maximum(b1_x2, b2_x2) - tf.minimum(b1_x1, b2_x1)
    ch = tf.maximum(b1_y2, b2_y2) - tf.minimum(b1_y1, b2_y1)
    c2 = cw * cw + ch * ch + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    v = (4 / math.pi**2) * tf.square(tf.atan(w2 / (h2 + eps)) - tf.atan(w1 / (h1 + eps)))
    alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def pairwise_iou_np(boxes1: np.ndarray, boxes2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Pairwise IoU for xyxy numpy boxes."""
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    b1 = boxes1[:, None, :]
    b2 = boxes2[None, :, :]
    inter_w = np.maximum(np.minimum(b1[..., 2], b2[..., 2]) - np.maximum(b1[..., 0], b2[..., 0]), 0)
    inter_h = np.maximum(np.minimum(b1[..., 3], b2[..., 3]) - np.maximum(b1[..., 1], b2[..., 1]), 0)
    inter = inter_w * inter_h
    a1 = np.maximum(b1[..., 2] - b1[..., 0], 0) * np.maximum(b1[..., 3] - b1[..., 1], 0)
    a2 = np.maximum(b2[..., 2] - b2[..., 0], 0) * np.maximum(b2[..., 3] - b2[..., 1], 0)
    return inter / (a1 + a2 - inter + eps)


def nms_numpy(pred: np.ndarray, conf: float = 0.25, iou: float = 0.45, max_det: int = 300) -> np.ndarray:
    """Simple class-aware NMS for predictions shaped [N, 6] = xyxy, conf, cls."""
    pred = pred[pred[:, 4] >= conf]
    if len(pred) == 0:
        return pred.reshape(0, 6)
    keep = []
    for cls in np.unique(pred[:, 5].astype(int)):
        det = pred[pred[:, 5].astype(int) == cls]
        order = det[:, 4].argsort()[::-1]
        while order.size > 0 and len(keep) < max_det:
            idx = order[0]
            keep.append(det[idx])
            if order.size == 1:
                break
            ious = pairwise_iou_np(det[idx : idx + 1, :4], det[order[1:], :4]).reshape(-1)
            order = order[1:][ious <= iou]
    if not keep:
        return np.zeros((0, 6), dtype=np.float32)
    out = np.stack(keep, axis=0).astype(np.float32)
    return out[out[:, 4].argsort()[::-1]][:max_det]
