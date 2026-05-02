"""Lightweight Ultralytics-style Instances container for detection augmentations."""

from __future__ import annotations

import numpy as np

from .ops import xywh2xyxy_np, xyxy2xywh_np


class Instances:
    """Container for boxes, optional segments and keypoints with Ultralytics-like transforms."""

    def __init__(self, bboxes, segments=None, keypoints=None, bbox_format: str = "xywh", normalized: bool = True):
        self.bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        self.segments = np.asarray(segments, dtype=np.float32) if segments is not None else np.zeros((0, 1000, 2), dtype=np.float32)
        self.keypoints = None if keypoints is None else np.asarray(keypoints, dtype=np.float32)
        self.bbox_format = bbox_format
        self.normalized = bool(normalized)

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, index):
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        return Instances(self.bboxes[index], segments, keypoints, self.bbox_format, self.normalized)

    @property
    def bbox_areas(self):
        xyxy = self.as_xyxy()
        return np.maximum(xyxy[:, 2] - xyxy[:, 0], 0) * np.maximum(xyxy[:, 3] - xyxy[:, 1], 0)

    def as_xyxy(self):
        if self.bbox_format == "xyxy":
            return self.bboxes.copy()
        if self.bbox_format == "xywh":
            return xywh2xyxy_np(self.bboxes)
        if self.bbox_format == "ltwh":
            out = self.bboxes.copy()
            out[:, 2] = out[:, 0] + out[:, 2]
            out[:, 3] = out[:, 1] + out[:, 3]
            return out
        raise ValueError(f"Unsupported bbox format {self.bbox_format}")

    def convert_bbox(self, format: str):
        if format == self.bbox_format:
            return
        xyxy = self.as_xyxy()
        if format == "xyxy":
            self.bboxes = xyxy
        elif format == "xywh":
            self.bboxes = xyxy2xywh_np(xyxy)
        elif format == "ltwh":
            out = xyxy.copy()
            out[:, 2] -= out[:, 0]
            out[:, 3] -= out[:, 1]
            self.bboxes = out
        else:
            raise ValueError(f"Unsupported bbox format {format}")
        self.bbox_format = format

    def denormalize(self, w: int, h: int):
        if not self.normalized:
            return
        self.bboxes *= np.asarray([w, h, w, h], dtype=np.float32)
        if len(self.segments):
            self.segments[..., 0] *= w
            self.segments[..., 1] *= h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w: int, h: int):
        if self.normalized:
            return
        self.bboxes *= np.asarray([1 / w, 1 / h, 1 / w, 1 / h], dtype=np.float32)
        if len(self.segments):
            self.segments[..., 0] /= w
            self.segments[..., 1] /= h
        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def scale(self, scale_w: float, scale_h: float, bbox_only: bool = False):
        self.bboxes *= np.asarray([scale_w, scale_h, scale_w, scale_h], dtype=np.float32)
        if bbox_only:
            return
        if len(self.segments):
            self.segments[..., 0] *= scale_w
            self.segments[..., 1] *= scale_h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h

    def add_padding(self, padw: float, padh: float):
        if self.normalized:
            raise AssertionError("you should add padding with absolute coordinates.")
        self.bboxes += np.asarray([padw, padh, padw, padh], dtype=np.float32)
        if len(self.segments):
            self.segments[..., 0] += padw
            self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def fliplr(self, w: int):
        if self.bbox_format == "xyxy":
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else:
            self.bboxes[:, 0] = w - self.bboxes[:, 0]
        if len(self.segments):
            self.segments[..., 0] = w - self.segments[..., 0]
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def flipud(self, h: int):
        if self.bbox_format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else:
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
        if len(self.segments):
            self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def clip(self, w: int, h: int):
        old_format = self.bbox_format
        self.convert_bbox("xyxy")
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        if old_format != "xyxy":
            self.convert_bbox(old_format)
        if len(self.segments):
            self.segments[..., 0] = self.segments[..., 0].clip(0, w)
            self.segments[..., 1] = self.segments[..., 1].clip(0, h)

    def remove_zero_area_boxes(self):
        keep = self.bbox_areas > 0
        self.bboxes = self.bboxes[keep]
        if len(self.segments):
            self.segments = self.segments[keep]
        if self.keypoints is not None:
            self.keypoints = self.keypoints[keep]
        return keep

    def update(self, bboxes, segments=None, keypoints=None):
        self.bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        if segments is not None:
            self.segments = np.asarray(segments, dtype=np.float32)
        if keypoints is not None:
            self.keypoints = np.asarray(keypoints, dtype=np.float32)

    @classmethod
    def concatenate(cls, instances_list, axis=0):
        if not instances_list:
            return cls(np.zeros((0, 4), dtype=np.float32))
        first = instances_list[0]
        boxes = np.concatenate([x.bboxes for x in instances_list], axis=axis)
        if all(len(x.segments) for x in instances_list):
            segments = np.concatenate([x.segments for x in instances_list], axis=axis)
        else:
            segments = np.zeros((0, 1000, 2), dtype=np.float32)
        keypoints = np.concatenate([x.keypoints for x in instances_list], axis=axis) if first.keypoints is not None else None
        return cls(boxes, segments, keypoints, first.bbox_format, first.normalized)
