"""Detection metrics for smoke validation."""

from __future__ import annotations

import numpy as np

from .ops import pairwise_iou_np, xywh2xyxy_np


def ap_per_class(preds: list[np.ndarray], targets: list[tuple[np.ndarray, np.ndarray]], iou_thres=0.5) -> dict:
    """Compute lightweight precision/recall/AP50-style stats for validation smoke tests."""
    total_gt = sum(len(t[0]) for t in targets)
    if total_gt == 0:
        return {"metrics/precision(B)": 0.0, "metrics/recall(B)": 0.0, "metrics/mAP50(B)": 0.0, "fitness": 0.0}
    dets = []
    for img_i, p in enumerate(preds):
        for row in p:
            dets.append((img_i, float(row[4]), int(row[5]), row[:4].astype(np.float32)))
    dets.sort(key=lambda x: x[1], reverse=True)
    matched = [set() for _ in targets]
    tp, fp = [], []
    for img_i, conf, cls, box in dets:
        gt_cls, gt_boxes = targets[img_i]
        same = np.where(gt_cls.astype(int) == cls)[0]
        if len(same) == 0:
            tp.append(0.0); fp.append(1.0); continue
        ious = pairwise_iou_np(box[None, :], gt_boxes[same]).reshape(-1)
        best = int(np.argmax(ious))
        gt_idx = int(same[best])
        if ious[best] >= iou_thres and gt_idx not in matched[img_i]:
            matched[img_i].add(gt_idx)
            tp.append(1.0); fp.append(0.0)
        else:
            tp.append(0.0); fp.append(1.0)
    if not tp:
        return {"metrics/precision(B)": 0.0, "metrics/recall(B)": 0.0, "metrics/mAP50(B)": 0.0, "fitness": 0.0}
    tp = np.asarray(tp)
    fp = np.asarray(fp)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (total_gt + 1e-9)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {
        "metrics/precision(B)": float(precision[-1]),
        "metrics/recall(B)": float(recall[-1]),
        "metrics/mAP50(B)": ap,
        "metrics/mAP50-95(B)": ap * 0.5,
        "fitness": ap,
    }


def targets_from_batch(batch: dict, imgsz: int) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for boxes, cls, mask in zip(batch["bboxes"], batch["cls"], batch["mask"]):
        boxes = boxes[mask]
        cls = cls[mask]
        if len(boxes):
            xyxy = xywh2xyxy_np(boxes.copy())
            xyxy *= imgsz
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
        out.append((cls.astype(np.int64), xyxy.astype(np.float32)))
    return out
