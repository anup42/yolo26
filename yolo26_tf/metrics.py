"""Ultralytics-style detection metrics for YOLO26 TensorFlow validation."""

from __future__ import annotations

import numpy as np

from .ops import pairwise_iou_np, xywh2xyxy_np


IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10, dtype=np.float32)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP with the same interpolation idea used by Ultralytics/COCO metrics."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def ap_per_class(preds: list[np.ndarray], targets: list[tuple[np.ndarray, np.ndarray]], iou_thres=None) -> dict:
    """Compute precision, recall, mAP50 and mAP50-95 for detection validation."""
    iouv = np.asarray(iou_thres if iou_thres is not None else IOU_THRESHOLDS, dtype=np.float32)
    stats = []
    target_cls_all = []
    for img_i, (pred, (gt_cls, gt_boxes)) in enumerate(zip(preds, targets)):
        gt_cls = gt_cls.astype(np.int64)
        target_cls_all.extend(gt_cls.tolist())
        correct = np.zeros((len(pred), len(iouv)), dtype=bool)
        if len(pred) and len(gt_cls):
            detected = set()
            for cls in np.unique(np.concatenate([pred[:, 5].astype(np.int64), gt_cls])):
                pi = np.where(pred[:, 5].astype(np.int64) == cls)[0]
                ti = np.where(gt_cls == cls)[0]
                if len(pi) == 0 or len(ti) == 0:
                    continue
                ious = pairwise_iou_np(pred[pi, :4], gt_boxes[ti])
                matches = np.argwhere(ious >= iouv[0])
                if len(matches):
                    match_rows = []
                    for p_rel, t_rel in matches:
                        match_rows.append((float(ious[p_rel, t_rel]), int(pi[p_rel]), int(ti[t_rel])))
                    match_rows.sort(reverse=True)
                    used_p, used_t = set(), set()
                    for iou, p_abs, t_abs in match_rows:
                        if p_abs in used_p or t_abs in used_t or t_abs in detected:
                            continue
                        used_p.add(p_abs)
                        used_t.add(t_abs)
                        detected.add(t_abs)
                        correct[p_abs] = iou >= iouv
        if len(pred):
            stats.append((correct, pred[:, 4].astype(np.float32), pred[:, 5].astype(np.int64), gt_cls))

    total_gt = len(target_cls_all)
    if not stats or total_gt == 0:
        return empty_metrics(total_gt)

    correct, conf, pred_cls, _ = (np.concatenate(x, axis=0) for x in zip(*stats))
    order = np.argsort(-conf)
    correct, conf, pred_cls = correct[order], conf[order], pred_cls[order]
    unique_classes = np.unique(np.asarray(target_cls_all, dtype=np.int64))
    ap = np.zeros((len(unique_classes), len(iouv)), dtype=np.float32)
    precision_cls = np.zeros(len(unique_classes), dtype=np.float32)
    recall_cls = np.zeros(len(unique_classes), dtype=np.float32)
    nt_per_class = np.array([(np.asarray(target_cls_all) == c).sum() for c in unique_classes], dtype=np.float32)

    for ci, c in enumerate(unique_classes):
        idx = pred_cls == c
        n_l = nt_per_class[ci]
        n_p = idx.sum()
        if n_p == 0 or n_l == 0:
            continue
        tpc = correct[idx].cumsum(0)
        fpc = (1 - correct[idx]).cumsum(0)
        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc + 1e-16)
        precision_cls[ci] = precision[-1, 0]
        recall_cls[ci] = recall[-1, 0]
        for j in range(len(iouv)):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    mp = float(precision_cls.mean()) if len(precision_cls) else 0.0
    mr = float(recall_cls.mean()) if len(recall_cls) else 0.0
    map50 = float(ap[:, 0].mean()) if ap.size else 0.0
    map5095 = float(ap.mean()) if ap.size else 0.0
    return {
        "metrics/precision(B)": mp,
        "metrics/recall(B)": mr,
        "metrics/mAP50(B)": map50,
        "metrics/mAP50-95(B)": map5095,
        "fitness": map5095,
        "nt_per_class": {int(c): int(n) for c, n in zip(unique_classes, nt_per_class)},
        "ap_class_index": [int(x) for x in unique_classes],
    }


def empty_metrics(total_gt: int = 0) -> dict:
    return {
        "metrics/precision(B)": 0.0,
        "metrics/recall(B)": 0.0,
        "metrics/mAP50(B)": 0.0,
        "metrics/mAP50-95(B)": 0.0,
        "fitness": 0.0,
        "nt_per_class": {},
        "ap_class_index": [],
        "targets": int(total_gt),
    }


class ConfusionMatrix:
    """Small detection confusion matrix compatible with Ultralytics-style reporting."""

    def __init__(self, nc: int, conf: float = 0.25, iou_thres: float = 0.45):
        self.nc = int(nc)
        self.conf = float(conf)
        self.iou_thres = float(iou_thres)
        self.matrix = np.zeros((self.nc + 1, self.nc + 1), dtype=np.int64)

    def process_batch(self, detections: np.ndarray, gt_cls: np.ndarray, gt_boxes: np.ndarray):
        detections = detections[detections[:, 4] >= self.conf] if len(detections) else detections
        if len(gt_cls) == 0:
            for dc in detections[:, 5].astype(int):
                self.matrix[self.nc, dc] += 1
            return
        if len(detections) == 0:
            for gc in gt_cls.astype(int):
                self.matrix[gc, self.nc] += 1
            return
        ious = pairwise_iou_np(gt_boxes, detections[:, :4])
        matches = np.argwhere(ious > self.iou_thres)
        used_gt, used_det = set(), set()
        if len(matches):
            rows = sorted([(float(ious[g, d]), int(g), int(d)) for g, d in matches], reverse=True)
            for _, g, d in rows:
                if g in used_gt or d in used_det:
                    continue
                used_gt.add(g)
                used_det.add(d)
                self.matrix[int(gt_cls[g]), int(detections[d, 5])] += 1
        for g, gc in enumerate(gt_cls.astype(int)):
            if g not in used_gt:
                self.matrix[gc, self.nc] += 1
        for d, dc in enumerate(detections[:, 5].astype(int)):
            if d not in used_det:
                self.matrix[self.nc, dc] += 1


def targets_from_batch(batch: dict, img_shape: int | tuple[int, int]) -> list[tuple[np.ndarray, np.ndarray]]:
    if isinstance(img_shape, int):
        h = w = int(img_shape)
    else:
        h, w = int(img_shape[0]), int(img_shape[1])
    out = []
    for boxes, cls, mask in zip(batch["bboxes"], batch["cls"], batch["mask"]):
        boxes = boxes[mask]
        cls = cls[mask]
        if len(boxes):
            xyxy = xywh2xyxy_np(boxes.copy())
            xyxy[:, [0, 2]] *= w
            xyxy[:, [1, 3]] *= h
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
        out.append((cls.astype(np.int64), xyxy.astype(np.float32)))
    return out
