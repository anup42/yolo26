"""YOLO26 detection losses rewritten for TensorFlow eager training."""

from __future__ import annotations

import numpy as np

from .ops import bbox2dist, bbox_iou, dist2bbox, make_anchors, pairwise_iou_np, xywh2xyxy_np
from .tf_import import require_tf

tf = require_tf()


def _to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


class TaskAlignedAssigner:
    """Numpy/eager TaskAlignedAssigner matching the public YOLO task-aligned assignment logic."""

    def __init__(self, topk: int = 10, num_classes: int = 80, alpha: float = 0.5, beta: float = 6.0, topk2: int | None = None):
        self.topk = int(topk)
        self.topk2 = int(topk2 or topk)
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = 1e-9

    def __call__(self, pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        scores = _to_numpy(tf.stop_gradient(pd_scores)).astype(np.float32)
        boxes = _to_numpy(tf.stop_gradient(pd_bboxes)).astype(np.float32)
        anchors = _to_numpy(tf.stop_gradient(anchor_points)).astype(np.float32)
        labels = _to_numpy(gt_labels).astype(np.int64)
        gt = _to_numpy(gt_bboxes).astype(np.float32)
        mask = _to_numpy(mask_gt).astype(bool)
        bsz, anchors_n, nc = scores.shape
        target_bboxes = np.zeros((bsz, anchors_n, 4), dtype=np.float32)
        target_scores = np.zeros((bsz, anchors_n, nc), dtype=np.float32)
        fg_mask = np.zeros((bsz, anchors_n), dtype=bool)
        target_gt_idx = np.zeros((bsz, anchors_n), dtype=np.int64)
        for b in range(bsz):
            valid = mask[b] & (gt[b].sum(axis=-1) > 0)
            if not np.any(valid):
                continue
            gtb = gt[b][valid]
            gtl = labels[b][valid].reshape(-1)
            n_gt = len(gtb)
            ap = anchors[None, :, :]
            in_gts = (
                (ap[..., 0] > gtb[:, None, 0])
                & (ap[..., 1] > gtb[:, None, 1])
                & (ap[..., 0] < gtb[:, None, 2])
                & (ap[..., 1] < gtb[:, None, 3])
            )
            ious = pairwise_iou_np(gtb, boxes[b]).clip(0.0)
            cls_scores = scores[b][None, :, :].repeat(n_gt, axis=0)
            cls_take = cls_scores[np.arange(n_gt)[:, None], np.arange(anchors_n)[None, :], gtl[:, None]]
            metric = (cls_take**self.alpha) * (ious**self.beta) * in_gts
            mask_pos = np.zeros_like(metric, dtype=bool)
            k = min(self.topk, anchors_n)
            for gi in range(n_gt):
                if metric[gi].max() <= self.eps:
                    continue
                idx = np.argpartition(-metric[gi], k - 1)[:k]
                mask_pos[gi, idx] = metric[gi, idx] > self.eps
            if self.topk2 != self.topk:
                refined = np.zeros_like(mask_pos)
                k2 = min(self.topk2, anchors_n)
                for gi in range(n_gt):
                    vals = metric[gi] * mask_pos[gi]
                    if vals.max() <= self.eps:
                        continue
                    idx = np.argpartition(-vals, k2 - 1)[:k2]
                    refined[gi, idx] = vals[idx] > self.eps
                mask_pos = refined
            if not mask_pos.any():
                continue
            # Resolve anchors assigned to multiple GTs by highest IoU.
            overlaps = ious * mask_pos
            best_gt = overlaps.argmax(axis=0)
            best_val = overlaps.max(axis=0)
            pos = best_val > 0
            assigned = best_gt[pos]
            anchor_idx = np.where(pos)[0]
            fg_mask[b, anchor_idx] = True
            target_gt_idx[b, anchor_idx] = assigned
            target_bboxes[b, anchor_idx] = gtb[assigned]
            # Normalized target score as in TAL: alignment metric normalized by best GT metric/IoU.
            for a, gi in zip(anchor_idx, assigned):
                gi_metric = metric[gi]
                norm = gi_metric[a] / (gi_metric.max() + self.eps) * ious[gi].max()
                target_scores[b, a, gtl[gi]] = float(max(norm, self.eps))
        return (
            tf.convert_to_tensor(target_bboxes, tf.float32),
            tf.convert_to_tensor(target_scores, tf.float32),
            tf.convert_to_tensor(fg_mask, tf.bool),
            tf.convert_to_tensor(target_gt_idx, tf.int64),
        )


class BboxLoss:
    def __init__(self, reg_max: int = 1):
        self.reg_max = int(reg_max)

    def __call__(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor):
        weight = tf.reduce_sum(target_scores, axis=-1)
        fg = tf.where(fg_mask)
        if tf.shape(fg)[0] == 0:
            return tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32)
        pd = tf.gather_nd(pred_bboxes, fg)
        target_bboxes_grid = target_bboxes / stride_tensor[None, :, :]
        tb = tf.gather_nd(target_bboxes_grid, fg)
        w = tf.gather_nd(weight, fg)[:, None]
        ciou = bbox_iou(pd, tb, ciou=True)
        loss_iou = tf.reduce_sum((1.0 - ciou) * w) / target_scores_sum
        target_ltrb = bbox2dist(anchor_points[None, :, :], target_bboxes_grid)
        pdist = pred_dist
        # YOLO26 has reg_max=1; upstream uses normalized L1 distance when DFL is disabled.
        target_ltrb = target_ltrb * stride_tensor[None, :, :]
        pdist = pdist * stride_tensor[None, :, :]
        norm = tf.reshape(tf.cast([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], pdist.dtype), [1, 1, 4])
        target_ltrb = target_ltrb / norm
        pdist = pdist / norm
        loss_l1 = tf.reduce_mean(tf.abs(tf.gather_nd(pdist, fg) - tf.gather_nd(target_ltrb, fg)), axis=-1, keepdims=True)
        loss_dfl = tf.reduce_sum(loss_l1 * w) / target_scores_sum
        return loss_iou, loss_dfl


class DetectionLoss:
    """YOLO detection loss for one branch."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, hyp: dict | None = None):
        self.nc = model.nc
        self.stride = list(model.stride)
        self.reg_max = int(model.detect_layer.reg_max)
        self.assigner = TaskAlignedAssigner(tal_topk, self.nc, alpha=0.5, beta=6.0, topk2=tal_topk2)
        self.bbox_loss = BboxLoss(self.reg_max)
        self.hyp = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 100}
        if hyp:
            self.hyp.update(hyp)

    def _targets_to_xyxy(self, batch, imgsz):
        bboxes = batch["bboxes"]
        cls = tf.cast(batch["cls"], tf.int64)
        mask = tf.cast(batch.get("mask", tf.reduce_sum(bboxes, axis=-1) > 0), tf.bool)
        scale = tf.reshape(tf.cast([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], bboxes.dtype), [1, 1, 4])
        xywh = bboxes * scale
        x, y, w, h = tf.split(xywh, 4, axis=-1)
        xyxy = tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)
        return cls, xyxy, mask

    def bbox_decode(self, anchor_points, pred_dist):
        return dist2bbox(pred_dist, anchor_points[None, :, :], xywh=False)

    def loss(self, preds: dict, batch: dict):
        pred_dist = preds["boxes"]
        pred_scores = preds["scores"]
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        imgsz = tf.cast(tf.shape(batch["img"])[1:3], pred_scores.dtype)
        pred_bboxes_grid = self.bbox_decode(anchor_points, pred_dist)
        pred_bboxes = pred_bboxes_grid * stride_tensor[None, :, :]
        gt_labels, gt_bboxes, mask_gt = self._targets_to_xyxy(batch, imgsz)
        target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            tf.sigmoid(pred_scores), pred_bboxes, anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )
        target_scores_sum = tf.maximum(tf.reduce_sum(target_scores), 1.0)
        cls_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_scores, logits=pred_scores)) / target_scores_sum
        box_loss, dfl_loss = self.bbox_loss(
            pred_dist,
            pred_bboxes_grid,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )
        loss_items = tf.stack(
            [box_loss * self.hyp["box"], cls_loss * self.hyp["cls"], dfl_loss * self.hyp["dfl"]]
        )
        return tf.reduce_sum(loss_items) * tf.cast(tf.shape(pred_scores)[0], tf.float32), loss_items

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        return self.loss(preds, batch)


class E2ELoss:
    """YOLO26 end-to-end loss combining one-to-many and one-to-one branches."""

    def __init__(self, model, hyp: dict | None = None):
        self.one2many = DetectionLoss(model, tal_topk=10, hyp=hyp)
        self.one2one = DetectionLoss(model, tal_topk=1, hyp=hyp)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        loss_m, items_m = self.one2many.loss(preds["one2many"], batch)
        loss_o, items_o = self.one2one.loss(preds["one2one"], batch)
        return loss_m + loss_o, items_m + items_o

    def update(self):
        return None
