"""YOLO26 detection losses rewritten for TensorFlow training.

The loss keeps the public Ultralytics detection behavior that matters for YOLO26:
end-to-end one-to-many/one-to-one branches, TaskAligned assignment, CIoU box
loss, BCE class loss, and the reg_max=1 normalized L1 distance term.
"""

from __future__ import annotations

from .ops import bbox2dist, bbox_iou, dist2bbox, make_anchors
from .tf_import import require_tf

tf = require_tf()


class TaskAlignedAssigner:
    """TensorFlow TaskAlignedAssigner for YOLO detection training."""

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        stride: list[float] | None = None,
        topk2: int | None = None,
    ):
        self.topk = int(topk)
        self.topk2 = int(topk2 or topk)
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.stride = [float(x) for x in (stride or [8.0, 16.0, 32.0])]
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = 1e-9

    def __call__(self, pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        pd_scores = tf.stop_gradient(tf.cast(pd_scores, tf.float32))
        pd_bboxes = tf.stop_gradient(tf.cast(pd_bboxes, tf.float32))
        anchor_points = tf.stop_gradient(tf.cast(anchor_points, tf.float32))
        gt_bboxes = tf.cast(gt_bboxes, tf.float32)
        gt_labels = tf.cast(gt_labels, tf.int32)
        mask_gt = tf.cast(mask_gt, tf.bool) & (tf.reduce_sum(gt_bboxes, axis=-1) > 0)

        bsz = tf.shape(pd_scores)[0]
        anchors_n = tf.shape(pd_scores)[1]
        max_gt = tf.shape(gt_bboxes)[1]

        candidate_gt_bboxes = self._candidate_boxes(gt_bboxes, mask_gt)
        ap = anchor_points[None, None, :, :]
        gtb = candidate_gt_bboxes[:, :, None, :]
        in_gts = (
            (ap[..., 0] > gtb[..., 0])
            & (ap[..., 1] > gtb[..., 1])
            & (ap[..., 0] < gtb[..., 2])
            & (ap[..., 1] < gtb[..., 3])
        )
        in_gts = in_gts & mask_gt[:, :, None]

        ious = pairwise_iou_tf(gt_bboxes, pd_bboxes)
        labels = tf.clip_by_value(gt_labels, 0, self.num_classes - 1)
        label_oh = tf.one_hot(labels, self.num_classes, dtype=pd_scores.dtype)
        cls_scores = tf.reduce_sum(pd_scores[:, None, :, :] * label_oh[:, :, None, :], axis=-1)
        metric = tf.pow(tf.maximum(cls_scores, 0.0), self.alpha) * tf.pow(tf.maximum(ious, 0.0), self.beta)
        metric = tf.where(in_gts, metric, tf.zeros_like(metric))

        mask_pos = self._topk_mask(metric, self.topk)
        if self.topk2 != self.topk:
            mask_pos = self._topk_mask(tf.where(mask_pos, metric, tf.zeros_like(metric)), self.topk2)
        mask_pos_f = tf.cast(mask_pos, tf.float32)

        overlaps = ious * mask_pos_f
        best_gt_idx = tf.argmax(overlaps, axis=1, output_type=tf.int32)
        best_overlap = tf.reduce_max(overlaps, axis=1)
        fg_mask = best_overlap > 0
        target_gt_idx = tf.cast(best_gt_idx, tf.int64)
        target_bboxes = tf.gather(gt_bboxes, best_gt_idx, batch_dims=1)
        target_labels = tf.gather(labels, best_gt_idx, batch_dims=1)

        gt_selector = tf.one_hot(best_gt_idx, max_gt, dtype=tf.float32)
        selected_metric = tf.reduce_sum(tf.transpose(metric, [0, 2, 1]) * gt_selector, axis=-1)
        pos_metric_max = tf.reduce_max(metric * mask_pos_f, axis=-1)
        pos_iou_max = tf.reduce_max(ious * mask_pos_f, axis=-1)
        selected_metric_max = tf.reduce_sum(pos_metric_max[:, None, :] * gt_selector, axis=-1)
        selected_iou_max = tf.reduce_sum(pos_iou_max[:, None, :] * gt_selector, axis=-1)
        norm = selected_metric / (selected_metric_max + self.eps) * selected_iou_max
        norm = tf.where(fg_mask, tf.maximum(norm, self.eps), tf.zeros_like(norm))
        target_scores = tf.one_hot(target_labels, self.num_classes, dtype=tf.float32) * norm[..., None]
        return target_bboxes, target_scores, fg_mask, target_gt_idx

    def _candidate_boxes(self, gt_bboxes, mask_gt):
        """Match Ultralytics TAL small-box expansion before anchor-in-GT checks."""
        x1, y1, x2, y2 = tf.split(gt_bboxes, 4, axis=-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        wh = tf.concat([x2 - x1, y2 - y1], axis=-1)
        min_stride = tf.cast(self.stride[0], gt_bboxes.dtype)
        stride_val = tf.cast(self.stride_val, gt_bboxes.dtype)
        wh = tf.where((wh < min_stride) & mask_gt[..., None], stride_val, wh)
        half = wh * 0.5
        return tf.concat([cx - half[..., 0:1], cy - half[..., 1:2], cx + half[..., 0:1], cy + half[..., 1:2]], axis=-1)

    def _topk_mask(self, metric, topk: int):
        k = tf.minimum(tf.cast(topk, tf.int32), tf.shape(metric)[-1])
        values, _ = tf.math.top_k(metric, k=k)
        threshold = values[..., -1:]
        return (metric >= tf.maximum(threshold, self.eps)) & (metric > self.eps)


def pairwise_iou_tf(boxes1, boxes2, eps: float = 1e-7):
    """Pairwise IoU for boxes shaped [B, N, 4] and [B, A, 4]."""
    b1 = boxes1[:, :, None, :]
    b2 = boxes2[:, None, :, :]
    inter_w = tf.maximum(tf.minimum(b1[..., 2], b2[..., 2]) - tf.maximum(b1[..., 0], b2[..., 0]), 0.0)
    inter_h = tf.maximum(tf.minimum(b1[..., 3], b2[..., 3]) - tf.maximum(b1[..., 1], b2[..., 1]), 0.0)
    inter = inter_w * inter_h
    area1 = tf.maximum(b1[..., 2] - b1[..., 0], 0.0) * tf.maximum(b1[..., 3] - b1[..., 1], 0.0)
    area2 = tf.maximum(b2[..., 2] - b2[..., 0], 0.0) * tf.maximum(b2[..., 3] - b2[..., 1], 0.0)
    return inter / (area1 + area2 - inter + eps)


class BboxLoss:
    def __init__(self, reg_max: int = 1):
        self.reg_max = int(reg_max)

    def __call__(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor):
        weight = tf.reduce_sum(target_scores, axis=-1)
        fg = tf.where(fg_mask)
        return tf.cond(
            tf.shape(fg)[0] > 0,
            lambda: self._loss_non_empty(pred_dist, pred_bboxes, anchor_points, target_bboxes, weight, target_scores_sum, fg, imgsz, stride_tensor),
            lambda: (tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32)),
        )

    def _loss_non_empty(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, weight, target_scores_sum, fg, imgsz, stride_tensor):
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
        return tf.cast(loss_iou, tf.float32), tf.cast(loss_dfl, tf.float32)


class DetectionLoss:
    """YOLO detection loss for one branch."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, hyp: dict | None = None):
        self.nc = model.nc
        self.stride = list(model.stride)
        self.reg_max = int(model.detect_layer.reg_max)
        self.assigner = TaskAlignedAssigner(tal_topk, self.nc, alpha=0.5, beta=6.0, stride=self.stride, topk2=tal_topk2)
        self.bbox_loss = BboxLoss(self.reg_max)
        self.hyp = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 100}
        if hyp:
            self.hyp.update(hyp)
        class_weights = self.hyp.get("class_weights")
        self.class_weights = None if class_weights is None else tf.reshape(tf.cast(class_weights, tf.float32), [1, 1, -1])

    def _targets_to_xyxy(self, batch, imgsz):
        bboxes = tf.cast(batch["bboxes"], tf.float32)
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
        pred_dist = tf.cast(preds["boxes"], tf.float32)
        pred_scores = tf.cast(preds["scores"], tf.float32)
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        imgsz = tf.cast(tf.shape(batch["img"])[1:3], pred_scores.dtype)
        pred_bboxes_grid = self.bbox_decode(anchor_points, pred_dist)
        pred_bboxes = pred_bboxes_grid * stride_tensor[None, :, :]
        gt_labels, gt_bboxes, mask_gt = self._targets_to_xyxy(batch, imgsz)
        target_bboxes, target_scores, fg_mask, _ = self.assigner(
            tf.sigmoid(pred_scores), pred_bboxes, anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )
        target_scores_sum = tf.maximum(tf.reduce_sum(target_scores), 1.0)
        cls_loss_raw = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_scores, logits=pred_scores)
        if self.class_weights is not None:
            cls_loss_raw = cls_loss_raw * self.class_weights
        cls_loss = tf.reduce_sum(cls_loss_raw) / target_scores_sum
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
        loss_items = tf.stack([box_loss * self.hyp["box"], cls_loss * self.hyp["cls"], dfl_loss * self.hyp["dfl"]])
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
