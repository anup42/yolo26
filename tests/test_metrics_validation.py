import numpy as np
import inspect

from scripts.make_tiny_dataset import create_tiny_dataset
from yolo26_tf.metrics import ConfusionMatrix, DetMetrics, ap_per_class, process_batch, targets_from_batch
from yolo26_tf.validation import prediction_to_detections, validate_detection_model
from yolo26_tf.tf_import import require_tf

tf = require_tf()


def test_ap_per_class_reports_map50_95():
    preds = [np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)]
    targets = [(np.array([0], dtype=np.int64), np.array([[0, 0, 10, 10]], dtype=np.float32))]
    metrics = ap_per_class(preds, targets)
    assert metrics["metrics/precision(B)"] > 0.99
    assert metrics["metrics/recall(B)"] > 0.99
    assert metrics["metrics/mAP50(B)"] > 0.99
    assert metrics["metrics/mAP50-95(B)"] > 0.99


def test_confusion_matrix_counts_background():
    cm = ConfusionMatrix(nc=2, conf=0.25, iou_thres=0.5)
    det = np.array([[0, 0, 10, 10, 0.9, 1]], dtype=np.float32)
    cm.process_batch(det, np.array([0], dtype=np.int64), np.array([[20, 20, 30, 30]], dtype=np.float32))
    assert cm.matrix[0, 2] == 1
    assert cm.matrix[2, 1] == 1


def test_targets_from_batch_uses_rectangular_input_shape():
    batch = {
        "bboxes": np.array([[[0.5, 0.5, 0.5, 0.25]]], dtype=np.float32),
        "cls": np.array([[1]], dtype=np.int64),
        "mask": np.array([[True]]),
    }
    targets = targets_from_batch(batch, (64, 128))
    cls, boxes = targets[0]
    assert cls.tolist() == [1]
    np.testing.assert_allclose(boxes, np.array([[32.0, 24.0, 96.0, 40.0]], dtype=np.float32))


def test_det_metrics_exposes_ultralytics_keys():
    preds = [np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)]
    targets = [(np.array([0], dtype=np.int64), np.array([[0, 0, 10, 10]], dtype=np.float32))]
    metrics = DetMetrics(names={0: "object"})
    metrics.update_stats(preds, targets)
    result = metrics.process()
    assert all(k in result for k in DetMetrics.keys)
    assert metrics.fitness > 0.99
    assert metrics.summary()[0]["Class"] == "object"


def test_det_metrics_stats_path_matches_prediction_path():
    det = np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.8, 1]], dtype=np.float32)
    gt_cls = np.array([0], dtype=np.int64)
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    correct = process_batch(det, gt_cls, gt_boxes)

    metrics = DetMetrics(names={0: "object", 1: "other"})
    metrics.update_stats(
        {
            "tp": correct,
            "conf": det[:, 4],
            "pred_cls": det[:, 5],
            "target_cls": gt_cls,
            "target_img": np.unique(gt_cls),
            "im_name": np.array(["img0.jpg"], dtype=object),
        }
    )
    result = metrics.process()
    assert result["metrics/mAP50(B)"] > 0.99
    assert result["nt_per_class"] == {0: 1}


def test_det_metrics_stats_path_keeps_target_counts_without_predictions():
    metrics = DetMetrics(names={0: "object"})
    metrics.update_stats(
        {
            "tp": np.zeros((0, 10), dtype=bool),
            "conf": np.zeros((0,), dtype=np.float32),
            "pred_cls": np.zeros((0,), dtype=np.float32),
            "target_cls": np.array([0], dtype=np.int64),
            "target_img": np.array([0], dtype=np.int64),
        }
    )
    result = metrics.process()
    assert result["metrics/mAP50(B)"] == 0.0
    assert result["nt_per_class"] == {0: 1}


def test_prediction_to_detections_supports_multi_label_nms():
    pred = np.array([[0, 0, 10, 10, 0.1, 0.8, 0.7]], dtype=np.float32)
    det = prediction_to_detections(pred, conf=0.5, iou=0.5, max_det=10, multi_label=True)
    assert len(det) == 2
    assert set(det[:, 5].astype(int).tolist()) == {1, 2}


def test_process_batch_matches_independently_per_iou_threshold():
    det = np.array([[0, 0, 10, 10, 0.9, 0], [0, 0, 8, 8, 0.8, 0]], dtype=np.float32)
    gt_cls = np.array([0], dtype=np.int64)
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    correct = process_batch(det, gt_cls, gt_boxes, iouv=np.array([0.5, 0.75, 0.95], dtype=np.float32))
    assert correct.shape == (2, 3)
    assert correct[0].tolist() == [True, True, True]
    assert correct[1].tolist() == [False, False, False]


def test_det_metrics_summary_uses_per_class_values():
    det = np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.8, 1]], dtype=np.float32)
    gt_cls = np.array([0, 1], dtype=np.int64)
    gt_boxes = np.array([[0, 0, 10, 10], [40, 40, 50, 50]], dtype=np.float32)
    correct = process_batch(det, gt_cls, gt_boxes)
    metrics = DetMetrics(names={0: "hit", 1: "miss"})
    metrics.update_stats(
        {
            "tp": correct,
            "conf": det[:, 4],
            "pred_cls": det[:, 5],
            "target_cls": gt_cls,
            "target_img": np.unique(gt_cls),
        }
    )
    metrics.process()
    rows = {x["Class"]: x for x in metrics.summary()}
    assert rows["hit"]["metrics/mAP50(B)"] > rows["miss"]["metrics/mAP50(B)"]
    assert rows["miss"]["metrics/mAP50(B)"] == 0.0


def test_validation_defaults_match_ultralytics_detect_val():
    sig = inspect.signature(validate_detection_model)
    assert sig.parameters["conf"].default == 0.001
    assert sig.parameters["iou"].default == 0.7
    assert sig.parameters["multi_label"].default is True


def test_validation_fast_nms_matches_numpy_path_on_simple_e2e(tmp_path):
    data = create_tiny_dataset(tmp_path / "tiny_val_nms", n=2, size=32)

    class OneBoxModel:
        def __call__(self, images, training=False):
            b = tf.shape(images)[0]
            row = tf.constant([[8.0, 8.0, 24.0, 24.0, 0.9, 0.0]], tf.float32)
            return tf.tile(row[None], [b, 1, 1])

    fast = validate_detection_model(OneBoxModel(), data, imgsz=32, batch=2, conf=0.1, fast_nms=True, verbose=False)
    slow = validate_detection_model(OneBoxModel(), data, imgsz=32, batch=2, conf=0.1, fast_nms=False, verbose=False)
    assert fast["predictions"] == slow["predictions"]
    assert fast["metrics/mAP50(B)"] == slow["metrics/mAP50(B)"]
