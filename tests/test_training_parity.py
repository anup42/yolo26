import numpy as np
import pytest

from yolo26_tf.losses import TaskAlignedAssigner
from yolo26_tf.trainer import resolve_optimizer_auto


def test_optimizer_auto_matches_ultralytics_defaults():
    assert resolve_optimizer_auto("auto", 0.123, 0.456, iterations=10001, nc=80) == ("musgd", 0.01, 0.9)
    assert resolve_optimizer_auto("auto", 0.123, 0.456, iterations=100, nc=1) == ("adamw", 0.002, 0.9)
    assert resolve_optimizer_auto("sgd", 0.123, 0.456, iterations=100, nc=1) == ("sgd", 0.123, 0.456)


def test_task_aligned_assigner_matches_ultralytics_small_box_behavior():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.utils.tal import TaskAlignedAssigner as TorchTaskAlignedAssigner

    pd_scores_np = np.array([[[0.9, 0.1], [0.8, 0.1], [0.2, 0.7], [0.1, 0.8]]], dtype=np.float32)
    pd_bboxes_np = np.array([[[0, 0, 4, 4], [8, 8, 12, 12], [20, 20, 24, 24], [30, 30, 34, 34]]], dtype=np.float32)
    anchors_np = np.array([[2, 2], [10, 10], [22, 22], [32, 32]], dtype=np.float32)
    gt_labels_np = np.array([[[0]]], dtype=np.int64)
    gt_bboxes_np = np.array([[[1, 1, 3, 3]]], dtype=np.float32)
    mask_gt_np = np.array([[[True]]])

    torch_assigner = TorchTaskAlignedAssigner(topk=2, num_classes=2, alpha=0.5, beta=6.0, stride=[8, 16, 32])
    _, torch_boxes, torch_scores, torch_fg, _ = torch_assigner(
        torch.tensor(pd_scores_np),
        torch.tensor(pd_bboxes_np),
        torch.tensor(anchors_np),
        torch.tensor(gt_labels_np),
        torch.tensor(gt_bboxes_np),
        torch.tensor(mask_gt_np),
    )

    tf_assigner = TaskAlignedAssigner(topk=2, num_classes=2, alpha=0.5, beta=6.0, stride=[8, 16, 32])
    tf_boxes, tf_scores, tf_fg, _ = tf_assigner(
        pd_scores_np,
        pd_bboxes_np,
        anchors_np,
        gt_labels_np[..., 0],
        gt_bboxes_np,
        mask_gt_np[..., 0],
    )

    np.testing.assert_array_equal(tf_fg.numpy(), torch_fg.numpy().astype(bool))
    np.testing.assert_allclose(tf_boxes.numpy(), torch_boxes.numpy(), atol=1e-6)
    np.testing.assert_allclose(tf_scores.numpy(), torch_scores.numpy(), atol=1e-6)
