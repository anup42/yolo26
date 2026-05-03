import numpy as np
import pytest

from yolo26_tf.losses import TaskAlignedAssigner
from yolo26_tf.optim import FastSGD, make_optimizer
from yolo26_tf.trainer import resolve_optimizer_auto, variable_decay_group


def test_optimizer_auto_matches_ultralytics_defaults():
    assert resolve_optimizer_auto("auto", 0.123, 0.456, iterations=10001, nc=80) == ("musgd", 0.01, 0.9)
    assert resolve_optimizer_auto("auto", 0.123, 0.456, iterations=100, nc=1) == ("adamw", 0.002, 0.9)
    assert resolve_optimizer_auto("sgd", 0.123, 0.456, iterations=100, nc=1) == ("sgd", 0.123, 0.456)


def test_musgd_defaults_match_ultralytics_coefficients():
    opt = make_optimizer("musgd", lr=0.01, momentum=0.9, weight_decay=0.0, iterations=10001)
    assert opt.muon == 0.2
    assert opt.sgd == 1.0


def test_sgd_default_uses_non_xla_fast_sgd():
    tf = pytest.importorskip("tensorflow")
    opt = make_optimizer("sgd", lr=0.1, momentum=0.0, weight_decay=0.0, iterations=100)
    assert isinstance(opt, FastSGD)
    var = tf.Variable([1.0], dtype=tf.float32)
    opt.build([var])
    opt.apply_gradients([(tf.constant([0.5], dtype=tf.float32), var)])
    np.testing.assert_allclose(var.numpy(), [0.95], atol=1e-6)
    keras_opt = make_optimizer("tfsgd", lr=0.1, momentum=0.0, weight_decay=0.0, iterations=100)
    assert isinstance(keras_opt, tf.keras.optimizers.Optimizer)


def test_variable_decay_group_matches_bias_norm_decay_split():
    tf = pytest.importorskip("tensorflow")
    kernel = tf.Variable(np.ones((3, 3, 1, 4), dtype=np.float32), name="conv/kernel")
    bias = tf.Variable(np.ones((4,), dtype=np.float32), name="conv/bias")
    norm = tf.Variable(np.ones((4,), dtype=np.float32), name="batch_normalization/gamma")
    assert variable_decay_group(kernel) == "decay"
    assert variable_decay_group(bias) == "bias"
    assert variable_decay_group(norm) == "norm"


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


def test_task_aligned_assigner_matches_ultralytics_multi_gt_conflict():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.utils.tal import TaskAlignedAssigner as TorchTaskAlignedAssigner

    pd_scores_np = np.array([[[0.9, 0.8], [0.85, 0.7], [0.2, 0.95]]], dtype=np.float32)
    pd_bboxes_np = np.array([[[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]]], dtype=np.float32)
    anchors_np = np.array([[5, 5], [6, 6], [25, 25]], dtype=np.float32)
    gt_labels_np = np.array([[[0], [1]]], dtype=np.int64)
    gt_bboxes_np = np.array([[[0, 0, 12, 12], [2, 2, 14, 14]]], dtype=np.float32)
    mask_gt_np = np.array([[[True], [True]]])

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
        gt_labels_np,
        gt_bboxes_np,
        mask_gt_np,
    )

    np.testing.assert_array_equal(tf_fg.numpy(), torch_fg.numpy().astype(bool))
    np.testing.assert_allclose(tf_boxes.numpy(), torch_boxes.numpy(), atol=1e-6)
    np.testing.assert_allclose(tf_scores.numpy(), torch_scores.numpy(), atol=1e-6)
