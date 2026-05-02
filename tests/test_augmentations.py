import random

import numpy as np

from scripts.make_tiny_dataset import create_tiny_dataset
from yolo26_tf.augment import Format, LetterBox, copy_paste_segments
from yolo26_tf.data import YOLODataset
from yolo26_tf.instances import Instances


def test_detection_augmentation_pipeline_preserves_valid_boxes(tmp_path):
    random.seed(0)
    np.random.seed(0)
    data_yaml = create_tiny_dataset(tmp_path / "tiny", n=8, size=64)
    hyp = {
        "mosaic": 1.0,
        "mixup": 1.0,
        "cutmix": 1.0,
        "copy_paste": 1.0,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.2,
        "shear": 2.0,
        "perspective": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 1.0,
        "flipud": 0.0,
        "bgr": 1.0,
    }
    ds = YOLODataset(data_yaml, split="train", imgsz=64, batch=2, augment=True, hyp=hyp, shuffle=False)
    batch = next(iter(ds))

    assert batch["img"].shape == (2, 64, 64, 3)
    assert np.isfinite(batch["img"]).all()
    assert np.isfinite(batch["bboxes"]).all()
    assert batch["mask"].sum() > 0
    valid_boxes = batch["bboxes"][batch["mask"]]
    assert (valid_boxes >= 0.0).all()
    assert (valid_boxes <= 1.0).all()
    assert (valid_boxes[:, 2:] > 0.0).all()


def test_instances_letterbox_format_pipeline():
    img = np.zeros((40, 80, 3), dtype=np.uint8)
    labels = {
        "img": img,
        "cls": np.array([0], dtype=np.int64),
        "instances": Instances(np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32), bbox_format="xywh", normalized=True),
    }
    labels = LetterBox(new_shape=(64, 64), scaleup=True)(labels)
    labels = Format(bbox_format="xywh", normalize=True, batch_idx=True)(labels)
    assert labels["img"].shape == (64, 64, 3)
    assert labels["bboxes"].shape == (1, 4)
    assert np.isfinite(labels["bboxes"]).all()
    assert (labels["bboxes"] >= 0).all()
    assert (labels["bboxes"] <= 1).all()


def test_copy_paste_segments_smoke():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, 16:] = 255
    segment = np.array([[[8, 8], [16, 8], [16, 16], [8, 16]]], dtype=np.float32) / 32.0
    instances = Instances(
        np.array([[0.375, 0.375, 0.25, 0.25]], dtype=np.float32),
        segments=segment,
        bbox_format="xywh",
        normalized=True,
    )
    out, merged, cls = copy_paste_segments(img, instances, np.array([0], dtype=np.int64), p=1.0)
    assert out.shape == img.shape
    assert len(merged) >= 1
    assert cls.shape[0] == len(merged)
