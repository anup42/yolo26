import random

import numpy as np

from scripts.make_tiny_dataset import create_tiny_dataset
from yolo26_tf.augment import Compose, Format, LetterBox, Mosaic, RandomFlip, RandomHSV, copy_paste_segments, resample_segments
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


def test_ultralytics_style_transform_and_collate_contract(tmp_path):
    data_yaml = create_tiny_dataset(tmp_path / "tiny_collate", n=2, size=32)
    ds = YOLODataset(data_yaml, split="train", imgsz=32, batch=2, augment=False, shuffle=False)
    transforms = Compose(
        [
            LetterBox(new_shape=(32, 32), scaleup=True),
            RandomHSV(0.0, 0.0, 0.0),
            RandomFlip(p=0.0),
            Format(bbox_format="xywh", normalize=True, batch_idx=True),
        ]
    )
    samples = [transforms(ds.get_image_and_label(i)) for i in range(2)]
    batch = YOLODataset.collate_fn(samples)
    assert batch["img"].shape == (2, 32, 32, 3)
    assert batch["bboxes"].ndim == 2
    assert batch["cls"].shape[0] == batch["bboxes"].shape[0]
    assert batch["batch_idx"].shape == (batch["bboxes"].shape[0], 1)


def test_resample_segments_fixed_count():
    segment = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
    out = resample_segments([segment], n=8)[0]
    assert out.shape == (8, 2)
    assert np.isfinite(out).all()


def test_mosaic_transform_supports_3_4_9(tmp_path):
    data_yaml = create_tiny_dataset(tmp_path / "tiny_mosaic_modes", n=9, size=32)
    ds = YOLODataset(data_yaml, split="train", imgsz=32, batch=1, augment=False, shuffle=False)
    base = ds.get_image_and_label(0)
    for n in (3, 4, 9):
        random.seed(0)
        labels = Mosaic(ds, imgsz=32, p=1.0, n=n)(dict(base))
        labels = Format(bbox_format="xywh", normalize=True, batch_idx=True)(labels)
        assert labels["img"].shape == (64, 64, 3)
        assert labels["bboxes"].ndim == 2
        assert (labels["bboxes"] >= 0).all()
        assert (labels["bboxes"] <= 1).all()
