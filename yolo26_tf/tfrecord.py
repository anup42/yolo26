"""TFRecord helpers for YOLO26 TensorFlow detection datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .tf_import import require_tf

tf = require_tf()


def _bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x) for x in np.asarray(values).reshape(-1)]))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in np.asarray(values).reshape(-1)]))


def write_yolo_tfrecord(
    data: str | Path | dict,
    split: str,
    output: str | Path,
    image_files: Iterable[str | Path] | None = None,
    compression: str | None = None,
) -> dict:
    """Write a YOLO split to TFRecord with image bytes and normalized labels."""
    from .data import img2label_path, list_images, load_data_yaml, read_label_file_with_segments, verify_image_label

    data_dict = load_data_yaml(data)
    files = [Path(x) for x in (image_files if image_files is not None else list_images(data_dict[split]))]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    options = tf.io.TFRecordOptions(compression_type=compression or "")
    examples = 0
    labels = 0
    with tf.io.TFRecordWriter(str(output), options=options) as writer:
        for img in files:
            verified = verify_image_label(img, nc=data_dict["nc"])
            if not verified.get("ok", False):
                continue
            cls, boxes, segments, _ = read_label_file_with_segments(img2label_path(img), nc=data_dict["nc"])
            image_bytes = img.read_bytes()
            shape = verified["shape"]
            features = {
                "image": _bytes_feature(image_bytes),
                "im_file": _bytes_feature(str(img.resolve()).encode("utf-8")),
                "shape": _int64_feature(shape),
                "cls": _int64_feature(cls),
                "bboxes": _float_feature(boxes.astype(np.float32).reshape(-1)),
                "segments": _bytes_feature(repr([np.asarray(s, dtype=np.float32).tolist() for s in segments]).encode("utf-8")),
            }
            writer.write(tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString())
            examples += 1
            labels += int(len(cls))
    return {"path": str(output), "images": examples, "labels": labels, "bytes": output.stat().st_size if output.exists() else 0}


def load_yolo_tfrecord(path: str | Path, max_bytes: int | None = None, compression: str | None = None) -> list[dict]:
    """Load TFRecord examples into RAM for random-access augmentation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise MemoryError(f"TFRecord {path} is {path.stat().st_size} bytes, above limit {max_bytes} bytes.")
    records = []
    total_bytes = 0
    ds = tf.data.TFRecordDataset([str(path)], compression_type=compression or "")
    for raw in ds:
        ex = tf.train.Example()
        ex.ParseFromString(bytes(raw.numpy()))
        feat = ex.features.feature
        image_bytes = bytes(feat["image"].bytes_list.value[0])
        total_bytes += len(image_bytes)
        if max_bytes is not None and total_bytes > max_bytes:
            raise MemoryError(f"TFRecord decoded image bytes exceed limit {max_bytes} bytes.")
        shape = tuple(int(x) for x in feat["shape"].int64_list.value)
        cls = np.asarray(feat["cls"].int64_list.value, dtype=np.int64)
        bboxes = np.asarray(feat["bboxes"].float_list.value, dtype=np.float32).reshape(-1, 4)
        records.append(
            {
                "image_bytes": image_bytes,
                "im_file": feat["im_file"].bytes_list.value[0].decode("utf-8"),
                "shape": shape,
                "cls": cls,
                "bboxes": bboxes,
                "segments": [],
                "normalized": True,
                "bbox_format": "xywh",
                "ok": True,
                "msg": "tfrecord",
            }
        )
    return records


def default_tfrecord_path(data: str | Path | dict, split: str, tfrecord_dir: str | Path | None = None) -> Path:
    """Resolve the default TFRecord path for a split."""
    from .data import load_data_yaml

    data_dict = load_data_yaml(data)
    key = f"{split}_tfrecord"
    if data_dict.get(key):
        return Path(data_dict[key])
    root = Path(tfrecord_dir) if tfrecord_dir else Path(data_dict["path"]) / "tfrecords"
    return root / f"{split}.tfrecord"
