"""Prepare COCO 2017 for YOLO26 TensorFlow scratch training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from yolo26_tf.coco import convert_coco_json_to_yolo, make_subset_file, write_coco_yaml
from yolo26_tf.data import list_images
from yolo26_tf.tfrecord import write_yolo_tfrecord


def run(args: argparse.Namespace) -> dict:
    root = Path(args.coco_root).resolve()
    image_root = root / "images"
    if not (image_root / "train2017").exists() and (root / "train2017").exists():
        image_root.mkdir(exist_ok=True)
        # Keep compatibility with standard COCO layout by using relative paths in YAML.
        train_images = root / "train2017"
        val_images = root / "val2017"
    else:
        train_images = image_root / "train2017"
        val_images = image_root / "val2017"
    if not train_images.exists():
        train_images = root / "train2017"
    if not val_images.exists():
        val_images = root / "val2017"
    ann_train = root / "annotations" / "instances_train2017.json"
    ann_val = root / "annotations" / "instances_val2017.json"
    if not ann_train.exists() or not ann_val.exists():
        raise FileNotFoundError("Missing COCO instances_train2017.json or instances_val2017.json under annotations/.")

    labels_train = root / "labels" / "train2017"
    labels_val = root / "labels" / "val2017"
    train_list = root / "train2017.txt"
    val_list = root / "val2017.txt"
    train_stats = convert_coco_json_to_yolo(ann_train, train_images, labels_train, train_list)
    val_stats = convert_coco_json_to_yolo(ann_val, val_images, labels_val, val_list)
    data_yaml = write_coco_yaml(root, args.output_yaml, train=str(train_list), val=str(val_list))
    tfrecords = {}
    write_full_records = not args.no_tfrecord and (args.full_tfrecord or not args.train_subset)
    if write_full_records:
        tfrecord_dir = Path(args.tfrecord_dir) if args.tfrecord_dir else root / "tfrecords"
        train_record = tfrecord_dir / "train.tfrecord"
        val_record = tfrecord_dir / "val.tfrecord"
        if args.rebuild_tfrecord or not train_record.exists():
            tfrecords["train"] = write_yolo_tfrecord(data_yaml, "train", train_record)
        else:
            tfrecords["train"] = {"path": str(train_record), "bytes": train_record.stat().st_size}
        if args.rebuild_tfrecord or not val_record.exists():
            tfrecords["val"] = write_yolo_tfrecord(data_yaml, "val", val_record)
        else:
            tfrecords["val"] = {"path": str(val_record), "bytes": val_record.stat().st_size}
        data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        data["train_tfrecord"] = str(train_record)
        data["val_tfrecord"] = str(val_record)
        Path(data_yaml).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    train_images_list = list_images(train_list)
    val_images_list = list_images(val_list)
    subsets = {}
    if args.train_subset:
        subset_train = make_subset_file(train_images_list, root / f"train2017_{args.train_subset}.txt", args.train_subset)
        subset_val = make_subset_file(val_images_list, root / f"val2017_{min(args.val_subset, len(val_images_list))}.txt", args.val_subset)
        subset_yaml = Path(args.output_yaml).with_name(f"coco_yolo26_subset{args.train_subset}.yaml") if args.output_yaml else root / f"coco_yolo26_subset{args.train_subset}.yaml"
        data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        data["train"] = str(subset_train)
        data["val"] = str(subset_val)
        if not args.no_tfrecord:
            tfrecord_dir = Path(args.tfrecord_dir) if args.tfrecord_dir else root / "tfrecords"
            subset_train_record = tfrecord_dir / f"train_subset{args.train_subset}.tfrecord"
            subset_val_record = tfrecord_dir / f"val_subset{min(args.val_subset, len(val_images_list))}.tfrecord"
            if args.rebuild_tfrecord or not subset_train_record.exists():
                tfrecords["train_subset"] = write_yolo_tfrecord(data, "train", subset_train_record)
            else:
                tfrecords["train_subset"] = {"path": str(subset_train_record), "bytes": subset_train_record.stat().st_size}
            if args.rebuild_tfrecord or not subset_val_record.exists():
                tfrecords["val_subset"] = write_yolo_tfrecord(data, "val", subset_val_record)
            else:
                tfrecords["val_subset"] = {"path": str(subset_val_record), "bytes": subset_val_record.stat().st_size}
            data["train_tfrecord"] = str(subset_train_record)
            data["val_tfrecord"] = str(subset_val_record)
        subset_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        subsets = {"train": str(subset_train), "val": str(subset_val), "yaml": str(subset_yaml)}

    result = {"data_yaml": str(data_yaml), "train": train_stats, "val": val_stats, "tfrecords": tfrecords, "subsets": subsets}
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO 2017 annotations to YOLO labels for yolo26-tf.")
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--output-yaml", default=None)
    parser.add_argument("--train-subset", type=int, default=100, help="Also create a deterministic train subset file. Use 0 to disable.")
    parser.add_argument("--val-subset", type=int, default=100)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--tfrecord-dir", default=None)
    parser.add_argument("--no-tfrecord", action="store_true")
    parser.add_argument("--rebuild-tfrecord", action="store_true")
    parser.add_argument("--full-tfrecord", action="store_true", help="Also write full train/val TFRecords when creating subsets.")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
