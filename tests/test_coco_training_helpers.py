import json
import argparse
from pathlib import Path

import pytest
from PIL import Image

from yolo26_tf.coco import convert_coco_json_to_yolo, write_coco_yaml
from yolo26_tf.cli import add_train_args
from yolo26_tf.data import YOLODataset, load_data_yaml
from yolo26_tf.tfrecord import load_yolo_tfrecord, write_yolo_tfrecord
from yolo26_tf.trainer import TrainConfig
from yolo26_tf.validation import validate_detection_model

tf = pytest.importorskip("tensorflow")


def test_coco_conversion_and_subset_dataset(tmp_path):
    root = tmp_path / "coco"
    image_dir = root / "train2017"
    ann_dir = root / "annotations"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    Image.new("RGB", (100, 80), (0, 0, 0)).save(image_dir / "000000000001.jpg")
    ann = {
        "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 100, "height": 80}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 20], "iscrowd": 0}],
        "categories": [{"id": 1, "name": "person"}],
    }
    ann_file = ann_dir / "instances_train2017.json"
    ann_file.write_text(json.dumps(ann), encoding="utf-8")
    stats = convert_coco_json_to_yolo(ann_file, image_dir, root / "labels" / "train2017", root / "train2017.txt")
    assert stats["images"] == 1
    label = (root / "labels" / "train2017" / "000000000001.txt").read_text(encoding="utf-8").strip()
    assert label.startswith("0 ")

    yaml_path = write_coco_yaml(root, root / "data.yaml", train=str(root / "train2017.txt"), val=str(root / "train2017.txt"))
    data = load_data_yaml(yaml_path)
    ds = YOLODataset(data, "train", imgsz=64, batch=1, augment=False, shuffle=False, cache=True)
    batch = next(iter(ds))
    assert batch["img"].shape == (1, 64, 64, 3)
    assert batch["mask"].sum() == 1
    assert batch["batch_idx"].shape == (1, 1)
    assert batch["flat_cls"].shape == (1, 1)
    assert batch["flat_bboxes"].shape == (1, 4)
    cache_file = root / "train.labels.cache.json"
    assert cache_file.exists()
    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "hash" in cache and "version" in cache and "results" in cache

    rect_ds = YOLODataset(data, "train", imgsz=64, batch=1, augment=False, shuffle=False, rect=True, classes=[0])
    assert rect_ds.batch_shapes is not None
    assert next(iter(rect_ds))["mask"].sum() == 1


def test_tfrecord_write_read_and_dataset_source(tmp_path):
    root = tmp_path / "tiny_tfrecord"
    data_yaml = __import__("scripts.make_tiny_dataset", fromlist=["create_tiny_dataset"]).create_tiny_dataset(root, n=3, size=32)
    record = root / "records" / "train.tfrecord"
    stats = write_yolo_tfrecord(data_yaml, "train", record)
    assert stats["images"] == 3
    records = load_yolo_tfrecord(record)
    assert len(records) == 3
    data = load_data_yaml(data_yaml)
    data["train_tfrecord"] = record
    ds = YOLODataset(data, "train", imgsz=32, batch=2, augment=False, shuffle=False, use_tfrecord=True, cache_ram_gb=1)
    batch = next(iter(ds))
    assert batch["img"].shape == (2, 32, 32, 3)
    assert batch["mask"].sum() == 2
    assert ds.image_cache_stats()["image_cache_items"] > 0
    ds._read_image(0)
    ds._read_image(0)
    stats_after = ds.image_cache_stats()
    assert stats_after["image_cache_hits"] >= 1
    assert stats_after["image_cache_misses"] >= 1
    ds.close()

    threaded_ds = YOLODataset(data, "train", imgsz=32, batch=2, augment=True, hyp={"mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0}, shuffle=False, use_tfrecord=True, cache_ram_gb=1, sample_workers=2)
    threaded = next(iter(threaded_ds))
    assert threaded["img"].shape == (2, 32, 32, 3)
    assert threaded["mask"].sum() == 2
    threaded_ds.close()


def test_train_config_exposes_coco_parity_knobs():
    assert TrainConfig().compile_train_step is False
    assert TrainConfig().fast_data is False
    assert TrainConfig().prefetch_data is True
    assert TrainConfig().sample_workers == 0
    assert TrainConfig().ema_update_interval == 1
    assert TrainConfig().graph_forward is True
    assert TrainConfig().graph_optimizer_apply is True
    cfg = TrainConfig(
        amp=True,
        multi_scale=0.5,
        gpus="0,1",
        val_coco=True,
        require_gpu=True,
        single_cls=True,
        freeze=2,
        time=1.0,
        compile_train_step=True,
        fast_data=True,
        prefetch_data=False,
        sample_workers=4,
        fast_nms=True,
        cache_images="auto",
        ema_update_interval=2,
        graph_forward=False,
        graph_optimizer_apply=False,
    )
    assert cfg.amp is True
    assert cfg.multi_scale == 0.5
    assert cfg.gpus == "0,1"
    assert cfg.val_coco is True
    assert cfg.require_gpu is True
    assert cfg.single_cls is True
    assert cfg.freeze == 2
    assert cfg.time == 1.0
    assert cfg.compile_train_step is True
    assert cfg.fast_data is True
    assert cfg.prefetch_data is False
    assert cfg.sample_workers == 4
    assert cfg.fast_nms is True
    assert cfg.cache_images == "auto"
    assert cfg.ema_update_interval == 2
    assert cfg.graph_forward is False
    assert cfg.graph_optimizer_apply is False


def test_cli_and_full_coco_runner_stability_defaults_are_stable():
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args(["--data", "data.yaml"])
    assert args.compile_train_step is False
    assert args.fast_data is False
    assert args.prefetch_data is True
    assert args.sample_workers == 0
    assert args.amp is True
    assert args.profile_stage is False
    assert args.profile_batches == 0
    assert args.ema_update_interval == 1
    assert args.graph_forward is True
    assert args.graph_optimizer_apply is True
    assert parser.parse_args(["--data", "data.yaml", "--compile"]).compile_train_step is True
    assert parser.parse_args(["--data", "data.yaml", "--fast-data"]).fast_data is True
    assert parser.parse_args(["--data", "data.yaml", "--no-prefetch-data"]).prefetch_data is False
    assert parser.parse_args(["--data", "data.yaml", "--sample-workers", "4"]).sample_workers == 4
    assert parser.parse_args(["--data", "data.yaml", "--no-graph-forward"]).graph_forward is False
    assert parser.parse_args(["--data", "data.yaml", "--no-graph-optimizer-apply"]).graph_optimizer_apply is False
    assert parser.parse_args(["--data", "data.yaml", "--no-amp"]).amp is False
    profile_args = parser.parse_args(["--data", "data.yaml", "--profile-stage", "--profile-batches", "2"])
    assert profile_args.profile_stage is True
    assert profile_args.profile_batches == 2

    script = Path("scripts/train_coco_yolo26n_linux.sh").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert 'PROFILE="${YOLO26_COCO_PROFILE:-full}"' in script
    assert 'BATCH="${YOLO26_COCO_BATCH:-32}"' in script
    assert 'AMP="${YOLO26_COCO_AMP:-0}"' in script
    assert 'COMPILE_STEP="${YOLO26_COCO_COMPILE:-0}"' in script
    assert 'FAST_DATA="${YOLO26_COCO_FAST_DATA:-0}"' in script
    assert 'PREFETCH_DATA="${YOLO26_COCO_PREFETCH_DATA:-1}"' in script
    assert 'SAMPLE_WORKERS="${YOLO26_COCO_SAMPLE_WORKERS:-8}"' in script
    assert '--sample-workers "$SAMPLE_WORKERS"' in script
    assert 'OPTIMIZER="${YOLO26_COCO_OPTIMIZER:-sgd}"' in script
    assert 'EMA_UPDATE_INTERVAL="${YOLO26_COCO_EMA_UPDATE_INTERVAL:-10}"' in script
    assert 'GRAPH_FORWARD="${YOLO26_COCO_GRAPH_FORWARD:-1}"' in script
    assert 'GRAPH_OPTIMIZER_APPLY="${YOLO26_COCO_GRAPH_OPTIMIZER_APPLY:-1}"' in script
    assert 'PROFILE_STAGE="${YOLO26_COCO_PROFILE_STAGE:-0}"' in script
    assert 'PROFILE_BATCHES="${YOLO26_COCO_PROFILE_BATCHES:-0}"' in script
    assert 'SYNC_PROFILE_STAGE="${YOLO26_COCO_SYNC_PROFILE_STAGE:-0}"' in script
    assert 'GPU_MONITOR="${YOLO26_COCO_GPU_MONITOR:-0}"' in script
    assert "nvidia-smi --query-gpu" in script
    assert "trap cleanup EXIT" in script
    assert "stale yolo26_tf package detected" in script
    assert "YOLO26 speed defaults:" in script
    assert "Repository commit:" in script
    assert "CUDA_LAUNCH_BLOCKING=1" in script
    assert '[[ "$AMP" == "1" ]] && echo "--amp" || echo "--no-amp"' in script
    assert "bash scripts/train_coco_yolo26n_linux.sh" in readme
    assert "YOLO26_COCO_BATCH=32" in readme
    assert "YOLO26_COCO_BATCH=16" in readme
    assert "YOLO26_COCO_AMP=0" in readme
    assert "YOLO26_COCO_FAST_DATA=0" in readme
    assert "YOLO26_COCO_PREFETCH_DATA=1" in readme
    assert "YOLO26_COCO_SAMPLE_WORKERS=8" in readme
    assert "YOLO26_COCO_OPTIMIZER=sgd" in readme
    assert "YOLO26_COCO_EMA_UPDATE_INTERVAL=10" in readme
    assert "YOLO26_COCO_GRAPH_FORWARD=1" in readme
    assert "YOLO26_COCO_GRAPH_OPTIMIZER_APPLY=1" in readme
    assert "YOLO26_COCO_COMPILE=0" in readme
    assert "YOLO26_COCO_PROFILE_BATCHES=200" in readme
    assert "YOLO26 package version: 0.1.5" in readme
    assert "graph_forward=True" in readme
    assert "graph_optimizer_apply=True" in readme
    assert "data_path=tf_data_prefetch_threaded" in readme
    assert "stage_profile.csv" in readme
    assert "gpu_stats.csv" in readme


def test_coco_validation_merges_native_and_coco_metrics(tmp_path, monkeypatch):
    root = tmp_path / "coco_val"
    image_dir = root / "images" / "val2017"
    label_dir = root / "labels" / "val2017"
    ann_dir = root / "annotations"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (0, 0, 0)).save(image_dir / "000000000001.jpg")
    (label_dir / "000000000001.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    ann = {
        "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 64, "height": 64}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [24, 24, 16, 16], "iscrowd": 0}],
        "categories": [{"id": 1, "name": "person"}],
    }
    ann_file = ann_dir / "instances_val2017.json"
    ann_file.write_text(json.dumps(ann), encoding="utf-8")
    data = {
        "path": str(root),
        "val": "images/val2017",
        "val_annotations": "annotations/instances_val2017.json",
        "nc": 1,
        "names": ["person"],
    }

    class EmptyModel:
        def __call__(self, images, training=False):
            return tf.zeros((tf.shape(images)[0], 0, 6), tf.float32)

    monkeypatch.setattr(
        "yolo26_tf.validation.evaluate_coco_predictions",
        lambda ann_path, rows, image_ids=None: {"metrics/mAP50-95(B)": 0.123, "metrics/mAP50(B)": 0.456, "fitness": 0.423},
    )
    result = validate_detection_model(EmptyModel(), data, imgsz=64, batch=1, use_coco=True, verbose=False)
    assert "metrics/precision(B)" in result
    assert result["metrics/mAP50-95(B)"] == 0.123
    assert result["nt_per_class"] == {0: 1}
