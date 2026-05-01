import json
from pathlib import Path

from PIL import Image

from yolo26_tf.coco import convert_coco_json_to_yolo, write_coco_yaml
from yolo26_tf.data import YOLODataset, load_data_yaml
from yolo26_tf.trainer import TrainConfig


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


def test_train_config_exposes_coco_parity_knobs():
    cfg = TrainConfig(amp=True, multi_scale=True, gpus="0,1", val_coco=True, require_gpu=True)
    assert cfg.amp is True
    assert cfg.multi_scale is True
    assert cfg.gpus == "0,1"
    assert cfg.val_coco is True
    assert cfg.require_gpu is True
