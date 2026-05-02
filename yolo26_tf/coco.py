"""COCO helpers for YOLO26 TensorFlow training and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml

COCO80_TO_COCO91 = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]
COCO91_TO_COCO80 = {v: i for i, v in enumerate(COCO80_TO_COCO91)}
COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


def load_coco_api():
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install pycocotools first, e.g. `pip install pycocotools`.") from exc
    return COCO, COCOeval


def convert_coco_json_to_yolo(ann_file: str | Path, image_dir: str | Path, label_dir: str | Path, image_list_file: str | Path | None = None) -> dict:
    """Convert one COCO instances JSON into YOLO labels."""
    ann_file = Path(ann_file)
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    images = {int(x["id"]): x for x in data.get("images", [])}
    grouped: dict[int, list[str]] = {image_id: [] for image_id in images}
    skipped = 0
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            skipped += 1
            continue
        image_id = int(ann["image_id"])
        img = images.get(image_id)
        if img is None:
            skipped += 1
            continue
        cls = COCO91_TO_COCO80.get(int(ann["category_id"]))
        if cls is None:
            skipped += 1
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        iw, ih = float(img["width"]), float(img["height"])
        cx = (x + w / 2) / iw
        cy = (y + h / 2) / ih
        grouped[image_id].append(f"{cls} {cx:.6f} {cy:.6f} {w / iw:.6f} {h / ih:.6f}\n")
    image_paths = []
    for image_id, img in images.items():
        label_path = label_dir / Path(img["file_name"]).with_suffix(".txt").name
        label_path.write_text("".join(grouped.get(image_id, [])), encoding="utf-8")
        image_paths.append(image_dir / img["file_name"])
    if image_list_file:
        image_list_file = Path(image_list_file)
        image_list_file.parent.mkdir(parents=True, exist_ok=True)
        image_list_file.write_text("\n".join(str(p.resolve()) for p in image_paths) + "\n", encoding="utf-8")
    return {"images": len(images), "labels": len(grouped), "skipped_annotations": skipped}


def write_coco_yaml(coco_root: str | Path, output: str | Path | None = None, train: str = "train2017", val: str = "val2017") -> Path:
    coco_root = Path(coco_root).resolve()
    output = Path(output) if output else coco_root / "coco_yolo26.yaml"

    def split_value(value: str) -> str:
        p = Path(value)
        if p.is_absolute() or p.suffix == ".txt":
            return str(p)
        return f"images/{value}" if (coco_root / "images" / value).exists() else value

    data = {
        "path": str(coco_root),
        "train": split_value(train),
        "val": split_value(val),
        "train_annotations": "annotations/instances_train2017.json",
        "val_annotations": "annotations/instances_val2017.json",
        "nc": 80,
        "names": [COCO_NAMES[i] for i in range(80)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output


def make_subset_file(images: Iterable[str | Path], output: str | Path, limit: int) -> Path:
    output = Path(output)
    selected = list(images)[: int(limit)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(str(Path(p).resolve()) for p in selected) + "\n", encoding="utf-8")
    return output


def coco_image_id_from_path(path: str | Path) -> int:
    stem = Path(path).stem
    try:
        return int(stem)
    except ValueError:
        return int(stem.split("_")[-1])


def evaluate_coco_predictions(ann_file: str | Path, predictions: list[dict], image_ids: list[int] | None = None) -> dict:
    if not predictions:
        return {"metrics/mAP50-95(B)": 0.0, "metrics/mAP50(B)": 0.0, "fitness": 0.0, "predictions": 0}
    COCO, COCOeval = load_coco_api()
    coco = COCO(str(ann_file))
    coco_dt = coco.loadRes(predictions)
    evaluator = COCOeval(coco, coco_dt, "bbox")
    if image_ids:
        evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats.tolist()
    map5095 = float(stats[0])
    map50 = float(stats[1])
    return {
        "metrics/mAP50-95(B)": map5095,
        "metrics/mAP50(B)": map50,
        "metrics/mAP75(B)": float(stats[2]),
        "metrics/mAP50-95_small(B)": float(stats[3]),
        "metrics/mAP50-95_medium(B)": float(stats[4]),
        "metrics/mAP50-95_large(B)": float(stats[5]),
        "metrics/AR1(B)": float(stats[6]),
        "metrics/AR10(B)": float(stats[7]),
        "metrics/AR100(B)": float(stats[8]),
        "fitness": float(0.1 * map50 + 0.9 * map5095),
        "predictions": len(predictions),
    }
