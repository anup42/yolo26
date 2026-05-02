"""High-level Ultralytics-like API for YOLO26 TensorFlow detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .data import load_data_yaml
from .export import export_model
from .model import DetectionModel, build_model
from .ops import letterbox, scale_boxes_np
from .tf_import import require_tf
from .trainer import DEFAULT_HYP, TrainConfig, YOLO26Trainer
from .validation import prediction_to_detections, validate_detection_model

tf = require_tf()


class YOLO26:
    """Ultralytics-style wrapper for TensorFlow YOLO26 object detection."""

    def __init__(self, model: str | Path | DetectionModel = "yolo26n.yaml", nc: int | None = None, imgsz: int = 640):
        self.model_ref = model
        self.imgsz = int(imgsz)
        if isinstance(model, DetectionModel):
            self.model = model
        else:
            model_path = Path(model)
            if model_path.suffix == ".pt":
                from .converter import convert_pt_to_tf

                self.model = convert_pt_to_tf(model_path, imgsz=imgsz, nc=nc)
            else:
                self.model = build_model(str(model), nc=nc, imgsz=imgsz)
                if model_path.exists() and model_path.suffix in {".h5", ".weights"}:
                    self.model.load_weights(str(model_path))
        self.names = getattr(self.model, "names", {i: str(i) for i in range(self.model.nc)})

    def train(
        self,
        data: str | Path,
        epochs: int = 100,
        imgsz: int | None = None,
        batch: int = 16,
        project: str | Path = "runs/detect",
        name: str = "train",
        **kwargs,
    ) -> dict:
        imgsz = int(imgsz or self.imgsz)
        data_dict = load_data_yaml(data)
        if getattr(self.model, "nc", None) != data_dict["nc"] and not isinstance(self.model_ref, DetectionModel):
            model_path = Path(self.model_ref)
            if model_path.suffix == ".pt":
                from .converter import convert_pt_to_tf

                self.model = convert_pt_to_tf(model_path, imgsz=imgsz, nc=data_dict["nc"])
            else:
                self.model = build_model(str(self.model_ref), nc=data_dict["nc"], imgsz=imgsz)
        self.model.names = data_dict["names"]
        self.model.nc = data_dict["nc"]
        hyp = dict(DEFAULT_HYP)
        for key in list(hyp):
            if key in kwargs:
                hyp[key] = kwargs[key]
        for key in ("box", "cls", "dfl", "class_weights"):
            if key in kwargs:
                hyp[key] = kwargs[key]
        cfg_keys = {f.name for f in TrainConfig.__dataclass_fields__.values()}
        cfg_values: dict[str, Any] = {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "project": project,
            "name": name,
        }
        for key in cfg_keys:
            if key in kwargs:
                cfg_values[key] = kwargs[key]
        trainer = YOLO26Trainer(self.model, data_dict, TrainConfig(**cfg_values), hyp=hyp)
        result = trainer.train()
        self.model = trainer.model
        self.names = getattr(self.model, "names", self.names)
        return result

    def val(
        self,
        data: str | Path | dict,
        imgsz: int | None = None,
        batch: int = 16,
        conf: float = 0.001,
        iou: float = 0.7,
        max_det: int = 300,
        coco: bool = False,
        save_json: bool = False,
        project: str | Path = "runs/detect",
        name: str = "val",
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        imgsz = int(imgsz or self.imgsz)
        return validate_detection_model(
            self.model,
            data,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            iou=iou,
            max_det=max_det,
            rect=kwargs.get("rect", True),
            use_coco=coco,
            save_json=save_json,
            save_txt=kwargs.get("save_txt", False),
            save_conf=kwargs.get("save_conf", False),
            single_cls=kwargs.get("single_cls", False),
            agnostic_nms=kwargs.get("agnostic_nms", False),
            multi_label=kwargs.get("multi_label", True),
            half=kwargs.get("half", False),
            project=project,
            name=name,
            fast_nms=kwargs.get("fast_nms", True),
            verbose=verbose,
        )

    def predict(
        self,
        source: str | Path | np.ndarray,
        imgsz: int | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        agnostic_nms: bool = False,
        multi_label: bool = False,
        single_cls: bool = False,
    ) -> list[dict]:
        imgsz = int(imgsz or self.imgsz)
        paths, images = self._load_sources(source)
        results = []
        for path, img0 in zip(paths, images):
            img, ratio, pad = letterbox(img0, imgsz, scaleup=False)
            x = img.astype(np.float32)[None] / 255.0
            raw = self.model(tf.convert_to_tensor(x), training=False).numpy()[0]
            det = prediction_to_detections(raw, conf=conf, iou=iou, max_det=max_det, agnostic=agnostic_nms or single_cls, multi_label=multi_label)
            if single_cls and len(det):
                det[:, 5] = 0
            if len(det):
                det[:, :4] = scale_boxes_np(det[:, :4], (imgsz, imgsz), img0.shape[:2], ((ratio[0], ratio[1]), pad))
            results.append({"path": path, "boxes": det[:, :4], "conf": det[:, 4], "cls": det[:, 5].astype(np.int64), "names": self.names})
        return results

    def export(self, format: str = "keras", output: str | Path | None = None, **kwargs):
        imgsz = kwargs.pop("imgsz", self.imgsz)
        return export_model(self.model, format=format, output=output, imgsz=imgsz, **kwargs)

    def _load_sources(self, source):
        if isinstance(source, np.ndarray):
            return ["array"], [source[..., :3].astype(np.uint8)]
        p = Path(source)
        if p.is_dir():
            files = sorted(x for x in p.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        else:
            files = [p]
        return [str(x) for x in files], [np.asarray(Image.open(x).convert("RGB")) for x in files]
