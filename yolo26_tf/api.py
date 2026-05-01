"""High-level Ultralytics-like API for YOLO26 TensorFlow detection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .data import YOLODataset, load_data_yaml
from .export import export_model
from .losses import E2ELoss
from .metrics import ap_per_class, targets_from_batch
from .model import DetectionModel, build_model
from .ops import letterbox, nms_numpy, scale_boxes_np
from .optim import ModelEMA, make_optimizer
from .tf_import import require_tf

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

    def train(self, data: str | Path, epochs: int = 100, imgsz: int | None = None, batch: int = 16, project: str | Path = "runs/detect", name: str = "train", **kwargs) -> dict:
        imgsz = int(imgsz or self.imgsz)
        data_dict = load_data_yaml(data)
        self.model.names = data_dict["names"]
        self.model.nc = data_dict["nc"]
        hyp = {
            "epochs": epochs,
            "box": kwargs.get("box", 7.5),
            "cls": kwargs.get("cls", 0.5),
            "dfl": kwargs.get("dfl", 1.5),
            "hsv_h": kwargs.get("hsv_h", 0.015),
            "hsv_s": kwargs.get("hsv_s", 0.7),
            "hsv_v": kwargs.get("hsv_v", 0.4),
            "fliplr": kwargs.get("fliplr", 0.5),
            "flipud": kwargs.get("flipud", 0.0),
            "mosaic": kwargs.get("mosaic", 1.0),
            "mixup": kwargs.get("mixup", 0.0),
            "cutmix": kwargs.get("cutmix", 0.0),
            "copy_paste": kwargs.get("copy_paste", 0.0),
            "degrees": kwargs.get("degrees", 0.0),
            "translate": kwargs.get("translate", 0.1),
            "scale": kwargs.get("scale", 0.5),
            "shear": kwargs.get("shear", 0.0),
            "perspective": kwargs.get("perspective", 0.0),
            "bgr": kwargs.get("bgr", 0.0),
        }
        train_ds = YOLODataset(data_dict, "train", imgsz, batch, augment=True, hyp=hyp, shuffle=True)
        val_ds = YOLODataset(data_dict, "val", imgsz, batch, augment=False, hyp=hyp, shuffle=False) if data_dict.get("val") else None
        criterion = E2ELoss(self.model, hyp=hyp) if self.model.end2end else None
        iterations = len(train_ds) * epochs
        optimizer = make_optimizer(
            kwargs.get("optimizer", "auto"),
            lr=kwargs.get("lr0", 0.01),
            momentum=kwargs.get("momentum", 0.937),
            weight_decay=kwargs.get("weight_decay", 5e-4),
            iterations=iterations,
        )
        ema = ModelEMA(self.model) if kwargs.get("ema", True) else None
        save_dir = Path(project) / name
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        history = []
        best_fitness = -1.0
        close_mosaic = int(kwargs.get("close_mosaic", 10))
        for epoch in range(epochs):
            if close_mosaic and epoch == max(epochs - close_mosaic, 0):
                train_ds.close_mosaic()
            losses = []
            start = time.time()
            for batch_data in train_ds:
                batch_tf = {k: tf.convert_to_tensor(v) if k in {"img", "bboxes", "cls", "mask"} else v for k, v in batch_data.items()}
                with tf.GradientTape() as tape:
                    preds = self.model(batch_tf["img"], training=True)
                    loss, loss_items = criterion(preds, batch_tf)
                    if not tf.reduce_all(tf.math.is_finite(loss)):
                        raise FloatingPointError("Non-finite YOLO26 loss encountered")
                grads = tape.gradient(loss, self.model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, kwargs.get("clip_grad", 10.0))
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                if ema:
                    ema.update(self.model)
                losses.append(loss_items.numpy())
            criterion.update()
            train_loss = np.mean(losses, axis=0).tolist() if losses else [0.0, 0.0, 0.0]
            metrics = self.val(data_dict, imgsz=imgsz, batch=batch, use_ema=ema, verbose=False) if val_ds else {"fitness": -float(sum(train_loss))}
            fitness = float(metrics.get("fitness", 0.0))
            last = weights_dir / "last.weights.h5"
            self.model.save_weights(str(last))
            if fitness >= best_fitness:
                best_fitness = fitness
                self.model.save_weights(str(weights_dir / "best.weights.h5"))
            row = {"epoch": epoch + 1, "box_loss": train_loss[0], "cls_loss": train_loss[1], "dfl_loss": train_loss[2], **metrics, "time": time.time() - start}
            history.append(row)
            (save_dir / "results.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(
                f"epoch {epoch + 1}/{epochs} box={train_loss[0]:.4f} cls={train_loss[1]:.4f} dfl={train_loss[2]:.4f} fitness={fitness:.4f}"
            )
            train_ds.on_epoch_end()
        return {"save_dir": str(save_dir), "best": str(weights_dir / "best.weights.h5"), "last": str(weights_dir / "last.weights.h5"), "history": history}

    def val(self, data: str | Path | dict, imgsz: int | None = None, batch: int = 16, conf: float = 0.25, iou: float = 0.45, use_ema: ModelEMA | None = None, verbose: bool = True) -> dict:
        imgsz = int(imgsz or self.imgsz)
        data_dict = load_data_yaml(data) if not isinstance(data, dict) else data
        ds = YOLODataset(data_dict, "val", imgsz, batch, augment=False, shuffle=False)
        if use_ema:
            use_ema.apply_to(self.model)
        preds_all, targets_all = [], []
        for b in ds:
            raw = self.model(tf.convert_to_tensor(b["img"], tf.float32), training=False).numpy()
            for pred in raw:
                if pred.shape[-1] == 6:
                    det = pred[pred[:, 4] >= conf]
                else:
                    boxes, scores = pred[:, :4], pred[:, 4:]
                    cls = scores.argmax(axis=-1)
                    score = scores.max(axis=-1)
                    det = nms_numpy(np.concatenate([boxes, score[:, None], cls[:, None]], axis=-1), conf, iou)
                preds_all.append(det.astype(np.float32))
            targets_all.extend(targets_from_batch(b, imgsz))
        if use_ema:
            use_ema.restore(self.model)
        metrics = ap_per_class(preds_all, targets_all, iou_thres=0.5)
        if verbose:
            print(metrics)
        return metrics

    def predict(self, source: str | Path | np.ndarray, imgsz: int | None = None, conf: float = 0.25, iou: float = 0.45, max_det: int = 300) -> list[dict]:
        imgsz = int(imgsz or self.imgsz)
        paths, images = self._load_sources(source)
        results = []
        for path, img0 in zip(paths, images):
            img, ratio, pad = letterbox(img0, imgsz, scaleup=False)
            x = img.astype(np.float32)[None] / 255.0
            raw = self.model(tf.convert_to_tensor(x), training=False).numpy()[0]
            if raw.shape[-1] == 6:
                det = raw[raw[:, 4] >= conf]
            else:
                boxes, scores = raw[:, :4], raw[:, 4:]
                cls = scores.argmax(axis=-1)
                score = scores.max(axis=-1)
                det = nms_numpy(np.concatenate([boxes, score[:, None], cls[:, None]], axis=-1), conf, iou, max_det)
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

