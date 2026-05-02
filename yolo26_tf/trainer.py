"""YOLO-style TensorFlow trainer for YOLO26 detection."""

from __future__ import annotations

import json
import math
import random
import time
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import YOLODataset, load_data_yaml
from .losses import E2ELoss
from .optim import ModelEMA, make_optimizer
from .tf_import import require_tf
from .validation import validate_detection_model

tf = require_tf()


@dataclass
class TrainConfig:
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    project: str | Path = "runs/detect"
    name: str = "train"
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    nbs: int = 64
    accumulate: int = 0
    patience: int = 100
    close_mosaic: int = 10
    multi_scale: float = 0.0
    amp: bool = True
    cos_lr: bool = False
    ema: bool = True
    resume: bool = False
    seed: int = 0
    fraction: float = 1.0
    cache: bool | str = False
    rect: bool = False
    workers: int = 8
    gpus: str | None = None
    require_gpu: bool = False
    classes: list[int] | None = None
    single_cls: bool = False
    freeze: int | list[int] = 0
    time: float = 0.0
    clip_grad: float = 10.0
    val: bool = True
    val_coco: bool = False
    val_conf: float = 0.001
    val_iou: float = 0.7
    max_det: int = 300
    save_period: int = -1
    log_interval: int = 10
    cls_pw: float = 0.0


DEFAULT_HYP = {
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "bgr": 0.0,
}


class YOLO26Trainer:
    """Ultralytics-like detection trainer for the TensorFlow port."""

    def __init__(self, model, data: str | Path | dict, cfg: TrainConfig | dict | None = None, hyp: dict | None = None):
        self.model = model
        self.data = load_data_yaml(data)
        self.cfg = cfg if isinstance(cfg, TrainConfig) else TrainConfig(**(cfg or {}))
        self.hyp = dict(DEFAULT_HYP)
        if hyp:
            self.hyp.update(hyp)
        self.hyp.update({"epochs": self.cfg.epochs})
        self.save_dir = Path(self.cfg.project) / self.cfg.name
        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.last = self.weights_dir / "last.weights.h5"
        self.best = self.weights_dir / "best.weights.h5"
        self.state_file = self.weights_dir / "trainer_state.json"
        self.optimizer_state = self.weights_dir / "optimizer_state.npz"
        self.ema_state = self.weights_dir / "ema_state.npz"
        self.args_file = self.save_dir / "args.json"
        self.csv_file = self.save_dir / "results.csv"
        self.best_fitness = -float("inf")
        self.start_epoch = 0
        self.history: list[dict[str, Any]] = []
        self.strategy = self._make_strategy()
        self._rebuild_model_for_strategy_if_needed()
        self.optimizer = None
        self.ema = None
        self.loss_fn = None
        self.accum_grads = None
        self.accum_counter = 0
        self.total_iterations = 1
        self.bias_lr = self.cfg.lr0
        self.train_time_start = 0.0

    def _make_strategy(self):
        gpus = tf.config.list_physical_devices("GPU")
        if self.cfg.require_gpu and not gpus:
            raise RuntimeError("TensorFlow found no visible GPUs. This training path is GPU-only.")
        if self.cfg.gpus:
            ids = [x.strip() for x in str(self.cfg.gpus).split(",") if x.strip()]
            logical = tf.config.list_logical_devices("GPU")
            devices = [f"/GPU:{i}" for i in ids if int(i) < len(logical)] if logical else None
            if devices:
                return tf.distribute.MirroredStrategy(devices=devices)
        if len(gpus) > 1:
            return tf.distribute.MirroredStrategy()
        return tf.distribute.get_strategy()

    def _rebuild_model_for_strategy_if_needed(self):
        if self.replicas <= 1:
            return
        from .model import build_model

        cfg_ref = getattr(self.model, "cfg_ref", "yolo26n.yaml")
        old_weights = self.model.get_weights()
        with self.strategy.scope():
            cloned = build_model(cfg_ref, nc=getattr(self.model, "nc", self.data["nc"]), imgsz=self.cfg.imgsz)
            try:
                cloned.set_weights(old_weights)
            except Exception:
                pass
            cloned.names = getattr(self.model, "names", self.data.get("names", {}))
        self.model = cloned

    @property
    def replicas(self) -> int:
        return int(getattr(self.strategy, "num_replicas_in_sync", 1))

    def train(self) -> dict[str, Any]:
        set_seed(self.cfg.seed)
        if self.cfg.amp:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        self._freeze_layers()
        self.args_file.write_text(json.dumps({"config": asdict(self.cfg), "hyp": self.hyp}, indent=2, default=str), encoding="utf-8")
        print(
            f"Starting YOLO26 training: epochs={self.cfg.epochs}, imgsz={self.cfg.imgsz}, "
            f"batch={self.cfg.batch}, replicas={self.replicas}, xla_jit={tf.config.optimizer.get_jit()}",
            flush=True,
        )
        train_ds = YOLODataset(
            self.data,
            "train",
            imgsz=self.cfg.imgsz,
            batch=self.cfg.batch,
            augment=True,
            hyp=self.hyp,
            shuffle=True,
            rect=self.cfg.rect,
            cache=self.cfg.cache,
            fraction=self.cfg.fraction,
            seed=self.cfg.seed,
            drop_last=self.replicas > 1,
            classes=self.cfg.classes,
            single_cls=self.cfg.single_cls,
        )
        self._set_class_weights(train_ds)
        if self.cfg.resume:
            self._load_resume_weights()
        iterations = max(len(train_ds) * self.cfg.epochs, 1)
        self.total_iterations = iterations
        with self.strategy.scope():
            optimizer_name, lr0, momentum = resolve_optimizer_auto(
                self.cfg.optimizer,
                self.cfg.lr0,
                self.cfg.momentum,
                iterations=iterations,
                nc=self.data.get("nc", 80),
            )
            if (self.cfg.optimizer or "auto").lower() == "auto" and optimizer_name == "adamw":
                self.cfg.warmup_bias_lr = 0.0
            self.optimizer = make_optimizer(
                optimizer_name,
                lr=lr0,
                momentum=momentum,
                weight_decay=0.0,
                iterations=iterations,
            )
            self.cfg.lr0 = lr0
            self.cfg.momentum = momentum
            if self.cfg.amp and isinstance(self.optimizer, tf.keras.optimizers.Optimizer):
                self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)
            self.loss_fn = E2ELoss(self.model, hyp=self.hyp)
            self.model.criterion = self.loss_fn
            self.ema = ModelEMA(self.model) if self.cfg.ema else None
            self.accum_grads = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.model.trainable_variables]
            self._build_optimizer_slots()
            if self.cfg.resume:
                self._load_resume_training_state()
        nw = max(round(self.cfg.warmup_epochs * len(train_ds)), 100 if self.cfg.warmup_epochs > 0 else 0)
        accumulate = self.cfg.accumulate or max(round(self.cfg.nbs / max(self.cfg.batch, 1)), 1)
        patience_count = 0
        self.train_time_start = time.time()
        for epoch in range(self.start_epoch, self.cfg.epochs):
            if self.cfg.close_mosaic and epoch == max(self.cfg.epochs - self.cfg.close_mosaic, 0):
                train_ds.close_mosaic()
            start = time.time()
            losses = []
            print(f"epoch {epoch + 1}/{self.cfg.epochs} starting, batches={len(train_ds)}", flush=True)
            iterator = train_ds.as_tf_dataset(prefetch=max(self.cfg.workers, 1)) if self.replicas > 1 else train_ds
            if self.replicas > 1:
                iterator = self.strategy.experimental_distribute_dataset(iterator)
            for i, batch_data in enumerate(iterator):
                ni = epoch * len(train_ds) + i
                lr = self._set_lr_momentum(ni, nw)
                current_accumulate = accumulate
                if nw and ni <= nw:
                    current_accumulate = max(1, int(round(np.interp(ni, [0, nw], [1, accumulate]).item())))
                try:
                    if self.replicas > 1:
                        per_replica = self.strategy.run(self._train_step, args=(batch_data, current_accumulate, lr))
                        loss_items = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)
                    else:
                        batch_tf = to_tensor_batch(batch_data)
                        loss_items = self._train_step(batch_tf, current_accumulate, lr)
                except FloatingPointError:
                    if self._recover_from_nan(epoch):
                        continue
                    raise
                losses.append(np.asarray(loss_items.numpy(), dtype=np.float32))
                if i == 0 or (self.cfg.log_interval > 0 and (i + 1) % self.cfg.log_interval == 0) or (i + 1) == len(train_ds):
                    li = losses[-1]
                    print(
                        f"epoch {epoch + 1}/{self.cfg.epochs} batch {i + 1}/{len(train_ds)} "
                        f"box={li[0]:.4f} cls={li[1]:.4f} dfl={li[2]:.4f} lr={lr:.6g}",
                        flush=True,
                    )
            self._flush_accumulated(lr=self._current_lr())
            if self.loss_fn:
                self.loss_fn.update()
            train_loss = np.mean(losses, axis=0).tolist() if losses else [0.0, 0.0, 0.0]
            metrics = self._validate(epoch) if self.cfg.val and self.data.get("val") else {"fitness": -float(sum(train_loss))}
            fitness = float(metrics.get("fitness", metrics.get("metrics/mAP50-95(B)", 0.0)))
            improved = fitness >= self.best_fitness
            if improved:
                self.best_fitness = fitness
                patience_count = 0
            else:
                patience_count += 1
            self._save_epoch(epoch, fitness, improved)
            row = {
                "epoch": epoch + 1,
                "lr": self._current_lr(),
                "box_loss": float(train_loss[0]),
                "cls_loss": float(train_loss[1]),
                "dfl_loss": float(train_loss[2]),
                **metrics,
                "time": time.time() - start,
            }
            self.history.append(row)
            (self.save_dir / "results.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            self._append_csv(row)
            print(
                f"epoch {epoch + 1}/{self.cfg.epochs} box={row['box_loss']:.4f} cls={row['cls_loss']:.4f} "
                f"dfl={row['dfl_loss']:.4f} fitness={fitness:.4f} lr={row['lr']:.6g}"
            )
            train_ds.on_epoch_end()
            if self.cfg.patience and patience_count >= self.cfg.patience:
                print(f"early stopping: no fitness improvement for {self.cfg.patience} epochs")
                break
            if self.cfg.time and (time.time() - self.train_time_start) > self.cfg.time * 3600:
                print(f"timed stopping: reached {self.cfg.time} training hours")
                break
        final_metrics = self._final_eval() if self.cfg.val and self.best.exists() and self.data.get("val") else {}
        return {"save_dir": str(self.save_dir), "best": str(self.best), "last": str(self.last), "history": self.history, "final_metrics": final_metrics}

    def _train_step(self, batch_tf: dict, accumulate: int, lr: float):
        batch_tf = to_tensor_batch(batch_tf)
        if self.cfg.multi_scale > 0.0:
            batch_tf = multiscale_batch(batch_tf, self.cfg.imgsz, factor=self.cfg.multi_scale)
        with tf.GradientTape() as tape:
            preds = self.model(batch_tf["img"], training=True)
            loss, loss_items = self.loss_fn(preds, batch_tf)
            loss = tf.cast(loss, tf.float32) / float(accumulate)
            scaled_loss = scale_loss(self.optimizer, loss)
        grads = tape.gradient(scaled_loss, self.model.trainable_variables)
        grads = unscale_grads(self.optimizer, grads)
        for acc, grad in zip(self.accum_grads, grads):
            if grad is not None:
                acc.assign_add(tf.cast(grad, acc.dtype))
        self.accum_counter += 1
        if self.accum_counter % accumulate == 0:
            self._apply_accumulated(lr)
        if not tf.reduce_all(tf.math.is_finite(loss_items)):
            raise FloatingPointError("Non-finite YOLO26 loss encountered")
        return tf.cast(loss_items, tf.float32)

    def _apply_accumulated(self, lr: float):
        grads = [g.read_value() for g in self.accum_grads]
        if lr > 0 and self.bias_lr != lr:
            bias_scale = float(self.bias_lr / lr)
            grads = [g * bias_scale if is_bias_variable(v) else g for g, v in zip(grads, self.model.trainable_variables)]
        grads, _ = tf.clip_by_global_norm(grads, self.cfg.clip_grad)
        apply_decoupled_weight_decay(self.model.trainable_variables, lr, self.cfg.weight_decay)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        for g in self.accum_grads:
            g.assign(tf.zeros_like(g))
        self.accum_counter = 0
        if self.ema:
            self.ema.update(self.model)

    def _flush_accumulated(self, lr: float):
        if self.accum_grads and any(float(tf.reduce_sum(tf.abs(g)).numpy()) > 0 for g in self.accum_grads):
            self._apply_accumulated(lr)

    def _set_lr_momentum(self, ni: int, nw: int) -> float:
        progress = min(max(ni / max(self.total_iterations, 1), 0.0), 1.0)
        if self.cfg.cos_lr:
            lf = self.cfg.lrf + 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - self.cfg.lrf)
        else:
            lf = max(1.0 - progress, 0.0) * (1.0 - self.cfg.lrf) + self.cfg.lrf
        lr = self.cfg.lr0 * lf
        bias_lr = lr
        momentum = self.cfg.momentum
        if nw and ni <= nw:
            target_lr = lr
            lr = np.interp(ni, [0, nw], [0.0, target_lr]).item()
            bias_lr = np.interp(ni, [0, nw], [self.cfg.warmup_bias_lr, target_lr]).item()
            momentum = np.interp(ni, [0, nw], [self.cfg.warmup_momentum, self.cfg.momentum]).item()
        self.bias_lr = float(bias_lr)
        set_optimizer_attr(self.optimizer, "learning_rate", lr)
        set_optimizer_attr(self.optimizer, "momentum", momentum)
        return float(lr)

    def _current_lr(self) -> float:
        opt = inner_optimizer(self.optimizer)
        lr = getattr(opt, "learning_rate", self.cfg.lr0)
        try:
            return float(tf.keras.backend.get_value(lr))
        except Exception:
            return float(lr)

    def _validate(self, epoch: int) -> dict[str, Any]:
        if self.ema:
            self.ema.apply_to(self.model)
        try:
            return validate_detection_model(
                self.model,
                self.data,
                imgsz=self.cfg.imgsz,
                batch=self.cfg.batch,
                conf=self.cfg.val_conf,
                iou=self.cfg.val_iou,
                max_det=self.cfg.max_det,
                rect=True,
                use_coco=self.cfg.val_coco,
                save_json=self.cfg.val_coco,
                project=self.save_dir,
                name=f"val_epoch{epoch + 1}",
                verbose=False,
                single_cls=self.cfg.single_cls,
            )
        finally:
            if self.ema:
                self.ema.restore(self.model)

    def _save_epoch(self, epoch: int, fitness: float, improved: bool):
        self.model.save_weights(str(self.last))
        if improved:
            self.model.save_weights(str(self.best))
        if self.cfg.save_period and self.cfg.save_period > 0 and (epoch + 1) % self.cfg.save_period == 0:
            self.model.save_weights(str(self.weights_dir / f"epoch{epoch + 1}.weights.h5"))
        state = {"epoch": epoch + 1, "best_fitness": self.best_fitness, "fitness": fitness, "config": asdict(self.cfg)}
        self.state_file.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        self._save_optimizer_state()
        self._save_ema_state()

    def _load_resume_weights(self):
        if self.last.exists():
            self.model.load_weights(str(self.last))
        if self.state_file.exists():
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.start_epoch = int(state.get("epoch", 0))
            self.best_fitness = float(state.get("best_fitness", -float("inf")))
            history_file = self.save_dir / "results.json"
            if history_file.exists():
                self.history = json.loads(history_file.read_text(encoding="utf-8"))

    def _load_resume_training_state(self):
        self._load_optimizer_state()
        self._load_ema_state()

    def _set_class_weights(self, train_ds: YOLODataset):
        """Compute Ultralytics-style inverse-frequency class weights when requested."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data.get("names", getattr(self.model, "names", {}))
        self.model.args = self.cfg
        self.model.task = "detect"
        if self.cfg.cls_pw <= 0:
            return
        if not 0 <= self.cfg.cls_pw <= 1.0:
            raise AssertionError("cls_pw must be in the range [0, 1]")
        classes = [lb["cls"].reshape(-1) for lb in train_ds.labels if len(lb.get("cls", []))]
        if not classes:
            return
        counts = np.bincount(np.concatenate(classes).astype(np.int64), minlength=self.data["nc"]).astype(np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        weights = (1.0 / counts) ** float(self.cfg.cls_pw)
        weights = weights / weights.mean()
        self.hyp["class_weights"] = weights.astype(np.float32)
        self.model.class_weights = self.hyp["class_weights"]

    def _recover_from_nan(self, epoch: int) -> bool:
        """Restore the last checkpoint after non-finite loss, matching Ultralytics recovery intent."""
        print(f"warning: non-finite loss at epoch {epoch + 1}; restoring last checkpoint", flush=True)
        if not self.last.exists():
            return False
        try:
            self.model.load_weights(str(self.last))
            if self.accum_grads:
                for g in self.accum_grads:
                    g.assign(tf.zeros_like(g))
            self.accum_counter = 0
            if self.ema:
                self.ema = ModelEMA(self.model)
            return True
        except Exception:
            return False

    def _build_optimizer_slots(self):
        opt = inner_optimizer(self.optimizer)
        if hasattr(opt, "build"):
            try:
                opt.build(self.model.trainable_variables)
            except Exception:
                pass

    def _optimizer_variables(self):
        opt = inner_optimizer(self.optimizer)
        vars_attr = getattr(opt, "variables", [])
        return list(vars_attr() if callable(vars_attr) else vars_attr)

    def _save_optimizer_state(self):
        opt = inner_optimizer(self.optimizer)
        try:
            if hasattr(opt, "state_dict"):
                state = opt.state_dict()
                np.savez_compressed(self.optimizer_state, **{k.replace("/", "__slash__"): v for k, v in state.items()})
                return
            variables = self._optimizer_variables()
            if variables:
                np.savez_compressed(self.optimizer_state, **{f"v{i}": v.numpy() for i, v in enumerate(variables)})
        except Exception:
            pass

    def _load_optimizer_state(self):
        if not self.optimizer_state.exists():
            return
        opt = inner_optimizer(self.optimizer)
        try:
            data = np.load(self.optimizer_state, allow_pickle=False)
            if hasattr(opt, "load_state_dict"):
                state = {k.replace("__slash__", "/"): data[k] for k in data.files}
                opt.load_state_dict(state)
                return
            variables = self._optimizer_variables()
            for i, v in enumerate(variables):
                key = f"v{i}"
                if key in data and tuple(v.shape) == tuple(data[key].shape):
                    v.assign(data[key])
        except Exception:
            pass

    def _save_ema_state(self):
        if not self.ema:
            return
        try:
            state = self.ema.state_dict()
            payload = {f"shadow{i}": v for i, v in enumerate(state["shadow"])}
            payload["updates"] = np.asarray(state["updates"], dtype=np.int64)
            np.savez_compressed(self.ema_state, **payload)
        except Exception:
            pass

    def _load_ema_state(self):
        if not self.ema or not self.ema_state.exists():
            return
        try:
            data = np.load(self.ema_state, allow_pickle=False)
            state = {"updates": int(data["updates"]) if "updates" in data else 0, "shadow": []}
            for i in range(len(self.ema.shadow)):
                key = f"shadow{i}"
                if key in data:
                    state["shadow"].append(data[key])
            self.ema.load_state_dict(state)
        except Exception:
            pass

    def _freeze_layers(self):
        freeze = self.cfg.freeze
        if not freeze:
            return
        if isinstance(freeze, int):
            freeze_ids = set(range(freeze))
        else:
            freeze_ids = {int(x) for x in freeze}
        for i, layer in enumerate(getattr(self.model, "layers_seq", [])):
            if i in freeze_ids:
                layer.trainable = False
                print(f"Freezing layer {i}: {layer.name}", flush=True)

    def _append_csv(self, row: dict[str, Any]):
        write_header = not self.csv_file.exists()
        fields = list(row.keys())
        with self.csv_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _final_eval(self) -> dict[str, Any]:
        current = self.model.get_weights()
        try:
            self.model.load_weights(str(self.best))
            metrics = validate_detection_model(
                self.model,
                self.data,
                imgsz=self.cfg.imgsz,
                batch=self.cfg.batch,
                conf=self.cfg.val_conf,
                iou=self.cfg.val_iou,
                max_det=self.cfg.max_det,
                rect=True,
                use_coco=self.cfg.val_coco,
                save_json=self.cfg.val_coco,
                project=self.save_dir,
                name="final_best_val",
                verbose=False,
                single_cls=self.cfg.single_cls,
            )
            (self.save_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            return metrics
        finally:
            self.model.set_weights(current)


def to_tensor_batch(batch: dict) -> dict:
    tensor_keys = {"img", "bboxes", "cls", "mask", "batch_idx", "flat_cls", "flat_bboxes"}
    return {k: tf.convert_to_tensor(v) if k in tensor_keys else v for k, v in batch.items()}


def multiscale_batch(batch: dict, base_imgsz: int, factor: float = 0.5, stride: int = 32) -> dict:
    factor = float(factor)
    low = max(stride, int(base_imgsz * (1.0 - factor)) // stride * stride)
    high = max(low, int(base_imgsz * (1.0 + factor)) // stride * stride)
    size = random.randrange(low, high + stride, stride)
    if int(batch["img"].shape[1]) == size:
        return batch
    out = dict(batch)
    out["img"] = tf.image.resize(batch["img"], [size, size], method="bilinear")
    return out


def resolve_optimizer_auto(name: str, lr: float, momentum: float, iterations: int, nc: int) -> tuple[str, float, float]:
    """Match Ultralytics optimizer=auto behavior for optimizer, lr0 and momentum."""
    if (name or "auto").lower() != "auto":
        return name, float(lr), float(momentum)
    if iterations > 10000:
        return "musgd", 0.01, 0.9
    lr_fit = round(0.002 * 5 / (4 + int(nc)), 6)
    return "adamw", lr_fit, 0.9


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def inner_optimizer(optimizer):
    return getattr(optimizer, "inner_optimizer", getattr(optimizer, "optimizer", optimizer))


def set_optimizer_attr(optimizer, name: str, value: float):
    opt = inner_optimizer(optimizer)
    if hasattr(opt, name):
        attr = getattr(opt, name)
        try:
            attr.assign(value)
        except Exception:
            try:
                setattr(opt, name, value)
            except Exception:
                pass
    if name == "learning_rate" and hasattr(opt, "learning_rate"):
        try:
            opt.learning_rate = value
        except Exception:
            pass
    if hasattr(opt, "learning_rate") and hasattr(opt.learning_rate, "assign") and name == "learning_rate":
        opt.learning_rate.assign(value)


def scale_loss(optimizer, loss):
    if hasattr(optimizer, "scale_loss"):
        return optimizer.scale_loss(loss)
    if hasattr(optimizer, "get_scaled_loss"):
        return optimizer.get_scaled_loss(loss)
    return loss


def unscale_grads(optimizer, grads):
    if hasattr(optimizer, "get_unscaled_gradients"):
        return optimizer.get_unscaled_gradients(grads)
    return grads


def apply_decoupled_weight_decay(variables, lr: float, weight_decay: float):
    if not weight_decay:
        return
    for v in variables:
        if variable_decay_group(v) != "decay":
            continue
        v.assign_sub(tf.cast(lr * weight_decay, v.dtype) * v)


def is_bias_variable(var) -> bool:
    name = str(getattr(var, "path", None) or getattr(var, "name", "")).lower()
    return name.endswith("bias") or "/bias" in name or ".bias" in name


def variable_decay_group(var) -> str:
    """Classify variables into Ultralytics-style decay/no-decay/bias groups."""
    name = str(getattr(var, "path", None) or getattr(var, "name", "")).lower()
    if is_bias_variable(var):
        return "bias"
    if len(var.shape) <= 1 or any(token in name for token in ("batch_normalization", "batchnorm", "/bn", ".bn", "layer_norm", "group_norm")):
        return "norm"
    return "decay"
