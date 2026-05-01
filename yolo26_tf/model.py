"""YOLO26 YAML parser and Keras model wrapper."""

from __future__ import annotations

import ast
import re
from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

from .ops import load_yaml, make_divisible
from .tf_import import require_tf
from .modules import Bottleneck, C2PSA, C2f, C3, C3k2, Concat, Conv, DWConv, Detect, SPPF

tf = require_tf()
keras = tf.keras

MODULES = {
    "Conv": Conv,
    "DWConv": DWConv,
    "Bottleneck": Bottleneck,
    "C2f": C2f,
    "C3": C3,
    "C3k2": C3k2,
    "SPPF": SPPF,
    "C2PSA": C2PSA,
    "Concat": Concat,
    "Detect": Detect,
}
BASE_MODULES = {Conv, DWConv, Bottleneck, C2f, C3, C3k2, SPPF, C2PSA}
REPEAT_MODULES = {C2f, C3, C3k2, C2PSA}


def config_path(model: str | Path) -> tuple[Path, str]:
    """Resolve YOLO26 shorthand like yolo26n-p2.yaml to bundled base YAML and scale."""
    path = Path(model)
    stem = path.stem
    scale = guess_model_scale(stem)
    if path.exists():
        return path, scale
    if "p2" in stem:
        name = "yolo26-p2.yaml"
    elif "p6" in stem:
        name = "yolo26-p6.yaml"
    else:
        name = "yolo26.yaml"
    with resources.as_file(resources.files("yolo26_tf.configs") / name) as p:
        return Path(p), scale or "n"


def guess_model_scale(model_name: str) -> str:
    match = re.search(r"yolo(?:v)?26([nslmx])", model_name)
    return match.group(1) if match else ""


class Upsample(keras.layers.Layer):
    def __init__(self, scale: int = 2, mode: str = "nearest", **kwargs):
        super().__init__(**kwargs)
        self.up = keras.layers.UpSampling2D(size=(scale, scale), interpolation=mode)

    def call(self, x, training=None):
        return self.up(x)


class DetectionModel(keras.Model):
    """Keras model generated from a YOLO26 detection YAML."""

    def __init__(self, cfg: str | Path | dict = "yolo26n.yaml", ch: int = 3, nc: int | None = None, imgsz: int = 640, verbose: bool = False):
        super().__init__(name="yolo26_detection")
        self.cfg_ref = cfg
        if isinstance(cfg, dict):
            self.yaml = deepcopy(cfg)
            self.scale = self.yaml.get("scale", "n")
        else:
            cfg_path, scale = config_path(cfg)
            self.yaml = load_yaml(cfg_path)
            self.scale = scale or self.yaml.get("scale", "n")
            self.yaml["yaml_file"] = str(cfg_path)
        if nc is not None:
            self.yaml["nc"] = int(nc)
        self.yaml["channels"] = ch
        self.nc = int(self.yaml["nc"])
        self.names = {i: str(i) for i in range(self.nc)}
        self.layers_seq, self.froms, self.out_channels = parse_model(self.yaml, ch=ch, scale=self.scale)
        self.detect_layer = self.layers_seq[-1] if isinstance(self.layers_seq[-1], Detect) else None
        self.stride = [32.0]
        self._initialize(imgsz, ch)
        if verbose:
            print(self.info())

    @property
    def end2end(self) -> bool:
        return bool(getattr(self.detect_layer, "end2end", False))

    @end2end.setter
    def end2end(self, value: bool):
        if self.detect_layer is not None:
            self.detect_layer.end2end = value

    def set_head_attr(self, **kwargs):
        if self.detect_layer is None:
            return
        for k, v in kwargs.items():
            if hasattr(self.detect_layer, k):
                setattr(self.detect_layer, k, v)

    def _initialize(self, imgsz: int, ch: int):
        init_imgsz = max(64, min(int(imgsz), 256))
        dummy = tf.zeros([1, init_imgsz, init_imgsz, ch], dtype=tf.float32)
        train_out = self(dummy, training=True)
        if self.detect_layer is not None:
            preds = train_out["one2many"] if self.end2end else train_out
            feats = preds["feats"]
            self.stride = [float(init_imgsz / int(feat.shape[1])) for feat in feats]
            self.detect_layer.stride = self.stride
            self.detect_layer.bias_init()
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = inputs
        outputs = []
        for f, layer in zip(self.froms, self.layers_seq):
            if isinstance(f, list):
                inp = [x if j == -1 else outputs[j] for j in f]
            else:
                inp = x if f == -1 else outputs[f]
            x = layer(inp, training=training)
            outputs.append(x)
        return x

    def predict_raw(self, images):
        return self(images, training=False)

    def info(self) -> str:
        params = int(sum(tf.size(v).numpy() for v in self.trainable_variables)) if self.built else 0
        return f"YOLO26 TensorFlow DetectionModel(scale={self.scale}, nc={self.nc}, params={params:,}, stride={self.stride})"


def _eval_arg(arg: Any, context: dict) -> Any:
    if not isinstance(arg, str):
        return arg
    if arg in context:
        return context[arg]
    try:
        return ast.literal_eval(arg)
    except Exception:
        return arg


def parse_model(d: dict, ch: int = 3, scale: str = "n"):
    """Parse an Ultralytics-style YOLO model YAML into Keras layers."""
    d = deepcopy(d)
    nc = int(d.get("nc", 80))
    end2end = bool(d.get("end2end", False))
    reg_max = int(d.get("reg_max", 16))
    depth, width, max_channels = 1.0, 1.0, float("inf")
    scales = d.get("scales")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
        depth, width, max_channels = scales[scale]
    context = {"nc": nc, "reg_max": reg_max, "end2end": end2end}
    layers, froms, out_channels = [], [], []
    prev_c = ch
    legacy = True

    def get_ch(idx):
        return prev_c if idx == -1 else out_channels[idx]

    for i, (f, n, module_name, args) in enumerate(d["backbone"] + d["head"]):
        args = [_eval_arg(a, context) for a in args]
        repeats = max(round(n * depth), 1) if n > 1 else n
        if isinstance(module_name, str) and module_name.startswith("nn.Upsample"):
            scale_factor = int(args[1]) if len(args) > 1 and args[1] is not None else 2
            mode = args[2] if len(args) > 2 else "nearest"
            layer = Upsample(scale_factor, mode, name=f"layer_{i}_upsample")
            c2 = get_ch(f)
        else:
            m = MODULES[module_name]
            if m in BASE_MODULES:
                c1, c2 = get_ch(f), int(args[0])
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                new_args = [c1, c2, *args[1:]]
                if m in REPEAT_MODULES:
                    new_args.insert(2, repeats)
                    repeats = 1
                if m is C3k2:
                    legacy = False
                    if scale in "mlx" and len(new_args) > 3:
                        new_args[3] = True
                layer = m(*new_args, name=f"layer_{i}_{module_name.lower()}") if repeats == 1 else keras.Sequential(
                    [m(*new_args) for _ in range(repeats)], name=f"layer_{i}_{module_name.lower()}"
                )
            elif m is Concat:
                c2 = sum(get_ch(x) for x in f)
                dim = args[0] if args else -1
                layer = Concat(dim, name=f"layer_{i}_concat")
            elif m is Detect:
                chs = [get_ch(x) for x in f]
                layer = Detect(args[0], reg_max, end2end, chs, legacy=legacy, name=f"layer_{i}_detect")
                c2 = sum(chs)
            else:  # pragma: no cover
                raise NotImplementedError(f"Unsupported module {module_name}")
        layers.append(layer)
        froms.append(f)
        out_channels.append(c2)
        prev_c = c2
    return layers, froms, out_channels


def build_model(model: str = "yolo26n.yaml", nc: int | None = None, imgsz: int = 640, ch: int = 3, verbose: bool = False) -> DetectionModel:
    return DetectionModel(model, ch=ch, nc=nc, imgsz=imgsz, verbose=verbose)
