"""PyTorch Ultralytics checkpoint conversion helpers.

This module is optional and requires torch + ultralytics installed from the pinned
upstream commit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import build_model


def convert_pt_to_tf(pt_file: str | Path, output: str | Path | None = None, imgsz: int = 640, nc: int | None = None, verify: bool = False):
    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("PT conversion requires `pip install -e .[convert]`.") from exc
    pt_file = Path(pt_file)
    yolo = YOLO(str(pt_file))
    torch_model = yolo.model.eval()
    names = getattr(torch_model, "names", None)
    nc = nc or getattr(torch_model.model[-1], "nc", None) or len(names or []) or 80
    tf_model = build_model(_yaml_name_from_pt(pt_file), nc=nc, imgsz=imgsz)
    copy_torch_to_tf(torch_model, tf_model)
    if names:
        tf_model.names = names
    if verify:
        max_diff = parity_check(torch_model, tf_model, imgsz=imgsz)
        print(f"PT->TF parity max_abs_diff={max_diff:.6g}")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        tf_model.save_weights(str(output))
    return tf_model


def _yaml_name_from_pt(path: Path) -> str:
    stem = path.stem
    if "p2" in stem:
        return f"{stem}.yaml"
    if "p6" in stem:
        return f"{stem}.yaml"
    return f"{stem}.yaml"


def parity_check(torch_model, tf_model, imgsz=640) -> float:
    import torch

    x = np.random.default_rng(0).random((1, imgsz, imgsz, 3), dtype=np.float32)
    with torch.no_grad():
        pt = torch_model(torch.from_numpy(np.transpose(x, (0, 3, 1, 2))))
        if isinstance(pt, (tuple, list)):
            pt = pt[0]
        pt_np = pt.detach().cpu().numpy()
    tf_np = tf_model(x, training=False).numpy()
    n = min(pt_np.size, tf_np.size)
    return float(np.max(np.abs(pt_np.reshape(-1)[:n] - tf_np.reshape(-1)[:n])))


def copy_torch_to_tf(torch_model, tf_model) -> None:
    for tm, tl in zip(list(torch_model.model), tf_model.layers_seq):
        _copy(tm, tl)


def _copy(tm, tl):
    name = tm.__class__.__name__
    if _is_tf_conv(tl) and hasattr(tm, "conv") and hasattr(tm, "bn"):
        _assign_conv(tl.conv, tm.conv)
        _assign_bn(tl.bn, tm.bn)
        return
    if name in {"Conv", "DWConv"}:
        _assign_conv(tl.conv, tm.conv); _assign_bn(tl.bn, tm.bn); return
    if name == "Bottleneck":
        _copy(tm.cv1, tl.cv1); _copy(tm.cv2, tl.cv2); return
    if name in {"C2f", "C3k2"}:
        _copy(tm.cv1, tl.cv1); _copy(tm.cv2, tl.cv2)
        for a, b in zip(list(tm.m), tl.m):
            _copy(a, b)
        return
    if name in {"C3", "C3k"}:
        _copy(tm.cv1, tl.cv1); _copy(tm.cv2, tl.cv2); _copy(tm.cv3, tl.cv3)
        for a, b in zip(list(tm.m), tl.m):
            _copy(a, b)
        return
    if name == "SPPF":
        _copy(tm.cv1, tl.cv1); _copy(tm.cv2, tl.cv2); return
    if name == "Attention":
        _copy(tm.qkv, tl.qkv); _copy(tm.proj, tl.proj); _copy(tm.pe, tl.pe); return
    if name == "PSABlock":
        _copy(tm.attn, tl.attn)
        _copy(tm.ffn[0], tl.ffn1); _copy(tm.ffn[1], tl.ffn2); return
    if name == "C2PSA":
        _copy(tm.cv1, tl.cv1); _copy(tm.cv2, tl.cv2)
        for a, b in zip(list(tm.m), tl.m):
            _copy(a, b)
        return
    if name == "Detect":
        for a, b in zip(list(tm.cv2), tl.cv2):
            _copy_sequence(a, b)
        for a, b in zip(list(tm.cv3), tl.cv3):
            _copy_sequence(a, b)
        if getattr(tl, "end2end", False):
            for a, b in zip(list(tm.one2one_cv2), tl.one2one_cv2):
                _copy_sequence(a, b)
            for a, b in zip(list(tm.one2one_cv3), tl.one2one_cv3):
                _copy_sequence(a, b)
        return
    if name == "Sequential" or hasattr(tm, "children") and hasattr(tl, "layers"):
        _copy_sequence(tm, tl)


def _copy_sequence(tm, tl):
    tchildren = list(tm.children()) if hasattr(tm, "children") else []
    for a, b in zip(tchildren, tl.layers):
        if a.__class__.__name__ == "Conv2d":
            _assign_conv(b, a)
        else:
            _copy(a, b)


def _is_tf_conv(layer) -> bool:
    return hasattr(layer, "conv") and hasattr(layer, "bn")


def _assign_conv(tf_conv, torch_conv):
    w = torch_conv.weight.detach().cpu().numpy()
    w = np.transpose(w, (2, 3, 1, 0))
    vars_ = [w]
    if getattr(torch_conv, "bias", None) is not None:
        vars_.append(torch_conv.bias.detach().cpu().numpy())
    if len(tf_conv.weights) != len(vars_) or any(tuple(a.shape) != tuple(b.shape) for a, b in zip(tf_conv.weights, vars_)):
        return
    tf_conv.set_weights(vars_)


def _assign_bn(tf_bn, torch_bn):
    weights = [
        torch_bn.weight.detach().cpu().numpy(),
        torch_bn.bias.detach().cpu().numpy(),
        torch_bn.running_mean.detach().cpu().numpy(),
        torch_bn.running_var.detach().cpu().numpy(),
    ]
    if len(tf_bn.weights) != len(weights) or any(tuple(a.shape) != tuple(b.shape) for a, b in zip(tf_bn.weights, weights)):
        return
    tf_bn.set_weights(weights)
