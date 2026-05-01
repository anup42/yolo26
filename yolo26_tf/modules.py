"""Keras modules used by the YOLO26 detection model."""

from __future__ import annotations

import copy
import math
from typing import Sequence

from .ops import dist2bbox, make_anchors
from .tf_import import require_tf

tf = require_tf()
keras = tf.keras


def autopad(k, p=None, d: int = 1):
    """Ultralytics-style symmetric padding for exact PyTorch Conv2d geometry."""
    if p is not None:
        return p
    if isinstance(k, tuple):
        return tuple(((x - 1) * d + 1) // 2 for x in k)
    return ((k - 1) * d + 1) // 2


class Conv(keras.layers.Layer):
    """Conv2D + BatchNorm + SiLU/identity, matching Ultralytics Conv semantics."""

    def __init__(self, c1: int, c2: int, k=1, s=1, p=None, g: int = 1, d: int = 1, act=True, **kwargs):
        super().__init__(**kwargs)
        p = autopad(k, p, d)
        self.c1, self.c2, self.k, self.s, self.g, self.d, self.p, self.act_arg = c1, c2, k, s, g, d, p, act
        self.conv = keras.layers.Conv2D(
            c2,
            k,
            strides=s,
            padding="valid",
            dilation_rate=d,
            groups=g,
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.pad = keras.layers.ZeroPadding2D(p) if p not in (None, 0, (0, 0)) else None
        self.bn = keras.layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.97)
        self.activation = keras.layers.Activation(tf.nn.silu) if act is True else (act if callable(act) else keras.layers.Activation("linear"))

    def call(self, x, training=None):
        if self.pad is not None:
            x = self.pad(x)
        return self.activation(self.bn(self.conv(x), training=training))


class DWConv(Conv):
    """Depth-wise/grouped convolution module."""

    def __init__(self, c1: int, c2: int, k=1, s=1, d: int = 1, act=True, **kwargs):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act, **kwargs)


class Bottleneck(keras.layers.Layer):
    """Standard YOLO bottleneck."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def call(self, x, training=None):
        y = self.cv2(self.cv1(x, training=training), training=training)
        return x + y if self.add else y


class C2f(keras.layers.Layer):
    """CSP bottleneck with two convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = [Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)]

    def call(self, x, training=None):
        y = list(tf.split(self.cv1(x, training=training), 2, axis=-1))
        for m in self.m:
            y.append(m(y[-1], training=training))
        return self.cv2(tf.concat(y, axis=-1), training=training)


class C3(keras.layers.Layer):
    """CSP bottleneck with three convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = [Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)]

    def call(self, x, training=None):
        y1 = self.cv1(x, training=training)
        for m in self.m:
            y1 = m(y1, training=training)
        y2 = self.cv2(x, training=training)
        return self.cv3(tf.concat([y1, y2], axis=-1), training=training)


class C3k(C3):
    """C3 block with configurable kernel size."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3, **kwargs):
        super().__init__(c1, c2, n, shortcut, g, e, **kwargs)
        c_ = int(c2 * e)
        self.m = [Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)]


class SPPF(keras.layers.Layer):
    """Spatial pyramid pooling fast layer."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False, **kwargs):
        super().__init__(**kwargs)
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.pool = keras.layers.MaxPool2D(pool_size=k, strides=1, padding="same")
        self.n = n
        self.add = shortcut and c1 == c2

    def call(self, x, training=None):
        y = [self.cv1(x, training=training)]
        for _ in range(self.n):
            y.append(self.pool(y[-1]))
        out = self.cv2(tf.concat(y, axis=-1), training=training)
        return out + x if self.add else out


class Attention(keras.layers.Layer):
    """Position-sensitive attention block used by YOLO26 C2PSA/C3k2."""

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = max(int(num_heads), 1)
        self.head_dim = dim // self.num_heads
        self.key_dim = max(int(self.head_dim * attn_ratio), 1)
        self.scale = self.key_dim**-0.5
        h = dim + self.key_dim * self.num_heads * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def call(self, x, training=None):
        shape = tf.shape(x)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        n = h * w
        qkv = self.qkv(x, training=training)
        qkv = tf.reshape(qkv, [b, n, self.num_heads, self.key_dim * 2 + self.head_dim])
        q, k, v = tf.split(qkv, [self.key_dim, self.key_dim, self.head_dim], axis=-1)
        attn = tf.einsum("bnhd,bmhd->bhnm", q, k) * tf.cast(self.scale, q.dtype)
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.einsum("bhnm,bmhd->bnhd", attn, v)
        out = tf.reshape(out, [b, h, w, c])
        pe = self.pe(tf.reshape(v, [b, h, w, c]), training=training)
        return self.proj(out + pe, training=training)


class PSABlock(keras.layers.Layer):
    """Attention + feed-forward block."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn1 = Conv(c, c * 2, 1)
        self.ffn2 = Conv(c * 2, c, 1, act=False)
        self.add = shortcut

    def call(self, x, training=None):
        y = self.attn(x, training=training)
        x = x + y if self.add else y
        y = self.ffn2(self.ffn1(x, training=training), training=training)
        return x + y if self.add else y


class C2PSA(keras.layers.Layer):
    """C2 block with stacked PSA blocks."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        if c1 != c2:
            raise ValueError("C2PSA requires c1 == c2")
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = [PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)]

    def call(self, x, training=None):
        a, b = tf.split(self.cv1(x, training=training), [self.c, self.c], axis=-1)
        for m in self.m:
            b = m(b, training=training)
        return self.cv2(tf.concat([a, b], axis=-1), training=training)


class C3k2(C2f):
    """YOLO26 C3k2 module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        **kwargs,
    ):
        super().__init__(c1, c2, n, shortcut, g, e, **kwargs)
        blocks = []
        for _ in range(n):
            if attn:
                blocks.append(keras.Sequential([Bottleneck(self.c, self.c, shortcut, g), PSABlock(self.c, 0.5, max(self.c // 64, 1))]))
            elif c3k:
                blocks.append(C3k(self.c, self.c, 2, shortcut, g))
            else:
                blocks.append(Bottleneck(self.c, self.c, shortcut, g))
        self.m = blocks


def _box_head(ch: int, c2: int, reg_max: int):
    return keras.Sequential([Conv(ch, c2, 3), Conv(c2, c2, 3), keras.layers.Conv2D(4 * reg_max, 1, padding="same")])


def _cls_head(ch: int, c3: int, nc: int, legacy: bool):
    if legacy:
        return keras.Sequential([Conv(ch, c3, 3), Conv(c3, c3, 3), keras.layers.Conv2D(nc, 1, padding="same")])
    return keras.Sequential(
        [
            keras.Sequential([DWConv(ch, ch, 3), Conv(ch, c3, 1)]),
            keras.Sequential([DWConv(c3, c3, 3), Conv(c3, c3, 1)]),
            keras.layers.Conv2D(nc, 1, padding="same"),
        ]
    )


class Detect(keras.layers.Layer):
    """YOLO detection head with YOLO26 end-to-end one-to-one branch support."""

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: Sequence[int] = (), legacy: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.nc = int(nc)
        self.nl = len(ch)
        self.reg_max = int(reg_max)
        self.no = self.nc + self.reg_max * 4
        self.end2end_enabled = bool(end2end)
        self.max_det = 300
        self.agnostic_nms = False
        self.dynamic = False
        self.xyxy = False
        self.stride = [8.0, 16.0, 32.0][: self.nl]
        self.anchors = None
        self.strides = None
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = [_box_head(x, c2, self.reg_max) for x in ch]
        self.cv3 = [_cls_head(x, c3, self.nc, legacy) for x in ch]
        if self.end2end_enabled:
            self.one2one_cv2 = [_box_head(x, c2, self.reg_max) for x in ch]
            self.one2one_cv3 = [_cls_head(x, c3, self.nc, legacy) for x in ch]

    @property
    def end2end(self) -> bool:
        return self.end2end_enabled and hasattr(self, "one2one_cv2")

    @end2end.setter
    def end2end(self, value: bool):
        self.end2end_enabled = bool(value)

    def forward_head(self, x: list, box_head=None, cls_head=None) -> dict:
        box_head = self.cv2 if box_head is None else box_head
        cls_head = self.cv3 if cls_head is None else cls_head
        boxes, scores = [], []
        for i in range(self.nl):
            b = box_head[i](x[i])
            s = cls_head[i](x[i])
            bs = tf.shape(b)[0]
            boxes.append(tf.reshape(b, [bs, -1, 4 * self.reg_max]))
            scores.append(tf.reshape(s, [bs, -1, self.nc]))
        return {"boxes": tf.concat(boxes, axis=1), "scores": tf.concat(scores, axis=1), "feats": x}

    def call(self, x: list, training=None):
        preds = self.forward_head(x, self.cv2, self.cv3)
        if self.end2end:
            x_detach = [tf.stop_gradient(xi) for xi in x]
            one2one = self.forward_head(x_detach, self.one2one_cv2, self.one2one_cv3)
            preds = {"one2many": preds, "one2one": one2one}
        if training:
            return preds
        infer_source = preds["one2one"] if self.end2end else preds
        y = self.inference(infer_source)
        return self.postprocess(y) if self.end2end else y

    def dfl(self, boxes):
        if self.reg_max <= 1:
            return boxes
        proj = tf.cast(tf.range(self.reg_max), boxes.dtype)
        b = tf.shape(boxes)[0]
        a = tf.shape(boxes)[1]
        boxes = tf.reshape(boxes, [b, a, 4, self.reg_max])
        return tf.reduce_sum(tf.nn.softmax(boxes, axis=-1) * proj, axis=-1)

    def inference(self, x: dict):
        anchor_points, stride_tensor = make_anchors(x["feats"], self.stride, 0.5)
        self.anchors, self.strides = anchor_points, stride_tensor
        boxes = dist2bbox(self.dfl(x["boxes"]), anchor_points[None, :, :], xywh=not self.end2end and not self.xyxy)
        boxes = boxes * stride_tensor[None, :, :]
        return tf.concat([boxes, tf.sigmoid(x["scores"])], axis=-1)

    def postprocess(self, preds):
        boxes, scores = preds[..., :4], preds[..., 4:]
        anchors = tf.shape(scores)[1]
        k = tf.minimum(tf.cast(self.max_det, tf.int32), anchors)
        max_scores = tf.reduce_max(scores, axis=-1)
        _, anchor_idx = tf.math.top_k(max_scores, k=k)
        boxes_k = tf.gather(boxes, anchor_idx, batch_dims=1)
        scores_k = tf.gather(scores, anchor_idx, batch_dims=1)
        flat_scores = tf.reshape(scores_k, [tf.shape(scores_k)[0], -1])
        top_scores, top_flat_idx = tf.math.top_k(flat_scores, k=k)
        top_anchor = top_flat_idx // self.nc
        top_cls = top_flat_idx % self.nc
        top_boxes = tf.gather(boxes_k, top_anchor, batch_dims=1)
        return tf.concat([top_boxes, top_scores[..., None], tf.cast(top_cls[..., None], preds.dtype)], axis=-1)

    def bias_init(self):
        """Initialize detection biases after the layer has been built by a dummy call."""
        for i, (box_head, cls_head) in enumerate(zip(self.cv2, self.cv3)):
            _assign_head_biases(box_head, cls_head, self.nc, self.stride[i])
        if self.end2end:
            for i, (box_head, cls_head) in enumerate(zip(self.one2one_cv2, self.one2one_cv3)):
                _assign_head_biases(box_head, cls_head, self.nc, self.stride[i])


def _last_conv(seq):
    layer = seq.layers[-1]
    if isinstance(layer, keras.Sequential):
        return _last_conv(layer)
    return layer


def _assign_head_biases(box_head, cls_head, nc: int, stride: float):
    box_conv = _last_conv(box_head)
    cls_conv = _last_conv(cls_head)
    if getattr(box_conv, "bias", None) is not None:
        box_conv.bias.assign(tf.ones_like(box_conv.bias) * 2.0)
    if getattr(cls_conv, "bias", None) is not None:
        value = math.log(5 / nc / (640 / stride) ** 2)
        cls_conv.bias.assign(tf.ones_like(cls_conv.bias) * value)


class Concat(keras.layers.Layer):
    def __init__(self, dimension=-1, **kwargs):
        super().__init__(**kwargs)
        self.dimension = -1 if dimension in (1, -1) else dimension

    def call(self, x, training=None):
        return tf.concat(x, axis=self.dimension)
