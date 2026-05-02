"""Optimizers and EMA helpers."""

from __future__ import annotations

from .tf_import import require_tf

tf = require_tf()


def zeropower_via_newtonschulz5(g, eps: float = 1e-7):
    """Muon Newton-Schulz orthogonalization."""
    x = tf.cast(g, tf.float32)
    norm = tf.norm(x) + eps
    x = x / norm
    transposed = False
    if tf.shape(x)[0] > tf.shape(x)[1]:
        x = tf.transpose(x)
        transposed = True
    for a, b, c in [(3.4445, -4.7750, 2.0315)] * 5:
        aa = tf.matmul(x, x, transpose_b=True)
        bmat = b * aa + c * tf.matmul(aa, aa)
        x = a * x + tf.matmul(bmat, x)
    if transposed:
        x = tf.transpose(x)
    return tf.cast(x, g.dtype)


def _var_key(var) -> str:
    return str(getattr(var, "path", None) or getattr(var, "name", id(var)))


class MuSGD:
    """Small TensorFlow implementation of Ultralytics MuSGD for eager training."""

    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True, muon=0.2, sgd=1.0):
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.muon = float(muon)
        self.sgd = float(sgd)
        self.state = {}
        self._pending_state = {}

    def _state(self, var, suffix):
        key = (_var_key(var), suffix)
        if key not in self.state:
            self.state[key] = tf.Variable(tf.zeros_like(var), trainable=False)
            pending = self._pending_state.pop("|".join(key), None)
            if pending is not None:
                self.state[key].assign(pending)
        return self.state[key]

    def state_dict(self):
        """Return numpy state for checkpoint resume."""
        return {"|".join(k): v.numpy() for k, v in self.state.items()}

    def load_state_dict(self, state):
        """Restore state values that already exist in this optimizer instance."""
        self._pending_state = dict(state)
        for key, value in state.items():
            parts = str(key).rsplit("|", 1)
            if len(parts) != 2:
                continue
            state_key = (parts[0], parts[1])
            if state_key in self.state:
                self.state[state_key].assign(value)
                self._pending_state.pop(key, None)

    def apply_gradients(self, grads_and_vars):
        for grad, var in grads_and_vars:
            if grad is None:
                continue
            grad = tf.cast(grad, var.dtype)
            lr = tf.cast(self.learning_rate, var.dtype)
            if len(var.shape) >= 2:
                mom = self._state(var, "muon")
                mom.assign(mom * self.momentum + grad * (1 - self.momentum))
                update = grad * (1 - self.momentum) + mom * self.momentum if self.nesterov else mom
                original_shape = tf.shape(update)
                if len(var.shape) == 4:
                    mat = tf.reshape(tf.transpose(update, [3, 0, 1, 2]), [tf.shape(update)[-1], -1])
                else:
                    mat = tf.reshape(update, [tf.shape(update)[0], -1])
                mu = zeropower_via_newtonschulz5(mat)
                if len(var.shape) == 4:
                    mu = tf.transpose(tf.reshape(mu, [tf.shape(update)[-1], tf.shape(update)[0], tf.shape(update)[1], tf.shape(update)[2]]), [1, 2, 3, 0])
                else:
                    mu = tf.reshape(mu, original_shape)
                var.assign_sub(lr * self.muon * tf.cast(mu, var.dtype))
                grad_sgd = grad + self.weight_decay * var if self.weight_decay else grad
                mom_sgd = self._state(var, "sgd")
                mom_sgd.assign(mom_sgd * self.momentum + grad_sgd)
                sgd_update = grad_sgd + mom_sgd * self.momentum if self.nesterov else mom_sgd
                var.assign_sub(lr * self.sgd * sgd_update)
            else:
                grad_sgd = grad + self.weight_decay * var if self.weight_decay else grad
                mom = self._state(var, "sgd_only")
                mom.assign(mom * self.momentum + grad_sgd)
                update = grad_sgd + mom * self.momentum if self.nesterov else mom
                var.assign_sub(lr * update)


def make_optimizer(name="auto", lr=0.01, momentum=0.9, weight_decay=5e-4, iterations=1000):
    name = (name or "auto").lower()
    if name == "auto":
        name = "musgd" if iterations > 10000 else "adamw"
    if name == "musgd":
        return MuSGD(lr, momentum, weight_decay, nesterov=True, muon=0.2, sgd=1.0)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay, beta_1=momentum)
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=momentum)
    raise ValueError(f"Unsupported optimizer '{name}'")


class ModelEMA:
    """Exponential moving average for Keras model weights."""

    def __init__(self, model, decay=0.9999):
        self.decay = float(decay)
        self.shadow = [tf.Variable(v, trainable=False) for v in model.weights]
        self.updates = 0

    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - tf.exp(-self.updates / 2000))
        for s, v in zip(self.shadow, model.weights):
            s.assign(s * d + v * (1 - d))

    def apply_to(self, model):
        self.backup = [tf.identity(v) for v in model.weights]
        for v, s in zip(model.weights, self.shadow):
            v.assign(s)

    def restore(self, model):
        for v, b in zip(model.weights, self.backup):
            v.assign(b)

    def state_dict(self):
        return {"updates": int(self.updates), "shadow": [v.numpy() for v in self.shadow]}

    def load_state_dict(self, state):
        self.updates = int(state.get("updates", 0))
        for dst, src in zip(self.shadow, state.get("shadow", [])):
            dst.assign(src)
