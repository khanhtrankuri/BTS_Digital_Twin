"""Checkpoint and exponential-moving-average utilities."""

from copy import deepcopy
import os
import torch


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = deepcopy(model).eval()
        self.decay = float(decay)
        for parameter in self.module.parameters(): parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            value.copy_(value * self.decay + source[name].detach() * (1.0 - self.decay))

    def state_dict(self): return self.module.state_dict()
    def load_state_dict(self, state): self.module.load_state_dict(state)


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, ema=None,
                    epoch=0, step=0, best_score=None, config=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"model": model.state_dict(), "epoch": epoch, "step": step,
               "best_score": best_score, "config": config}
    for name, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None: payload[name] = obj.state_dict()
    if ema is not None: payload["ema"] = ema.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, ema=None,
                    map_location="cpu", strict=True):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)
    for name, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None and payload.get(name) is not None: obj.load_state_dict(payload[name])
    if ema is not None and payload.get("ema") is not None: ema.load_state_dict(payload["ema"])
    return payload
