"""Split a model between CPU memory and the GPU when it does not fit the GPU (E6 Gemma 4, D-032;
E4 Qwen3-VL-32B, D-041). Exact: weights are only moved, never changed.

- `CpuLookup`: an embedding table kept on the CPU; only the looked-up rows go to the GPU.
- `stream_weights`: a module's weights stay in CPU memory and are copied to the GPU only while it
  runs. Unlike accelerate's `cpu_offload`, the call's arguments are left untouched (Gemma 4 layers fill
  a `shared_kv_states` dict in place). Pinned memory is faster but cannot be swapped out, so on a
  shared machine it is used only when enough RAM is free (`pin=None` decides; 8 GB margin). Unpinned,
  memory-mapped weights stay in the OS file cache, which the OS can reclaim.
- `place_layers`: everything except the given CPU-only modules and the decoder layers that do not fit
  `gpu_budget_gb` goes to the GPU; the rest is streamed.
"""
from __future__ import annotations

import os

import torch.nn as nn


class CpuLookup(nn.Module):
    """Wraps an embedding table kept on the CPU: ids go to the CPU, looked-up rows come back."""

    def __init__(self, emb, device):
        super().__init__()
        self.emb, self.device_out = emb, device

    @property
    def weight(self):
        return self.emb.weight

    def forward(self, ids):
        return self.emb(ids.to(self.emb.weight.device)).to(self.device_out)


def _nbytes(module, skip=()) -> int:
    """Bytes of parameters and buffers under `module`, leaving out every module in `skip` together
    with everything below it (shared tensors counted once)."""
    seen, n = set(), 0

    def walk(m):
        nonlocal n
        if any(m is s for s in skip):
            return
        for t in list(m.parameters(recurse=False)) + list(m.buffers(recurse=False)):
            if id(t) not in seen:
                seen.add(id(t))
                n += t.numel() * t.element_size()
        for c in m.children():
            walk(c)

    walk(module)
    return n


def _move_except(module, device, skip) -> None:
    """Move parameters and buffers to `device`, leaving the modules in `skip` (and below) untouched."""
    if any(module is s for s in skip):
        return
    for p in module.parameters(recurse=False):
        p.data = p.data.to(device)
    for name, b in list(module.named_buffers(recurse=False)):
        module._buffers[name] = b.to(device)
    for c in module.children():
        _move_except(c, device, skip)


def mem_available_gb() -> float:
    """RAM the OS can still give out (Linux MemAvailable; psutil elsewhere)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    import psutil
    return psutil.virtual_memory().available / 1e9


def stream_weights(module, device, pin: bool = True) -> None:
    """Keep `module`'s weights in CPU memory (pinned if `pin`) and copy them to `device` while it runs."""
    import torch
    tensors = [(m, kind, name, t) for m in module.modules()
               for kind, items in (("p", m._parameters), ("b", m._buffers)) for name, t in items.items() if t is not None]
    cpu = {}
    for m, kind, name, t in tensors:
        host = t.data.cpu()
        cpu[(id(m), kind, name)] = host.pin_memory() if pin and torch.cuda.is_available() else host
        if kind == "p":
            t.data = cpu[(id(m), kind, name)]
        else:
            m._buffers[name] = cpu[(id(m), kind, name)]

    def load(_mod, _args):
        for m, kind, name, t in tensors:
            host = cpu[(id(m), kind, name)]
            if kind == "p":
                t.data = host.to(device, non_blocking=True)
            else:
                m._buffers[name] = host.to(device, non_blocking=True)

    def unload(_mod, _args, _out):
        for m, kind, name, t in tensors:
            if kind == "p":
                t.data = cpu[(id(m), kind, name)]
            else:
                m._buffers[name] = cpu[(id(m), kind, name)]

    module.register_forward_pre_hook(load)
    module.register_forward_hook(unload)


def place_layers(model, layers, device: str, gpu_budget_gb: float | None, cpu_only=(), pin: bool | None = None,
                 ram_margin_gb: float = 8.0) -> dict:
    """Move a CPU-loaded model to `device`, except `cpu_only` modules and the decoder `layers` that do not
    fit `gpu_budget_gb` (those are streamed). `pin=None`: pin only if the streamed bytes leave at least
    `ram_margin_gb` of RAM free. Returns the placement summary."""
    layers = list(layers)
    cpu_only = list(cpu_only)
    fixed = _nbytes(model, skip=cpu_only + layers)
    budget = float("inf") if gpu_budget_gb is None else gpu_budget_gb * 1e9
    on_gpu, used = [], fixed
    for layer in layers:
        size = _nbytes(layer)
        if used + size > budget:
            break
        on_gpu.append(layer)
        used += size
    offloaded = layers[len(on_gpu):]
    streamed = sum(_nbytes(layer) for layer in offloaded)
    if pin is None:
        pin = bool(offloaded) and mem_available_gb() - streamed / 1e9 >= ram_margin_gb
    _move_except(model, device, skip=cpu_only + offloaded)
    for layer in offloaded:
        stream_weights(layer, device, pin=pin)
    return {"device": device, "gpu_budget_gb": gpu_budget_gb, "gpu_weights_gb": round(used / 1e9, 2),
            "layers_on_gpu": len(on_gpu), "layers_offloaded": len(offloaded),
            "streamed_gb": round(streamed / 1e9, 2), "pinned": bool(pin and offloaded), "pid": os.getpid()}
