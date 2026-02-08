from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn


def _unscale_and_check_(params: Iterable[nn.Parameter], inv_scale: float) -> bool:
    """
    In-place unscale of grads and a single GPU->CPU sync at the end.

    Args:
        params: iterable of parameters whose .grad will be unscaled in-place.
        inv_scale: 1/scale.

    Returns:
        found_inf: True if any gradient contains NaN/Inf after unscale.
    """
    found_inf_t: Optional[torch.Tensor] = None  # 0-dim bool tensor on grad device

    for p in params:
        g = p.grad
        if g is None:
            continue

        if found_inf_t is None:
            found_inf_t = torch.zeros((), device=g.device, dtype=torch.bool)

        if g.is_sparse:
            # Operate on stored values only.
            values = g._values()
            values.mul_(inv_scale)
            found_inf_t |= (~torch.isfinite(values)).any()
        else:
            g.mul_(inv_scale)
            found_inf_t |= (~torch.isfinite(g)).any()

    if found_inf_t is None:
        return False

    # Single synchronization point.
    return bool(found_inf_t.item())


@torch.no_grad()
def grad_zero_fraction(params: Iterable[nn.Parameter]) -> tuple[float, int]:
    """
    Diagnostic helper: returns (fraction_of_exact_zeros, total_elements) across all non-None grads.
    """
    total = 0
    zeros = 0
    for p in params:
        g = p.grad
        if g is None:
            continue
        if g.is_sparse:
            g = g._values()
        total += g.numel()
        zeros += (g == 0).sum().item()
    frac = (zeros / total) if total > 0 else 0.
    return frac, total


@dataclass
class StaticLossScaler:
    scale: float = 8192.

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * float(self.scale)

    @torch.no_grad()
    def unscale_(self, params: Iterable[nn.Parameter]) -> bool:
        inv_scale = 1. / float(self.scale)
        return _unscale_and_check_(params, inv_scale)

    @torch.no_grad()
    def update(self, found_inf: bool) -> None:
        _ = found_inf

    @torch.no_grad()
    def state_dict(self) -> dict[str, float]:
        return {"scale": float(self.scale)}

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = float(state["scale"])


@dataclass
class DynamicLossScaler:
    init_scale: float = 65536.
    growth_factor: float = 2.
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    min_scale: float = 1.
    max_scale: float = 2.**24

    def __post_init__(self) -> None:
        self.scale = float(self.init_scale)
        self._growth_tracker = 0  # consecutive finite steps

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * float(self.scale)

    @torch.no_grad()
    def unscale_(self, params: Iterable[nn.Parameter]) -> bool:
        inv_scale = 1. / float(self.scale)
        return _unscale_and_check_(params, inv_scale)

    @torch.no_grad()
    def update(self, found_inf: bool) -> None:
        if found_inf:
            self.scale = max(self.scale * float(self.backoff_factor), float(self.min_scale))
            self._growth_tracker = 0
            return

        self._growth_tracker += 1
        if self._growth_tracker >= int(self.growth_interval):
            self.scale = min(self.scale * float(self.growth_factor), float(self.max_scale))
            self._growth_tracker = 0

    @torch.no_grad()
    def state_dict(self) -> dict[str, float | int]:
        return {
            "scale": float(self.scale),
            "growth_tracker": int(self._growth_tracker),
            "init_scale": float(self.init_scale),
            "growth_factor": float(self.growth_factor),
            "backoff_factor": float(self.backoff_factor),
            "growth_interval": int(self.growth_interval),
            "min_scale": float(self.min_scale),
            "max_scale": float(self.max_scale),
        }

    @torch.no_grad()
    def load_state_dict(self, state: Dict[str, float | int]) -> None:
        self.scale = float(state["scale"])
        self._growth_tracker = int(state.get("growth_tracker", 0))
