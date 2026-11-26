"""Recursive layer injection utilities for Qwen2 / Llama transformers."""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.layers.basis_shield import BasisShieldKey, BasisShieldLoRA

TARGET_MODULES: Tuple[str, ...] = (
    "q_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class PlainLoRALinear(nn.Module):
    """Standard unshielded LoRA wrapper around an nn.Linear layer."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        dev = base_linear.weight.device
        dt = lora_dtype if lora_dtype is not None else base_linear.weight.dtype
        self.A = nn.Parameter(torch.empty((self.in_features, rank), device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros((rank, self.out_features), device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base_linear(x)
        x_lora = x if x.dtype == self.A.dtype else x.to(self.A.dtype)
        lora_out = (x_lora @ self.A) @ self.B
        if lora_out.dtype != base_out.dtype:
            lora_out = lora_out.to(base_out.dtype)
        return base_out + self.scaling * lora_out


def inject_plain_lora(
    model: nn.Module,
    rank: int = 16,
    lora_alpha: float = 16.0,
    target_modules: Sequence[str] = TARGET_MODULES,
    freeze_base: bool = True,
    lora_dtype: Optional[torch.dtype] = None,
    name_filter: Optional[Callable[[str], bool]] = None,
) -> nn.Module:
    """Recursively wrap matching nn.Linear layers with PlainLoRALinear."""
    target_set = set(target_modules)

    def _recurse(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and child_name in target_set:
                if name_filter is not None and not name_filter(full_name):
                    continue
                if freeze_base:
                    for p in child.parameters():
                        p.requires_grad_(False)
                setattr(
                    parent,
                    child_name,
                    PlainLoRALinear(
                        child,
                        rank=rank,
                        lora_alpha=lora_alpha,
                        lora_dtype=lora_dtype,
                    ),
                )
            elif not isinstance(child, (PlainLoRALinear, BasisShieldLoRA)):
                _recurse(child, full_name)

    _recurse(model, "")
    return model


def inject_basis_shield(
    model: nn.Module,
    secret: Union[str, bytes],
    rank: int = 16,
    lora_alpha: float = 16.0,
    target_modules: Sequence[str] = TARGET_MODULES,
    use_dynamic_q: bool = True,
    bind_key: bool = True,
    freeze_base: bool = True,
    lora_dtype: Optional[torch.dtype] = None,
    name_filter: Optional[Callable[[str], bool]] = None,
) -> Tuple[nn.Module, Dict[str, BasisShieldKey]]:
    """Recursively replace target nn.Linear or PlainLoRALinear layers with BasisShieldLoRA.

    If a matching child is already a ``PlainLoRALinear``, its trained ``A`` and
    ``B`` factors are converted directly into obfuscated ``A'`` and ``B'``
    without ever retaining ``A`` or ``B``.
    """
    target_set = set(target_modules)
    keys: Dict[str, BasisShieldKey] = {}

    def _recurse(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, PlainLoRALinear) and child_name in target_set:
                if name_filter is not None and not name_filter(full_name):
                    continue
                if freeze_base:
                    for p in child.base_linear.parameters():
                        p.requires_grad_(False)
                dt = lora_dtype if lora_dtype is not None else child.A.dtype
                key = BasisShieldKey.from_secret(
                    secret=secret,
                    layer_id=full_name,
                    in_features=child.in_features,
                    out_features=child.out_features,
                    rank=child.rank,
                    device=child.base_linear.weight.device,
                    dtype=dt,
                )
                shield = BasisShieldLoRA.from_plain_lora(
                    base_linear=child.base_linear,
                    A=child.A.detach(),
                    B=child.B.detach(),
                    key=key,
                    lora_alpha=child.lora_alpha,
                    use_dynamic_q=use_dynamic_q,
                    bind_key=bind_key,
                    lora_dtype=dt,
                )
                keys[full_name] = key
                setattr(parent, child_name, shield)
            elif isinstance(child, nn.Linear) and child_name in target_set:
                if name_filter is not None and not name_filter(full_name):
                    continue
                if freeze_base:
                    for p in child.parameters():
                        p.requires_grad_(False)
                dev = child.weight.device
                dt = lora_dtype if lora_dtype is not None else child.weight.dtype
                A = torch.empty((child.in_features, rank), device=dev, dtype=dt)
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                B = torch.zeros((rank, child.out_features), device=dev, dtype=dt)
                key = BasisShieldKey.from_secret(
                    secret=secret,
                    layer_id=full_name,
                    in_features=child.in_features,
                    out_features=child.out_features,
                    rank=rank,
                    device=dev,
                    dtype=dt,
                )
                shield = BasisShieldLoRA.from_plain_lora(
                    base_linear=child,
                    A=A,
                    B=B,
                    key=key,
                    lora_alpha=lora_alpha,
                    use_dynamic_q=use_dynamic_q,
                    bind_key=bind_key,
                    lora_dtype=dt,
                )
                keys[full_name] = key
                setattr(parent, child_name, shield)
            elif not isinstance(child, BasisShieldLoRA):
                _recurse(child, full_name)

    _recurse(model, "")
    return model, keys
