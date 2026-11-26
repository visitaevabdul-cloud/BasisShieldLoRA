"""Tests verifying >99% extraction error under static and gray-box attacks."""

from __future__ import annotations

import torch
import torch.nn as nn

from benchmarks.attack_extraction import (
    run_dynamic_hook_attacks,
    run_static_svd_attack,
)
from src.layers.basis_shield import BasisShieldKey, BasisShieldLoRA


def test_extraction_attacks_fail_without_key() -> None:
    torch.manual_seed(123)
    d_in, d_out, rank = 256, 256, 64
    base = nn.Linear(d_in, d_out, bias=False)
    A = torch.randn(d_in, rank) * 0.05
    B = torch.randn(rank, d_out) * 0.05

    key = BasisShieldKey.from_secret(
        secret="stage2-test-secret",
        layer_id="layers.0.mlp.down_proj",
        in_features=d_in,
        out_features=d_out,
        rank=rank,
    )
    shield = BasisShieldLoRA.from_plain_lora(
        base_linear=base,
        A=A,
        B=B,
        key=key,
        lora_alpha=float(rank),
        use_dynamic_q=True,
        bind_key=True,
    )

    x_eval = torch.randn(512, d_in)
    static_metrics = run_static_svd_attack(
        A, B, shield.A_prime.detach(), shield.B_prime.detach(), x_eval
    )
    assert static_metrics.delta_w_rel_error_pct > 99.0
    assert static_metrics.output_rel_error_pct > 99.0

    dyn_metrics = run_dynamic_hook_attacks(
        shield, A, B, n_tokens=10_000, batch_size=500
    )
    # Gray-box internal hook (B' before P2^T) and un-isolated y_total must fail (>99% error)
    assert dyn_metrics[0].delta_w_rel_error_pct > 99.0
    assert dyn_metrics[0].output_rel_error_pct > 99.0
    assert dyn_metrics[1].delta_w_rel_error_pct > 99.0
