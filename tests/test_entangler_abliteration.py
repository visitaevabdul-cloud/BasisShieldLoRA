"""Tests for Stage 3: inject_basis_shield, SpectralEntanglerLoss, and Abliteration."""

from __future__ import annotations

import torch

from benchmarks.attack_abliteration import MiniQwen2LM, run_abliteration_benchmark
from src.layers.basis_shield import BasisShieldLoRA
from src.layers.injector import TARGET_MODULES, inject_basis_shield
from src.training.entangler import SpectralEntanglerLoss


def test_inject_basis_shield_replaces_target_layers() -> None:
    model = MiniQwen2LM(vocab_size=32, d_model=64, n_layers=2, n_heads=4, d_ff=128)
    shielded, keys = inject_basis_shield(
        model,
        secret="unit-test-secret",
        rank=8,
        lora_alpha=8.0,
    )
    expected_count = 2 * len(TARGET_MODULES)
    assert len(keys) == expected_count

    shield_modules = [
        m for m in shielded.modules() if isinstance(m, BasisShieldLoRA)
    ]
    assert len(shield_modules) == expected_count

    x = torch.randint(0, 32, (2, 6))
    logits, _ = shielded(x)
    assert logits.shape == (2, 6, 32)


def test_spectral_entangler_loss_differentiable() -> None:
    loss_fn = SpectralEntanglerLoss(lambda_entangle=1.5, top_k=4)
    util = torch.randn(32, 16, requires_grad=True)
    safe = torch.randn(32, 16, requires_grad=True)
    loss = loss_fn(util, safe)
    assert loss.item() >= 0.0
    loss.backward()
    assert util.grad is not None and torch.isfinite(util.grad).all()
    assert safe.grad is not None and torch.isfinite(safe.grad).all()


def test_entangled_model_degrades_utility_more_under_abliteration() -> None:
    res_a, res_b = run_abliteration_benchmark()

    # Both models must learn strong refusal and low utility perplexity before attack
    assert res_a.refusal_before > 0.85
    assert res_b.refusal_before > 0.85
    assert res_a.ppl_before < 5.0
    assert res_b.ppl_before < 5.0

    # Under Abliteration, Model B's utility perplexity must degrade >40% more than Model A
    assert res_b.ppl_increase_pct > res_a.ppl_increase_pct + 40.0, (
        f"Expected Model B PPL degradation >40% higher than Model A; "
        f"got A={res_a.ppl_increase_pct:.2f}%, B={res_b.ppl_increase_pct:.2f}%"
    )
