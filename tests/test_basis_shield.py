"""Unit tests for Stage 1: BasisShieldLoRA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.basis_shield import BasisShieldKey, BasisShieldLoRA


def _plain_lora_forward(
    base_linear: nn.Linear,
    A: torch.Tensor,
    B: torch.Tensor,
    x: torch.Tensor,
    lora_alpha: float,
) -> torch.Tensor:
    rank = A.shape[1]
    scaling = lora_alpha / rank
    return base_linear(x) + scaling * ((x @ A) @ B)


def test_mathematical_equivalence_valid_key() -> None:
    """||y_plain - y_shield|| < 1e-5 in float64 and relative error < 1e-5 in float32."""
    torch.manual_seed(42)
    d_in, d_out, rank = 256, 384, 64
    lora_alpha = 32.0

    for dtype, tol in [(torch.float64, 1e-10), (torch.float32, 1e-5)]:
        base = nn.Linear(d_in, d_out, bias=True, dtype=dtype)
        A = (torch.randn(d_in, rank, dtype=dtype) * 0.05)
        B = (torch.randn(rank, d_out, dtype=dtype) * 0.05)

        key = BasisShieldKey.from_secret(
            secret="corp-enclave-master-secret-2026",
            layer_id="model.layers.0.self_attn.q_proj",
            in_features=d_in,
            out_features=d_out,
            rank=rank,
            dtype=dtype,
        )

        shield = BasisShieldLoRA.from_plain_lora(
            base_linear=base,
            A=A,
            B=B,
            key=key,
            lora_alpha=lora_alpha,
            use_dynamic_q=True,
        )

        # Test 3D transformer-style activation tensor: (batch, seq_len, d_in)
        x = torch.randn(4, 16, d_in, dtype=dtype)
        y_plain = _plain_lora_forward(base, A, B, x, lora_alpha)
        y_shield = shield(x, step_seed=12345)

        diff_norm = torch.norm(y_plain - y_shield).item()
        rel_err = diff_norm / torch.norm(y_plain).item()
        max_abs_err = (y_plain - y_shield).abs().max().item()

        if dtype == torch.float64:
            assert diff_norm < 1e-5, f"float64 L2 diff too high: {diff_norm}"
        assert rel_err < tol, f"{dtype} relative L2 error too high: {rel_err}"
        assert max_abs_err < tol, f"{dtype} max abs error too high: {max_abs_err}"


def test_wrong_key_degrades_lora_output_to_noise() -> None:
    """Wrong key must destroy the LoRA delta signal (cosine similarity ~ 0)."""
    torch.manual_seed(7)
    d_in, d_out, rank = 256, 256, 64
    lora_alpha = 64.0

    # Zero out base weights to isolate the LoRA delta signal cleanly
    base = nn.Linear(d_in, d_out, bias=False)
    nn.init.zeros_(base.weight)

    A = torch.randn(d_in, rank)
    B = torch.randn(rank, d_out)

    valid_key = BasisShieldKey.from_secret(
        secret="valid-key-alpha",
        layer_id="layers.5.mlp.gate_proj",
        in_features=d_in,
        out_features=d_out,
        rank=rank,
    )
    wrong_key = BasisShieldKey.from_secret(
        secret="attacker-guessed-key",
        layer_id="layers.5.mlp.gate_proj",
        in_features=d_in,
        out_features=d_out,
        rank=rank,
    )

    shield = BasisShieldLoRA.from_plain_lora(
        base_linear=base,
        A=A,
        B=B,
        key=valid_key,
        lora_alpha=lora_alpha,
        bind_key=False,
    )

    x = torch.randn(8, 32, d_in)
    y_plain = _plain_lora_forward(base, A, B, x, lora_alpha)
    y_valid = shield(x, key=valid_key)
    y_wrong = shield(x, key=wrong_key)

    valid_rel_err = (
        torch.norm(y_plain - y_valid) / torch.norm(y_plain)
    ).item()
    assert valid_rel_err < 1e-5, f"Valid key rel_err too high: {valid_rel_err}"

    rel_error = torch.norm(y_plain - y_wrong) / torch.norm(y_plain)
    cos_sim = F.cosine_similarity(
        y_plain.reshape(-1), y_wrong.reshape(-1), dim=0
    ).abs()

    assert rel_error.item() > 1.0, f"Expected full degradation, got rel_error={rel_error}"
    assert cos_sim.item() < 0.05, f"Expected noise-level correlation, got cos_sim={cos_sim}"


def test_no_plain_lora_or_key_in_state_dict_or_memory() -> None:
    """Plain A, B and key transforms must never appear in state_dict or attributes."""
    torch.manual_seed(99)
    d_in, d_out, rank = 128, 128, 32
    base = nn.Linear(d_in, d_out, bias=True)
    A = torch.randn(d_in, rank)
    B = torch.randn(rank, d_out)

    key = BasisShieldKey.from_secret(
        secret=b"binary-enclave-key",
        layer_id="layers.1.mlp.down_proj",
        in_features=d_in,
        out_features=d_out,
        rank=rank,
    )

    shield = BasisShieldLoRA.from_plain_lora(
        base_linear=base,
        A=A,
        B=B,
        key=key,
        bind_key=False,
    )

    sd = shield.state_dict()
    expected_keys = {"base_linear.weight", "base_linear.bias", "A_prime", "B_prime"}
    assert set(sd.keys()) == expected_keys

    # Ensure A_prime and B_prime are genuinely transformed, not copies of A and B
    assert not torch.allclose(sd["A_prime"], A)
    assert not torch.allclose(sd["B_prime"], B)

    # Ensure no hidden plain A/B or dense (d, d) matrices exist on the object
    for attr_name, attr_val in shield.__dict__.items():
        if isinstance(attr_val, torch.Tensor):
            assert attr_name not in {"A", "B", "perm1", "perm2", "Q0"}
            assert attr_val.shape != (d_in, d_in)
            assert attr_val.shape != (d_out, d_out)

    # Ensure all parameter/buffer tensors are strictly smaller than d x d for LoRA
    for name, param in list(shield.named_parameters()) + list(shield.named_buffers()):
        assert name in expected_keys
        assert "A_plain" not in name and "B_plain" not in name
