"""Stage 2: Weight extraction attack simulator against BasisShieldLoRA.

Simulates two key-less attack scenarios:
1. Static analysis of obfuscated weights (A', B', and A' @ B') via SVD.
2. Dynamic forward-hook interception on 10,000 random tokens using Ridge
   Regression on internal hooked activations and final outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.basis_shield import BasisShieldKey, BasisShieldLoRA


@dataclass
class AttackMetrics:
    scenario: str
    delta_w_rel_error_pct: float
    output_rel_error_pct: float
    cosine_similarity: float


def _rel_error_pct(target: torch.Tensor, estimate: torch.Tensor) -> float:
    denom = torch.norm(target).item()
    if denom == 0.0:
        return 0.0
    return (torch.norm(target - estimate).item() / denom) * 100.0


def _ridge_regression(
    X: torch.Tensor,
    Y: torch.Tensor,
    l2_reg: float = 1e-2,
) -> torch.Tensor:
    """Solve W_hat = argmin_W ||X W - Y||_F^2 + l2_reg ||W||_F^2."""
    d_in = X.shape[1]
    xtx = X.T @ X + l2_reg * torch.eye(d_in, device=X.device, dtype=X.dtype)
    xty = X.T @ Y
    return torch.linalg.solve(xtx, xty)


def run_static_svd_attack(
    A_plain: torch.Tensor,
    B_plain: torch.Tensor,
    A_prime: torch.Tensor,
    B_prime: torch.Tensor,
    x_eval: torch.Tensor,
) -> AttackMetrics:
    """Attempt to recover Delta W = A @ B directly from A', B' via SVD."""
    delta_w_true = A_plain @ B_plain
    rank = A_plain.shape[1]

    # Attacker computes product of obfuscated factors and runs truncated SVD
    delta_w_obf = A_prime @ B_prime
    u_obf, s_obf, vh_obf = torch.linalg.svd(delta_w_obf, full_matrices=False)
    u_r = u_obf[:, :rank]
    s_r = s_obf[:rank]
    vh_r = vh_obf[:rank, :]

    # Reconstructed Delta W without knowing P1, P2, Q0
    delta_w_hat = (u_r * s_r.unsqueeze(0)) @ vh_r

    y_true = x_eval @ delta_w_true
    y_hat = x_eval @ delta_w_hat

    cos_sim = F.cosine_similarity(
        delta_w_true.reshape(-1), delta_w_hat.reshape(-1), dim=0
    ).abs().item()

    return AttackMetrics(
        scenario="Static SVD on (A', B')",
        delta_w_rel_error_pct=_rel_error_pct(delta_w_true, delta_w_hat),
        output_rel_error_pct=_rel_error_pct(y_true, y_hat),
        cosine_similarity=cos_sim,
    )


def run_dynamic_hook_attacks(
    shield: BasisShieldLoRA,
    A_plain: torch.Tensor,
    B_plain: torch.Tensor,
    n_tokens: int = 10_000,
    batch_size: int = 500,
) -> List[AttackMetrics]:
    """Attach forward hooks during inference and attempt Ridge recovery."""
    d_in = shield.in_features
    d_out = shield.out_features
    delta_w_true = A_plain @ B_plain

    hooked_traces: Dict[str, List[torch.Tensor]] = {
        "x_in": [],
        "A_prime_out": [],  # h_masked = (x P1) @ A'
        "B_prime_out": [],  # out_perm = h_unmasked @ B' (before P2^T)
        "base_out": [],
        "total_out": [],
    }

    def hook_shield(
        _mod: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        hooked_traces["x_in"].append(inputs[0].detach())
        hooked_traces["total_out"].append(output.detach())

    def hook_base(
        _mod: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        hooked_traces["base_out"].append(output.detach())

    h1 = shield.register_forward_hook(hook_shield)
    h2 = shield.base_linear.register_forward_hook(hook_base)

    # Also intercept internal intermediate states via a lightweight wrapper
    # simulating memory-read hooks on A_prime and B_prime matmuls.
    orig_forward = shield.forward

    def instrumented_forward(x: torch.Tensor) -> torch.Tensor:
        active_key = shield._shield_key
        assert active_key is not None
        x_perm = x[..., active_key.perm1] * active_key.signs1
        h_masked = x_perm @ shield.A_prime
        hooked_traces["A_prime_out"].append(h_masked.detach())

        shield._step_counter += 1
        q_t = active_key.derive_dynamic_q(
            shield._step_counter, device=x.device, dtype=x.dtype
        )
        h_unmasked = (h_masked @ q_t) @ q_t.T
        out_perm = h_unmasked @ shield.B_prime
        hooked_traces["B_prime_out"].append(out_perm.detach())

        base_out = shield.base_linear(x)
        lora_out = (out_perm * active_key.signs2)[..., active_key.inv_perm2]
        return base_out + shield.scaling * lora_out

    try:
        shield.forward = instrumented_forward  # type: ignore[method-assign]
        with torch.no_grad():
            for start in range(0, n_tokens, batch_size):
                cur_bs = min(batch_size, n_tokens - start)
                x_batch = torch.randn(cur_bs, d_in)
                _ = shield(x_batch)
    finally:
        shield.forward = orig_forward  # type: ignore[method-assign]
        h1.remove()
        h2.remove()

    X = torch.cat(hooked_traces["x_in"], dim=0)
    B_out_perm = torch.cat(hooked_traces["B_prime_out"], dim=0)
    Y_base = torch.cat(hooked_traces["base_out"], dim=0)
    Y_total = torch.cat(hooked_traces["total_out"], dim=0)

    x_eval = torch.randn(1024, d_in)
    y_lora_eval_true = shield.scaling * (x_eval @ delta_w_true)

    results: List[AttackMetrics] = []

    # 2A. Gray-box hook on internal B_prime output (in P2-permuted space)
    w_hat_perm = _ridge_regression(X, B_out_perm * shield.scaling)
    y_hat_perm = x_eval @ w_hat_perm
    results.append(
        AttackMetrics(
            scenario="Gray-box Hook (B' out before P2^T, 10k tokens)",
            delta_w_rel_error_pct=_rel_error_pct(
                delta_w_true * shield.scaling, w_hat_perm
            ),
            output_rel_error_pct=_rel_error_pct(y_lora_eval_true, y_hat_perm),
            cosine_similarity=F.cosine_similarity(
                (delta_w_true * shield.scaling).reshape(-1),
                w_hat_perm.reshape(-1),
                dim=0,
            )
            .abs()
            .item(),
        )
    )

    # 2B. Gray-box hook on total output y_total without isolating y_base
    w_hat_total = _ridge_regression(X, Y_total)
    y_hat_total = x_eval @ w_hat_total
    results.append(
        AttackMetrics(
            scenario="Gray-box Hook (y_total without base subtraction, 10k tokens)",
            delta_w_rel_error_pct=_rel_error_pct(
                delta_w_true * shield.scaling, w_hat_total
            ),
            output_rel_error_pct=_rel_error_pct(y_lora_eval_true, y_hat_total),
            cosine_similarity=F.cosine_similarity(
                (delta_w_true * shield.scaling).reshape(-1),
                w_hat_total.reshape(-1),
                dim=0,
            )
            .abs()
            .item(),
        )
    )

    # 2C. Authorized-enclave Black-box Oracle: (X, Y_total - Y_base)
    # Measures the theoretical linear-oracle limit when an attacker can
    # freely query an authorized enclave AND isolate the exact LoRA delta.
    Y_lora_isolated = Y_total - Y_base
    w_hat_oracle = _ridge_regression(X, Y_lora_isolated)
    y_hat_oracle = x_eval @ w_hat_oracle
    results.append(
        AttackMetrics(
            scenario="Oracle Limit (authorized enclave + isolated LoRA delta, 10k tokens)",
            delta_w_rel_error_pct=_rel_error_pct(
                delta_w_true * shield.scaling, w_hat_oracle
            ),
            output_rel_error_pct=_rel_error_pct(y_lora_eval_true, y_hat_oracle),
            cosine_similarity=F.cosine_similarity(
                (delta_w_true * shield.scaling).reshape(-1),
                w_hat_oracle.reshape(-1),
                dim=0,
            )
            .abs()
            .item(),
        )
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate weight-extraction attacks on BasisShieldLoRA"
    )
    parser.add_argument("--d-in", type=int, default=256)
    parser.add_argument("--d-out", type=int, default=256)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=10_000)
    args = parser.parse_args()

    torch.manual_seed(2026)
    base = nn.Linear(args.d_in, args.d_out, bias=False)
    A = torch.randn(args.d_in, args.rank) * 0.05
    B = torch.randn(args.rank, args.d_out) * 0.05

    key = BasisShieldKey.from_secret(
        secret="enclave-secret-key-stage2",
        layer_id="layers.0.self_attn.q_proj",
        in_features=args.d_in,
        out_features=args.d_out,
        rank=args.rank,
    )
    shield = BasisShieldLoRA.from_plain_lora(
        base_linear=base,
        A=A,
        B=B,
        key=key,
        lora_alpha=float(args.rank),
        use_dynamic_q=True,
        bind_key=True,
    )

    x_eval = torch.randn(1024, args.d_in)
    static_res = run_static_svd_attack(
        A, B, shield.A_prime.detach(), shield.B_prime.detach(), x_eval
    )
    dynamic_res = run_dynamic_hook_attacks(
        shield, A, B, n_tokens=args.tokens
    )

    print("\n=== BasisShieldLoRA Extraction Attack Benchmark ===")
    print(f"Dims: d_in={args.d_in}, d_out={args.d_out}, rank={args.rank}, tokens={args.tokens}")
    print("-" * 92)
    print(
        f"{'Scenario':<62} | {'dW Err %':>8} | {'Out Err %':>9} | {'CosSim':>6}"
    )
    print("-" * 92)
    for m in [static_res, *dynamic_res]:
        print(
            f"{m.scenario:<62} | {m.delta_w_rel_error_pct:7.2f}% | "
            f"{m.output_rel_error_pct:8.2f}% | {m.cosine_similarity:6.4f}"
        )
    print("-" * 92)


if __name__ == "__main__":
    main()
