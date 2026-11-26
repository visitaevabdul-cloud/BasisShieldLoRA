"""BasisShieldLoRA: LoRA computation in a secret transformed basis.

Executes low-rank adapter projections ``Delta W = A @ B`` directly in a
secret signed-permutation and orthogonal subspace without ever materializing
plain ``A`` or ``B`` matrices in disk checkpoints or GPU/CPU memory.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


def _hmac_seed(master_key: bytes, layer_id: str, purpose: str) -> int:
    """Derive a deterministic 63-bit integer seed via HMAC-SHA256."""
    mac = hmac.new(
        master_key,
        f"{layer_id}:{purpose}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return int.from_bytes(mac[:8], "big") & ((1 << 63) - 1)


def _sample_orthogonal(
    rank: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Sample an orthogonal matrix Q in O(rank) using QR decomposition."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn((rank, rank), generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(mat)
    # Fix sign ambiguity of QR so Q is uniformly distributed in O(rank)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.unsqueeze(0)
    return q.to(device=device, dtype=dtype)


def _sample_signed_permutation(
    dim: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a signed permutation without allocating a dense (dim, dim) matrix.

    Returns:
        perm: LongTensor of shape (dim,)
        inv_perm: LongTensor of shape (dim,) satisfying perm[inv_perm[i]] == i
        signs: Tensor of shape (dim,) with values in {-1.0, +1.0}
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(dim, generator=gen, dtype=torch.long)
    inv_perm = torch.argsort(perm)
    signs_bits = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.int64)
    signs = (signs_bits * 2 - 1).to(dtype=dtype)
    return perm.to(device), inv_perm.to(device), signs.to(device)


@dataclass
class BasisShieldKey:
    """Cryptographic key material for a single BasisShieldLoRA layer.

    This object is intentionally kept outside ``nn.Module`` parameters/buffers
    so that ``state_dict()`` never contains permutation indices, signs, or
    orthogonal rotation matrices.
    """

    master_key: bytes
    layer_id: str
    in_features: int
    out_features: int
    rank: int
    perm1: Tensor
    signs1: Tensor
    perm2: Tensor
    inv_perm2: Tensor
    signs2: Tensor
    Q0: Tensor

    @classmethod
    def from_secret(
        cls,
        secret: Union[str, bytes],
        layer_id: str,
        in_features: int,
        out_features: int,
        rank: int,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "BasisShieldKey":
        """Derive all layer-specific basis transformations from a secret key."""
        master_key = secret.encode("utf-8") if isinstance(secret, str) else secret
        dev = torch.device(device)

        perm1, _, signs1 = _sample_signed_permutation(
            in_features,
            _hmac_seed(master_key, layer_id, "P1"),
            dev,
            dtype,
        )
        perm2, inv_perm2, signs2 = _sample_signed_permutation(
            out_features,
            _hmac_seed(master_key, layer_id, "P2"),
            dev,
            dtype,
        )
        q0 = _sample_orthogonal(
            rank,
            _hmac_seed(master_key, layer_id, "Q0"),
            dev,
            dtype,
        )

        return cls(
            master_key=master_key,
            layer_id=layer_id,
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            perm1=perm1,
            signs1=signs1,
            perm2=perm2,
            inv_perm2=inv_perm2,
            signs2=signs2,
            Q0=q0,
        )

    def derive_dynamic_q(
        self,
        step_seed: Union[int, bytes],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Generate a dynamic orthogonal matrix Q_t in O(rank) for a token/batch."""
        dev = device if device is not None else self.Q0.device
        dt = dtype if dtype is not None else self.Q0.dtype
        seed_str = (
            step_seed.hex()
            if isinstance(step_seed, bytes)
            else str(step_seed)
        )
        seed = _hmac_seed(self.master_key, self.layer_id, f"Qt:{seed_str}")
        return _sample_orthogonal(self.rank, seed, dev, dt)

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "BasisShieldKey":
        """Return a copy of the key material moved to target device/dtype."""
        dev = torch.device(device) if device is not None else self.Q0.device
        dt = dtype if dtype is not None else self.Q0.dtype
        return BasisShieldKey(
            master_key=self.master_key,
            layer_id=self.layer_id,
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            perm1=self.perm1.to(dev),
            signs1=self.signs1.to(device=dev, dtype=dt),
            perm2=self.perm2.to(dev),
            inv_perm2=self.inv_perm2.to(dev),
            signs2=self.signs2.to(device=dev, dtype=dt),
            Q0=self.Q0.to(device=dev, dtype=dt),
        )


class BasisShieldLoRA(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with basis-obfuscated LoRA weights.

    Stores only ``base_linear`` and obfuscated matrices:
        A' = P1^T @ A @ Q0   (shape: [in_features, rank])
        B' = Q0^T @ B @ P2   (shape: [rank, out_features])

    Plain matrices ``A`` and ``B`` are never stored in the module or its
    ``state_dict()``. Signed permutations ``P1`` and ``P2`` are applied on the
    fly to activation vectors via indexing and elementwise sign flips without
    ever constructing dense ``(d, d)`` matrices.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        lora_alpha: float = 1.0,
        key: Optional[BasisShieldKey] = None,
        use_dynamic_q: bool = True,
        lora_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.use_dynamic_q = use_dynamic_q

        # Keep key outside parameters/buffers so state_dict() stays clean.
        self._shield_key: Optional[BasisShieldKey] = key
        self._step_counter: int = 0

        device = base_linear.weight.device
        dtype = lora_dtype if lora_dtype is not None else base_linear.weight.dtype
        self.A_prime = nn.Parameter(
            torch.zeros((self.in_features, rank), device=device, dtype=dtype)
        )
        self.B_prime = nn.Parameter(
            torch.zeros((rank, self.out_features), device=device, dtype=dtype)
        )

    @classmethod
    def from_plain_lora(
        cls,
        base_linear: nn.Linear,
        A: Tensor,
        B: Tensor,
        key: BasisShieldKey,
        lora_alpha: float = 1.0,
        use_dynamic_q: bool = True,
        bind_key: bool = True,
        lora_dtype: Optional[torch.dtype] = None,
    ) -> "BasisShieldLoRA":
        """Construct a shielded layer from plain LoRA factors A and B.

        Computes A' = P1^T @ A @ Q0 and B' = Q0^T @ B @ P2 without retaining
        any reference to A or B.
        """
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        rank = A.shape[1]

        if A.shape != (in_features, rank):
            raise ValueError(
                f"Expected A shape {(in_features, rank)}, got {tuple(A.shape)}"
            )
        if B.shape != (rank, out_features):
            raise ValueError(
                f"Expected B shape {(rank, out_features)}, got {tuple(B.shape)}"
            )
        if (
            key.in_features != in_features
            or key.out_features != out_features
            or key.rank != rank
        ):
            raise ValueError("BasisShieldKey dimensions do not match layer/LoRA")

        target_dtype = lora_dtype if lora_dtype is not None else A.dtype
        module = cls(
            base_linear=base_linear,
            rank=rank,
            lora_alpha=lora_alpha,
            key=key if bind_key else None,
            use_dynamic_q=use_dynamic_q,
            lora_dtype=target_dtype,
        )

        with torch.no_grad():
            dtype = target_dtype
            device = base_linear.weight.device
            k = key.to(device=device, dtype=dtype)

            # A' = P1^T @ A @ Q0  (apply P1^T via row indexing + sign flip)
            a_perm = A.to(device=device, dtype=dtype)[k.perm1] * k.signs1.unsqueeze(1)
            a_prime = a_perm @ k.Q0

            # B' = Q0^T @ B @ P2  (apply P2 via column indexing + sign flip)
            b_rot = k.Q0.T @ B.to(device=device, dtype=dtype)
            b_prime = b_rot[:, k.perm2] * k.signs2.unsqueeze(0)

            module.A_prime.copy_(a_prime)
            module.B_prime.copy_(b_prime)

        return module

    def set_key(self, key: Optional[BasisShieldKey]) -> None:
        """Attach or detach the runtime key object."""
        self._shield_key = key

    def forward(
        self,
        x: Tensor,
        key: Optional[BasisShieldKey] = None,
        step_seed: Optional[Union[int, bytes]] = None,
    ) -> Tensor:
        """Compute y = x W_base + scaling * (((x P1) A') Q_t Q_t^T (B' P2^T))."""
        active_key = key if key is not None else self._shield_key
        if active_key is None:
            raise RuntimeError(
                "BasisShieldLoRA requires a BasisShieldKey at runtime"
            )

        base_out = self.base_linear(x)

        lora_dtype = self.A_prime.dtype
        x_lora = x if x.dtype == lora_dtype else x.to(lora_dtype)

        # Ensure key tensors match LoRA compute device/dtype
        if (
            active_key.Q0.device != x_lora.device
            or active_key.Q0.dtype != lora_dtype
        ):
            active_key = active_key.to(device=x_lora.device, dtype=lora_dtype)

        # 1. Apply P1 to input activations on the fly: (x P1)_j = s1_j * x_{perm1[j]}
        x_perm = x_lora[..., active_key.perm1] * active_key.signs1

        # 2. Project into Q0-rotated rank-r subspace: h0 = (x P1) @ A'
        h_masked = x_perm @ self.A_prime

        # 3. Dynamic per-token/per-batch masking in O(r) so rank-r state varies
        if self.use_dynamic_q:
            if step_seed is None:
                self._step_counter += 1
                seed_val: Union[int, bytes] = self._step_counter
            else:
                seed_val = step_seed
            q_t = active_key.derive_dynamic_q(
                seed_val, device=x_lora.device, dtype=lora_dtype
            )
            h_dyn = h_masked @ q_t
            h_unmasked = h_dyn @ q_t.T
        else:
            h_unmasked = h_masked

        # 4. Project back to P2-permuted output space and apply P2^T on the fly
        out_perm = h_unmasked @ self.B_prime
        lora_out = (out_perm * active_key.signs2)[..., active_key.inv_perm2]
        if lora_out.dtype != base_out.dtype:
            lora_out = lora_out.to(base_out.dtype)

        return base_out + self.scaling * lora_out
