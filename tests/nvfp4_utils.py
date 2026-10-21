# SPDX-License-Identifier: Apache-2.0
"""NVFP4 reference helpers for the software-emulated NVFP4 MoE path on XPU.

The XPU stack has no native NVFP4 support: the Xe3P ``bdpas`` instruction only
accepts UE8M0 scale factors with a fixed block size of 32, while NVFP4 uses
E4M3 scale factors with a block size of 16 plus one FP32 per-tensor global
scale.  The emulation route implemented in
``nvfp4_woq_mxfp8_scaled_grouped_mm_xe35`` is weight-only-quantized (WoQ): the FP4
weights are dequantized to BF16 and requantized to MXFP8 inside the GEMM
mainloop, and the MMA itself runs as a normal MXFP8 (E4M3 x UE8M0/32)
block-scaled ``bdpas``.

Numerics contract (also implemented by the device kernels)
---------------------------------------------------------
NVFP4 weight, as shipped by the checkpoint::

    w[n, k] = e2m1_lut_xe35[q[n, k]] * sf16[n, k // 16] * gs_dq

    q     : (N, K/2) uint8, two E2M1 nibbles per byte, low nibble = even k
    sf16  : (N, K/16) float8_e4m3fn, per-16 block scale
    gs_dq : per-tensor FP32 *dequant* multiplier (reciprocal of the more
            common "global scale" that checkpoints store)

The one-time prep pass (:func:`nvfp4_woq_prep_ref_xe35`) rewrites the two scale
arrays into the pair the mainloop wants::

    amax32[n, j] = 6 * gs_dq * max(sf16[n, 2j], sf16[n, 2j + 1])
    kexp [n, j] = ceil(log2(amax32[n, j] / 448))          # UE8M0, no clipping
    c    [n, i] = gs_dq * sf16[n, i] * 2 ** -kexp[n, i // 2]   # BF16

``6`` is the largest E2M1 magnitude, so ``amax32`` upper-bounds the group and
``kexp`` is chosen so that requantization never clips.  NVFP4 quantizers pick
``sf16 = block_amax / 6``, which makes the bound tight in practice.

The mainloop then computes, per 32-element K group::

    w_mx[n, k] = to_e4m3(c[n, k // 16] * e2m1_lut_xe35[q[n, k]])
    sfb [n, j] = 2 ** kexp[n, j]                     # UE8M0 byte, kexp + 127

so ``w_mx * sfb`` reproduces ``w`` up to one E4M3 rounding per element.
"""

from typing import Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

NVFP4_BLOCK_SIZE = 16
MXFP8_BLOCK_SIZE = 32

FP8_E4M3_MAX = 448.0
E2M1_MAX = 6.0

# UE8M0 stores a biased exponent; byte 127 == 2**0.
UE8M0_BIAS = 127

# E2M1 magnitudes indexed by the 3 low bits; bit 3 is the sign.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1_lut_xe35(device="cpu", dtype=torch.float32) -> torch.Tensor:
    """16-entry signed E2M1 lookup table indexed by the raw nibble."""
    mags = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=device)
    return torch.cat([mags, -mags]).to(dtype)


# ---------------------------------------------------------------------------
# NVFP4 quantize / dequantize (reference)
# ---------------------------------------------------------------------------


def nvfp4_global_scale_xe35(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor NVFP4 global scale, ``(448 * 6) / amax``, as fp32 scalar."""
    amax = x.abs().float().amax().clamp_min(1e-12)
    return ((FP8_E4M3_MAX * E2M1_MAX) / amax).to(torch.float32)


def quantize_nvfp4_xe35(
    x: torch.Tensor, global_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to NVFP4.

    Returns ``(q, sf16, global_scale)`` where ``q`` is (N, K/2) uint8 packed
    E2M1 (low nibble = even k), ``sf16`` is (N, K/16) float8_e4m3fn and
    ``global_scale`` is the fp32 scalar such that
    ``dequantize_nvfp4_xe35(q, sf16, global_scale) ~= x``.
    """
    assert x.dim() == 2, "quantize_nvfp4_xe35 expects a 2D tensor"
    n, k = x.shape
    assert k % NVFP4_BLOCK_SIZE == 0, f"K ({k}) must be a multiple of 16"

    xf = x.float()
    if global_scale is None:
        global_scale = nvfp4_global_scale_xe35(xf)
    global_scale = global_scale.to(torch.float32).reshape(())

    blocks = xf.reshape(n, k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=-1)

    # sf lives in E4M3 after being brought into range by the global scale.
    sf = (block_amax / E2M1_MAX) * global_scale
    sf16 = sf.clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    # Per-element scale actually used on dequant, so quantization inverts it.
    inv = (sf16.float() / global_scale).clamp_min(1e-30).unsqueeze(-1)
    scaled = blocks / inv

    q_nibbles = _round_to_e2m1_nibbles_xe35(scaled).reshape(n, k)
    packed = q_nibbles[:, 0::2] | (q_nibbles[:, 1::2] << 4)
    return packed.to(torch.uint8).contiguous(), sf16, global_scale


def _round_to_e2m1_nibbles_xe35(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even into E2M1 and return raw nibbles (uint8)."""
    mags = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=x.device)
    sign = (x < 0).to(torch.uint8)
    a = x.abs().clamp(max=E2M1_MAX)

    # Midpoints between consecutive magnitudes.  bucketize(side='left') maps a
    # value strictly inside (mid[j-1], mid[j]] to magnitude index j, which is
    # round-to-nearest; exact midpoints need the tie broken to the even
    # encoding (mantissa LSB 0), matching IEEE RNE on a 1-bit mantissa.
    mid = (mags[:-1] + mags[1:]) / 2  # 7 boundaries, all exact in binary
    bucket = torch.bucketize(a, mid, right=False).to(torch.int64)
    at_mid = (a.unsqueeze(-1) == mid).any(dim=-1)
    # a == mid[j] sits between magnitude indices j and j+1.
    even_pick = torch.where(bucket % 2 == 0, bucket, bucket + 1)
    idx = torch.where(at_mid, even_pick, bucket).to(torch.uint8)
    return (sign << 3) | idx


def unpack_nvfp4_xe35(q: torch.Tensor) -> torch.Tensor:
    """Unpack (…, K/2) uint8 into (…, K) raw E2M1 nibbles (uint8)."""
    lo = q & 0xF
    hi = (q >> 4) & 0xF
    out = torch.stack([lo, hi], dim=-1)
    return out.reshape(*q.shape[:-1], q.shape[-1] * 2)


def dequantize_nvfp4_xe35(
    q: torch.Tensor,
    sf16: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize NVFP4 back to ``dtype`` (fp32 math)."""
    nibbles = unpack_nvfp4_xe35(q).long()
    lut = e2m1_lut_xe35(device=q.device)
    vals = lut[nibbles]

    k = vals.shape[-1]
    blocks = vals.reshape(*vals.shape[:-1], k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    sf = sf16.float().unsqueeze(-1) / global_scale.to(torch.float32)
    return (blocks * sf).reshape(*vals.shape).to(dtype)


# ---------------------------------------------------------------------------
# WoQ prep: NVFP4 scales -> (BF16 per-16 multiplier, UE8M0 per-32 scale)
# ---------------------------------------------------------------------------


def ue8m0_exponent_xe35(amax: torch.Tensor) -> torch.Tensor:
    """Smallest integer k with ``448 * 2**k >= amax``, computed exactly.

    ``frexp`` gives ``amax = m * 2**e`` with ``m`` in [0.5, 1).  Since
    ``448 == 1.75 * 2**8``, the condition is ``1.75 >= m * 2**(e - 8 - k)``, and
    with ``m < 1`` the smallest admissible ``k`` is ``e - 9`` when ``2m <= 1.75``
    and ``e - 8`` otherwise.  Mirrors ``ue8m0_exponent_xe35`` in src/sycl/xe35/nvfp4_woq.cpp.
    """
    m, e = torch.frexp(amax.float())
    k = torch.where(m <= 0.875, e - 9, e - 8).float()
    k = torch.where(amax > 0, k, torch.zeros_like(k))
    return k.clamp(-UE8M0_BIAS, 127)


def nvfp4_woq_prep_ref_xe35(
    sf16: torch.Tensor,
    gs_dq: torch.Tensor,
    transpose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference for the one-time NVFP4 -> WoQ scale prep pass.

    Args:
        sf16: (…, N, K/16) float8_e4m3fn NVFP4 block scales.
        gs_dq: per-tensor dequant multiplier, scalar or (E,) fp32.  This is the
            reciprocal of the checkpoint's ``global_scale``.
        transpose: emit the kernel layouts — ``c`` stays row-major
            ``(…, N, K/16)`` and ``sfmx`` becomes MN-major ``(…, K/32, N)``,
            which is what the Xe block-2D scale load needs.  ``False`` keeps
            both row-major, which is easier to compare against.

    Returns:
        ``(c, sfmx)`` with ``c`` bfloat16 and ``sfmx`` uint8 (UE8M0).
    """
    sf = sf16.float()
    assert sf.shape[-1] % 2 == 0, "K/16 must be even (K must be a multiple of 32)"

    gs = gs_dq.to(torch.float32)
    while gs.dim() < sf.dim() - 1:
        gs = gs.unsqueeze(-1)
    if gs.dim() == sf.dim() - 1:
        gs = gs.unsqueeze(-1)

    sf_g = sf * gs  # dequant scale per 16-block, global scale folded in

    pairs = sf_g.reshape(*sf_g.shape[:-1], sf_g.shape[-1] // 2, 2)
    amax32 = E2M1_MAX * pairs.amax(dim=-1)

    kexp = ue8m0_exponent_xe35(amax32)

    sfmx = (kexp + UE8M0_BIAS).to(torch.uint8)
    c = (sf_g / torch.exp2(kexp).repeat_interleave(2, dim=-1)).to(torch.bfloat16)

    if transpose:
        sfmx = sfmx.transpose(-1, -2).contiguous()
    return c.contiguous(), sfmx


def nvfp4_to_mxfp8_ref_xe35(
    q: torch.Tensor,
    sf16: torch.Tensor,
    gs_dq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Requantize NVFP4 to MXFP8, exactly as the fused mainloop does in registers.

    Returns ``(w_e4m3, sfmx)`` with ``w_e4m3`` (…, N, K) float8_e4m3fn and
    ``sfmx`` (…, N, K/32) uint8 UE8M0, row-major — i.e. drop-in MXFP8 B operands
    for ``fp8_blockwise_scaled_grouped_mm``.  Feeding these to the MXFP8 grouped
    GEMM gives a *bit-exact* oracle for the fused NVFP4 WoQ kernel: both paths
    multiply the same E4M3 values by the same UE8M0 scales, so any difference is
    a layout or race bug rather than rounding.
    """
    c, sfmx = nvfp4_woq_prep_ref_xe35(sf16, gs_dq, transpose=False)

    nibbles = unpack_nvfp4_xe35(q).long()
    lut = e2m1_lut_xe35(device=q.device)
    vals = lut[nibbles]  # exact dequant of the FP4 payload

    # ``c`` is BF16 in memory but the product is evaluated in FP32: c has 8
    # mantissa bits and an E2M1 value has 2, so the product is exact in FP32
    # and only the final E4M3 rounding is lossy.
    k = vals.shape[-1]
    per16 = vals.reshape(*vals.shape[:-1], k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    scaled = (per16 * c.float().unsqueeze(-1)).reshape(*vals.shape)
    return scaled.to(torch.float8_e4m3fn), sfmx


def dequantize_nvfp4_via_bf16_prepared_xe35(
    q: torch.Tensor,
    c: torch.Tensor,
    sfmx: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Direct-BF16 reference using the prepared ``(c, sfmx)`` scale pair.

    ``sfmx`` uses the device layout ``(..., K/32, N)``. The BF16 mainloop
    reconstructs the full per-16 multiplier, converts E2M1 exactly to BF16,
    and rounds the scaled value back to BF16 before DPAS.
    """
    nibbles = unpack_nvfp4_xe35(q).long()
    vals = e2m1_lut_xe35(device=q.device, dtype=torch.bfloat16)[nibbles]
    k = vals.shape[-1]
    kexp = sfmx.transpose(-1, -2).to(torch.int32) - UE8M0_BIAS
    full_scale = (c.float() * torch.exp2(kexp.float()).repeat_interleave(2, dim=-1)).to(
        torch.bfloat16
    )
    blocks = vals.reshape(*vals.shape[:-1], k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    return (blocks * full_scale.unsqueeze(-1)).reshape(*vals.shape).to(dtype)


def dequantize_nvfp4_via_mxfp8_xe35(
    q: torch.Tensor,
    sf16: torch.Tensor,
    gs_dq: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Emulation reference: NVFP4 -> BF16 -> MXFP8 -> dequantized value.

    Mirrors what the fused mainloop computes, so a test can separate
    "requantization error" from "kernel bug".
    """
    w_e4m3, sfmx = nvfp4_to_mxfp8_ref_xe35(q, sf16, gs_dq)
    return dequantize_mxfp8_xe35(w_e4m3, sfmx, dtype)


# ---------------------------------------------------------------------------
# MXFP8 activation helpers (reference; the kernel path uses sgl_kernel)
# ---------------------------------------------------------------------------


def quantize_mxfp8_xe35(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize (…, K) to MXFP8: E4M3 values + UE8M0 per-32 scales."""
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    xf = x.float()
    blocks = xf.reshape(
        *xf.shape[:-1], xf.shape[-1] // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE
    )
    amax = blocks.abs().amax(dim=-1)
    kexp = torch.ceil(torch.log2((amax / FP8_E4M3_MAX).clamp_min(1e-38)))
    kexp = torch.where(amax > 0, kexp, torch.zeros_like(kexp)).clamp(-UE8M0_BIAS, 127)
    q = (blocks / torch.exp2(kexp).unsqueeze(-1)).reshape(*xf.shape)
    return q.to(torch.float8_e4m3fn), (kexp + UE8M0_BIAS).to(torch.uint8)


def dequantize_mxfp8_xe35(
    q: torch.Tensor, sf: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Dequantize MXFP8 (E4M3 + UE8M0/32)."""
    vals = q.float()
    blocks = vals.reshape(
        *vals.shape[:-1], vals.shape[-1] // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE
    )
    scale = torch.exp2(sf.float() - UE8M0_BIAS).unsqueeze(-1)
    return (blocks * scale).reshape(*vals.shape).to(dtype)
