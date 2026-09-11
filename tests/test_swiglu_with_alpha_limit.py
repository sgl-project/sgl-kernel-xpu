import itertools
import sys

import pytest
import torch
from sgl_kernel import swiglu_gpt_oss_sigmoid_alpha


def swiglu_gpt_oss_sigmoid_alpha_ref(x, gemm1_alpha, gemm1_limit):
    """Reference implementation using native PyTorch"""
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


def swiglu_gpt_oss_sigmoid_alpha_ref_fp32(x, gemm1_alpha, gemm1_limit):
    """Same reference, evaluated in fp32 and narrowed -- the kernel's contract."""
    return swiglu_gpt_oss_sigmoid_alpha_ref(x.float(), gemm1_alpha, gemm1_limit).to(
        x.dtype
    )


# One output ULP. The kernel computes in fp32 and narrows once, so it should land
# on the fp32-then-narrow reference to within the rounding of that single cast;
# the 2-byte paths additionally use sycl::native::exp (~1e-6 relative, well
# inside one ULP of a bf16/fp16 output).
#
# These are *relative* ULPs at the unfavourable end of a binade. For a p-bit
# mantissa and v in [2^e, 2^(e+1)), one ULP is 2^(e-p), so ulp/v lies in
# (2^-(p+1), 2^-p]: 2^-10 for fp16 (p=10) and 2^-7 for bf16 (p=7). Using the
# favourable end (2^-11 / 2^-8) makes these tests fail on roughly 1 element in
# 32k, whenever the kernel's fp32 intermediate and torch's land either side of
# an output rounding boundary -- a genuine 1-ULP disagreement, not a kernel bug.
# That is a half-ULP tolerance masquerading as a one-ULP one, so it is set to a
# true ULP here.
#
# This is still far tighter than the 1e-1 of the sweep above and retains the
# power the tight-tolerance tests were added for: a tail that computed the wrong
# element, or a de-interleave that swapped gate and up, is off by O(1), not by
# one ULP.
TIGHT_TOL = {
    torch.float32: (1e-6, 1e-6),
    torch.bfloat16: (2**-7, 2**-7),
    torch.float16: (2**-10, 2**-10),
}


@pytest.mark.parametrize(
    "batch_size, hidden_size, alpha, limit, dtype",
    list(
        itertools.product(
            [1, 16, 128, 512, 1024],  # batch_size
            # hidden_size is the 2H dim (must be even). The powers of two here
            # all make B*H a multiple of the kernel's vector width, so they
            # never reach the tail path -- see test_swiglu_vector_tail.
            [64, 128, 256, 512, 1024, 2048, 4096],  # hidden_size
            [0.5, 1.0, 2.0],  # alpha
            [1.0, 5.0, 10.0],  # limit
            [torch.float32, torch.bfloat16, torch.float16],  # dtype
        )
    ),
)
def test_swiglu_gpt_oss_sigmoid_alpha(batch_size, hidden_size, alpha, limit, dtype):
    # Ensure hidden_size is even for gate/up split
    if hidden_size % 2 != 0:
        pytest.skip("hidden_size must be even")

    x = torch.randn((batch_size, hidden_size), dtype=dtype, device="xpu")

    # Call the kernel
    output = swiglu_gpt_oss_sigmoid_alpha(x, alpha, limit)

    # Reference implementation
    output_ref = swiglu_gpt_oss_sigmoid_alpha_ref(x, alpha, limit)

    # Verify the outputs match
    atol = 1e-1 if dtype in [torch.bfloat16, torch.float16] else 1e-4
    rtol = 1e-1 if dtype in [torch.bfloat16, torch.float16] else 1e-4
    assert torch.allclose(
        output_ref, output, atol=atol, rtol=rtol
    ), f"dtype = {dtype}Output mismatch: max_diff={torch.max(torch.abs(output_ref - output))}"


# B * H values that are NOT a multiple of the kernel's vector width. The kernel
# processes 16 B of output per work-item, so the width is 8 pairs at
# bf16/fp16 and 4 at fp32; these shapes cover every nonzero remainder for both,
# plus B*H strictly smaller than one vector.
#
# The pre-existing parameterization above can never reach this path: it varies
# the 2H dim over powers of two, so H is always a multiple of 4 and B*H is
# always a multiple of 8. That is why the previous kernel's block-granular
# bound (`vec_idx * 4 >= total_pairs`, which lets the last work-item read and
# write up to 3 pairs past the end of x and y) went unnoticed.
TAIL_SHAPES = [
    (1, 1),  # B*H = 1     -- smaller than one vector, tail does everything
    (2, 3),  # B*H = 6     -- %8 = 6, %4 = 2
    (1, 5),  # B*H = 5     -- %8 = 5, %4 = 1
    (3, 7),  # B*H = 21    -- %8 = 5, %4 = 1
    (7, 9),  # B*H = 63    -- %8 = 7, %4 = 3
    (1, 255),  # B*H = 255   -- %8 = 7, %4 = 3
    (17, 6),  # B*H = 102   -- %8 = 6, %4 = 2
    (128, 253),  # B*H = 32384 -- %8 = 0, %4 = 0 (control: no tail)
    (16, 721),  # B*H = 11536 -- %8 = 0 (control), production-shaped H
]


@pytest.mark.parametrize("batch_size, hidden", TAIL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_swiglu_vector_tail(batch_size, hidden, dtype):
    """B*H not a multiple of the vector width: exercises the scalar tail.

    Tolerance is one output ULP, not the 1e-1 of the sweep above -- a tail that
    computed the wrong element, or a de-interleave that swapped gate and up,
    can produce a value that is still within 1e-1 of the reference for small
    inputs.
    """
    alpha, limit = 1.702, 7.0
    # Seeded: at a one-ULP tolerance an unseeded draw makes this test
    # intermittent rather than wrong, and an intermittent failure in CI is
    # indistinguishable from an unrelated flake.
    torch.manual_seed(0)
    x = torch.randn((batch_size, hidden * 2), dtype=dtype, device="xpu")

    output = swiglu_gpt_oss_sigmoid_alpha(x, alpha, limit)
    output_ref = swiglu_gpt_oss_sigmoid_alpha_ref_fp32(x, alpha, limit)

    assert output.shape == (batch_size, hidden)
    rtol, atol = TIGHT_TOL[dtype]
    torch.testing.assert_close(output, output_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_swiglu_unaligned_input_view(dtype):
    """Contiguous view with an odd storage_offset -> unvectorizable input.

    x passes is_contiguous() but its data_ptr is not 32 B aligned, so the
    kernel must fall back to the elementwise path rather than misreading or
    rejecting it.
    """
    alpha, limit = 1.702, 7.0
    hidden = 61  # odd, so B*H also lands on the tail
    torch.manual_seed(0)  # see the note on seeding in test_swiglu_vector_tail
    big = torch.randn((1, hidden * 2 + 8), dtype=dtype, device="xpu")
    x = big[:, 1 : 1 + hidden * 2]
    assert x.is_contiguous()
    assert x.data_ptr() % 32 != 0, "view did not produce an unaligned pointer"

    output = swiglu_gpt_oss_sigmoid_alpha(x, alpha, limit)
    output_ref = swiglu_gpt_oss_sigmoid_alpha_ref_fp32(x, alpha, limit)

    rtol, atol = TIGHT_TOL[dtype]
    torch.testing.assert_close(output, output_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_swiglu_no_write_past_output(dtype):
    """The kernel must not write outside y.

    Guard, not a reproducer. The previous kernel's out-of-bounds write was at
    most (4 - B*H % 4) elements, and torch's XPU caching allocator rounds every
    allocation up to 512 B; since 4 * sizeof(scalar_t) divides 512, that write
    always landed inside y's own padded block and could never reach a
    neighbouring tensor. So this assertion passed before the fix as well as
    after it. It is here to catch a future tail bug whose overrun is large
    enough to escape the padding.
    """
    alpha, limit = 1.702, 7.0
    sentinel = 1.25  # exactly representable in fp32/fp16/bf16
    total_pairs = 253  # % 4 == 1: largest overrun, smallest allocator padding

    torch.xpu.empty_cache()
    torch.manual_seed(0)  # see the note on seeding in test_swiglu_vector_tail
    x = torch.randn((1, total_pairs * 2), dtype=dtype, device="xpu")
    first = swiglu_gpt_oss_sigmoid_alpha(x, alpha, limit)
    first_ptr = first.data_ptr()
    neighbour = torch.full((8192,), sentinel, dtype=dtype, device="xpu")
    del first

    # Re-run so the output reuses the freed block, which the allocator placed
    # immediately before `neighbour`.
    second = swiglu_gpt_oss_sigmoid_alpha(x, alpha, limit)
    torch.xpu.synchronize()
    if second.data_ptr() != first_ptr:
        pytest.skip("allocator did not reuse the block; layout assumption void")

    assert torch.equal(
        neighbour, torch.full_like(neighbour, sentinel)
    ), f"kernel wrote past the end of y (dtype={dtype}, B*H={total_pairs})"
    torch.testing.assert_close(
        second,
        swiglu_gpt_oss_sigmoid_alpha_ref_fp32(x, alpha, limit),
        rtol=TIGHT_TOL[dtype][0],
        atol=TIGHT_TOL[dtype][1],
    )


def _assert_clamps_bind(x, limit):
    """Check the premise of the clamp tests: that the clamps actually engage.

    Without this the tests below silently stop testing anything if the input
    scale or `limit` is ever changed -- which is exactly how `limit = 7.0`
    against unscaled `randn` came to look like clamp coverage.
    """
    gate, up = x[..., ::2].float(), x[..., 1::2].float()
    assert (gate > limit).any(), "gate's upper clamp never binds"
    assert (gate < -limit).any(), "no gate values below -limit: asymmetry untested"
    assert (up > limit).any(), "up's upper clamp never binds"
    assert (up < -limit).any(), "up's lower clamp never binds"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_swiglu_clamp_asymmetry(dtype):
    """limit small relative to input scale: forces gate's upper-only clamp
    and up's symmetric clamp to both bind, on both positive and negative
    inputs, so a clamp-shape bug (e.g. accidentally symmetric gate clamp)
    is caught at ULP precision.

    Every other tight-tolerance test in this file uses limit = 7.0 against
    unscaled randn (|x| <~ 4), so the clamps never engage in any of them.
    """
    alpha, limit = 1.702, 0.1  # small limit vs input scale
    hidden = 128
    torch.manual_seed(0)
    x = (
        torch.randn((256, hidden * 2), dtype=dtype, device="xpu") * 5.0
    )  # scale up so |x| >> limit
    _assert_clamps_bind(x, limit)

    output = swiglu_gpt_oss_sigmoid_alpha(x, alpha, limit)
    output_ref = swiglu_gpt_oss_sigmoid_alpha_ref_fp32(x, alpha, limit)

    rtol, atol = TIGHT_TOL[dtype]
    torch.testing.assert_close(output, output_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size, hidden", TAIL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_swiglu_clamp_with_vector_tail(batch_size, hidden, dtype):
    """Clamp crossed with the scalar tail.

    test_swiglu_clamp_asymmetry uses B*H = 32768, a multiple of both vector
    widths, so it only ever exercises the clamp on the vector path. These
    shapes put a binding clamp in the tail work-item as well.
    """
    alpha, limit = 1.702, 0.1
    torch.manual_seed(0)  # B*H = 1 needs a draw where the clamp actually binds
    x = torch.randn((batch_size, hidden * 2), dtype=dtype, device="xpu") * 5.0

    output = swiglu_gpt_oss_sigmoid_alpha(x, alpha, limit)
    output_ref = swiglu_gpt_oss_sigmoid_alpha_ref_fp32(x, alpha, limit)

    assert output.shape == (batch_size, hidden)
    rtol, atol = TIGHT_TOL[dtype]
    torch.testing.assert_close(output, output_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_swiglu_clamp_partially_binding(dtype):
    """A limit that binds on only part of the data, with O(1) outputs.

    In test_swiglu_clamp_asymmetry every unclamped-negative gate produces an
    output near 1e-10, so those elements pass on `atol` alone. Here the clamp
    binds on roughly half the elements while outputs stay O(0.1-1), making the
    comparison rtol-dominated and the clamp boundary itself the thing under
    test.
    """
    alpha, limit = 1.702, 0.7
    torch.manual_seed(0)
    x = torch.randn((256, 256), dtype=dtype, device="xpu")
    _assert_clamps_bind(x, limit)

    output = swiglu_gpt_oss_sigmoid_alpha(x, alpha, limit)
    output_ref = swiglu_gpt_oss_sigmoid_alpha_ref_fp32(x, alpha, limit)

    rtol, atol = TIGHT_TOL[dtype]
    torch.testing.assert_close(output, output_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
