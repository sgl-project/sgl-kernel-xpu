# OP-7: respect the user's input data types (no silent precision reduction)

Why:

- **Precision is the user's decision.** A model's accuracy is validated
  with specific data types. Silently lowering the bit width (e.g. quantizing
  a BF16 query to FP8) degrades accuracy in ways unit tests with loose
  tolerances do not catch; it shows up only in end-to-end evaluation, such
  as long-context retrieval or reasoning tasks.
- **Implicit promotion or demotion hides cost and errors.** A cast inside
  the op adds kernels and memory traffic the user did not ask for. A missing
  FP8 scale silently defaulted to 1.0 can saturate or underflow and give
  wrong results with no error.
- **PyTorch makes reduced precision opt-in.** `allow_tf32` for matmul
  defaults to `False` and `torch.set_float32_matmul_precision()` defaults
  to `"highest"`: users explicitly authorize any precision loss.

## Kernel choice follows the input dtypes

Attention as the example; other ops apply the same pattern (the compute
path is chosen from the input dtypes, and every FP8 operand comes with its
scale).

| Query | Key / Value (or KV cache) | Kernel / compute dtype | Required scales |
|---|---|---|---|
| BF16 / FP16 | same dtype as the query | BF16 / FP16 | none |
| BF16 / FP16 | FP8 (e4m3 / e5m2) | FP8 KV-cache kernel: dequantize K/V with the user's scales, compute in the query dtype | `k_scale`, `v_scale` |
| FP8 | FP8 | FP8 attention kernel | `q_scale`, `k_scale`, `v_scale` |
| any other combination | | rejected with `TORCH_CHECK` unless the interface documents it | |

## No silent precision reduction

- Never quantize or downcast inputs inside the op (BF16 → FP8,
  FP32 → BF16 / TF32) unless the user opts in through the API.
- Never cast mismatched inputs to a common dtype; reject unsupported
  combinations.
- An op whose documented contract is quantized compute (e.g. a per-token
  FP8 quant op, or a W8A16 linear) satisfies this rule: the precision is
  explicit in its interface.

## FP8 operands need user-provided scales

- Never default a missing scale to 1.0; a required scale is a required
  argument, or its absence is rejected. If the upstream CUDA `sgl-kernel` op
  has an optional scale with a documented default, follow upstream and say
  so in the docstring.
- Validate each scale's dtype, device and documented granularity
  (per-tensor, per-head, per-block) at the op entry.

## Output dtype follows the API contract

- The output dtype equals the query (activation) dtype, or the dtype the
  interface documents (e.g. an explicit `out_dtype` argument, or the dtype
  of a user-provided `out`).
- Accumulate in FP32. Intermediate narrowing follows CONTRIBUTING.md "Data
  types and precision".

## Lower precision for benchmarking: opt-in, off by default

- Prefer an explicit API argument (e.g. an FP8-query mode that also takes
  `q_scale`).
- An environment flag that forces lower-precision compute is allowed only
  for performance work. It must be off by default, be read once on the host
  so it is constant across graph replays (OP-2), and never be enabled in
  correctness tests or CI.

```cpp
// ALLOWED: dtype-driven dispatch, required scales validated
TORCH_CHECK(k.scalar_type() == q.scalar_type() || is_fp8(k.scalar_type()),
            "attn: K/V must match the query dtype or be FP8");
if (is_fp8(k.scalar_type())) {
  TORCH_CHECK(k_scale.has_value() && v_scale.has_value(),
              "attn: FP8 K/V requires k_scale and v_scale");
  check_scale(*k_scale, q.device(), /*granularity=*/kPerHead);
  check_scale(*v_scale, q.device(), /*granularity=*/kPerHead);
}
auto out = out_.has_value() ? *out_ : at::empty_like(q);   // query dtype

// DISALLOWED
auto q8 = q.to(at::kFloat8_e4m3fn);                 // silent downcast of input
auto ks = k_scale.value_or(1.0);                    // missing scale -> 1.0
auto k2 = k.to(q.scalar_type());                    // cast mismatch to common dtype
auto out = at::empty_like(q, q.options().dtype(at::kBFloat16));  // FP32 q, BF16 out
```
