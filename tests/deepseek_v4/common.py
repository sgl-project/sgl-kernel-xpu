from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, NamedTuple, Optional, Tuple

import torch

_is_xpu = True


# ----------------------------------------------------------------------------
# Plan tensor sizes (must match the C++ structs in compress.cuh).
# ----------------------------------------------------------------------------
_PREFILL_PLAN_BYTES = 24


class CompressorDecodePlan(NamedTuple):
    compress_ratio: int
    plan_d: torch.Tensor  # [batch_size, 16] uint8 --- DecodePlan

    def copy_(self, other) -> None:
        assert isinstance(other, CompressorDecodePlan)
        assert self.compress_ratio == other.compress_ratio
        self.plan_d.copy_(other.plan_d)

    @staticmethod
    def generate(
        compress_ratio: Literal[4, 128],
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        full_to_swa: torch.Tensor,
        seq_lens: torch.Tensor,
        swa_page_size: int,
        ring_size: int,
    ) -> CompressorDecodePlan:
        if _is_xpu:
            from sgl_kernel import plan_compress_decode

            fn = plan_compress_decode

        plan_d = fn(
            req_pool_indices,
            req_to_token,
            full_to_swa,
            seq_lens,
            int(compress_ratio),
            int(swa_page_size),
            int(ring_size),
        )
        return CompressorDecodePlan(compress_ratio, torch.from_dlpack(plan_d))

    @staticmethod
    def generate_legacy(
        compress_ratio: Literal[4, 128],
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> CompressorDecodePlan:
        if _is_xpu:
            from sgl_kernel import plan_compress_decode_legacy

            fn = plan_compress_decode_legacy

        plan_d = fn(req_pool_indices, seq_lens, compress_ratio)
        return CompressorDecodePlan(compress_ratio, torch.from_dlpack(plan_d))

    @property
    def is_decode(self) -> bool:
        return True


class CompressorPrefillPlan(NamedTuple):
    compress_ratio: int
    plan_c: torch.Tensor  # [num_q_tokens, 16] uint8 --- CompressPlan
    plan_w: torch.Tensor  # [num_q_tokens,  8] uint8 --- WritePlan
    pin_buffer: Optional[torch.Tensor] = None  # keep alive

    def copy_(self, other) -> None:
        assert isinstance(other, CompressorPrefillPlan)
        assert self.compress_ratio == other.compress_ratio
        self.plan_c.copy_(other.plan_c)
        self.plan_w.copy_(other.plan_w)

    @staticmethod
    def generate(
        compress_ratio: Literal[4, 128],
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        req_to_token: torch.Tensor,
        full_to_swa: torch.Tensor,
        swa_page_size: int,
        ring_size: int,
        num_q_tokens: int,
        use_cuda_graph: bool = False,
    ) -> CompressorPrefillPlan:
        is_gpu_input = seq_lens.device.type in ["cuda", "xpu"]
        pin_buffer = torch.empty(
            0 if is_gpu_input else num_q_tokens * _PREFILL_PLAN_BYTES,
            dtype=torch.uint8,
            pin_memory=not is_gpu_input,
        )
        if _is_xpu:
            from sgl_kernel import plan_compress_prefill

            fn = plan_compress_prefill

        plan_c, plan_w = fn(
            req_pool_indices,
            req_to_token,
            full_to_swa,
            seq_lens,
            extend_lens,
            pin_buffer,
            int(num_q_tokens),
            int(compress_ratio),
            int(swa_page_size),
            int(ring_size),
            bool(use_cuda_graph),
        )
        return CompressorPrefillPlan(
            compress_ratio,
            torch.from_dlpack(plan_c) if not _is_xpu else plan_c,
            torch.from_dlpack(plan_w) if not _is_xpu else plan_w,
            pin_buffer,
        )

    @staticmethod
    def generate_legacy(
        compress_ratio: Literal[4, 128],
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        num_q_tokens: int,
        device: torch.device,
        use_cuda_graph: bool = False,
    ) -> CompressorPrefillPlan:
        pin_buffer = torch.empty(
            num_q_tokens * _PREFILL_PLAN_BYTES,
            dtype=torch.uint8,
            pin_memory=True,
        )
        if _is_xpu:
            from sgl_kernel import plan_compress_prefill_legacy

            fn = plan_compress_prefill_legacy

        plan_c, plan_w = fn(
            req_pool_indices,
            seq_lens,
            extend_lens,
            pin_buffer,
            int(num_q_tokens),
            int(compress_ratio),
            bool(use_cuda_graph),
        )
        return CompressorPrefillPlan(
            compress_ratio,
            torch.from_dlpack(plan_c) if not _is_xpu else plan_c,
            torch.from_dlpack(plan_w) if not _is_xpu else plan_w,
            pin_buffer,
        )

    @property
    def is_decode(self) -> bool:
        return False


def compress_forward(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    *,
    head_dim: int,
    compress_ratio: Literal[4, 128],
    out: Optional[torch.Tensor] = None,
    is_online: bool = False,
) -> torch.Tensor:
    if out is None:
        num_q_tokens = plan[1].shape[0]  # NOTE: decode = bs, prefill = dynamic
        out = kv_score_input.new_empty((num_q_tokens, head_dim))
    assert plan.compress_ratio == compress_ratio
    if _is_xpu:
        if compress_ratio == 128:
            from sgl_kernel import flash_compress128_decode, flash_compress128_prefill

            flash_compress_decode = flash_compress128_decode
            flash_compress_prefill = flash_compress128_prefill
        else:
            from sgl_kernel import flash_compress4_decode, flash_compress4_prefill

            flash_compress_decode = flash_compress4_decode
            flash_compress_prefill = flash_compress4_prefill

    if _is_xpu:
        fn = flash_compress_decode if plan.is_decode else flash_compress_prefill

    fn(kv_score_buffer, kv_score_input, out, ape, *plan[1:3])
    return out


def get_device():
    return "xpu"


@dataclass
class LegacyContext:
    """Per-request ring buffer (no req_to_token / full_to_swa).

    `req_pool_indices[i]` directly maps to the request's ring base slot.
    """

    bs: int
    head_dim: int
    compress_ratio: int
    req_pool_indices: torch.Tensor  # int64 [bs] on cuda
    pages_per_req: int

    @property
    def num_pages(self) -> int:
        # Reserve enough pages to hold all batched requests' rings.
        return int(self.req_pool_indices.max().item() + 1) * self.pages_per_req

    def state_loc(self, b: int, position: int) -> int:
        rid = int(self.req_pool_indices[b].item())
        if self.compress_ratio == 4:
            page = rid * 2 + (position // 4) % 2
        else:
            page = rid
        return page * self.compress_ratio + position % self.compress_ratio

    def make_prefill_plan(
        self,
        seq_lens_cpu: torch.Tensor,
        extend_lens_cpu: torch.Tensor,
        num_q_tokens: int,
    ) -> CompressorPrefillPlan:
        return CompressorPrefillPlan.generate_legacy(
            compress_ratio=self.compress_ratio,  # type: ignore
            req_pool_indices=self.req_pool_indices,
            seq_lens=seq_lens_cpu,
            extend_lens=extend_lens_cpu,
            num_q_tokens=num_q_tokens,
            device=torch.device(get_device()),
        )

    def make_decode_plan(self, seq_lens_gpu: torch.Tensor) -> CompressorDecodePlan:
        return CompressorDecodePlan.generate_legacy(
            compress_ratio=self.compress_ratio,  # type: ignore
            req_pool_indices=self.req_pool_indices,
            seq_lens=seq_lens_gpu,
        )


@dataclass
class PagedContext:
    """SWA paged layout with identity req_to_token + identity full_to_swa.

    Each request occupies `num_swa_pages_per_req` contiguous swa_pages, so
    `req_to_token[r, p] = r * (num_swa_pages_per_req * swa_page_size) + p`.
    """

    bs: int
    head_dim: int
    compress_ratio: int
    swa_page_size: int
    ring_size: int
    num_swa_pages_per_req: int
    req_pool_indices: torch.Tensor  # int64 [bs] on cuda
    req_to_token: torch.Tensor  # int64 [num_reqs_capacity, max_tokens_per_req] on cuda
    full_to_swa: torch.Tensor  # int64 [num_swa_slots] on cuda

    @property
    def num_pages(self) -> int:
        # Upper bound: every (request, position) state slot fits.
        max_state_loc = (
            self.bs * self.num_swa_pages_per_req * self.ring_size
            + self.swa_page_size  # slack for the largest tail
        )
        return max_state_loc // self.compress_ratio + 1

    def state_loc(self, b: int, position: int) -> int:
        rid = int(self.req_pool_indices[b].item())
        loc = int(self.req_to_token[rid, position].item())
        swa_loc = int(self.full_to_swa[loc].item())
        swa_page = swa_loc // self.swa_page_size
        return swa_page * self.ring_size + swa_loc % self.ring_size

    def make_prefill_plan(
        self,
        seq_lens_cpu: torch.Tensor,
        extend_lens_cpu: torch.Tensor,
        num_q_tokens: int,
    ) -> CompressorPrefillPlan:
        return CompressorPrefillPlan.generate(
            compress_ratio=self.compress_ratio,  # type: ignore
            req_pool_indices=self.req_pool_indices,
            seq_lens=seq_lens_cpu,
            extend_lens=extend_lens_cpu,
            req_to_token=self.req_to_token,
            full_to_swa=self.full_to_swa,
            swa_page_size=self.swa_page_size,
            ring_size=self.ring_size,
            num_q_tokens=num_q_tokens,
        )

    def make_decode_plan(self, seq_lens_gpu: torch.Tensor) -> CompressorDecodePlan:
        return CompressorDecodePlan.generate(
            compress_ratio=self.compress_ratio,  # type: ignore
            req_pool_indices=self.req_pool_indices,
            req_to_token=self.req_to_token,
            full_to_swa=self.full_to_swa,
            seq_lens=seq_lens_gpu,
            swa_page_size=self.swa_page_size,
            ring_size=self.ring_size,
        )


def make_legacy_context(
    bs: int,
    compress_ratio: Literal[4, 128],
    head_dim: int = 512,
) -> LegacyContext:
    pages_per_req = 2 if compress_ratio == 4 else 1
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=get_device())
    return LegacyContext(
        bs=bs,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        req_pool_indices=req_pool_indices,
        pages_per_req=pages_per_req,
    )


def make_paged_context(
    bs: int,
    compress_ratio: Literal[4, 128],
    head_dim: int = 512,
    swa_page_size: int = 256,
    ring_size: Optional[int] = None,
    num_swa_pages_per_req: int = 8,
    max_tokens_per_req: int = 8192,
    num_reqs_capacity: int = 16,
) -> PagedContext:
    if ring_size is None:
        ring_size = 8 if compress_ratio == 4 else 128
    assert swa_page_size % ring_size == 0
    assert ring_size % compress_ratio == 0
    assert num_swa_pages_per_req * swa_page_size <= max_tokens_per_req

    stride = num_swa_pages_per_req * swa_page_size
    req_to_token = torch.zeros(
        (num_reqs_capacity, max_tokens_per_req), dtype=torch.int32
    )
    for r in range(bs):
        req_to_token[r, :stride] = torch.arange(
            r * stride, (r + 1) * stride, dtype=torch.int32
        )
    total_swa_slots = num_reqs_capacity * stride
    full_to_swa = torch.arange(total_swa_slots, dtype=torch.int64)
    req_pool_indices = torch.arange(bs, dtype=torch.int64)
    return PagedContext(
        bs=bs,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        swa_page_size=swa_page_size,
        ring_size=ring_size,
        num_swa_pages_per_req=num_swa_pages_per_req,
        req_pool_indices=req_pool_indices.to(get_device()),
        req_to_token=req_to_token.to(get_device()),
        full_to_swa=full_to_swa.to(get_device()),
    )


def make_state_pool(num_pages: int, compress_ratio: int, head_dim: int) -> torch.Tensor:
    last_dim = head_dim * (4 if compress_ratio == 4 else 2)
    return torch.zeros(
        (num_pages, compress_ratio, last_dim),
        dtype=torch.float32,
        device=get_device(),
    )


def to_seq_extend(
    seq_extend_pairs: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seq_lens = torch.tensor([s for s, _ in seq_extend_pairs], dtype=torch.int64)
    extend_lens = torch.tensor([e for _, e in seq_extend_pairs], dtype=torch.int64)
    num_q = int(extend_lens.sum().item())
    return seq_lens, extend_lens, num_q
