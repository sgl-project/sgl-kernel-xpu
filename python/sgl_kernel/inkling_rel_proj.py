from __future__ import annotations

import torch


def rel_proj_small_t(
    r: torch.Tensor,
    proj: torch.Tensor,
    tau: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project the production Inkling packed-r view with per-token tau."""
    if out is None:
        out = torch.empty(
            (r.shape[0], r.shape[1], proj.shape[1]),
            dtype=r.dtype,
            device=r.device,
        )
    return torch.ops.sgl_kernel.inkling_rel_proj_small_t.default(r, proj, tau, out)
