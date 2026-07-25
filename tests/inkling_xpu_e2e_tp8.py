from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import socket
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


@dataclass(frozen=True)
class InklingE2EConfig:
    vocab_size: int = 4096
    hidden_size: int = 6144
    intermediate_size: int = 3072
    num_hidden_layers: int = 2
    num_attention_heads: int = 48
    num_key_value_heads: int = 4
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    rms_norm_eps: float = 1.0e-6
    dtype: str = "bf16"
    embedding_scale: float = 0.02
    weight_scale: float = 0.005
    rel_scale: float = 0.005


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _default_checkpoint_dir() -> Path:
    base = Path("/workspace/tmp") if Path("/workspace").is_dir() else Path("/tmp")
    return base / "inkling_xpu_e2e_tp8"


def _validate_config(config: InklingE2EConfig, tp_size: int) -> None:
    if config.hidden_size != config.num_attention_heads * config.head_dim:
        raise ValueError(
            "hidden_size must equal num_attention_heads * head_dim: "
            f"{config.hidden_size} != {config.num_attention_heads} * {config.head_dim}"
        )
    for name, value in (
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_attention_heads", config.num_attention_heads),
        ("vocab_size", config.vocab_size),
    ):
        if value % tp_size != 0:
            raise ValueError(f"{name}={value} must be divisible by tp_size={tp_size}")
    if config.num_key_value_heads > tp_size and config.num_key_value_heads % tp_size:
        raise ValueError("num_key_value_heads must divide tp_size or be divisible by it")
    if config.num_key_value_heads < tp_size and tp_size % config.num_key_value_heads:
        raise ValueError("tp_size must be divisible by num_key_value_heads")
    if config.num_key_value_heads <= 0:
        raise ValueError("num_key_value_heads must be positive")


def _rand(shape: tuple[int, ...], dtype: torch.dtype, scale: float) -> torch.Tensor:
    out = torch.empty(shape, dtype=dtype)
    out.normal_(mean=0.0, std=scale)
    return out


def _checkpoint_config_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "config.json"


def _checkpoint_weights_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "model.safetensors"


def _checkpoint_matches(checkpoint_dir: Path, config: InklingE2EConfig) -> bool:
    config_path = _checkpoint_config_path(checkpoint_dir)
    weights_path = _checkpoint_weights_path(checkpoint_dir)
    if not config_path.is_file() or not weights_path.is_file():
        return False
    try:
        saved = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return False
    return saved == asdict(config)


def generate_fake_checkpoint(
    checkpoint_dir: Path,
    config: InklingE2EConfig,
    *,
    seed: int,
) -> Path:
    """Write an upstream-style dense Inkling text checkpoint.

    The keys mirror the Inkling loader mapping in SGLang:
    wq_du/wk_dv/wv_dv/wr_du become qkvr slices, w13_dn/w2_md become the
    dense SwiGLU MLP, and all tensors are stored full-width before TP slicing.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    dtype = _dtype(config.dtype)
    h = config.hidden_size
    f = config.intermediate_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    r = config.num_attention_heads * config.d_rel

    tensors: dict[str, torch.Tensor] = {
        "llm.embed_tokens.weight": _rand(
            (config.vocab_size, h), dtype, config.embedding_scale
        ),
        "llm.norm.weight": torch.ones((h,), dtype=dtype),
        "llm.lm_head.weight": _rand(
            (config.vocab_size, h), dtype, config.weight_scale
        ),
    }

    for layer_id in range(config.num_hidden_layers):
        prefix = f"llm.layers.{layer_id}"
        tensors[f"{prefix}.attn_norm.weight"] = torch.ones((h,), dtype=dtype)
        tensors[f"{prefix}.mlp_norm.weight"] = torch.ones((h,), dtype=dtype)
        tensors[f"{prefix}.attn.wq_du.weight"] = _rand(
            (q, h), dtype, config.weight_scale
        )
        tensors[f"{prefix}.attn.wk_dv.weight"] = _rand(
            (kv, h), dtype, config.weight_scale
        )
        tensors[f"{prefix}.attn.wv_dv.weight"] = _rand(
            (kv, h), dtype, config.weight_scale
        )
        tensors[f"{prefix}.attn.wr_du.weight"] = _rand(
            (r, h), dtype, config.weight_scale
        )
        tensors[f"{prefix}.attn.wo_ud.weight"] = _rand(
            (h, q), dtype, config.weight_scale
        )
        tensors[f"{prefix}.attn.rel_logits_proj.proj"] = _rand(
            (config.d_rel, config.rel_extent), dtype, config.rel_scale
        )
        tensors[f"{prefix}.attn.q_norm.weight"] = torch.ones(
            (config.head_dim,), dtype=dtype
        )
        tensors[f"{prefix}.attn.k_norm.weight"] = torch.ones(
            (config.head_dim,), dtype=dtype
        )
        tensors[f"{prefix}.mlp.w13_dn.weight"] = _rand(
            (2 * f, h), dtype, config.weight_scale
        )
        tensors[f"{prefix}.mlp.w2_md.weight"] = _rand(
            (h, f), dtype, config.weight_scale
        )

    save_file(tensors, str(_checkpoint_weights_path(checkpoint_dir)))
    _checkpoint_config_path(checkpoint_dir).write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"
    )
    return _checkpoint_weights_path(checkpoint_dir)


def _install_local_sgl_kernel() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_dir = repo_root / "python" / "sgl_kernel"
    build_dir = repo_root / "build" / "src"
    if not package_dir.is_dir():
        return

    package_paths = [str(package_dir)]
    if build_dir.is_dir():
        package_paths.append(str(build_dir))

    pkg = sys.modules.get("sgl_kernel")
    if pkg is None:
        pkg = types.ModuleType("sgl_kernel")
        pkg.__path__ = package_paths  # type: ignore[attr-defined]
        sys.modules["sgl_kernel"] = pkg
        sys.modules["sgl_kernel.common_ops"] = types.ModuleType("sgl_kernel.common_ops")
    else:
        path = getattr(pkg, "__path__", None)
        if path is not None:
            for entry in package_paths:
                if entry not in path:
                    path.append(entry)

    rel_attn = build_dir / "inkling_relative_attention_ops.abi3.so"
    if rel_attn.is_file() and not hasattr(
        torch.ops.sgl_kernel, "inkling_relative_attention"
    ):
        torch.ops.load_library(str(rel_attn))


def _load_tensor(
    f: safe_open,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    rows: slice | None = None,
    cols: slice | None = None,
) -> torch.Tensor:
    tensor = f.get_tensor(name)
    if rows is not None:
        tensor = tensor[rows]
    if cols is not None:
        tensor = tensor[:, cols]
    return tensor.contiguous().to(device=device, dtype=dtype)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(variance + eps)
    return (out * weight.float()).to(x.dtype)


def _rms_norm_with_residual(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual is None:
        residual = x
    else:
        residual = x + residual
    return _rms_norm(residual, weight, eps), residual


class TPAttentionLayer:
    def __init__(
        self,
        f: safe_open,
        config: InklingE2EConfig,
        *,
        layer_id: int,
        rank: int,
        tp_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        prefix = f"llm.layers.{layer_id}.attn"
        h = config.hidden_size
        d = config.head_dim
        q_heads_local = config.num_attention_heads // tp_size
        q_rows = q_heads_local * d
        q_start = rank * q_rows
        r_rows = q_heads_local * config.d_rel
        r_start = rank * r_rows

        if config.num_key_value_heads >= tp_size:
            kv_heads_local = config.num_key_value_heads // tp_size
            kv_head_start = rank * kv_heads_local
        else:
            replicas = tp_size // config.num_key_value_heads
            kv_heads_local = 1
            kv_head_start = rank // replicas
        kv_rows = kv_heads_local * d
        kv_start = kv_head_start * d

        self.num_heads = q_heads_local
        self.num_kv_heads = kv_heads_local
        self.head_dim = d
        self.d_rel = config.d_rel
        self.rel_extent = config.rel_extent
        self.softmax_scale = 1.0 / float(d)
        self.eps = config.rms_norm_eps

        self.wq = _load_tensor(
            f,
            f"{prefix}.wq_du.weight",
            device,
            dtype,
            rows=slice(q_start, q_start + q_rows),
        )
        self.wk = _load_tensor(
            f,
            f"{prefix}.wk_dv.weight",
            device,
            dtype,
            rows=slice(kv_start, kv_start + kv_rows),
        )
        self.wv = _load_tensor(
            f,
            f"{prefix}.wv_dv.weight",
            device,
            dtype,
            rows=slice(kv_start, kv_start + kv_rows),
        )
        self.wr = _load_tensor(
            f,
            f"{prefix}.wr_du.weight",
            device,
            dtype,
            rows=slice(r_start, r_start + r_rows),
        )
        self.wo = _load_tensor(
            f,
            f"{prefix}.wo_ud.weight",
            device,
            dtype,
            cols=slice(q_start, q_start + q_rows),
        )
        self.rel_proj = _load_tensor(f, f"{prefix}.rel_logits_proj.proj", device, dtype)
        self.q_norm = _load_tensor(f, f"{prefix}.q_norm.weight", device, dtype)
        self.k_norm = _load_tensor(f, f"{prefix}.k_norm.weight", device, dtype)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        q_to_seq: torch.Tensor,
        q_pos: torch.Tensor,
        cu_k: torch.Tensor,
    ) -> torch.Tensor:
        from sgl_kernel.inkling_relative_attention import inkling_relative_attention

        t = hidden_states.shape[0]
        q = F.linear(hidden_states, self.wq).view(t, self.num_heads, self.head_dim)
        k = F.linear(hidden_states, self.wk).view(t, self.num_kv_heads, self.head_dim)
        v = F.linear(hidden_states, self.wv).view(t, self.num_kv_heads, self.head_dim)
        r = F.linear(hidden_states, self.wr).view(t, self.num_heads, self.d_rel)

        q = _rms_norm(q, self.q_norm, self.eps).contiguous()
        k = _rms_norm(k, self.k_norm, self.eps).contiguous()
        v = v.contiguous()
        rel_bias = (r.float() @ self.rel_proj.float()).contiguous()

        attn = inkling_relative_attention(
            q,
            k,
            v,
            q_to_seq,
            q_pos,
            cu_k,
            rel_bias=rel_bias,
            softmax_scale=self.softmax_scale,
            causal=True,
            window_size=(-1, -1),
        )
        return F.linear(attn.reshape(t, -1).contiguous(), self.wo)


class TPDenseMLP:
    def __init__(
        self,
        f: safe_open,
        config: InklingE2EConfig,
        *,
        layer_id: int,
        rank: int,
        tp_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        prefix = f"llm.layers.{layer_id}.mlp"
        local_intermediate = config.intermediate_size // tp_size
        w13_rows = 2 * local_intermediate
        w13_start = rank * w13_rows
        w2_col_start = rank * local_intermediate
        self.w13 = _load_tensor(
            f,
            f"{prefix}.w13_dn.weight",
            device,
            dtype,
            rows=slice(w13_start, w13_start + w13_rows),
        )
        self.w2 = _load_tensor(
            f,
            f"{prefix}.w2_md.weight",
            device,
            dtype,
            cols=slice(w2_col_start, w2_col_start + local_intermediate),
        )

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        z = F.linear(hidden_states, self.w13)
        x = (F.silu(z[..., 0::2].float()) * z[..., 1::2].float()).to(z.dtype)
        return F.linear(x.contiguous(), self.w2)


class TPDecoderLayer:
    def __init__(
        self,
        f: safe_open,
        config: InklingE2EConfig,
        *,
        layer_id: int,
        rank: int,
        tp_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        prefix = f"llm.layers.{layer_id}"
        self.attn_norm = _load_tensor(f, f"{prefix}.attn_norm.weight", device, dtype)
        self.mlp_norm = _load_tensor(f, f"{prefix}.mlp_norm.weight", device, dtype)
        self.attn = TPAttentionLayer(
            f,
            config,
            layer_id=layer_id,
            rank=rank,
            tp_size=tp_size,
            device=device,
            dtype=dtype,
        )
        self.mlp = TPDenseMLP(
            f,
            config,
            layer_id=layer_id,
            rank=rank,
            tp_size=tp_size,
            device=device,
            dtype=dtype,
        )
        self.eps = config.rms_norm_eps

    def __call__(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        q_to_seq: torch.Tensor,
        q_pos: torch.Tensor,
        cu_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = _rms_norm_with_residual(
            hidden_states, residual, self.attn_norm, self.eps
        )
        hidden_states = self.attn(hidden_states, q_to_seq, q_pos, cu_k)
        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM)

        hidden_states, residual = _rms_norm_with_residual(
            hidden_states, residual, self.mlp_norm, self.eps
        )
        hidden_states = self.mlp(hidden_states)
        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM)
        return hidden_states, residual


class TPInklingModel:
    def __init__(
        self,
        checkpoint_path: Path,
        config: InklingE2EConfig,
        *,
        rank: int,
        tp_size: int,
        device: torch.device,
    ) -> None:
        dtype = _dtype(config.dtype)
        vocab_per_rank = config.vocab_size // tp_size
        vocab_start = rank * vocab_per_rank
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
            self.embed_tokens = _load_tensor(
                f, "llm.embed_tokens.weight", device, dtype
            )
            self.layers = [
                TPDecoderLayer(
                    f,
                    config,
                    layer_id=i,
                    rank=rank,
                    tp_size=tp_size,
                    device=device,
                    dtype=dtype,
                )
                for i in range(config.num_hidden_layers)
            ]
            self.norm = _load_tensor(f, "llm.norm.weight", device, dtype)
            self.lm_head = _load_tensor(
                f,
                "llm.lm_head.weight",
                device,
                dtype,
                rows=slice(vocab_start, vocab_start + vocab_per_rank),
            )
        self.eps = config.rms_norm_eps

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        q_to_seq: torch.Tensor,
        q_pos: torch.Tensor,
        cu_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = F.embedding(input_ids, self.embed_tokens)
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, q_to_seq, q_pos, cu_k
            )
        hidden_states, _ = _rms_norm_with_residual(
            hidden_states, residual, self.norm, self.eps
        )
        logits_local = F.linear(hidden_states, self.lm_head)
        return logits_local, hidden_states


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(
    rank: int,
    world_size: int,
    port: int,
    checkpoint_path: str,
    config_dict: dict[str, Any],
    batch_size: int,
    seq_len: int,
    seed: int,
) -> None:
    _install_local_sgl_kernel()
    config = InklingE2EConfig(**config_dict)
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("this e2e test requires torch.xpu")
    if torch.xpu.device_count() < world_size:
        raise RuntimeError(
            f"need {world_size} XPU devices, found {torch.xpu.device_count()}"
        )

    device = torch.device("xpu", rank)
    torch.xpu.set_device(device)
    backend = dist.get_default_backend_for_device(device)
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(seed)
        model = TPInklingModel(
            Path(checkpoint_path),
            config,
            rank=rank,
            tp_size=world_size,
            device=device,
        )

        total_tokens = batch_size * seq_len
        input_ids = torch.randint(
            0, config.vocab_size, (total_tokens,), dtype=torch.long
        ).to(device)
        q_to_seq = torch.arange(batch_size, dtype=torch.int32).repeat_interleave(seq_len)
        q_to_seq = q_to_seq.to(device)
        q_pos = torch.arange(seq_len, dtype=torch.int32).repeat(batch_size).to(device)
        cu_k = torch.arange(
            0, total_tokens + 1, seq_len, dtype=torch.int32, device=device
        )

        logits_local, hidden_states = model(input_ids, q_to_seq, q_pos, cu_k)
        torch.xpu.synchronize()

        nonfinite = (
            (~torch.isfinite(logits_local.float())).sum()
            + (~torch.isfinite(hidden_states.float())).sum()
        ).to(torch.int64)
        dist.all_reduce(nonfinite, op=dist.ReduceOp.SUM)
        max_abs = torch.stack(
            (
                torch.nan_to_num(hidden_states.float().abs()).max(),
                torch.nan_to_num(logits_local.float().abs()).max(),
            )
        )
        dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)

        if rank == 0:
            print(
                f"Inkling XPU E2E TP={world_size}: "
                f"layers={config.num_hidden_layers} batch={batch_size} seq={seq_len} "
                f"hidden={tuple(hidden_states.shape)} "
                f"local_logits={tuple(logits_local.shape)} "
                f"max_abs_hidden={float(max_abs[0].item()):.6f} "
                f"max_abs_logits={float(max_abs[1].item()):.6f} "
                f"nonfinite={int(nonfinite.item())}"
            )
        if int(nonfinite.item()) != 0:
            raise RuntimeError(
                f"non-finite output detected: count={int(nonfinite.item())}"
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_e2e(
    checkpoint_path: Path,
    config: InklingE2EConfig,
    *,
    world_size: int,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> None:
    mp.set_start_method("spawn", force=True)
    port = _open_port()
    processes: list[mp.Process] = []
    for rank in range(world_size):
        process = mp.Process(
            target=_worker,
            args=(
                rank,
                world_size,
                port,
                str(checkpoint_path),
                asdict(config),
                batch_size,
                seq_len,
                seed,
            ),
            name=f"inkling-xpu-e2e-rank-{rank}",
        )
        process.start()
        processes.append(process)

    failed: list[tuple[int, int | None]] = []
    for rank, process in enumerate(processes):
        process.join()
        if process.exitcode != 0:
            failed.append((rank, process.exitcode))
    if failed:
        raise RuntimeError(f"worker failures: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dense Inkling forward-pass smoke test on XPU TP=8."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=_default_checkpoint_dir())
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-heads", type=int, default=48)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--d-rel", type=int, default=16)
    parser.add_argument("--rel-extent", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--embedding-scale", type=float, default=0.02)
    parser.add_argument("--weight-scale", type=float, default=0.005)
    parser.add_argument("--rel-scale", type=float, default=0.005)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = InklingE2EConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        d_rel=args.d_rel,
        rel_extent=args.rel_extent,
        dtype=args.dtype,
        embedding_scale=args.embedding_scale,
        weight_scale=args.weight_scale,
        rel_scale=args.rel_scale,
    )
    _validate_config(config, args.world_size)

    if args.regenerate or not _checkpoint_matches(args.checkpoint_dir, config):
        path = generate_fake_checkpoint(args.checkpoint_dir, config, seed=args.seed)
        size_gib = path.stat().st_size / (1024**3)
        print(f"generated fake checkpoint: {path} ({size_gib:.2f} GiB)")
    else:
        path = _checkpoint_weights_path(args.checkpoint_dir)
        print(f"using existing fake checkpoint: {path}")

    run_e2e(
        path,
        config,
        world_size=args.world_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed + 17,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
