# utils.py
# Flexible config loader: supports
#   1. Hugging Face model config (--model-name)
#   2. Manual override via CLI args (e.g., --num-experts)
#   3. Safe fallback defaults

import argparse

from transformers import AutoConfig


def get_model_config(args):
    """
    Get model config with priority:
    1. CLI args override (e.g., --num-experts)
    2. Hugging Face config (if --model-name given)
    3. Hardcoded defaults (last resort)

    Args:
        args: Parsed command-line arguments

    Returns:
        dict: Standardized model config
    """
    config_dict = {}

    # Step 1: Load from Hugging Face model (if provided)
    if args.model_name:
        print(f"📡 Loading config from Hugging Face: {args.model_name}")
        try:
            hf_config = AutoConfig.from_pretrained(args.model_name)
        except Exception as e:
            raise ValueError(f"Failed to load {args.model_name}: {e}")

        # Extract with fallbacks
        config_dict.update(
            {
                "num_experts": getattr(hf_config, "moe_num_experts", None)
                or getattr(hf_config, "num_experts", None)
                or getattr(hf_config, "num_local_experts", None),
                "top_k": getattr(hf_config, "moe_top_k", None)
                or getattr(hf_config, "top_k", None)
                or getattr(hf_config, "num_experts_per_tok", None),
                "num_layers": getattr(hf_config, "num_hidden_layers", None)
                or getattr(hf_config, "num_layers", None),
                "hidden_size": getattr(hf_config, "hidden_size", None)
                or getattr(hf_config, "d_model", None),
                "ffn_hidden_size": getattr(hf_config, "intermediate_size", None)
                or getattr(hf_config, "ffn_dim", None),
                "num_heads": getattr(hf_config, "num_attention_heads", None),
                "num_kv_heads": getattr(hf_config, "num_key_value_heads", None)
                or getattr(hf_config, "num_attention_heads", None),
                "head_dim": getattr(hf_config, "head_dim", None)
                or (
                    getattr(hf_config, "hidden_size", None)
                    // getattr(hf_config, "num_attention_heads", 1)
                    if getattr(hf_config, "hidden_size", None)
                    and getattr(hf_config, "num_attention_heads")
                    else None
                ),
                "vocab_size": getattr(hf_config, "vocab_size", None),
                "max_seq_len": getattr(hf_config, "max_position_embeddings", None)
                or getattr(hf_config, "n_positions", 32768),
                "norm_eps": getattr(hf_config, "rms_norm_eps", None)
                or getattr(hf_config, "layer_norm_eps", 1e-6),
                "architectures": getattr(hf_config, "architectures", ["Unknown"]),
                "dtype": str(getattr(hf_config, "torch_dtype", "float16")),
            }
        )
    else:
        print("🔧 No --model-name provided. Using CLI args or defaults.")

    # Step 2: CLI args override everything
    cli_overrides = {
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "ffn_hidden_size": args.ffn_hidden_size,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "vocab_size": args.vocab_size,
        "max_seq_len": args.max_seq_len,
        "norm_eps": args.norm_eps,
    }

    for k, v in cli_overrides.items():
        if v is not None:
            config_dict[k] = v
            print(f"⚙️ Overriding {k} = {v} (from CLI)")

    # Step 3: Fill missing with safe defaults
    defaults = {
        "num_experts": 64,
        "top_k": 2,
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 11008,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
        "max_seq_len": 32768,
        "norm_eps": 1e-6,
        "architectures": ["LlamaForCausalLM"],
        "dtype": "float16",
    }

    for k, v in defaults.items():
        if k not in config_dict or config_dict[k] is None:
            config_dict[k] = v
            if args.model_name or any(
                getattr(args, field) is not None
                for field in ["num_experts", "top_k", "num_layers"]
            ):
                pass  # Don't log if user expected override
            else:
                print(f"💡 Using default {k} = {v}")

    # Add model name
    config_dict["model_name"] = args.model_name

    sweepable_config = {
        k: [v] if isinstance(v, (int, float, str)) else v
        for k, v in config_dict.items()
    }

    return sweepable_config


def parse_args():
    """Parse all possible model and benchmark arguments (support list values)."""
    parser = argparse.ArgumentParser(
        description="Flexible benchmark with model config support"
    )

    # Model source
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Hugging Face model name (e.g., deepseek-ai/DeepSeek-R1). If not set, use CLI args.",
    )

    # MoE parameters (support list)
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        nargs="*",
        help="Number of experts (can provide multiple values for sweep)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        nargs="*",
        help="Top-k experts per token (multiple values allowed)",
    )

    parser.add_argument(
        "--num-tokens",
        type=int,
        default=[100],
        nargs="*",
        help="Number of tokens (multiple values)",
    )

    # Transformer parameters (support list)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        nargs="*",
        help="Number of transformer layers (multiple values)",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=None, nargs="*", help="Hidden size (d_model)"
    )
    parser.add_argument(
        "--ffn-hidden-size",
        type=int,
        default=None,
        nargs="*",
        help="FFN/intermediate size",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        nargs="*",
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        nargs="*",
        help="Number of KV heads (for GQA)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        nargs="*",
        help="Dimension per attention head",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=None, nargs="*", help="Vocabulary size"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        nargs="*",
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--norm-eps",
        type=float,
        default=None,
        nargs="*",
        help="Normalization epsilon (rms_norm_eps)",
    )

    # Benchmark settings
    parser.add_argument(
        "--device", type=str, default="xpu", help="Device (default: xpu)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="torch.bfloat16",
        choices=["torch.float32", "torch.float16", "torch.bfloat16"],
        help="Data type",
    )

    return parser.parse_args()
