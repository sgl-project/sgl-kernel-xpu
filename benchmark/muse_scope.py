import os


def require_muse_b8_xe35():
    expected = {
        "BATCHES": "8",
        "BATCH_SIZE": "8",
        "BATCH": "8",
        "SERVING_BATCH": "8",
        "SERVING_BATCH_SIZE": "8",
        "SGLANG_BATCH_SIZE": "8",
        "ISL": "4096",
        "OSL": "16",
        "TP": "1",
        "EP": "1",
    }
    for name, value in expected.items():
        if os.environ.get(name, value) != value:
            raise SystemExit(f"scope error: {name} must be {value}; Muse B8 only")


def require_muse_arch_xe35():
    from _bench import current_arch

    architecture = current_arch()
    if architecture != "xe35":
        raise SystemExit(
            f"QK Muse port requires Xe35 XPU architecture 3.5; running arch={architecture}"
        )
