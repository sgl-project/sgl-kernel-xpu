"""Host-side logic of the SGLang monokernel plugin (no GPU): the memory budget
that refuses a --mem-fraction-static without physical headroom, the wait
deadline, the fail-stop watch that turns a failed in-kernel wait into a server
stop, and the spec's configuration checks."""

import threading
import time
import types

import pytest
import torch

memory = pytest.importorskip("sgl_kernel.monokernel.memory")
failure = pytest.importorskip("sgl_kernel.monokernel.failure")

GIB = 1 << 30
CARD = int(31.89 * GIB)  # B70


# ----------------------------------------------------------------------------
# memory budget


def test_graph_counts():
    g = memory.Graphs(decode_bs=tuple(range(1, 9)), prefill_shapes=50, speculative=True)
    assert g.decode_graphs == 24  # verify, draft decode, draft extend
    assert g.prefill_graphs == 100  # target and draft prefill
    g = memory.Graphs(decode_bs=(1, 2, 4), prefill_shapes=0, speculative=False)
    assert (g.decode_graphs, g.prefill_graphs) == (3, 0)


@pytest.mark.parametrize(
    "fraction, bs, prefill, fits",
    [
        # measured on B70 (rank 0 free memory at steady state): 8 MTP graph
        # sizes at 0.85 left 0 GiB, at 0.90 ran out; 0.80 keeps the reserve
        (0.80, 8, 0, True),
        (0.83, 8, 0, False),
        (0.85, 8, 0, False),
        (0.85, 8, 50, False),
        (0.90, 8, 50, False),
        # the default 23 decode sizes up to 48: 0.82 left about 1.3 GiB
        (0.82, 23, 0, False),
        (0.77, 23, 0, True),
    ],
)
def test_budget_mtp(fraction, bs, prefill, fits):
    g = memory.Graphs(
        decode_bs=tuple(range(bs)), prefill_shapes=prefill, speculative=True
    )
    b = memory.budget(CARD, fraction, g, reserve=1.5)
    assert b.fits is fits
    assert b.max_fraction <= (b.total_gib - b.outside_torch_gib - 1.5) / b.total_gib
    assert memory.budget(CARD, b.max_fraction, g, reserve=1.5).fits


def test_budget_without_mtp():
    # 12 decode sizes at 0.85 left 1.52 GiB free on B70: at the reserve
    g = memory.Graphs(decode_bs=tuple(range(12)), prefill_shapes=0, speculative=False)
    assert memory.budget(CARD, 0.84, g, reserve=1.5).fits
    assert not memory.budget(CARD, 0.86, g, reserve=1.5).fits


def test_cap_leaves_outside_torch_and_reserve():
    g = memory.Graphs(decode_bs=tuple(range(8)), prefill_shapes=0, speculative=True)
    b = memory.budget(CARD, 0.80, g, reserve=1.5)
    cap = b.torch_cap_fraction * b.total_gib
    assert cap == pytest.approx(
        b.total_gib - b.outside_torch_gib - memory.CAP_MARGIN_GIB
    )
    assert b.fits and b.torch_gib + b.reserve_gib <= cap + memory.CAP_MARGIN_GIB


def test_refusal_names_the_remedy():
    g = memory.Graphs(decode_bs=tuple(range(23)), prefill_shapes=50, speculative=True)
    b = memory.budget(CARD, 0.85, g, reserve=1.5)
    text = memory.refusal(b, g, 48)
    assert f"--mem-fraction-static {b.max_fraction:.2f}" in text
    assert "--cuda-graph-max-bs-decode 8" in text and "--max-running-requests 8" in text
    assert (
        "--cuda-graph-backend-prefill disabled" in text and memory.RESERVE_ENV in text
    )
    small = memory.Graphs(decode_bs=tuple(range(8)), prefill_shapes=0, speculative=True)
    assert "fewer graphs" not in memory.refusal(
        memory.budget(CARD, 0.9, small, 1.5), small, 8
    )


def test_reserve_env(monkeypatch):
    monkeypatch.delenv(memory.RESERVE_ENV, raising=False)
    assert memory.reserve_gib() == memory.DEFAULT_RESERVE_GIB == 1.5
    monkeypatch.setenv(memory.RESERVE_ENV, "3")
    assert memory.reserve_gib() == 3.0
    monkeypatch.setenv(memory.RESERVE_ENV, "-1")
    with pytest.raises(ValueError):
        memory.reserve_gib()


# ----------------------------------------------------------------------------
# deadline and fail-stop


def test_deadline(monkeypatch):
    monkeypatch.delenv(failure.DEADLINE_ENV, raising=False)
    assert failure.deadline_s() == 2.0
    assert failure.deadline_s(60.0) == 2.0  # not tied to the watchdog
    assert failure.deadline_s(1.5) == 1.5  # but never past SGLang's limit
    monkeypatch.setenv(failure.DEADLINE_ENV, "20")
    assert failure.deadline_s(60.0) == 20.0
    monkeypatch.setenv(failure.DEADLINE_ENV, "0")
    with pytest.raises(ValueError):
        failure.deadline_s()


# health word 6: the low 32 bits of the mailbox word's address (int32: may be
# negative)
WORD_AT = 0x9ABCDEF0 - (1 << 32)


def _words(failed=0, peer=0, tag=7, layer=0x1234):
    w = torch.zeros(16, dtype=torch.int32)
    w[0], w[1], w[2], w[3], w[6], w[7] = failed, failed, peer, tag, WORD_AT, layer
    return w


def test_describe():
    text = failure.describe(
        _words(1).tolist(), 1, 5.0, lambda a: "layer 3" if a == 0x1234 else None
    )
    assert "rank 1" in text and "layer 3" in text and "5 s" in text and "tag 7" in text
    assert "mailbox word at 0x9abcdef0" in text
    assert "--mem-fraction-static" in text and "NaN" in text
    peer = failure.describe(_words(1, peer=1).tolist(), 0, 5.0, lambda a: None)
    assert "peer rank gave up" in peer


def _watch(words, stopped):
    # words: the host copy of the health words (a workspace's health_host)
    return failure.FailureWatch(
        words, 0, 5.0, lambda a: None, on_failure=stopped.set, poll_s=0.01
    )


def test_watch_quiet_while_healthy():
    stopped = threading.Event()
    w = _watch(_words(0), stopped)
    w.check()
    assert not stopped.wait(0.1)
    w.close()


def test_watch_stops_the_server():
    words, stopped = _words(0), threading.Event()
    w = _watch(words, stopped)
    words.copy_(_words(1))  # a kernel marks the device failed
    assert stopped.wait(2.0)
    with pytest.raises(RuntimeError, match="gave up"):
        w.check()
    w.close()


# ----------------------------------------------------------------------------
# spec checks


def _spec():
    return pytest.importorskip("sgl_kernel.monokernel.qwen3_5.spec")


def _cfg(**kw):
    base = dict(
        context_length=None,
        speculative_algorithm=None,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
        dtype="auto",
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _hf():
    spec = _spec()
    tc = types.SimpleNamespace(
        **spec.TEXT_CONFIG,
        num_hidden_layers=40,
        rope_parameters={"rope_theta": spec.ROPE_THETA},
    )
    return types.SimpleNamespace(
        text_config=tc,
        quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
    )


def test_spec_accepts_the_model(monkeypatch):
    spec = _spec()
    monkeypatch.setattr(spec, "_conv_dtype", lambda: None)
    assert spec.config_disable_reasons(_cfg(), _hf()) == []
    cfg = _cfg(
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
    )
    assert spec.config_disable_reasons(cfg, _hf()) == []
    small = _hf()
    small.text_config.num_hidden_layers = 4  # the load test's 4-layer model
    assert spec.config_disable_reasons(_cfg(), small) == []


def test_spec_reasons(monkeypatch):
    spec = _spec()
    monkeypatch.setattr(spec, "_conv_dtype", lambda: None)
    hf = _hf()
    hf.text_config.hidden_size = 4096
    hf.quantization_config = {"quant_method": "fp8", "weight_block_size": [1, 128]}
    cfg = _cfg(
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=2,
        speculative_num_draft_tokens=5,
    )
    reasons = " | ".join(spec.config_disable_reasons(cfg, hf))
    assert "hidden_size" in reasons and "block-FP8" in reasons
    assert "topk above 1" in reasons and "speculative_num_draft_tokens 5" in reasons
    monkeypatch.setattr(spec, "_conv_dtype", lambda: "float16")
    assert any(
        "SGLANG_MAMBA_CONV_DTYPE" in r
        for r in spec.config_disable_reasons(_cfg(dtype="bfloat16"), _hf())
    )


@pytest.mark.parametrize("field", ["rope_parameters", "rope_scaling"])
@pytest.mark.parametrize(
    "scaling",
    [
        {"rope_type": "linear", "factor": 2.0},
        {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 262144,
        },
        {"type": "dynamic", "factor": 2.0},
    ],
)
def test_spec_rejects_scaled_rope(monkeypatch, field, scaling):
    # the kernels compute the plain frequencies of theta; SGLang's attention
    # (prefill) would apply the scaling
    spec = _spec()
    monkeypatch.setattr(spec, "_conv_dtype", lambda: None)
    hf = _hf()
    setattr(hf.text_config, field, {"rope_theta": spec.ROPE_THETA, **scaling})
    reasons = spec.config_disable_reasons(_cfg(), hf)
    assert any("rope_type" in r for r in reasons), reasons


def test_spec_accepts_the_checkpoints_default_rope(monkeypatch):
    spec = _spec()
    monkeypatch.setattr(spec, "_conv_dtype", lambda: None)
    hf = _hf()
    hf.text_config.rope_parameters = {  # Qwen3.6-35B-A3B's config.json
        "rope_type": "default",
        "rope_theta": spec.ROPE_THETA,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
    }
    assert spec.config_disable_reasons(_cfg(), hf) == []
    hf.text_config.rope_scaling = None  # a dict without a type is unscaled too
    hf.text_config.rope_parameters = {"rope_theta": spec.ROPE_THETA}
    assert spec.config_disable_reasons(_cfg(), hf) == []


@pytest.mark.parametrize(
    "dtype, kv, ok",
    [
        ("bfloat16", "auto", True),
        ("bfloat16", "bfloat16", True),
        ("bfloat16", "float16", False),
        ("float16", "bfloat16", False),
        ("float16", "float16", True),
        ("auto", "float16", False),  # auto: the checkpoint's bf16
    ],
)
def test_spec_kv_dtype_must_match_the_model(monkeypatch, dtype, kv, ok):
    spec = _spec()
    monkeypatch.setattr(
        spec, "_conv_dtype", lambda: "float16" if dtype == "float16" else None
    )
    reasons = spec.config_disable_reasons(_cfg(dtype=dtype, kv_cache_dtype=kv), _hf())
    assert (not any("kv_cache_dtype" in r for r in reasons)) is ok, reasons


def test_spec_refuses_fp16(monkeypatch):
    # SGLang's block-FP8 MoE (prefill) runs bf16 only: an fp16 server cannot
    # serve, for --dtype float16 and for auto on an fp16 checkpoint
    spec = _spec()
    monkeypatch.setattr(spec, "_conv_dtype", lambda: "float16")
    hf = _hf()
    for cfg, h in ((_cfg(dtype="float16"), _hf()), (_cfg(dtype="auto"), hf)):
        hf.torch_dtype = "float16"
        reasons = spec.config_disable_reasons(cfg, h)
        assert any("bfloat16 only" in r for r in reasons), reasons
    assert "float16" not in spec.SPEC.requires.dtypes
    monkeypatch.setattr(spec, "_conv_dtype", lambda: None)
    assert spec.config_disable_reasons(_cfg(dtype="auto"), _hf()) == []


def test_spec_layer_count_has_an_upper_bound(monkeypatch):
    # one kernel (layer table and counters row) per layer
    spec = _spec()
    monkeypatch.setattr(spec, "_conv_dtype", lambda: None)
    for layers, ok in (
        (1, True),
        (spec.MAX_LAYERS, True),
        (spec.MAX_LAYERS + 1, False),
    ):
        hf = _hf()
        hf.text_config.num_hidden_layers = layers
        got = spec.config_disable_reasons(_cfg(), hf)
        assert (got == []) is ok, (layers, got)


# ----------------------------------------------------------------------------
# runtime arguments and the peer handshake


def test_forward_refuses_arguments_it_does_not_serve():
    runtime = pytest.importorskip("sgl_kernel.monokernel.qwen3_5.runtime")
    runtime._check_call(False, None, {})
    for args in ((True, None, {}), (False, object(), {}), (False, None, {"x": 1})):
        with pytest.raises(RuntimeError, match="does not serve"):
            runtime._check_call(*args)
    with pytest.raises(RuntimeError, match="no kernel"):  # not an assert
        runtime._require(False, "no kernel")


def test_require_ptrace_refuses_without_ptrace(monkeypatch):
    """Without ptrace permission every rank fails before the symmetric-memory
    rendezvous of the workspace (which would otherwise fail inside the driver),
    with the fix in the message."""
    import torch.distributed as dist

    peer = pytest.importorskip("sgl_kernel.monokernel.peer")
    monkeypatch.setattr(peer, "ptrace_problem", lambda: "no CAP_SYS_PTRACE")
    dist.init_process_group(
        "gloo", store=dist.HashStore(), rank=0, world_size=1, group_name="ptrace_test"
    )
    try:
        group = types.SimpleNamespace(
            cpu_group=dist.group.WORLD, device_group=dist.group.WORLD
        )
        with pytest.raises(RuntimeError, match="ptrace"):
            peer.require_ptrace(group)
        monkeypatch.setattr(peer, "ptrace_problem", lambda: "")
        peer.require_ptrace(group)  # every rank can: no error
    finally:
        dist.destroy_process_group()


def test_failure_watch_reports_when_describing_fails():
    # the watch is the only fail-stop of a replayed graph: an error while
    # describing the failure must not kill it silently
    words = torch.zeros(16, dtype=torch.int32)
    words[0] = 1
    stopped = []
    w = failure.FailureWatch(
        words,
        0,
        2.0,
        lambda a: 1 / 0,
        on_failure=lambda: stopped.append(1),
        poll_s=0.01,
    )
    for _ in range(200):
        if stopped:
            break
        time.sleep(0.01)
    w.close()
    assert stopped and "describing it failed" in w.message


def test_counter_wait_failure_is_described():
    words = [1, 1, 0, 64, 32, 0, 1, 0x1234, 0, 0, 1, 0, 0, 0, 0, 0]
    m = failure.describe(words, 1, 2.0, lambda a: "layer 3")
    assert (
        "other work-groups of this card in layer 3" in m and "counter B: 32 of 64" in m
    )
    words[10] = 0  # a mailbox wait
    assert "a wait for the peer card in layer 3" in failure.describe(
        words, 1, 2.0, lambda a: "layer 3"
    )


def test_bad_environment_values_name_the_variable(monkeypatch):
    monkeypatch.setenv(failure.DEADLINE_ENV, "2s")
    with pytest.raises(ValueError, match=failure.DEADLINE_ENV):
        failure.deadline_s()
    monkeypatch.setenv(memory.RESERVE_ENV, "one")
    with pytest.raises(ValueError, match=memory.RESERVE_ENV):
        memory.reserve_gib()
    monkeypatch.setenv(failure.FAULT_ENV, "0-60")
    with pytest.raises(ValueError, match=failure.FAULT_ENV):
        failure.schedule_fault(0, 0, None)
