"""Host-side logic of the SGLang monokernel plugin (no GPU): the memory budget
that refuses a --mem-fraction-static without physical headroom, the wait
deadline, the fail-stop watch that turns a failed in-kernel wait into a server
stop, and the spec's configuration checks."""

import threading
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
    assert failure.deadline_s() == 5.0
    assert failure.deadline_s(60.0) == 5.0  # not tied to the watchdog
    assert failure.deadline_s(3.0) == 3.0  # but never past SGLang's limit
    monkeypatch.setenv(failure.DEADLINE_ENV, "20")
    assert failure.deadline_s(60.0) == 20.0
    monkeypatch.setenv(failure.DEADLINE_ENV, "0")
    with pytest.raises(ValueError):
        failure.deadline_s()


def _words(failed=0, peer=0, tag=7, layer=0x1234):
    w = torch.zeros(16, dtype=torch.int32)
    w[0], w[1], w[2], w[3], w[6], w[7] = failed, failed, peer, tag, 42, layer
    return w


def test_describe():
    text = failure.describe(
        _words(1).tolist(), 1, 5.0, lambda a: "layer 3" if a == 0x1234 else None
    )
    assert "rank 1" in text and "layer 3" in text and "5 s" in text and "tag 7" in text
    assert "--mem-fraction-static" in text and "NaN" in text
    peer = failure.describe(_words(1, peer=1).tolist(), 0, 5.0, lambda a: None)
    assert "peer rank gave up" in peer


def _watch(monkeypatch, words, stopped):
    monkeypatch.setattr(failure.mono_decode, "health_host", lambda device: words)
    return failure.FailureWatch(
        0, 0, 5.0, lambda a: None, on_failure=stopped.set, poll_s=0.01
    )


def test_watch_quiet_while_healthy(monkeypatch):
    stopped = threading.Event()
    w = _watch(monkeypatch, _words(0), stopped)
    w.check()
    assert not stopped.wait(0.1)
    w.close()


def test_watch_stops_the_server(monkeypatch):
    words, stopped = _words(0), threading.Event()
    w = _watch(monkeypatch, words, stopped)
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
    monkeypatch.setattr(spec, "_conv_dtype", lambda: "bfloat16")
    assert any(
        "SGLANG_MAMBA_CONV_DTYPE" in r
        for r in spec.config_disable_reasons(_cfg(dtype="float16"), _hf())
    )
