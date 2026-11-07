"""Fail-stop of the mono decode kernels: the in-kernel waits for the peer card
give up after a deadline (``SGLANG_XPU_MONO_DEADLINE_S``, default 5 s), the failed
step's outputs are NaN, and the wait that gives up writes a host copy of the
device's health words. A thread polls that copy and stops the server with an
error that names the likely cause, so a stalled peer never hangs it."""

import logging
import os
import signal
import threading
import time
from typing import Callable, Optional

import torch
from sgl_kernel import mono_decode

logger = logging.getLogger(__name__)

DEADLINE_ENV = "SGLANG_XPU_MONO_DEADLINE_S"
DEFAULT_DEADLINE_S = 5.0
POLL_S = 0.05
# test hook, "RANK:SECONDS": that many seconds after the build, rank RANK
# shifts the first layer's mailbox tags, so both ranks' next waits there fail
FAULT_ENV = "SGLANG_XPU_MONO_FAULT_INJECT"


def deadline_s(limit: Optional[float] = None) -> float:
    """The wait deadline: ``SGLANG_XPU_MONO_DEADLINE_S`` (default 5 s), at most
    ``limit`` (SGLang's: half the watchdog timeout)."""
    value = os.environ.get(DEADLINE_ENV)
    seconds = float(value) if value else DEFAULT_DEADLINE_S
    if not 0.0 < seconds <= 600.0:
        raise ValueError(f"{DEADLINE_ENV}={value}: the deadline must be in (0, 600] s")
    return min(seconds, limit) if limit else seconds


def describe(
    words: list, rank: int, deadline: float, layer_of: Callable[[int], Optional[str]]
) -> str:
    """The error of a failed device from its health words."""
    if words[2]:
        what = "the peer rank gave up a wait first"
    else:
        layer = layer_of(words[7]) or "an unknown layer"
        what = (
            f"a wait for the peer card in {layer} gave up after {deadline:g} s "
            f"(tag {words[3]}, mailbox word {words[6]})"
        )
    return (
        f"XPU monokernel: rank {rank}: {what}. The outputs of the failed step are NaN and the "
        "server stops. The usual cause is GPU memory pressure: the driver evicted buffers "
        "(e.g. the peer card's mappings) and preempted the resident layer kernels. Lower "
        "--mem-fraction-static, set --max-running-requests and --cuda-graph-max-bs-decode to "
        f"the concurrency you serve (the monokernels cover up to 8 rows), or raise "
        f"{DEADLINE_ENV} if the peer is only slow."
    )


def stop_server() -> None:
    """SGLang's fail-stop: the scheduler's parent tears the server down on
    SIGQUIT (as for a scheduler exception)."""
    os.kill(os.getppid(), signal.SIGQUIT)


class FailureWatch:
    """Polls the host copy of a device's health words; ``check`` raises in the
    caller's thread, and the polling thread stops the server, so a failure in a
    replayed graph (no Python runs) stops it too."""

    def __init__(
        self,
        device: int,
        rank: int,
        deadline: float,
        layer_of: Callable[[int], Optional[str]],
        on_failure: Callable[[], None] = stop_server,
        poll_s: float = POLL_S,
    ):
        self.words = mono_decode.health_host(device)
        self.rank, self.deadline, self.layer_of = rank, deadline, layer_of
        self.on_failure, self.poll_s = on_failure, poll_s
        self.message: Optional[str] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="xpu-monokernel-watch"
        )
        self._thread.start()

    def failed(self) -> bool:
        return int(self.words[0]) != 0

    def _describe(self) -> str:
        if self.message is None:
            self.message = describe(
                self.words.tolist(), self.rank, self.deadline, self.layer_of
            )
        return self.message

    def check(self) -> None:
        if self.failed():
            raise RuntimeError(self._describe())

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            if self.failed():
                logger.error(self._describe())
                self.on_failure()
                return

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def schedule_fault(rank: int, device: int, layer: int) -> None:
    """Arms the ``SGLANG_XPU_MONO_FAULT_INJECT`` test hook on this rank, if set."""
    value = os.environ.get(FAULT_ENV)
    if not value:
        return
    target, after = value.split(":")
    if int(target) != rank:
        return

    def inject():
        time.sleep(float(after))
        torch.xpu.set_device(device)
        torch.ops.sgl_kernel.mono_fault_inject(layer, 7)
        logger.warning(
            "XPU monokernel: rank %d: %s: shifted the mailbox tags of kernel %d",
            rank,
            FAULT_ENV,
            layer,
        )

    threading.Thread(target=inject, daemon=True, name="xpu-monokernel-fault").start()
