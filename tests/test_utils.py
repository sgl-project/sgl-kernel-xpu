"""Common utilities for testing and benchmarking"""

import os
import signal
import subprocess
import threading
import time
from typing import Callable, List, Optional


class TestFile:
    name: str
    estimated_time: float = 60


# Retry a signal-killed (likely GPU-faulted) test file this many times.
MAX_INFRA_RETRIES = int(os.environ.get("SGL_KERNEL_INFRA_RETRIES", "1"))

# Wait for the GPU engine reset / L0 teardown to settle before retrying.
_XPU_RECOVER_WAIT = float(os.environ.get("SGL_KERNEL_XPU_RECOVER_WAIT", "20"))


def _recover_xpu() -> None:
    """Wait for the XPU to reset; log health via xpu-smi if available."""
    time.sleep(_XPU_RECOVER_WAIT)
    try:
        subprocess.run(
            ["xpu-smi", "health", "-d", "0"],
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _terminate_process_tree(process: Optional[subprocess.Popen]) -> None:
    # SIGKILL the whole PGID: the child owns an L0 context that wedges
    # the XPU for later CI jobs if left alive.
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = None,
):
    """Run a function with timeout."""
    ret_value = []

    def _target_func():
        ret_value.append(func(*args, **(kwargs or {})))

    t = threading.Thread(target=_target_func)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError()

    if not ret_value:
        raise RuntimeError()

    return ret_value[0]


def run_unittest_files(files: List[TestFile], timeout_per_file: float):
    tic = time.perf_counter()
    success = True

    for i, file in enumerate(files):
        filename, estimated_time = file.name, file.estimated_time
        process = None

        def run_one_file(filename):
            nonlocal process

            filename = os.path.join(os.getcwd(), filename)
            print(
                f".\n.\nBegin ({i}/{len(files) - 1}):\npython3 {filename}\n.\n.\n",
                flush=True,
            )
            tic = time.perf_counter()

            # -x stops at first failure so an XPU-wedging case does not
            # cascade into thousands of downstream failures / a saferepr SIGSEGV.
            process = subprocess.Popen(
                ["python3", "-m", "pytest", "-x", "--tb=short", filename],
                stdout=None,
                stderr=None,
                env=os.environ,
                start_new_session=True,
            )
            process.wait()
            elapsed = time.perf_counter() - tic

            print(
                f".\n.\nEnd ({i}/{len(files) - 1}):\n{filename=}, {elapsed=:.0f}, {estimated_time=}\n.\n.\n",
                flush=True,
            )
            return process.returncode

        # Negative return code = killed by signal (GPU fault); retry.
        # Positive return code = real test failure; do not retry.
        crashed = False
        for attempt in range(MAX_INFRA_RETRIES + 1):
            try:
                ret_code = run_with_timeout(
                    run_one_file, args=(filename,), timeout=timeout_per_file
                )
            except TimeoutError:
                _terminate_process_tree(process)
                time.sleep(5)
                print(
                    f"\nTimeout after {timeout_per_file} seconds when running {filename}\n",
                    flush=True,
                )
                crashed = True
                break

            if ret_code >= 0:
                break

            if attempt < MAX_INFRA_RETRIES:
                print(
                    f"\n{filename} was killed by signal {-ret_code} "
                    f"(likely a GPU/driver fault, not a test failure). "
                    f"Recovering device and retrying "
                    f"(attempt {attempt + 1}/{MAX_INFRA_RETRIES}).\n",
                    flush=True,
                )
                _recover_xpu()
            else:
                print(
                    f"\n{filename} still crashing with signal {-ret_code} after "
                    f"{MAX_INFRA_RETRIES} retry(ies); treating as failure.\n",
                    flush=True,
                )

        if crashed:
            success = False
            break

        try:
            assert (
                ret_code == 0
            ), f"expected return code 0, but {filename} returned {ret_code}"
        except AssertionError as e:
            print(f"\n{e}\n", flush=True)
            success = False
            break

    if success:
        print(f"Success. Time elapsed: {time.perf_counter() - tic:.2f}s", flush=True)
    else:
        print(f"Fail. Time elapsed: {time.perf_counter() - tic:.2f}s", flush=True)

    return 0 if success else -1
