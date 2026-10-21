import importlib.util
import tempfile
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

source = Path(__file__).resolve().parents[1] / "benchmark/bench_bf16_gemm_nt.py"
specification = importlib.util.spec_from_file_location("bf16_benchmark", source)
benchmark = importlib.util.module_from_spec(specification)
specification.loader.exec_module(benchmark)


class TestBf16GemmBenchmark(unittest.TestCase):
    def test_cpu_reference_bounds_transfers_and_includes_tail_columns(self):
        transfers = []
        writes = []
        allocations = []

        class Matrix:
            def __init__(self, rows, columns, weight=False):
                self.shape = (rows, columns)
                self.weight = weight

            def element_size(self):
                return 2

            def cpu(self):
                if self.weight:
                    transfers.append(self.shape)
                return self

            def float(self):
                return self

            @property
            def T(self):
                return SimpleNamespace(shape=tuple(reversed(self.shape)))

            def __getitem__(self, rows):
                return Matrix(rows.stop - rows.start, self.shape[1], self.weight)

        class Reference:
            def __setitem__(self, index, value):
                writes.append((index[1].start, index[1].stop, value))

        reference = Reference()

        def allocate(shape, **kwargs):
            allocations.append((shape, kwargs))
            return reference

        fake_torch = SimpleNamespace(
            empty=allocate,
            float32="float32",
            mm=lambda activation, weight: (activation.shape[0], weight.shape[1]),
        )
        with patch.object(benchmark, "torch", fake_torch, create=True):
            actual = benchmark.cpu_reference_xe35(
                Matrix(3, 16), Matrix(5, 16, weight=True), chunk_bytes=64
            )
        self.assertIs(actual, reference)
        self.assertEqual(transfers, [(2, 16), (2, 16), (1, 16)])
        self.assertEqual(writes, [(0, 2, (3, 2)), (2, 4, (3, 2)), (4, 5, (3, 1))])
        self.assertEqual(allocations, [((3, 5), {"dtype": "float32", "device": "cpu"})])

    def test_cpu_reference_rejects_nonpositive_chunk_budget(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            benchmark.cpu_reference_xe35(None, None, chunk_bytes=0)

    def test_case_graphs_reset_and_release_cache_after_replay(self):
        cleanup = []
        graphs = [
            SimpleNamespace(reset=lambda: cleanup.append("first")),
            SimpleNamespace(reset=lambda: cleanup.append("second")),
        ]
        fake_torch = SimpleNamespace(
            xpu=SimpleNamespace(
                synchronize=lambda: cleanup.append("synchronize"),
                empty_cache=lambda: cleanup.append("empty_cache"),
            )
        )
        functions = [{"first": lambda: None, "second": lambda: None}]
        with patch.object(benchmark, "torch", fake_torch, create=True):
            with patch.object(benchmark, "capture_xe35", side_effect=graphs):
                with benchmark.case_graphs_xe35(functions, ("first", "second"), 1, 2):
                    cleanup.append("replay")
        self.assertEqual(
            cleanup, ["replay", "synchronize", "first", "second", "empty_cache"]
        )

    def test_partial_capture_failure_releases_previous_graph(self):
        cleanup = []
        graph = SimpleNamespace(reset=lambda: cleanup.append("reset"))
        fake_torch = SimpleNamespace(
            xpu=SimpleNamespace(
                synchronize=lambda: None,
                empty_cache=lambda: cleanup.append("empty_cache"),
            )
        )
        functions = [{"first": lambda: None, "second": lambda: None}]
        with patch.object(benchmark, "torch", fake_torch, create=True):
            with patch.object(
                benchmark, "capture_xe35", side_effect=[graph, RuntimeError("capture")]
            ):
                with self.assertRaisesRegex(RuntimeError, "capture"):
                    with benchmark.case_graphs_xe35(
                        functions, ("first", "second"), 1, 2
                    ):
                        self.fail("failed capture must not yield")
        self.assertEqual(cleanup, ["reset", "empty_cache"])

    def test_capture_replay_failure_resets_new_graph(self):
        cleanup = []

        def fail_replay():
            raise RuntimeError("replay")

        graph = SimpleNamespace(
            replay=fail_replay, reset=lambda: cleanup.append("reset")
        )
        fake_torch = SimpleNamespace(
            xpu=SimpleNamespace(
                synchronize=lambda: None,
                XPUGraph=lambda: graph,
                graph=lambda value: nullcontext(),
            )
        )
        with patch.object(benchmark, "torch", fake_torch, create=True):
            with self.assertRaisesRegex(RuntimeError, "replay"):
                benchmark.capture_xe35([lambda: None], 1, 2)
        self.assertEqual(cleanup, ["reset"])

    def test_shared_kernel_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.so"
            baseline = Path(directory) / "baseline.so"
            candidate.write_bytes(b"_ZTS17GemmUniversalCandidate\x00")
            baseline.write_bytes(candidate.read_bytes())
            with self.assertRaisesRegex(ValueError, "share 1 SYCL GEMM kernel IDs"):
                benchmark.validate_kernel_isolation_xe35(candidate, baseline)

    def test_missing_kernel_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "empty.so"
            library.write_bytes(b"no device image")
            with self.assertRaisesRegex(ValueError, "No SYCL GEMM"):
                benchmark.gemm_kernel_ids_xe35(library)

    def test_distinct_but_unloaded_libraries_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.so"
            baseline = Path(directory) / "baseline.so"
            candidate.write_bytes(b"_ZTS17GemmUniversalCandidate\x00")
            baseline.write_bytes(b"_ZTS17GemmUniversalBaseline\x00")
            with self.assertRaisesRegex(ValueError, "not loaded"):
                benchmark.validate_kernel_isolation_xe35(candidate, baseline)

    def test_loaded_older_package_kernel_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.so"
            baseline = Path(directory) / "baseline.so"
            installed = Path(directory) / "libsgl-ops-sycl-bf16_gemm_nt.so"
            candidate.write_bytes(b"_ZTS17GemmUniversalCandidate\x00")
            baseline.write_bytes(b"_ZTS17GemmUniversalBaseline\x00")
            installed.write_bytes(candidate.read_bytes())
            maps = "\n".join(
                f"0000-1000 r-xp 0 00:00 1 {library}"
                for library in (candidate, baseline, installed)
            )
            with patch.object(Path, "read_text", return_value=maps):
                with self.assertRaisesRegex(ValueError, "shares candidate kernel IDs"):
                    benchmark.validate_kernel_isolation_xe35(candidate, baseline)

    def test_distinct_loaded_kernel_ids_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.so"
            baseline = Path(directory) / "baseline.so"
            candidate.write_bytes(b"_ZTS17GemmUniversalCandidate\x00")
            baseline.write_bytes(b"_ZTS17GemmUniversalBaseline\x00")
            maps = "\n".join(
                f"0000-1000 r-xp 0 00:00 1 {library}"
                for library in (candidate, baseline)
            )
            with patch.object(Path, "read_text", return_value=maps):
                with patch("builtins.print") as print_result:
                    benchmark.validate_kernel_isolation_xe35(candidate, baseline)
            self.assertIn("candidate=1 baseline=1", print_result.call_args.args[0])

    def test_shared_storage_releases_original_baseline_weight(self):
        class PackedWeight:
            shape = (96, 256)

            def stride(self):
                return (256, 1)

        candidate = PackedWeight()
        original = PackedWeight()
        reference = weakref.ref(original)
        baseline = [original]
        del original
        with patch.object(
            benchmark, "torch", SimpleNamespace(equal=lambda *args: True), create=True
        ):
            result = benchmark.share_matching_packed_weights_xe35([candidate], baseline)
        self.assertIs(result[0], candidate)
        self.assertIsNone(reference())

    def test_cold_ring_requires_enough_calls(self):
        arguments = [
            "benchmark",
            "--cache-mode",
            "cold",
            "--shape",
            "4608",
            "8192",
            "--calls",
            "7",
        ]
        with patch("sys.argv", arguments), patch.dict("os.environ", {"BATCHES": "8"}):
            with self.assertRaises(SystemExit):
                benchmark.parse_args_xe35()
        with (
            patch("sys.argv", arguments[:-1] + ["8"]),
            patch.dict("os.environ", {"BATCHES": "8"}),
        ):
            self.assertEqual(benchmark.parse_args_xe35().calls, 8)

    def test_rejects_small_working_set(self):
        with patch("sys.argv", ["benchmark", "--working-set-mib", "511"]):
            with self.assertRaises(SystemExit):
                benchmark.parse_args_xe35()

    def test_paired_comparison_requires_baseline(self):
        with patch("sys.argv", ["benchmark", "--compare-baseline"]):
            with self.assertRaises(SystemExit):
                benchmark.parse_args_xe35()

    def test_paired_functions_use_the_matching_weight_and_operator(self):
        activations = SimpleNamespace(shape=(8, 96), dtype="bf16", device="xpu")
        weight = SimpleNamespace(shape=(257, 96), T="transposed")
        packed = SimpleNamespace(shape=(96, 512))
        baseline_packed = SimpleNamespace(shape=(96, 257))
        fake_torch = SimpleNamespace(empty=lambda shape, **kwargs: shape)
        dispatch_calls = []
        kernel_calls = []
        candidate_dispatch = lambda *args, **kwargs: dispatch_calls.append(
            ("candidate", args)
        )
        baseline_dispatch = lambda *args, **kwargs: dispatch_calls.append(
            ("baseline", args)
        )
        candidate_operator = lambda *args: kernel_calls.append(("candidate", args))
        baseline_operator = lambda *args: kernel_calls.append(("baseline", args))
        with patch.multiple(
            benchmark,
            create=True,
            torch=fake_torch,
            bf16_gemm_nt_xe35=candidate_dispatch,
            baseline_dispatch=baseline_dispatch,
            gemm_operator=candidate_operator,
            baseline_operator=baseline_operator,
        ):
            functions = benchmark.case_functions_xe35(
                activations, weight, packed, baseline_packed
            )
            functions["packed_dispatch"]()
            functions["baseline_packed_dispatch"]()
            self.assertEqual(functions["packed_cute"](), (8, 512))
            self.assertEqual(functions["baseline_packed_cute"](), (8, 257))
            functions["native_cute"]()
            functions["baseline_native_cute"]()
        self.assertIs(dispatch_calls[0][1][1], packed)
        self.assertIs(dispatch_calls[1][1][1], baseline_packed)
        self.assertEqual(
            [entry[0] for entry in kernel_calls],
            ["candidate", "baseline", "candidate", "baseline"],
        )
        self.assertIsNot(kernel_calls[-2][1][0], kernel_calls[-1][1][0])

    def test_capture_rotates_and_warms_all_weights(self):
        invocations = []
        functions = [
            lambda index=index: invocations.append(index) for index in range(3)
        ]
        replays = []
        graph = SimpleNamespace(replay=lambda: replays.append(True))
        fake_torch = SimpleNamespace(
            xpu=SimpleNamespace(
                synchronize=lambda: None,
                XPUGraph=lambda: graph,
                graph=lambda value: nullcontext(),
            )
        )
        with patch.object(benchmark, "torch", fake_torch, create=True):
            self.assertIs(benchmark.capture_xe35(functions, 1, 5), graph)
        self.assertEqual(invocations, [0, 1, 2, 0, 1, 2, 0, 1])
        self.assertEqual(len(replays), 1)


if __name__ == "__main__":
    unittest.main()
