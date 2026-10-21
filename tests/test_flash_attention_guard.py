import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "benchmark/bench_muse_fmha.py"


class TestServingGuard_xe35(unittest.TestCase):
    def benchmark_module_xe35(self):
        spec = importlib.util.spec_from_file_location("bench_xe35", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_architecture_guard_xe35(self):
        module = self.benchmark_module_xe35()
        for available, architecture in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(available=available, architecture=architecture):
                torch_module = SimpleNamespace(xpu=SimpleNamespace(is_available=lambda: available))
                architecture_check = Mock(return_value=architecture)
                self.assertEqual(module.is_fmha_arch_xe35(torch_module, architecture_check), available and architecture)
                self.assertEqual(architecture_check.call_count, int(available))
        architecture_check = Mock(return_value=True)
        self.assertFalse(module.is_fmha_arch_xe35(SimpleNamespace(), architecture_check))
        architecture_check.assert_not_called()

    def test_reject_before_torch_xe35(self):
        for flag, value in [("--serving-batch", "7"), ("--isl", "2048"), ("--osl", "8"), ("--tp", "2"), ("--ep", "2")]:
            with self.subTest(flag=flag):
                result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--guard-only", flag, value], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("rejected before GPU initialization", result.stderr)
                self.assertNotIn("torch", result.stderr)

    def test_environment_overrides_xe35(self):
        for key in ("BATCHES", "SERVING_BATCH", "BATCH_SIZE", "ISL", "OSL", "TP", "EP"):
            with self.subTest(key=key):
                environment = dict(os.environ)
                environment[key] = "7"
                result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--guard-only"], env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("rejected before GPU initialization", result.stderr)
                self.assertNotIn("torch", result.stderr)

    def test_accept_fixed_workload_xe35(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--guard-only"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)

    def test_reject_unleased_gpu_before_torch_xe35(self):
        for device in ("0", "1", "2", "3", "4", "6"):
            with self.subTest(device=device):
                environment = dict(os.environ)
                environment["ZE_AFFINITY_MASK"] = device
                result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--output", "unused.json"],
                                        env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn("--gpu-lock --gpu 5 before GPU initialization", result.stderr)
                self.assertNotIn("torch", result.stderr)


    def test_production_cases_xe35(self):
        module = self.benchmark_module_xe35()
        cases = list(module.production_cases_xe35())
        self.assertEqual(len(cases), 38)
        self.assertEqual(sum(case.query_length == 1 for case in cases), 32)
        self.assertEqual({case.requests for case in cases if case.query_length > 1}, {1, 7, 8})
        self.assertTrue(all(case.requests == 8 for case in cases if case.query_length == 1))


if __name__ == "__main__":
    unittest.main()
