import re
import shutil
import subprocess
import unittest
from pathlib import Path


class TestDeviceArchitectureGuard(unittest.TestCase):
    def test_compiler_release_guard(self):
        source = Path(__file__).resolve().parents[1] / "src/sycl/Device.cpp"
        content = source.read_text()
        match = re.search(
            r"(?m)^#if[^\n]*__INTEL_LLVM_COMPILER[^\n]*(?:\\\n[^\n]*)*\n(?=\s*case syclex::architecture::intel_gpu_cri:)",
            content,
        )
        self.assertIsNotNone(match)
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler)
        program = match.group() + "CRI_ENABLED\n#else\nCRI_DISABLED\n#endif\n"
        cases = (
            ({}, False),
            ({"__SYCL_COMPILER_VERSION": 20260724}, False),
            ({"__INTEL_LLVM_COMPILER": 20260100}, False),
            (
                {
                    "__INTEL_LLVM_COMPILER": 20260103,
                    "__SYCL_COMPILER_VERSION": 20260724,
                },
                False,
            ),
            ({"__INTEL_LLVM_COMPILER": 20260200}, True),
            ({"__INTEL_LLVM_COMPILER": 20260300}, True),
            ({"__INTEL_LLVM_COMPILER_VERSION": 20260100}, False),
            (
                {
                    "__INTEL_LLVM_COMPILER_VERSION": 20260103,
                    "__SYCL_COMPILER_VERSION": 20260724,
                },
                False,
            ),
            ({"__INTEL_LLVM_COMPILER_VERSION": 20260200}, True),
            ({"__INTEL_LLVM_COMPILER_VERSION": 20260300}, True),
            (
                {
                    "__INTEL_LLVM_COMPILER": 20260103,
                    "__INTEL_LLVM_COMPILER_VERSION": 20260103,
                },
                False,
            ),
            (
                {
                    "__INTEL_LLVM_COMPILER": 20260200,
                    "__INTEL_LLVM_COMPILER_VERSION": 20260200,
                },
                True,
            ),
        )
        for definitions, enabled in cases:
            with self.subTest(definitions=definitions):
                result = subprocess.run(
                    [
                        compiler,
                        "-E",
                        "-P",
                        "-x",
                        "c++",
                        "-",
                        *(f"-D{name}={value}" for name, value in definitions.items()),
                    ],
                    input=program,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertEqual(
                    result.stdout.strip(), "CRI_ENABLED" if enabled else "CRI_DISABLED"
                )


if __name__ == "__main__":
    unittest.main()
