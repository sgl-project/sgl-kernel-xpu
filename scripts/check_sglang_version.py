"""Fail the image build when the installed SGLang is older than 0.5.22."""

import importlib.metadata
import sys


def parse(version: str) -> tuple:
    core = version.split("+")[0].split("-")[0]
    parts = []
    for piece in core.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits or "0"))
    return tuple((parts + [0, 0, 0])[:3])


def main() -> None:
    version = importlib.metadata.version("sglang")
    if parse(version) < (0, 5, 22):
        print(f"SGLang {version} is below 0.5.22", file=sys.stderr)
        raise SystemExit(1)
    print(f"SGLang {version}")


if __name__ == "__main__":
    main()
