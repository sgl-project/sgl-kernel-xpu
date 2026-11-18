"""Prepare the dense noncausal FP16/BF16 attention overlay."""

import argparse
import shutil
from pathlib import Path


def prepare(source, destination):
    node = Path(__file__).resolve().parent
    source = Path(source)
    destination = Path(destination)
    relative = Path("flash_attention_v2/collective")
    root = source / "applications" / relative
    text = (root / "xe_fmha_fwd_mainloop.hpp").read_text()
    partial = "  using FragSPartialRow = decltype(reduce<1, ReduceMode::Vertical>(FragS{}, sycl::plus<void>{}));"
    if text.count(partial) != 1:
        raise RuntimeError("current pin partial-sum declaration changed")
    text = text.replace(
        partial,
        """#if !SGLD_ATTENTION_TWO_EDIT
  using FragSPartialRow = FragSRow;
#else
  using FragSPartialRow = decltype(reduce<1, ReduceMode::Vertical>(FragS{}, sycl::plus<void>{}));
#endif""",
    )
    begin = text.index(
        "    using namespace sycl::ext::oneapi::this_work_item;",
        text.index("  operator()"),
    )
    end = text.index("\n  // Single step of blocked softmax.", begin)
    body = (node / "diffusion_mainloop.inc").read_text()
    soft_end = text.index("\n};", end)
    text = text[:begin] + body + text[soft_end:]
    output = destination / relative
    output.mkdir(parents=True, exist_ok=True)
    (output / "xe_fmha_fwd_mainloop.hpp").write_text(text)
    shutil.copyfile(root / "fmha_fusion.hpp", output / "fmha_fusion.hpp")
    kernel = (
        source / "applications/flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
    ).read_text()
    floor = "k_blocks = static_cast<int>(static_cast<unsigned>(seq_len) / static_cast<unsigned>(get<1>(TileShapeQK{})));"
    if kernel.count(floor) != 1:
        raise RuntimeError("current pin dense block count changed")
    kernel = kernel.replace(
        floor, "k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));"
    )
    epilogue = "      epilogue(O(_,_,head_q,l_coord),"
    if kernel.count(epilogue) != 1:
        raise RuntimeError("current pin dense epilogue call changed")
    kernel = kernel.replace(
        epilogue,
        """#if !SGLD_ATTENTION_TWO_EDIT
      epilogue.template operator()<true>(O(_,_,head_q,l_coord),
#else
      epilogue(O(_,_,head_q,l_coord),
#endif""",
    )
    kernel_output = destination / "flash_attention_v2/kernel"
    kernel_output.mkdir(parents=True, exist_ok=True)
    (kernel_output / "xe_fmha_fwd_kernel.hpp").write_text(kernel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    prepare(args.source, args.destination)
