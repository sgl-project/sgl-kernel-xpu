import pytest
import torch
import utils
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_all_layer_lf_pf,
    transfer_kv_all_layer_lf_ph,
    transfer_kv_all_layer_mla,
    transfer_kv_all_layer_mla_lf_pf,
    transfer_kv_direct,
    transfer_kv_per_layer,
    transfer_kv_per_layer_direct_pf_lf,
    transfer_kv_per_layer_mla,
    transfer_kv_per_layer_mla_pf_lf,
    transfer_kv_per_layer_pf_lf,
    transfer_mamba_state,
    transfer_mamba_state_all_layer,
    transfer_mamba_state_all_layer_lf_pf,
    transfer_mamba_state_per_layer_pf_lf,
)

from sglang.srt.utils import is_hip

device = utils.get_device()


def ref_copy_with_indices(src_pool, dst_pool, src_indices, dst_indices):
    dst_pool[dst_indices] = src_pool[src_indices].to(dst_pool.device)


def ref_copy_with_indices_pf_direct(
    src_pool, dst_pool, src_indices, dst_indices, page_size, layer_id, lf_to_pf=False
):
    if lf_to_pf:
        for i in range(0, len(src_indices), page_size):
            dst_pool[dst_indices[i] // page_size][layer_id] = src_pool[layer_id][
                src_indices[i : i + page_size]
            ].to(dst_pool.device)
    else:
        for i in range(0, len(src_indices), page_size):
            dst_pool[layer_id][dst_indices[i : i + page_size]] = src_pool[
                src_indices[i] // page_size
            ][layer_id].to(dst_pool.device)


def ref_copy_with_indices_pf(
    src_pool, dst_pool, src_indices, dst_indices, layer_id, pf_to_lf=True
):
    """Reference for the page-first kernel layout.

    pf pool is [num_tokens, num_layers, item_size] (per-token addressing:
    base + token * (num_layers * item_size) + layer * item_size).
    lf pool is [num_layers, num_tokens, item_size].
    """
    if pf_to_lf:
        # src is pf [tokens, layers, item], dst is lf [layers, tokens, item]
        dst_pool[layer_id][dst_indices] = src_pool[src_indices, layer_id].to(
            dst_pool.device
        )
    else:
        # src is lf [layers, tokens, item], dst is pf [tokens, layers, item]
        dst_pool[dst_indices, layer_id] = src_pool[layer_id][src_indices].to(
            dst_pool.device
        )


def ref_copy_with_indices_page_head(
    src_pool,
    dst_pool,
    src_indices,
    dst_indices,
    page_size,
    layer_id,
    head_num,
    lf_to_ph=False,
):
    if lf_to_ph:
        for head_id in range(head_num):
            for i in range(0, len(src_indices)):
                dst_pool[dst_indices[i] // page_size][head_id][
                    dst_indices[i] % page_size
                ][layer_id] = src_pool[layer_id][src_indices[i]][head_id].to(
                    dst_pool.device
                )
    else:
        for head_id in range(head_num):
            for i in range(0, len(src_indices)):
                dst_pool[layer_id][dst_indices[i]][head_id] = src_pool[
                    src_indices[i] // page_size
                ][head_id][src_indices[i] % page_size][layer_id].to(dst_pool.device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [1, 128, 1024])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("item_size", [256])
@pytest.mark.parametrize("total_items_in_pool", [10240])
@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("all_layers", [False, True])
def test_transfer_kv(
    dtype: torch.dtype,
    num_items_to_transfer: int,
    item_size: int,
    page_size: int,
    total_items_in_pool: int,
    is_mla: bool,
    all_layers: bool,
):
    """
    Tests the per-layer transfer functions, treating tensors as memory pools.
    """

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)

    torch.manual_seed(42)

    num_layers = 4  # A small number of layers for pool creation

    total_pages_in_pool = total_items_in_pool // page_size
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return
    page_indices = torch.randperm(total_pages_in_pool, dtype=torch.int64)
    src_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[:num_pages_to_transfer]
        ]
    )
    src_indices_device = src_indices_host.to(device)
    dst_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[num_pages_to_transfer : 2 * num_pages_to_transfer]
        ]
    )
    dst_indices_device = dst_indices_host.to(device)

    # Prepare memory pools based on whether it's an MLA case.
    if is_mla:
        src_pool_host = torch.randn(
            num_layers, total_items_in_pool, item_size
        ).pin_memory()
        dst_pool_ref = torch.zeros_like(src_pool_host).to(device)
        dst_pool_kernel = torch.zeros_like(dst_pool_ref)
        dst_pool_direct = torch.zeros_like(dst_pool_ref)
    else:
        src_k_pool = torch.randn(
            num_layers, total_items_in_pool, item_size
        ).pin_memory()
        src_v_pool = torch.randn(
            num_layers, total_items_in_pool, item_size
        ).pin_memory()
        dst_k_pool_ref = torch.zeros_like(src_k_pool).to(device)
        dst_v_pool_ref = torch.zeros_like(src_v_pool).to(device)
        dst_k_pool_kernel = torch.zeros_like(dst_k_pool_ref)
        dst_v_pool_kernel = torch.zeros_like(dst_v_pool_ref)
        dst_k_pool_direct = torch.zeros_like(dst_k_pool_ref)
        dst_v_pool_direct = torch.zeros_like(dst_v_pool_ref)

    torch.accelerator.synchronize()

    # We will test the per-layer function on the first layer (index 0) of the pool.
    layer_idx_to_test = 0

    if is_mla:
        if not all_layers:
            ref_copy_with_indices(
                src_pool_host[layer_idx_to_test],
                dst_pool_ref[layer_idx_to_test],
                src_indices_host,
                dst_indices_device,
            )
            transfer_kv_per_layer_mla(
                src_pool_host[layer_idx_to_test],
                dst_pool_kernel[layer_idx_to_test],
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
            )
            transfer_kv_direct(
                [src_pool_host[layer_idx_to_test]],
                [dst_pool_direct[layer_idx_to_test]],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        else:
            for layer_id in range(num_layers):
                ref_copy_with_indices(
                    src_pool_host[layer_id],
                    dst_pool_ref[layer_id],
                    src_indices_host,
                    dst_indices_device,
                )
            src_layers_device = torch.tensor(
                [src_pool_host[layer_id].data_ptr() for layer_id in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            dst_layers_device = torch.tensor(
                [
                    dst_pool_kernel[layer_id].data_ptr()
                    for layer_id in range(num_layers)
                ],
                dtype=torch.uint64,
                device=device,
            )
            transfer_kv_all_layer_mla(
                src_layers_device,
                dst_layers_device,
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
                num_layers=num_layers,
            )
            transfer_kv_direct(
                [src_pool_host[layer_id] for layer_id in range(num_layers)],
                [dst_pool_direct[layer_id] for layer_id in range(num_layers)],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        torch.accelerator.synchronize()
        torch.testing.assert_close(dst_pool_kernel, dst_pool_ref)
        torch.testing.assert_close(dst_pool_direct, dst_pool_ref)
    else:
        if not all_layers:
            ref_copy_with_indices(
                src_k_pool[layer_idx_to_test],
                dst_k_pool_ref[layer_idx_to_test],
                src_indices_host,
                dst_indices_device,
            )
            ref_copy_with_indices(
                src_v_pool[layer_idx_to_test],
                dst_v_pool_ref[layer_idx_to_test],
                src_indices_host,
                dst_indices_device,
            )
            transfer_kv_per_layer(
                src_k_pool[layer_idx_to_test],
                dst_k_pool_kernel[layer_idx_to_test],
                src_v_pool[layer_idx_to_test],
                dst_v_pool_kernel[layer_idx_to_test],
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
            )
            transfer_kv_direct(
                [src_k_pool[layer_idx_to_test], src_v_pool[layer_idx_to_test]],
                [
                    dst_k_pool_direct[layer_idx_to_test],
                    dst_v_pool_direct[layer_idx_to_test],
                ],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        else:
            for layer_id in range(num_layers):
                ref_copy_with_indices(
                    src_k_pool[layer_id],
                    dst_k_pool_ref[layer_id],
                    src_indices_host,
                    dst_indices_device,
                )
                ref_copy_with_indices(
                    src_v_pool[layer_id],
                    dst_v_pool_ref[layer_id],
                    src_indices_host,
                    dst_indices_device,
                )

            src_k_layers_device = torch.tensor(
                [src_k_pool[layer_id].data_ptr() for layer_id in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            src_v_layers_device = torch.tensor(
                [src_v_pool[layer_id].data_ptr() for layer_id in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            dst_k_layers_device = torch.tensor(
                [
                    dst_k_pool_kernel[layer_id].data_ptr()
                    for layer_id in range(num_layers)
                ],
                dtype=torch.uint64,
                device=device,
            )
            dst_v_layers_device = torch.tensor(
                [
                    dst_v_pool_kernel[layer_id].data_ptr()
                    for layer_id in range(num_layers)
                ],
                dtype=torch.uint64,
                device=device,
            )
            transfer_kv_all_layer(
                src_k_layers_device,
                dst_k_layers_device,
                src_v_layers_device,
                dst_v_layers_device,
                src_indices_device,
                dst_indices_device,
                item_size=item_size * dtype.itemsize,
                num_layers=num_layers,
            )
            transfer_kv_direct(
                [src_k_pool[layer_id] for layer_id in range(num_layers)]
                + [src_v_pool[layer_id] for layer_id in range(num_layers)],
                [dst_k_pool_direct[layer_id] for layer_id in range(num_layers)]
                + [dst_v_pool_direct[layer_id] for layer_id in range(num_layers)],
                src_indices_host,
                dst_indices_device,
                page_size=page_size,
            )
        torch.accelerator.synchronize()
        torch.testing.assert_close(dst_k_pool_kernel, dst_k_pool_ref)
        torch.testing.assert_close(dst_v_pool_kernel, dst_v_pool_ref)
        torch.testing.assert_close(dst_k_pool_direct, dst_k_pool_ref)
        torch.testing.assert_close(dst_v_pool_direct, dst_v_pool_ref)

    torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [128, 1024, 8192])
@pytest.mark.parametrize("page_size", [16, 64, 128])
@pytest.mark.parametrize("item_size", [256])
@pytest.mark.parametrize("total_items_in_pool", [20480])
@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("lf_to_pf", [False, True])
def test_transfer_kv_pf_direct(
    dtype: torch.dtype,
    num_items_to_transfer: int,
    item_size: int,
    page_size: int,
    total_items_in_pool: int,
    is_mla: bool,
    lf_to_pf: bool,
):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)

    torch.manual_seed(42)

    num_layers = 4

    total_pages_in_pool = total_items_in_pool // page_size
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return
    page_indices = torch.randperm(total_pages_in_pool, dtype=torch.int64)
    src_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[:num_pages_to_transfer]
        ]
    )
    src_indices_device = src_indices_host.to(device)
    dst_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[num_pages_to_transfer : 2 * num_pages_to_transfer]
        ]
    )
    dst_indices_device = dst_indices_host.to(device)

    # We will test the per-layer function on the first layer (index 0) of the pool.
    layer_idx_to_test = 0

    if lf_to_pf:
        if is_mla:
            src_pool = torch.randn(num_layers, total_items_in_pool, item_size).to(
                device
            )
            src_pool_ptrs = [src_pool[i] for i in range(num_layers)]
            dst_pool_ref = torch.zeros(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()
            dst_pool_direct = torch.zeros_like(dst_pool_ref)
            torch.accelerator.synchronize()

            transfer_kv_all_layer_direct_lf_pf(
                src_pool_ptrs,
                [dst_pool_direct],
                src_indices_host,
                dst_indices_host,
                page_size,
            )
            for i in range(num_layers):
                ref_copy_with_indices_pf_direct(
                    src_pool,
                    dst_pool_ref,
                    src_indices_device,
                    dst_indices_host,
                    page_size,
                    i,
                    lf_to_pf=True,
                )
            torch.accelerator.synchronize()
            torch.testing.assert_close(dst_pool_direct, dst_pool_ref)

        else:
            src_k_pool = torch.randn(num_layers, total_items_in_pool, item_size).to(
                device
            )
            src_k_pool_ptrs = [src_k_pool[i] for i in range(num_layers)]
            src_v_pool = torch.randn(num_layers, total_items_in_pool, item_size).to(
                device
            )
            src_v_pool_ptrs = [src_v_pool[i] for i in range(num_layers)]
            dst_k_pool_ref = torch.zeros(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()
            dst_v_pool_ref = torch.zeros_like(dst_k_pool_ref)
            dst_k_pool_direct = torch.zeros_like(dst_k_pool_ref)
            dst_v_pool_direct = torch.zeros_like(dst_v_pool_ref)
            torch.accelerator.synchronize()

            transfer_kv_all_layer_direct_lf_pf(
                src_k_pool_ptrs + src_v_pool_ptrs,
                [dst_k_pool_direct, dst_v_pool_direct],
                src_indices_host,
                dst_indices_host,
                page_size,
            )
            for i in range(num_layers):
                ref_copy_with_indices_pf_direct(
                    src_k_pool,
                    dst_k_pool_ref,
                    src_indices_device,
                    dst_indices_host,
                    page_size,
                    i,
                    lf_to_pf=True,
                )
                ref_copy_with_indices_pf_direct(
                    src_v_pool,
                    dst_v_pool_ref,
                    src_indices_device,
                    dst_indices_host,
                    page_size,
                    i,
                    lf_to_pf=True,
                )
            torch.accelerator.synchronize()
            torch.testing.assert_close(dst_k_pool_direct, dst_k_pool_ref)
            torch.testing.assert_close(dst_v_pool_direct, dst_v_pool_ref)
    else:
        if is_mla:
            src_pool = torch.randn(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()

            dst_pool_ref = torch.zeros(num_layers, total_items_in_pool, item_size).to(
                device
            )
            dst_pool_direct = torch.zeros_like(dst_pool_ref)
            dst_pool_direct_ptrs = [dst_pool_direct[i] for i in range(num_layers)]
            torch.accelerator.synchronize()

            transfer_kv_per_layer_direct_pf_lf(
                [src_pool],
                [dst_pool_direct_ptrs[layer_idx_to_test]],
                src_indices_host,
                dst_indices_host,
                layer_idx_to_test,
                page_size,
            )
            ref_copy_with_indices_pf_direct(
                src_pool,
                dst_pool_ref,
                src_indices_host,
                dst_indices_device,
                page_size,
                layer_idx_to_test,
                lf_to_pf=False,
            )
            torch.accelerator.synchronize()
            torch.testing.assert_close(dst_pool_direct, dst_pool_ref)
        else:
            src_k_pool = torch.randn(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()
            src_v_pool = torch.randn(
                total_pages_in_pool, num_layers, page_size, item_size
            ).pin_memory()

            dst_k_pool_ref = torch.zeros(num_layers, total_items_in_pool, item_size).to(
                device
            )
            dst_k_pool_direct = torch.zeros_like(dst_k_pool_ref)
            dst_k_pool_direct_ptrs = [dst_k_pool_direct[i] for i in range(num_layers)]

            dst_v_pool_ref = torch.zeros_like(dst_k_pool_ref)
            dst_v_pool_direct = torch.zeros_like(dst_v_pool_ref)
            dst_v_pool_direct_ptrs = [dst_v_pool_direct[i] for i in range(num_layers)]
            torch.accelerator.synchronize()

            transfer_kv_per_layer_direct_pf_lf(
                [src_k_pool, src_v_pool],
                [
                    dst_k_pool_direct_ptrs[layer_idx_to_test],
                    dst_v_pool_direct_ptrs[layer_idx_to_test],
                ],
                src_indices_host,
                dst_indices_host,
                layer_idx_to_test,
                page_size,
            )

            ref_copy_with_indices_pf_direct(
                src_k_pool,
                dst_k_pool_ref,
                src_indices_host,
                dst_indices_device,
                page_size,
                layer_idx_to_test,
                lf_to_pf=False,
            )
            ref_copy_with_indices_pf_direct(
                src_v_pool,
                dst_v_pool_ref,
                src_indices_host,
                dst_indices_device,
                page_size,
                layer_idx_to_test,
                lf_to_pf=False,
            )

            torch.accelerator.synchronize()
            torch.testing.assert_close(dst_k_pool_direct, dst_k_pool_ref)
            torch.testing.assert_close(dst_v_pool_direct, dst_v_pool_ref)
    torch.set_default_dtype(original_dtype)


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [256, 1024])
@pytest.mark.parametrize("page_size", [16, 64, 128])
@pytest.mark.parametrize("item_size", [1024])
@pytest.mark.parametrize("head_num", [8, 16])
@pytest.mark.parametrize("total_items_in_pool", [4096])
@pytest.mark.parametrize("lf_to_ph", [False, True])
def test_transfer_kv_page_head(
    dtype: torch.dtype,
    num_items_to_transfer: int,
    page_size: int,
    item_size: int,
    head_num: int,
    total_items_in_pool: int,
    lf_to_ph: bool,
):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)

    torch.manual_seed(42)

    num_layers = 4

    total_pages_in_pool = total_items_in_pool // page_size
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return

    assert item_size % head_num == 0
    head_dim = item_size // head_num

    page_indices = torch.randperm(total_pages_in_pool, dtype=torch.int64)
    src_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[:num_pages_to_transfer]
        ]
    )
    src_indices_device = src_indices_host.to(device)
    dst_indices_host = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[num_pages_to_transfer : 2 * num_pages_to_transfer]
        ]
    )
    dst_indices_device = dst_indices_host.to(device)

    # We will test the per-layer function on the first layer (index 0) of the pool.
    layer_idx_to_test = 0

    if lf_to_ph:
        src_k_pool = torch.randn(
            num_layers, total_items_in_pool, head_num, head_dim
        ).to(device)
        src_v_pool = torch.randn(
            num_layers, total_items_in_pool, head_num, head_dim
        ).to(device)
        src_k_pool_ptrs = [src_k_pool[i] for i in range(num_layers)]
        src_k_pool_ptrs = torch.tensor(
            [x.data_ptr() for x in src_k_pool_ptrs],
            dtype=torch.uint64,
            device=device,
        )
        src_v_pool_ptrs = [src_v_pool[i] for i in range(num_layers)]
        src_v_pool_ptrs = torch.tensor(
            [x.data_ptr() for x in src_v_pool_ptrs],
            dtype=torch.uint64,
            device=device,
        )

        dst_k_pool_ref = torch.zeros(
            total_pages_in_pool, head_num, page_size, num_layers, head_dim
        ).pin_memory()
        dst_v_pool_ref = torch.zeros_like(dst_k_pool_ref).pin_memory()

        dst_k_pool_kernel = torch.zeros_like(dst_k_pool_ref).pin_memory()
        dst_v_pool_kernel = torch.zeros_like(dst_v_pool_ref).pin_memory()
        torch.accelerator.synchronize()

        transfer_kv_all_layer_lf_ph(
            src_k_pool_ptrs,
            dst_k_pool_kernel,
            src_v_pool_ptrs,
            dst_v_pool_kernel,
            src_indices_device,
            dst_indices_device,
            item_size * dtype.itemsize,
            item_size * num_layers * dtype.itemsize,
            num_layers,
            page_size,
            head_num,
        )
        torch.accelerator.synchronize()

        for i in range(num_layers):
            ref_copy_with_indices_page_head(
                src_k_pool,
                dst_k_pool_ref,
                src_indices_device,
                dst_indices_host,
                page_size,
                i,
                head_num,
                lf_to_ph=True,
            )
            ref_copy_with_indices_page_head(
                src_v_pool,
                dst_v_pool_ref,
                src_indices_device,
                dst_indices_host,
                page_size,
                i,
                head_num,
                lf_to_ph=True,
            )
        torch.accelerator.synchronize()
        torch.testing.assert_close(dst_k_pool_kernel, dst_k_pool_ref)
        torch.testing.assert_close(dst_v_pool_kernel, dst_v_pool_ref)
    else:
        from sgl_kernel.kvcacheio import transfer_kv_per_layer_ph_lf

        src_k_pool = torch.randn(
            total_pages_in_pool, head_num, page_size, num_layers, head_dim
        ).pin_memory()
        src_v_pool = torch.randn(
            total_pages_in_pool, head_num, page_size, num_layers, head_dim
        ).pin_memory()

        dst_k_pool_ref = torch.zeros(
            num_layers, total_items_in_pool, head_num, head_dim
        ).to(device)
        dst_v_pool_ref = torch.zeros_like(dst_k_pool_ref)
        dst_k_pool_kernel = torch.zeros_like(dst_k_pool_ref)
        dst_v_pool_kernel = torch.zeros_like(dst_v_pool_ref)
        dst_k_pool_kernel_ptrs = [dst_k_pool_kernel[i] for i in range(num_layers)]
        dst_v_pool_kernel_ptrs = [dst_v_pool_kernel[i] for i in range(num_layers)]
        torch.accelerator.synchronize()

        transfer_kv_per_layer_ph_lf(
            src_k_pool,
            dst_k_pool_kernel_ptrs[layer_idx_to_test],
            src_v_pool,
            dst_v_pool_kernel_ptrs[layer_idx_to_test],
            src_indices_device,
            dst_indices_device,
            layer_idx_to_test,
            item_size * dtype.itemsize,
            item_size * num_layers * dtype.itemsize,
            page_size,
            head_num,
        )

        ref_copy_with_indices_page_head(
            src_k_pool,
            dst_k_pool_ref,
            src_indices_host,
            dst_indices_device,
            page_size,
            layer_idx_to_test,
            head_num,
            lf_to_ph=False,
        )
        ref_copy_with_indices_page_head(
            src_v_pool,
            dst_v_pool_ref,
            src_indices_host,
            dst_indices_device,
            page_size,
            layer_idx_to_test,
            head_num,
            lf_to_ph=False,
        )
        torch.accelerator.synchronize()
        torch.testing.assert_close(dst_k_pool_kernel, dst_k_pool_ref)
        torch.testing.assert_close(dst_v_pool_kernel, dst_v_pool_ref)
    torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_items_to_transfer", [256, 1024])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("item_size", [256])
@pytest.mark.parametrize("total_items_in_pool", [10240])
@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("lf_to_pf", [False, True])
def test_transfer_kv_pf_kernel(
    dtype: torch.dtype,
    num_items_to_transfer: int,
    page_size: int,
    item_size: int,
    total_items_in_pool: int,
    is_mla: bool,
    lf_to_pf: bool,
):
    """Device-kernel page-first transfers (transfer_kv_*_pf_lf / *_lf_pf).

    pf pool layout is [num_tokens, num_layers, item_size]; lf pool is
    [num_layers, num_tokens, item_size].  Validated against a pure-torch
    reference (ref_copy_with_indices_pf).
    """
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)

    torch.manual_seed(42)

    num_layers = 4
    layer_idx_to_test = 0
    item_bytes = item_size * dtype.itemsize
    layout_dim_bytes = item_size * num_layers * dtype.itemsize

    total_pages_in_pool = total_items_in_pool // page_size
    num_pages_to_transfer = num_items_to_transfer // page_size
    if num_pages_to_transfer == 0:
        torch.set_default_dtype(original_dtype)
        return

    page_indices = torch.randperm(total_pages_in_pool, dtype=torch.int64)
    src_indices = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[:num_pages_to_transfer]
        ]
    ).to(device)
    dst_indices = torch.cat(
        [
            torch.arange(p * page_size, (p + 1) * page_size)
            for p in page_indices[num_pages_to_transfer : 2 * num_pages_to_transfer]
        ]
    ).to(device)

    def make_lf_tbl(pool):
        return torch.tensor(
            [pool[i].data_ptr() for i in range(num_layers)],
            dtype=torch.uint64,
            device=device,
        )

    if lf_to_pf:
        # src lf [layers, tokens, item] -> dst pf [tokens, layers, item], all layers.
        src_k = torch.randn(num_layers, total_items_in_pool, item_size).to(device)
        dst_k_ref = torch.zeros(total_items_in_pool, num_layers, item_size).to(device)
        dst_k_kernel = torch.zeros_like(dst_k_ref)
        if is_mla:
            torch.accelerator.synchronize()
            transfer_kv_all_layer_mla_lf_pf(
                make_lf_tbl(src_k),
                dst_k_kernel,
                src_indices,
                dst_indices,
                item_bytes,
                layout_dim_bytes,
                num_layers,
            )
            for i in range(num_layers):
                ref_copy_with_indices_pf(
                    src_k, dst_k_ref, src_indices, dst_indices, i, pf_to_lf=False
                )
            torch.accelerator.synchronize()
            torch.testing.assert_close(dst_k_kernel, dst_k_ref)
        else:
            src_v = torch.randn(num_layers, total_items_in_pool, item_size).to(device)
            dst_v_ref = torch.zeros_like(dst_k_ref)
            dst_v_kernel = torch.zeros_like(dst_v_ref)
            torch.accelerator.synchronize()
            transfer_kv_all_layer_lf_pf(
                make_lf_tbl(src_k),
                dst_k_kernel,
                make_lf_tbl(src_v),
                dst_v_kernel,
                src_indices,
                dst_indices,
                item_bytes,
                layout_dim_bytes,
                num_layers,
            )
            for i in range(num_layers):
                ref_copy_with_indices_pf(
                    src_k, dst_k_ref, src_indices, dst_indices, i, pf_to_lf=False
                )
                ref_copy_with_indices_pf(
                    src_v, dst_v_ref, src_indices, dst_indices, i, pf_to_lf=False
                )
            torch.accelerator.synchronize()
            torch.testing.assert_close(dst_k_kernel, dst_k_ref)
            torch.testing.assert_close(dst_v_kernel, dst_v_ref)
    else:
        # src pf [tokens, layers, item] -> dst lf [layers, tokens, item], one layer.
        src_k = torch.randn(total_items_in_pool, num_layers, item_size).to(device)
        dst_k_ref = torch.zeros(num_layers, total_items_in_pool, item_size).to(device)
        dst_k_kernel = torch.zeros_like(dst_k_ref)
        if is_mla:
            torch.accelerator.synchronize()
            transfer_kv_per_layer_mla_pf_lf(
                src_k,
                dst_k_kernel[layer_idx_to_test],
                src_indices,
                dst_indices,
                layer_idx_to_test,
                item_bytes,
                layout_dim_bytes,
            )
            ref_copy_with_indices_pf(
                src_k, dst_k_ref, src_indices, dst_indices, layer_idx_to_test
            )
            torch.accelerator.synchronize()
            torch.testing.assert_close(dst_k_kernel, dst_k_ref)
        else:
            src_v = torch.randn(total_items_in_pool, num_layers, item_size).to(device)
            dst_v_ref = torch.zeros_like(dst_k_ref)
            dst_v_kernel = torch.zeros_like(dst_v_ref)
            torch.accelerator.synchronize()
            transfer_kv_per_layer_pf_lf(
                src_k,
                dst_k_kernel[layer_idx_to_test],
                src_v,
                dst_v_kernel[layer_idx_to_test],
                src_indices,
                dst_indices,
                layer_idx_to_test,
                item_bytes,
                layout_dim_bytes,
            )
            ref_copy_with_indices_pf(
                src_k, dst_k_ref, src_indices, dst_indices, layer_idx_to_test
            )
            ref_copy_with_indices_pf(
                src_v, dst_v_ref, src_indices, dst_indices, layer_idx_to_test
            )
            torch.accelerator.synchronize()
            torch.testing.assert_close(dst_k_kernel, dst_k_ref)
            torch.testing.assert_close(dst_v_kernel, dst_v_ref)
    torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_transfer_kv_empty_indices(dtype: torch.dtype):
    """Empty index tensors must be a no-op, not a div-by-zero crash.

    Regression test: num_items==0 previously produced num_wgs = div_up(0, 0)
    in the SYCL launcher and crashed the process with SIGFPE.
    """
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)

    item_size = 256
    src_k = torch.randn(16, item_size).to(device)
    src_v = torch.randn(16, item_size).to(device)
    dst_k = torch.zeros_like(src_k)
    dst_v = torch.zeros_like(src_v)
    empty = torch.empty(0, dtype=torch.int64, device=device)

    transfer_kv_per_layer(
        src_k, dst_k, src_v, dst_v, empty, empty, item_size * dtype.itemsize
    )
    transfer_kv_per_layer_mla(src_k, dst_k, empty, empty, item_size * dtype.itemsize)
    torch.accelerator.synchronize()

    # Destinations stay zero: nothing was transferred.
    torch.testing.assert_close(dst_k, torch.zeros_like(dst_k))
    torch.testing.assert_close(dst_v, torch.zeros_like(dst_v))
    torch.set_default_dtype(original_dtype)


def test_transfer_kv_rejects_invalid_index_metadata():
    src = torch.zeros(8, 8, device=device)
    dst = torch.zeros_like(src)
    xpu_indices = torch.arange(4, dtype=torch.int64, device=device)

    with pytest.raises(RuntimeError, match="src_indices must be an XPU tensor"):
        transfer_kv_per_layer_mla(
            src, dst, xpu_indices.cpu(), xpu_indices, item_size=32
        )

    noncontiguous = torch.arange(8, dtype=torch.int64, device=device).view(2, 4).t()
    with pytest.raises(RuntimeError, match="src_indices must be contiguous"):
        transfer_kv_per_layer_mla(
            src, dst, noncontiguous, noncontiguous, item_size=32
        )


def test_transfer_kv_rejects_invalid_pointer_table():
    indices = torch.arange(1, dtype=torch.int64, device=device)
    pointer_table = torch.tensor([0], dtype=torch.uint64)

    with pytest.raises(RuntimeError, match="src_layers must be an XPU tensor"):
        transfer_kv_all_layer_mla(
            pointer_table,
            pointer_table.to(device),
            indices,
            indices,
            item_size=8,
            num_layers=1,
        )


def test_transfer_kv_page_head_rejects_unaligned_heads():
    indices = torch.arange(1, dtype=torch.int64, device=device)
    src = torch.zeros(1, 24, device=device)
    ptrs = torch.tensor([src.data_ptr()], dtype=torch.uint64, device=device)
    dst_k = torch.zeros(1, 2, 1, 1, 12).pin_memory()
    dst_v = torch.zeros_like(dst_k).pin_memory()

    with pytest.raises(RuntimeError, match="per-head item size must be divisible by 8"):
        transfer_kv_all_layer_lf_ph(
            ptrs,
            dst_k,
            ptrs,
            dst_v,
            indices,
            indices,
            item_size=24,
            dst_layout_dim=24,
            num_layers=1,
            page_size=1,
            head_num=2,
        )

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("item_size_bytes", [
    512,     # Typical KV cache (256 elements × 2 bytes)
    1024,    # Larger attention head
    8192,    # Upper bound of typical KV
    65536,   # 64 KB boundary - kernel design limit
    # 1048576,  # 1 MB - would timeout (commented out)
    # 3145728,  # Mamba temporal state (~3 MB) - would timeout (commented out)
])
@pytest.mark.parametrize("num_items", [1, 16])
def test_transfer_kv_large_item_size(dtype: torch.dtype, item_size_bytes: int, num_items: int):
    """Test kernel behavior with varying item_size values.

    Regression test for Mamba + HiCache DEVICE_LOST bug. The kernel was originally
    designed for KV cache (item_size ~512 bytes) but was called with Mamba states
    (item_size ~3 MB), causing the kernel's inner loop to run 24,576+ iterations
    per lane and triggering a GPU watchdog timeout.

    Known limits:
    - item_size <= 64 KB: Kernel completes in ~1 ms, safe
    - item_size = 1 MB: Kernel runs ~50+ ms, may timeout on strict TDR
    - item_size = 3 MB: Kernel runs 200+ ms, guaranteed DEVICE_LOST on XPU

    For item_size > 64 KB, callers should use PyTorch copy_ fallback or a
    chunked transfer approach instead of this kernel.

    See: docs/mamba-hicache-device-lost-root-cause.md
    """
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)

    torch.manual_seed(42)

    # item_size in elements (kernel takes bytes, we compute from dtype)
    item_size_elements = item_size_bytes // dtype.itemsize
    total_pool_size = 256

    # Create pools
    src_pool = torch.randn(total_pool_size, item_size_elements).pin_memory()
    dst_pool_ref = torch.zeros(total_pool_size, item_size_elements, device=device)
    dst_pool_kernel = torch.zeros_like(dst_pool_ref)

    # Select random indices
    indices = torch.randperm(total_pool_size, dtype=torch.int64)[:num_items]
    src_indices = indices.to(device)
    dst_indices = indices.to(device)

    # Reference implementation
    for si, di in zip(indices.tolist(), indices.tolist()):
        dst_pool_ref[di] = src_pool[si].to(device)

    # Kernel implementation
    transfer_kv_per_layer_mla(
        src=src_pool,
        dst=dst_pool_kernel,
        src_indices=src_indices,
        dst_indices=dst_indices,
        item_size=item_size_bytes,
    )
    torch.accelerator.synchronize()

    # Verify
    torch.testing.assert_close(dst_pool_kernel, dst_pool_ref)

    torch.set_default_dtype(original_dtype)


# =============================================================================
# Mamba State Transfer Kernel Tests (Tier 2: 64 KB - 16 MB)
# =============================================================================
# HiCache uses three transfer directions for Mamba states:
#   - Device→Host (backup/write from L1 device to L2 host pinned)
#   - Host→Device (restore/read from L2 host pinned to L1 device)
#   - Device→Device (for L1 internal operations, less common)


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_items", [1, 8, 32])
@pytest.mark.parametrize("item_size_bytes", [524288, 1048576])  # 512 KB, 1 MB
def test_transfer_mamba_state_device_to_device(
    dtype: torch.dtype,
    num_items: int,
    item_size_bytes: int,
):
    """Test Mamba state transfer: Device → Device (L1 internal)."""
    torch.manual_seed(42)
    item_size_elements = item_size_bytes // dtype.itemsize
    pool_size = 128

    src = torch.randn(pool_size, item_size_elements, dtype=dtype, device=device)
    dst = torch.zeros_like(src)

    indices = torch.randperm(pool_size, dtype=torch.int64, device=device)[:num_items]

    # Reference
    dst_ref = dst.clone()
    for i in indices.tolist():
        dst_ref[i] = src[i]

    transfer_mamba_state(src, dst, indices, indices, item_size=item_size_bytes)
    torch.accelerator.synchronize()

    torch.testing.assert_close(dst, dst_ref)


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_items", [1, 8, 32])
@pytest.mark.parametrize("item_size_bytes", [524288, 1048576])  # 512 KB, 1 MB
def test_transfer_mamba_state_device_to_host(
    dtype: torch.dtype,
    num_items: int,
    item_size_bytes: int,
):
    """Test Mamba state transfer: Device → Host (HiCache backup/write L1→L2)."""
    torch.manual_seed(42)
    item_size_elements = item_size_bytes // dtype.itemsize
    pool_size = 128

    src_dev = torch.randn(pool_size, item_size_elements, dtype=dtype, device=device)
    dst_host = torch.zeros(pool_size, item_size_elements, dtype=dtype).pin_memory()

    indices = torch.randperm(pool_size, dtype=torch.int64, device=device)[:num_items]

    # Reference
    dst_ref = dst_host.clone()
    for i in indices.tolist():
        dst_ref[i] = src_dev[i].cpu()

    transfer_mamba_state(src_dev, dst_host, indices, indices, item_size=item_size_bytes)
    torch.accelerator.synchronize()

    torch.testing.assert_close(dst_host, dst_ref)


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_items", [1, 8, 32])
@pytest.mark.parametrize("item_size_bytes", [524288, 1048576])  # 512 KB, 1 MB
def test_transfer_mamba_state_host_to_device(
    dtype: torch.dtype,
    num_items: int,
    item_size_bytes: int,
):
    """Test Mamba state transfer: Host → Device (HiCache restore/read L2→L1)."""
    torch.manual_seed(42)
    item_size_elements = item_size_bytes // dtype.itemsize
    pool_size = 128

    src_host = torch.randn(pool_size, item_size_elements, dtype=dtype).pin_memory()
    dst_dev = torch.zeros(pool_size, item_size_elements, dtype=dtype, device=device)

    indices = torch.randperm(pool_size, dtype=torch.int64, device=device)[:num_items]

    # Reference
    dst_ref = dst_dev.clone()
    for i in indices.tolist():
        dst_ref[i] = src_host[i].to(device)

    transfer_mamba_state(src_host, dst_dev, indices, indices, item_size=item_size_bytes)
    torch.accelerator.synchronize()

    torch.testing.assert_close(dst_dev, dst_ref)


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_items", [1, 16, 64])
@pytest.mark.parametrize(
    "item_size_bytes",
    [
        131072,   # 128 KB — small Mamba (mamba-1.4b)
        1048576,  # 1 MB — Qwen3.6-35B Mamba (d_inner=2048, d_state=256)
        1572864,  # 1.5 MB — Falcon-H1-1.5B Mamba (d_inner=3072, d_state=256)
        3145728,  # 3 MB — Falcon-H1-7B equivalent
    ],
)
def test_transfer_mamba_state(
    dtype: torch.dtype,
    num_items: int,
    item_size_bytes: int,
):
    """
    Test the new Mamba state transfer kernel designed for large item sizes.

    This kernel uses work-group cooperative copy (256 work-items per token)
    instead of sub-group parallelism (16 lanes per token) to handle
    Mamba's large temporal state (~1-3 MB per token).

    Model-derived shapes:
    - Qwen3.6-35B: d_inner=2048, d_state=256, dtype=bf16 → 1 MB/token
    - Falcon-H1-1.5B: d_inner=3072, d_state=256, dtype=bf16 → 1.5 MB/token
    """
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)

    torch.manual_seed(42)

    item_size_elements = item_size_bytes // dtype.itemsize
    total_pool_size = 256

    src_pool = torch.randn(total_pool_size, item_size_elements).pin_memory()
    dst_pool_ref = torch.zeros(total_pool_size, item_size_elements, device=device)
    dst_pool_kernel = torch.zeros_like(dst_pool_ref)

    indices = torch.randperm(total_pool_size, dtype=torch.int64)[:num_items]
    src_indices = indices.to(device)
    dst_indices = indices.to(device)

    for si, di in zip(indices.tolist(), indices.tolist()):
        dst_pool_ref[di] = src_pool[si].to(device)

    transfer_mamba_state(
        src=src_pool,
        dst=dst_pool_kernel,
        src_indices=src_indices,
        dst_indices=dst_indices,
        item_size=item_size_bytes,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(dst_pool_kernel, dst_pool_ref)

    torch.set_default_dtype(original_dtype)


@pytest.mark.skipif(is_hip(), reason="HIP is not supported for this test")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_items", [16])
@pytest.mark.parametrize("num_layers", [4, 8])
@pytest.mark.parametrize("item_size_bytes", [1048576])  # 1 MB — Qwen3.6-35B
def test_transfer_mamba_state_all_layer(
    dtype: torch.dtype,
    num_items: int,
    num_layers: int,
    item_size_bytes: int,
):
    """Test multi-layer Mamba state transfer via layer pointer table."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)

    torch.manual_seed(42)

    item_size_elements = item_size_bytes // dtype.itemsize
    total_pool_size = 256

    src_pools = [
        torch.randn(total_pool_size, item_size_elements).pin_memory()
        for _ in range(num_layers)
    ]
    dst_pools = [
        torch.zeros(total_pool_size, item_size_elements, device=device)
        for _ in range(num_layers)
    ]
    dst_ref = [
        torch.zeros(total_pool_size, item_size_elements, device=device)
        for _ in range(num_layers)
    ]

    # Create pointer tensors on CPU first to avoid overflow when converting
    # large unsigned device addresses. Kernel expects uint64.
    src_ptrs = torch.tensor(
        [p.data_ptr() for p in src_pools], dtype=torch.uint64
    ).to(device)
    dst_ptrs = torch.tensor(
        [p.data_ptr() for p in dst_pools], dtype=torch.uint64
    ).to(device)

    indices = torch.randperm(total_pool_size, dtype=torch.int64)[:num_items]
    src_indices = indices.to(device)
    dst_indices = indices.to(device)

    for layer in range(num_layers):
        for si, di in zip(indices.tolist(), indices.tolist()):
            dst_ref[layer][di] = src_pools[layer][si].to(device)

    transfer_mamba_state_all_layer(
        src_layers=src_ptrs,
        dst_layers=dst_ptrs,
        src_indices=src_indices,
        dst_indices=dst_indices,
        item_size=item_size_bytes,
        num_layers=num_layers,
    )
    torch.accelerator.synchronize()

    for layer in range(num_layers):
        torch.testing.assert_close(dst_pools[layer], dst_ref[layer])

    torch.set_default_dtype(original_dtype)


if __name__ == "__main__":
    pytest.main([__file__])
