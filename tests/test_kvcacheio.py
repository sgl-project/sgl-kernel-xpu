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
    transfer_kv_per_layer_ph_lf,
)

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
    """Reference for the page-head layout, as one indexed copy.

    ph pool is [num_pages, head_num, page_size, num_layers, head_dim];
    lf pool is [num_layers, num_tokens, head_num, head_dim]. Both sides of the
    copy are [len(indices), head_num, head_dim], so all heads move at once
    (head_num is implied by the pool shapes).
    """
    src_idx = src_indices.to(src_pool.device)
    dst_idx = dst_indices.to(dst_pool.device)
    if lf_to_ph:
        dst_pool[dst_idx // page_size, :, dst_idx % page_size, layer_id] = src_pool[
            layer_id
        ][src_idx].to(dst_pool.device)
    else:
        dst_pool[layer_id][dst_idx] = src_pool[
            src_idx // page_size, :, src_idx % page_size, layer_id
        ].to(dst_pool.device)


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


def _strided_lf_pool(num_layers, num_tokens, item_size, row_stride_mult, fill):
    """lf [layers, tokens, item] view whose rows sit row_stride_mult rows apart.

    Returns (view, backing); the gap rows of `backing` stay zero so a stride bug
    that writes past a row shows up as a non-zero gap.
    """
    backing = torch.zeros(num_layers, num_tokens, row_stride_mult, item_size).to(device)
    view = backing[:, :, 0, :]
    if fill:
        view.copy_(torch.randn(num_layers, num_tokens, item_size).to(device))
    return view, backing


def _assert_gaps_untouched(backing):
    gaps = backing[:, :, 1:, :]
    torch.testing.assert_close(gaps, torch.zeros_like(gaps))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("row_stride_mult", [2, 3])
@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("lf_to_pf", [False, True])
def test_transfer_kv_pf_kernel_strided_device_rows(
    dtype: torch.dtype, row_stride_mult: int, is_mla: bool, lf_to_pf: bool
):
    """Page-first transfers whose lf (device) rows are strided, as in a unified-memory pool.

    The lf side steps by src/dst_row_stride; the pf side keeps layout_dim.
    """
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    num_layers, num_tokens, item_size, num_items = 4, 4096, 256, 512
    layer_idx_to_test = 1
    item_bytes = item_size * dtype.itemsize
    layout_dim_bytes = item_size * num_layers * dtype.itemsize
    row_stride = row_stride_mult * item_bytes
    perm = torch.randperm(num_tokens, dtype=torch.int64)
    src_indices = perm[:num_items].to(device)
    dst_indices = perm[num_items : 2 * num_items].to(device)
    num_kv = 1 if is_mla else 2

    def make_lf_tbl(pool):
        return torch.tensor(
            [pool[i].data_ptr() for i in range(num_layers)],
            dtype=torch.uint64,
            device=device,
        )

    if lf_to_pf:
        srcs = [
            _strided_lf_pool(num_layers, num_tokens, item_size, row_stride_mult, True)[
                0
            ]
            for _ in range(num_kv)
        ]
        dsts = [torch.zeros(num_tokens, num_layers, item_size).to(device) for _ in srcs]
        refs = [torch.zeros_like(d) for d in dsts]
        if is_mla:
            transfer_kv_all_layer_mla_lf_pf(
                make_lf_tbl(srcs[0]),
                dsts[0],
                src_indices,
                dst_indices,
                item_bytes,
                layout_dim_bytes,
                num_layers,
                src_row_stride=row_stride,
            )
        else:
            transfer_kv_all_layer_lf_pf(
                make_lf_tbl(srcs[0]),
                dsts[0],
                make_lf_tbl(srcs[1]),
                dsts[1],
                src_indices,
                dst_indices,
                item_bytes,
                layout_dim_bytes,
                num_layers,
                src_row_stride=row_stride,
            )
        for src, ref in zip(srcs, refs):
            for i in range(num_layers):
                ref_copy_with_indices_pf(
                    src, ref, src_indices, dst_indices, i, pf_to_lf=False
                )
        torch.accelerator.synchronize()
        for dst, ref in zip(dsts, refs):
            torch.testing.assert_close(dst, ref)
    else:
        srcs = [
            torch.randn(num_tokens, num_layers, item_size).to(device)
            for _ in range(num_kv)
        ]
        dst_pools = [
            _strided_lf_pool(num_layers, num_tokens, item_size, row_stride_mult, False)
            for _ in srcs
        ]
        refs = [torch.zeros(num_layers, num_tokens, item_size).to(device) for _ in srcs]
        dsts = [view[layer_idx_to_test] for view, _ in dst_pools]
        if is_mla:
            transfer_kv_per_layer_mla_pf_lf(
                srcs[0],
                dsts[0],
                src_indices,
                dst_indices,
                layer_idx_to_test,
                item_bytes,
                layout_dim_bytes,
                dst_row_stride=row_stride,
            )
        else:
            transfer_kv_per_layer_pf_lf(
                srcs[0],
                dsts[0],
                srcs[1],
                dsts[1],
                src_indices,
                dst_indices,
                layer_idx_to_test,
                item_bytes,
                layout_dim_bytes,
                dst_row_stride=row_stride,
            )
        for src, ref in zip(srcs, refs):
            ref_copy_with_indices_pf(
                src, ref, src_indices, dst_indices, layer_idx_to_test
            )
        torch.accelerator.synchronize()
        for (view, backing), ref in zip(dst_pools, refs):
            torch.testing.assert_close(view, ref)
            _assert_gaps_untouched(backing)
    torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("all_layers", [False, True])
@pytest.mark.parametrize("strided_side", ["src", "dst"])
def test_transfer_kv_lf_strided_rows(
    dtype: torch.dtype, is_mla: bool, all_layers: bool, strided_side: str
):
    """lf→lf transfers with one side's rows strided by src/dst_row_stride."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    num_layers, num_tokens, item_size, num_items = 4, 4096, 256, 512
    row_stride_mult = 2
    item_bytes = item_size * dtype.itemsize
    stride_kw = {f"{strided_side}_row_stride": row_stride_mult * item_bytes}
    perm = torch.randperm(num_tokens, dtype=torch.int64)
    src_indices = perm[:num_items].to(device)
    dst_indices = perm[num_items : 2 * num_items].to(device)
    num_kv = 1 if is_mla else 2

    def pool(strided, fill):
        if strided:
            return _strided_lf_pool(
                num_layers, num_tokens, item_size, row_stride_mult, fill
            )
        view = torch.zeros(num_layers, num_tokens, item_size).to(device)
        if fill:
            view.copy_(torch.randn_like(view))
        return view, None

    srcs = [pool(strided_side == "src", True)[0] for _ in range(num_kv)]
    dst_pools = [pool(strided_side == "dst", False) for _ in range(num_kv)]
    dsts = [view for view, _ in dst_pools]
    refs = [
        torch.zeros(num_layers, num_tokens, item_size).to(device) for _ in range(num_kv)
    ]
    layers = range(num_layers) if all_layers else [0]

    def tbl(view):
        return torch.tensor(
            [view[i].data_ptr() for i in range(num_layers)],
            dtype=torch.uint64,
            device=device,
        )

    if all_layers and is_mla:
        transfer_kv_all_layer_mla(
            tbl(srcs[0]),
            tbl(dsts[0]),
            src_indices,
            dst_indices,
            item_bytes,
            num_layers,
            **stride_kw,
        )
    elif all_layers:
        transfer_kv_all_layer(
            tbl(srcs[0]),
            tbl(dsts[0]),
            tbl(srcs[1]),
            tbl(dsts[1]),
            src_indices,
            dst_indices,
            item_bytes,
            num_layers,
            **stride_kw,
        )
    elif is_mla:
        transfer_kv_per_layer_mla(
            srcs[0][0], dsts[0][0], src_indices, dst_indices, item_bytes, **stride_kw
        )
    else:
        transfer_kv_per_layer(
            srcs[0][0],
            dsts[0][0],
            srcs[1][0],
            dsts[1][0],
            src_indices,
            dst_indices,
            item_bytes,
            **stride_kw,
        )
    for src, ref in zip(srcs, refs):
        for i in layers:
            ref_copy_with_indices(src[i], ref[i], src_indices, dst_indices)
    torch.accelerator.synchronize()
    for (dst, backing), ref in zip(dst_pools, refs):
        torch.testing.assert_close(dst, ref)
        if backing is not None:
            _assert_gaps_untouched(backing)
    torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("all_layers", [False, True])
def test_transfer_kv_lf_strided_rows_both_sides(dtype: torch.dtype, all_layers: bool):
    """lf→lf with both sides strided by different amounts (src x2, dst x3)."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    num_layers, num_tokens, item_size, num_items = 4, 4096, 256, 512
    item_bytes = item_size * dtype.itemsize
    perm = torch.randperm(num_tokens, dtype=torch.int64)
    src_indices = perm[:num_items].to(device)
    dst_indices = perm[num_items : 2 * num_items].to(device)

    srcs = [
        _strided_lf_pool(num_layers, num_tokens, item_size, 2, True)[0]
        for _ in range(2)
    ]
    dst_pools = [
        _strided_lf_pool(num_layers, num_tokens, item_size, 3, False) for _ in range(2)
    ]
    dsts = [view for view, _ in dst_pools]
    refs = [torch.zeros(num_layers, num_tokens, item_size).to(device) for _ in range(2)]
    strides = dict(src_row_stride=2 * item_bytes, dst_row_stride=3 * item_bytes)

    def tbl(view):
        return torch.tensor(
            [view[i].data_ptr() for i in range(num_layers)],
            dtype=torch.uint64,
            device=device,
        )

    if all_layers:
        transfer_kv_all_layer(
            tbl(srcs[0]),
            tbl(dsts[0]),
            tbl(srcs[1]),
            tbl(dsts[1]),
            src_indices,
            dst_indices,
            item_bytes,
            num_layers,
            **strides,
        )
    else:
        transfer_kv_per_layer(
            srcs[0][0],
            dsts[0][0],
            srcs[1][0],
            dsts[1][0],
            src_indices,
            dst_indices,
            item_bytes,
            **strides,
        )
    for src, ref in zip(srcs, refs):
        for i in range(num_layers) if all_layers else [0]:
            ref_copy_with_indices(src[i], ref[i], src_indices, dst_indices)
    torch.accelerator.synchronize()
    for (dst, backing), ref in zip(dst_pools, refs):
        torch.testing.assert_close(dst, ref)
        _assert_gaps_untouched(backing)
    torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("lf_to_ph", [False, True])
def test_transfer_kv_page_head_strided_device_rows(dtype: torch.dtype, lf_to_ph: bool):
    """Page-head transfers whose lf (device) rows are strided; the ph side keeps layout_dim."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    num_layers, num_tokens, item_size, head_num, page_size = 4, 1024, 1024, 8, 16
    num_pages_to_transfer, row_stride_mult = 4, 2
    head_dim = item_size // head_num
    num_pages = num_tokens // page_size
    item_bytes = item_size * dtype.itemsize
    layout_dim_bytes = item_size * num_layers * dtype.itemsize
    row_stride = row_stride_mult * item_bytes
    layer_idx_to_test = 1

    pages = torch.randperm(num_pages, dtype=torch.int64)

    def tokens(ps):
        return torch.cat([torch.arange(p * page_size, (p + 1) * page_size) for p in ps])

    src_indices = tokens(pages[:num_pages_to_transfer])
    dst_indices = tokens(pages[num_pages_to_transfer : 2 * num_pages_to_transfer])
    ph_shape = (num_pages, head_num, page_size, num_layers, head_dim)

    if lf_to_ph:
        srcs = [
            _strided_lf_pool(num_layers, num_tokens, item_size, row_stride_mult, True)[
                0
            ].unflatten(-1, (head_num, head_dim))
            for _ in range(2)
        ]
        dsts = [torch.zeros(ph_shape).pin_memory() for _ in range(2)]
        refs = [torch.zeros(ph_shape) for _ in range(2)]
        tbls = [
            torch.tensor(
                [src[i].data_ptr() for i in range(num_layers)],
                dtype=torch.uint64,
                device=device,
            )
            for src in srcs
        ]
        transfer_kv_all_layer_lf_ph(
            tbls[0],
            dsts[0],
            tbls[1],
            dsts[1],
            src_indices.to(device),
            dst_indices.to(device),
            item_bytes,
            layout_dim_bytes,
            num_layers,
            page_size,
            head_num,
            src_row_stride=row_stride,
        )
        for src, ref in zip(srcs, refs):
            for i in range(num_layers):
                ref_copy_with_indices_page_head(
                    src.cpu(),
                    ref,
                    src_indices,
                    dst_indices,
                    page_size,
                    i,
                    head_num,
                    lf_to_ph=True,
                )
        torch.accelerator.synchronize()
        for dst, ref in zip(dsts, refs):
            torch.testing.assert_close(dst, ref)
    else:
        srcs = [torch.randn(ph_shape).pin_memory() for _ in range(2)]
        dst_pools = [
            _strided_lf_pool(num_layers, num_tokens, item_size, row_stride_mult, False)
            for _ in range(2)
        ]
        views = [view.unflatten(-1, (head_num, head_dim)) for view, _ in dst_pools]
        refs = [
            torch.zeros(num_layers, num_tokens, head_num, head_dim) for _ in range(2)
        ]
        transfer_kv_per_layer_ph_lf(
            srcs[0],
            views[0][layer_idx_to_test],
            srcs[1],
            views[1][layer_idx_to_test],
            src_indices.to(device),
            dst_indices.to(device),
            layer_idx_to_test,
            item_bytes,
            layout_dim_bytes,
            page_size,
            head_num,
            dst_row_stride=row_stride,
        )
        for src, ref in zip(srcs, refs):
            ref_copy_with_indices_page_head(
                src,
                ref,
                src_indices,
                dst_indices,
                page_size,
                layer_idx_to_test,
                head_num,
                lf_to_ph=False,
            )
        torch.accelerator.synchronize()
        for view, (_, backing), ref in zip(views, dst_pools, refs):
            torch.testing.assert_close(view.cpu(), ref)
            _assert_gaps_untouched(backing)
    torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_transfer_kv_direct_strided_device_rows(dtype: torch.dtype):
    """The Python copy_ direct pf ops follow tensor strides, so strided lf views just work."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    num_layers, num_tokens, item_size, page_size, num_pages_to_transfer = (
        4,
        1024,
        256,
        16,
        8,
    )
    num_pages = num_tokens // page_size
    pages = torch.randperm(num_pages, dtype=torch.int64)

    def tokens(ps):
        return torch.cat([torch.arange(p * page_size, (p + 1) * page_size) for p in ps])

    src_indices = tokens(pages[:num_pages_to_transfer])
    dst_indices = tokens(pages[num_pages_to_transfer : 2 * num_pages_to_transfer])
    pf_shape = (num_pages, num_layers, page_size, item_size)

    # lf (strided device) -> pf (host), all layers
    lf = [
        _strided_lf_pool(num_layers, num_tokens, item_size, 2, True)[0]
        for _ in range(2)
    ]
    pf = [torch.zeros(pf_shape).pin_memory() for _ in range(2)]
    transfer_kv_all_layer_direct_lf_pf(
        [lf[0][i] for i in range(num_layers)] + [lf[1][i] for i in range(num_layers)],
        pf,
        src_indices,
        dst_indices,
        page_size,
    )
    torch.accelerator.synchronize()
    for src, dst in zip(lf, pf):
        flat = dst.permute(1, 0, 2, 3).reshape(num_layers, num_tokens, item_size)
        torch.testing.assert_close(
            flat[:, dst_indices], src[:, src_indices.to(device)].cpu()
        )

    # pf (host) -> lf (strided device), one layer
    layer_id = 1
    dst_pools = [
        _strided_lf_pool(num_layers, num_tokens, item_size, 2, False) for _ in range(2)
    ]
    transfer_kv_per_layer_direct_pf_lf(
        pf,
        [dst_pools[0][0][layer_id], dst_pools[1][0][layer_id]],
        dst_indices,
        src_indices,
        layer_id,
        page_size,
    )
    torch.accelerator.synchronize()
    for src, (view, backing) in zip(lf, dst_pools):
        idx = src_indices.to(device)
        ref = torch.zeros_like(src)
        ref[layer_id][idx] = src[layer_id][idx]
        torch.testing.assert_close(view, ref)
        _assert_gaps_untouched(backing)
    torch.set_default_dtype(original_dtype)


def test_transfer_kv_pf_ph_side_row_stride_rejected():
    """A row stride on the pf/ph side would be ignored, so it is rejected."""
    dtype = torch.bfloat16
    num_layers, num_tokens, item_size, head_num, page_size = 2, 64, 256, 8, 16
    item_bytes = item_size * dtype.itemsize
    layout_dim = item_bytes * num_layers
    idx = torch.arange(page_size, dtype=torch.int64, device=device)
    lf = torch.zeros(num_layers, num_tokens, item_size, dtype=dtype, device=device)
    tbl = torch.tensor(
        [lf[i].data_ptr() for i in range(num_layers)], dtype=torch.uint64, device=device
    )
    pf = torch.zeros(num_tokens, num_layers, item_size, dtype=dtype, device=device)
    ph = torch.zeros(
        num_tokens // page_size,
        head_num,
        page_size,
        num_layers,
        item_size // head_num,
        dtype=dtype,
        device=device,
    )
    bad = 2 * item_bytes
    calls = {
        "dst_row_stride": [
            lambda: transfer_kv_all_layer_lf_pf(
                tbl,
                pf,
                tbl,
                pf,
                idx,
                idx,
                item_bytes,
                layout_dim,
                num_layers,
                dst_row_stride=bad,
            ),
            lambda: transfer_kv_all_layer_mla_lf_pf(
                tbl,
                pf,
                idx,
                idx,
                item_bytes,
                layout_dim,
                num_layers,
                dst_row_stride=bad,
            ),
            lambda: transfer_kv_all_layer_lf_ph(
                tbl,
                ph,
                tbl,
                ph,
                idx,
                idx,
                item_bytes,
                layout_dim,
                num_layers,
                page_size,
                head_num,
                dst_row_stride=bad,
            ),
        ],
        "src_row_stride": [
            lambda: transfer_kv_per_layer_pf_lf(
                pf,
                lf[0],
                pf,
                lf[0],
                idx,
                idx,
                0,
                item_bytes,
                layout_dim,
                src_row_stride=bad,
            ),
            lambda: transfer_kv_per_layer_mla_pf_lf(
                pf, lf[0], idx, idx, 0, item_bytes, layout_dim, src_row_stride=bad
            ),
            lambda: transfer_kv_per_layer_ph_lf(
                ph,
                lf[0],
                ph,
                lf[0],
                idx,
                idx,
                0,
                item_bytes,
                layout_dim,
                page_size,
                head_num,
                src_row_stride=bad,
            ),
        ],
    }
    for name, fns in calls.items():
        for fn in fns:
            with pytest.raises(RuntimeError, match=f"{name} must be 0"):
                fn()


def test_transfer_kv_invalid_row_stride():
    """A row stride must be 0 (packed) or an 8-byte multiple no smaller than item_size."""
    item_size = 256
    item_bytes = item_size * torch.bfloat16.itemsize
    src = torch.randn(64, item_size, dtype=torch.bfloat16).to(device)
    dst = torch.zeros_like(src)
    idx = torch.arange(8, dtype=torch.int64, device=device)
    for bad in (item_bytes - 8, item_bytes + 4):
        with pytest.raises(RuntimeError, match="row_stride"):
            transfer_kv_per_layer_mla(
                src, dst, idx, idx, item_bytes, src_row_stride=bad
            )


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


if __name__ == "__main__":
    pytest.main([__file__])
