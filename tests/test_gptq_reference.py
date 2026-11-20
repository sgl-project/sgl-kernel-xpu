"""CPU unpack formula. This file is not Intel Arc Pro B70 evidence."""

import pytest
import torch

pytestmark = pytest.mark.non_device


def formula(q, stored, scale, zero_format):
    effective = stored + 1 if zero_format == 1 else stored
    return (q - effective) * scale


def test_v1_adds_one_and_v2_uses_stored():
    q = torch.tensor([0.0, 7.0, 15.0])
    stored = torch.tensor([0.0, 3.0, 8.0])
    scale = torch.tensor([0.5, 0.25, 2.0])
    v1 = formula(q, stored, scale, 1)
    v2 = formula(q, stored, scale, 2)
    assert torch.equal(v1, (q - (stored + 1)) * scale)
    assert torch.equal(v2, (q - stored) * scale)
    assert not torch.equal(v1, v2)


def test_noncontiguous_group_index_selects_scale_row():
    q = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    stored = torch.tensor([[1.0, 1.0], [4.0, 0.0]])
    scales = torch.tensor([[0.5, 1.0], [2.0, 0.25]])
    g_idx = torch.tensor([1, 0, 1])
    rows = []
    for k in range(q.shape[0]):
        group = int(g_idx[k])
        rows.append(formula(q[k], stored[group], scales[group], 2))
    got = torch.stack(rows)
    expected = torch.tensor(
        [
            [(1 - 4) * 2.0, (2 - 0) * 0.25],
            [(3 - 1) * 0.5, (4 - 1) * 1.0],
            [(5 - 4) * 2.0, (6 - 0) * 0.25],
        ]
    )
    assert torch.equal(got, expected)
