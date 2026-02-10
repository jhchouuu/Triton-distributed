################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""
Uses mori_shmem backend.
"""

import os
import sys

os.environ.setdefault("TRITON_DIST_SHMEM_BACKEND", "mori_shmem")
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "4G")

_test_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.abspath(os.path.join(_test_dir, "../../../.."))
_triton_dist_python_path = os.path.join(_workspace_root, "python")
if _triton_dist_python_path not in sys.path:
    sys.path.insert(0, _triton_dist_python_path)
_current_pythonpath = os.environ.get("PYTHONPATH", "")
if _triton_dist_python_path not in _current_pythonpath:
    os.environ["PYTHONPATH"] = (
        f"{_triton_dist_python_path}:{_current_pythonpath}" if _current_pythonpath else _triton_dist_python_path
    )

_triton_python_path = os.path.join(_workspace_root, "3rdparty/triton/python")
if os.path.exists(_triton_python_path):
    sys.path.insert(0, _triton_python_path)

import random
import torch
import torch.distributed
from triton_dist.utils import initialize_distributed, finalize_distributed
from triton_dist.layers.amd.ep_a2a_layer import EPAll2AllLayer


def test_ep_a2a_layer_init():
    """Test EPAll2AllLayer init and reallocate_dispatch_output_buf (mori_shmem)."""
    EP_GROUP = initialize_distributed()
    RANK = EP_GROUP.rank()
    WORLD_SIZE = EP_GROUP.size()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", WORLD_SIZE))

    max_tokens = 4096
    hidden = 7168
    topk = 8
    num_tot_experts = 256
    assert num_tot_experts % WORLD_SIZE == 0, f"num_tot_experts {num_tot_experts} % WORLD_SIZE {WORLD_SIZE} != 0"

    layer = EPAll2AllLayer(
        EP_GROUP,
        max_tokens=max_tokens,
        hidden=hidden,
        topk=topk,
        rank=RANK,
        num_tot_experts=num_tot_experts,
        local_world_size=LOCAL_WORLD_SIZE,
        world_size=WORLD_SIZE,
        dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        num_sm=20,
        enable_local_combine=False,
    )
    assert layer.world_size == WORLD_SIZE
    assert layer.experts_per_rank == num_tot_experts // WORLD_SIZE

    # Test reallocate_dispatch_output_buf in same test
    ctx = layer.a2a_ctx
    init_cap = ctx.dispatch_output_buf.shape[0]
    if RANK == 0:
        print(f"  before reallocate: dispatch_output_buf {ctx.dispatch_output_buf.shape}, weight_recv_buf {ctx.weight_recv_buf.shape}")
    # Case 1: no realloc (capacity sufficient)
    small_tokens = min(1000, init_cap)
    out_buf, weight_buf = ctx.reallocate_dispatch_output_buf(small_tokens)
    if RANK == 0:
        print(f"  after reallocate(small_tokens={small_tokens}): out_buf {out_buf.shape}, weight_buf {weight_buf.shape} (no realloc)")
    assert out_buf.shape[0] == init_cap and out_buf.shape[1] == hidden
    assert out_buf.data_ptr() == ctx.dispatch_output_buf.data_ptr()
    assert weight_buf.data_ptr() == ctx.weight_recv_buf.data_ptr()
    # Case 2: realloc (need larger capacity)
    large_tokens = init_cap + 10000
    out_buf2, weight_buf2 = ctx.reallocate_dispatch_output_buf(large_tokens)
    if RANK == 0:
        print(f"  after reallocate(large_tokens={large_tokens}): out_buf {out_buf2.shape}, weight_buf {weight_buf2.shape} (realloc)")
    assert out_buf2.shape[0] >= large_tokens and out_buf2.shape[1] == hidden
    assert weight_buf2.shape[0] == out_buf2.shape[0] and weight_buf2.shape[1] == topk
    assert out_buf2.data_ptr() == ctx.dispatch_output_buf.data_ptr()
    assert weight_buf2.data_ptr() == ctx.weight_recv_buf.data_ptr()

    layer.finalize()
    finalize_distributed()
    print(f"RANK[{RANK}] EPAll2AllLayer init + reallocate_dispatch_output_buf passed.")


def _generate_random_exp_indices(token_num: int, total_num_experts: int, topk: int) -> torch.Tensor:
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for _ in range(token_num):
        top_selected = random.sample(exp_list, topk)
        exp_indices.append(top_selected)
    return torch.tensor(exp_indices, dtype=torch.int32)


def calc_scatter_index_stable(chosen_experts: torch.Tensor):
    """Compute stable scatter index by double-argsort of flattened expert indices.

    Adapted from NVIDIA test_ep_a2a.py.
    """
    return (
        chosen_experts.flatten()
        .argsort(stable=True)
        .argsort()
        .int()
        .view(chosen_experts.shape)
    )


def calc_full_scatter_indices(
    exp_indices: torch.Tensor, max_tokens: int, world_size: int
):
    """All-gather expert indices from all ranks and compute global stable scatter indices.

    Adapted from NVIDIA test_ep_a2a.py calc_full_scatter_indices.
    Each rank contributes its own exp_indices (possibly different token_num);
    we pad to max_tokens before all-gather, then trim and concatenate.
    """
    n_token = exp_indices.size(0)
    topk = exp_indices.size(1)

    # All-gather per-rank token counts
    input_len = torch.tensor(
        [n_token], dtype=torch.int32, device=exp_indices.device
    )
    ag_input_len = torch.zeros(
        world_size, dtype=torch.int32, device=exp_indices.device
    )
    torch.distributed.all_gather_into_tensor(ag_input_len, input_len)
    ag_input_len_list = ag_input_len.cpu().tolist()

    # Pad to max_tokens and all-gather expert indices
    padded_indices = torch.empty(
        [max_tokens, topk], dtype=torch.int32, device=exp_indices.device
    )
    padded_indices[:n_token] = exp_indices
    ag_padded_indices = [
        torch.empty_like(padded_indices) for _ in range(world_size)
    ]
    torch.distributed.all_gather(ag_padded_indices, padded_indices)

    # Trim padding, concatenate, compute scatter index
    ag_indices = torch.cat(
        [t[: ag_input_len_list[i], :] for i, t in enumerate(ag_padded_indices)]
    )
    return calc_scatter_index_stable(ag_indices)


def test_ep_a2a_preprocess():
    """Test EPAll2AllLayer preprocess only (mori_shmem)."""
    EP_GROUP = initialize_distributed()
    RANK = EP_GROUP.rank()
    WORLD_SIZE = EP_GROUP.size()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", WORLD_SIZE))

    max_tokens = 4096
    hidden = 7168
    topk = 8
    num_tot_experts = 256
    assert num_tot_experts % WORLD_SIZE == 0, f"num_tot_experts {num_tot_experts} % WORLD_SIZE {WORLD_SIZE} != 0"

    layer = EPAll2AllLayer(
        EP_GROUP,
        max_tokens=max_tokens,
        hidden=hidden,
        topk=topk,
        rank=RANK,
        num_tot_experts=num_tot_experts,
        local_world_size=LOCAL_WORLD_SIZE,
        world_size=WORLD_SIZE,
        dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        num_sm=20,
        enable_local_combine=False,
    )

    token_num = min(512, max_tokens)
    exp_indices = _generate_random_exp_indices(token_num, num_tot_experts, topk).to("cuda")
    input = torch.randn(token_num, hidden, device="cuda", dtype=torch.bfloat16)

    recv_buf_offset_per_expert, num_recv_tokens_per_rank, ep_a2a_layout_desc = layer.preprocess(
        input, exp_indices, full_scatter_indices=None
    )

    # Basic shape / dtype checks
    assert recv_buf_offset_per_expert.shape == (
        WORLD_SIZE,
        num_tot_experts // WORLD_SIZE,
        WORLD_SIZE,
    )
    assert num_recv_tokens_per_rank.is_cpu
    assert ep_a2a_layout_desc.num_input_tokens_per_rank.shape[0] == WORLD_SIZE

    # Reference: compute splits and expected counts on CPU
    local_splits = torch.bincount(
        exp_indices.view(-1), minlength=num_tot_experts
    ).to(torch.int32)
    full_splits = torch.empty(
        (WORLD_SIZE, num_tot_experts), dtype=torch.int32, device="cuda"
    )
    torch.distributed.all_gather_into_tensor(full_splits, local_splits)

    expected_num_input = torch.empty((WORLD_SIZE,), dtype=torch.int32, device="cuda")
    for target_rank in range(WORLD_SIZE):
        total_topk = full_splits[target_rank, :].sum().item()
        expected_num_input[target_rank] = total_topk // topk

    torch.testing.assert_close(
        ep_a2a_layout_desc.num_input_tokens_per_rank, expected_num_input
    )

    experts_per_rank = num_tot_experts // WORLD_SIZE
    recv_counts = torch.zeros(
        (WORLD_SIZE, experts_per_rank, WORLD_SIZE), dtype=torch.int32
    )
    for ep_rank in range(WORLD_SIZE):
        for eir in range(experts_per_rank):
            expert_idx = ep_rank * experts_per_rank + eir
            for src_rank in range(WORLD_SIZE):
                recv_counts[ep_rank, eir, src_rank] = full_splits[
                    src_rank, expert_idx
                ].item()

    expected_offsets = torch.zeros_like(recv_counts)
    expected_recv_tokens = torch.empty((WORLD_SIZE,), dtype=torch.int32)
    for ep_rank in range(WORLD_SIZE):
        row = recv_counts[ep_rank].view(-1).float()
        recv_tokens = int(row.sum().item())
        cusum = row.cumsum(0)
        cusum_excl = (cusum - row).to(torch.int32)
        expected_offsets[ep_rank].view(-1).copy_(cusum_excl)
        expected_recv_tokens[ep_rank] = recv_tokens

    torch.testing.assert_close(
        recv_buf_offset_per_expert.cpu(), expected_offsets
    )
    torch.testing.assert_close(
        num_recv_tokens_per_rank.cpu(), expected_recv_tokens.cpu()
    )

    layer.finalize()
    finalize_distributed()
    print(f"RANK[{RANK}] EPAll2AllLayer preprocess passed.")


def test_ep_a2a_preprocess_with_scatter_indices():
    """Test EPAll2AllLayer preprocess with full_scatter_indices path (mori_shmem).

    Verifies token_dst_scatter_idx in addition to recv_buf_offset_per_expert,
    num_recv_tokens_per_rank, and num_input_tokens_per_rank.
    Reference: NVIDIA test_ep_a2a.py --with-scatter-indices path.
    """
    EP_GROUP = initialize_distributed()
    RANK = EP_GROUP.rank()
    WORLD_SIZE = EP_GROUP.size()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", WORLD_SIZE))

    max_tokens = 4096
    hidden = 7168
    topk = 8
    num_tot_experts = 256
    assert num_tot_experts % WORLD_SIZE == 0, (
        f"num_tot_experts {num_tot_experts} % WORLD_SIZE {WORLD_SIZE} != 0"
    )

    layer = EPAll2AllLayer(
        EP_GROUP,
        max_tokens=max_tokens,
        hidden=hidden,
        topk=topk,
        rank=RANK,
        num_tot_experts=num_tot_experts,
        local_world_size=LOCAL_WORLD_SIZE,
        world_size=WORLD_SIZE,
        dtype=torch.bfloat16,
        weight_dtype=torch.float32,
        num_sm=20,
        enable_local_combine=False,
    )

    token_num = min(512, max_tokens)
    exp_indices = _generate_random_exp_indices(token_num, num_tot_experts, topk).to("cuda")
    input = torch.randn(token_num, hidden, device="cuda", dtype=torch.bfloat16)

    # Generate full_scatter_indices via all-gather + stable double-argsort
    full_scatter_indices = calc_full_scatter_indices(exp_indices, max_tokens, WORLD_SIZE)

    recv_buf_offset_per_expert, num_recv_tokens_per_rank, ep_a2a_layout_desc = (
        layer.preprocess(input, exp_indices, full_scatter_indices=full_scatter_indices)
    )

    # ----------------------------------------------------------------
    # 1. Basic shape / dtype checks (same as test_ep_a2a_preprocess)
    # ----------------------------------------------------------------
    experts_per_rank = num_tot_experts // WORLD_SIZE
    assert recv_buf_offset_per_expert.shape == (
        WORLD_SIZE,
        experts_per_rank,
        WORLD_SIZE,
    )
    assert num_recv_tokens_per_rank.is_cpu
    assert ep_a2a_layout_desc.num_input_tokens_per_rank.shape[0] == WORLD_SIZE

    # token_dst_scatter_idx must be present when full_scatter_indices is given
    token_dst_scatter_idx = ep_a2a_layout_desc.token_dst_scatter_idx
    assert token_dst_scatter_idx is not None
    nnodes = WORLD_SIZE // LOCAL_WORLD_SIZE
    assert token_dst_scatter_idx.shape == (nnodes, max_tokens, topk)

    # ----------------------------------------------------------------
    # 2. Reference: compute splits and expected quantities on CPU
    # ----------------------------------------------------------------
    local_splits = torch.bincount(
        exp_indices.view(-1), minlength=num_tot_experts
    ).to(torch.int32)
    full_splits = torch.empty(
        (WORLD_SIZE, num_tot_experts), dtype=torch.int32, device="cuda"
    )
    torch.distributed.all_gather_into_tensor(full_splits, local_splits)

    # num_input_tokens_per_rank[r] = sum(full_splits[r, :]) // topk
    expected_num_input = torch.zeros(WORLD_SIZE, dtype=torch.int32)
    for r in range(WORLD_SIZE):
        expected_num_input[r] = full_splits[r, :].sum().item() // topk

    torch.testing.assert_close(
        ep_a2a_layout_desc.num_input_tokens_per_rank.cpu(), expected_num_input
    )

    # recv_counts[ep_rank, eir, src_rank] = full_splits[src_rank, ep_rank * epr + eir]
    recv_counts = torch.zeros(
        (WORLD_SIZE, experts_per_rank, WORLD_SIZE), dtype=torch.int32
    )
    for ep_rank in range(WORLD_SIZE):
        for eir in range(experts_per_rank):
            expert_idx = ep_rank * experts_per_rank + eir
            for src_rank in range(WORLD_SIZE):
                recv_counts[ep_rank, eir, src_rank] = full_splits[
                    src_rank, expert_idx
                ].item()

    expected_offsets = torch.zeros_like(recv_counts)
    expected_recv_tokens = torch.empty((WORLD_SIZE,), dtype=torch.int32)
    for ep_rank in range(WORLD_SIZE):
        row = recv_counts[ep_rank].view(-1).float()
        recv_tokens = int(row.sum().item())
        cusum = row.cumsum(0)
        cusum_excl = (cusum - row).to(torch.int32)
        expected_offsets[ep_rank].view(-1).copy_(cusum_excl)
        expected_recv_tokens[ep_rank] = recv_tokens

    torch.testing.assert_close(
        recv_buf_offset_per_expert.cpu(), expected_offsets
    )
    torch.testing.assert_close(
        num_recv_tokens_per_rank.cpu(), expected_recv_tokens
    )

    # ----------------------------------------------------------------
    # 3. Reference: expected token_dst_scatter_idx
    #
    # Kernel logic (ep_a2a_intra_node.py lines 562-587):
    #   tokens_start = cumsum_input_tokens_per_rank[rank] * topk
    #   for token_idx in range(num_input_tokens_per_rank[rank] * topk):
    #       scatter_idx   = full_scatter_indices.flat[tokens_start + token_idx]
    #       expert_idx    = topk_indices.flat[token_idx]
    #       expert_rank   = expert_idx // experts_per_rank
    #       if expert_rank < world_size:
    #           dst[token_idx] = scatter_idx - cumsum_recv_tokens_per_rank[expert_rank]
    # ----------------------------------------------------------------

    # cumsum_input_tokens_per_rank = exclusive prefix-sum of expected_num_input
    cumsum_input = torch.zeros(WORLD_SIZE, dtype=torch.int32)
    for r in range(1, WORLD_SIZE):
        cumsum_input[r] = cumsum_input[r - 1] + expected_num_input[r - 1]

    # cumsum_recv_tokens_per_rank = exclusive prefix-sum of expected_recv_tokens
    cumsum_recv = torch.zeros(WORLD_SIZE, dtype=torch.int32)
    for r in range(1, WORLD_SIZE):
        cumsum_recv[r] = cumsum_recv[r - 1] + expected_recv_tokens[r - 1]

    # Current rank's slice in the all-gathered scatter indices
    tokens_start = cumsum_input[RANK].item() * topk
    my_scatter = full_scatter_indices.flatten().cpu()[
        tokens_start : tokens_start + token_num * topk
    ]
    my_experts = exp_indices.flatten().cpu()
    expert_ranks = my_experts // experts_per_rank

    # Vectorised reference: dst = scatter_idx - cumsum_recv[expert_rank]
    valid_mask = expert_ranks < WORLD_SIZE
    cumsum_recv_indexed = cumsum_recv[expert_ranks.clamp(max=WORLD_SIZE - 1)]
    expected_dst = my_scatter - cumsum_recv_indexed
    # Entries for dropped tokens (expert_rank >= world_size) are never written
    # by the kernel, so mask them out for comparison.
    expected_dst[~valid_mask] = 0

    # Extract actual values from (nnodes, max_tokens, topk) tensor
    actual_dst = token_dst_scatter_idx[0, :token_num, :].flatten().cpu()
    actual_dst_masked = actual_dst.clone()
    actual_dst_masked[~valid_mask] = 0

    torch.testing.assert_close(actual_dst_masked, expected_dst)

    layer.finalize()
    finalize_distributed()
    print(f"RANK[{RANK}] EPAll2AllLayer preprocess with scatter_indices passed.")


if __name__ == "__main__":
    if "--with-scatter-indices" in sys.argv:
        test_ep_a2a_preprocess_with_scatter_indices()
    else:
        test_ep_a2a_preprocess()
    print("All tests passed!")
