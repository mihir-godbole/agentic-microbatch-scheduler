"""Utility for custom CUDA kernels used by the scheduler.
All kernels are JIT-compiled with torch.utils.cpp_extension for simplicity.
"""
from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Compile kernels once per process
# -----------------------------------------------------------------------------

def _compile():
    cuda_src = r"""
    extern "C" __global__ void reduce_sum_max(const int* lengths, int n, int* sum_out, int* max_out) {
        // Shared memory per block storing partial reductions
        __shared__ int s_sum[256];
        __shared__ int s_max[256];

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;

        // Strided loop over the array
        int local_sum = 0;
        int local_max = 0;
        for (; idx < n; idx += blockDim.x * gridDim.x) {
            int val = lengths[idx];
            local_sum += val;
            if (val > local_max) local_max = val;
        }

        s_sum[tid] = local_sum;
        s_max[tid] = local_max;
        __syncthreads();

        // In-block reduction
        for (int s = blockDim.x/2; s>0; s >>= 1) {
            if (tid < s) {
                s_sum[tid] += s_sum[tid + s];
                if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(sum_out, s_sum[0]);
            atomicMax(max_out, s_max[0]);
        }
    }
    """

    return load_inline(
        name="microbatch_kernels",
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=["reduce_sum_max"],
        verbose=False,
    )


_module = _compile()

# -----------------------------------------------------------------------------
# Python wrappers
# -----------------------------------------------------------------------------

def batch_stats(lengths: torch.Tensor) -> tuple[int, int]:
    """Compute (sum, max) of int lengths on GPU using the custom kernel.

    Parameters
    ----------
    lengths: 1-D torch.int32 tensor on CUDA device
    """
    assert lengths.is_cuda and lengths.dtype == torch.int32 and lengths.dim() == 1

    n = lengths.numel()
    sum_out = torch.zeros(1, dtype=torch.int32, device="cuda")
    max_out = torch.zeros(1, dtype=torch.int32, device="cuda")

    threads = 256
    grid = (max(1, (n + threads - 1) // threads), 1, 1)

    _module.reduce_sum_max(
        lengths,
        n,
        sum_out,
        max_out,
        block=(threads, 1, 1),
        grid=grid,
    )
    # make sure kernel done
    torch.cuda.synchronize()
    return int(sum_out.item()), int(max_out.item())
