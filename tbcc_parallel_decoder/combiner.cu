#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>


// Helper device function to implement atomicMin for floats.
// This allows robust block-wide minimum tracking regardless of architecture.
__device__ __forceinline__ void atomicMinFloat(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        // Compare current value with the incoming value and only update if smaller
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

/**
 * Kernel to combine pairs of survivor trellis stages into a single stage.
 * 
 * @tparam M Number of states in the trellis (e.g., 32, 64)
 * @param input_stage_left Metric costs of the left stage pair
 * @param input_stage_right Metric costs of the right stage pair
 * @param output_stage Computed optimal metrics
 * @param output_argmin Best intermediate states (r) for traceback
 * @param N Number of independent stage pairs
 */
template <int M>
__global__ void combineTrellisStagesKernel(
    const float* __restrict__ input_stage_left,
    const float* __restrict__ input_stage_right,
    float* __restrict__ output_stage,
    int* __restrict__ output_argmin,
    int N)
{
    // i is the stage pair index. 1 block maps to 1 stage pair.
    int i = blockIdx.x;
    
    // Safety boundary check in case grid block sizing isn't a perfect fit
    if (i >= N) return;

    // Use shared memory to load the M x M matrices for cooperative caching.
    // Static allocation assumes M won't exceed hardware bounds (e.g. M=64 takes ~32KB total)
    __shared__ float smem_left[M][M];
    __shared__ float smem_right[M][M];
    
    // Shared variable to store the local minimum of the block for normalization
    __shared__ float block_min;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int tid = ty * bw + tx;

    // Initialize the block reduction target
    if (tid == 0) {
        block_min = FLT_MAX;
    }

    // Cooperative Memory Loading: 
    // Threads collaborate to load the 2D structures into __shared__ memory.
    // We use a block-stride loop to properly handle configurations where M*M > block size.
    for (int row = ty; row < M; row += bh) {
        for (int col = tx; col < M; col += bw) {
            smem_left[row][col]  = input_stage_left[i * M * M + row * M + col];
            smem_right[row][col] = input_stage_right[i * M * M + row * M + col];
        }
    }

    // Ensure all elements are loaded into __shared__ memory
    __syncthreads();

    // Variable to track the min over all computations done uniquely by this thread
    float thread_min_metric = FLT_MAX;

    // Process assigned (s, d) combination pairs, computing min path over intermediates.
    for (int s = ty; s < M; s += bh) {
        for (int d = tx; d < M; d += bw) {
            
            float min_val = FLT_MAX;
            int best_r = 0;

            // Find the optimal intermediate connecting state `r`
            for (int r = 0; r < M; r++) {
                float val = smem_left[s][r] + smem_right[r][d];
                if (val < min_val) {
                    min_val = val;
                    best_r = r;
                }
            }

            // Write results cleanly to temporary global memory spot
            // (We write it unnormalized, we will normalize it in the final loop)
            int target_idx = i * M * M + s * M + d;
            output_stage[target_idx] = min_val;
            output_argmin[target_idx] = best_r;

            // Track this local minimum for the normalization pass
            if (min_val < thread_min_metric) {
                thread_min_metric = min_val;
            }
        }
    }

    // Reduce the individual thread minimum metrics into a single block-level minimum
    atomicMinFloat(&block_min, thread_min_metric);

    // Ensure thread writes and the block minimal reduction are complete
    __syncthreads();

    // Normalization Step: 
    // Subtract the absolute minimum of the current matrix to avoid overflow over many stages.
    float b_min = block_min;
    
    for (int s = ty; s < M; s += bh) {
        for (int d = tx; d < M; d += bw) {
            int target_idx = i * M * M + s * M + d;
            // Read back and deduct block minimum, updating to normalized metric
            output_stage[target_idx] = output_stage[target_idx] - b_min;
        }
    }
}

// Host launcher – sets block/grid dims and dispatches the kernel.
template <int M>
void launchCombineTrellisStages(
    const float* d_input_stage_left,
    const float* d_input_stage_right,
    float* d_output_stage,
    int* d_output_argmin,
    int N,
    cudaStream_t stream = 0)
{
    int block_x = (M < 32) ? M : 32;
    int block_y = (M < 32) ? M : 32;
    dim3 block(block_x, block_y);
    dim3 grid(N);
    if (N > 0) {
        combineTrellisStagesKernel<M><<<grid, block, 0, stream>>>(
            d_input_stage_left,
            d_input_stage_right,
            d_output_stage,
            d_output_argmin,
            N
        );
    }
}

template void launchCombineTrellisStages< 8>(const float*, const float*, float*, int*, int, cudaStream_t);
template void launchCombineTrellisStages<16>(const float*, const float*, float*, int*, int, cudaStream_t);
template void launchCombineTrellisStages<32>(const float*, const float*, float*, int*, int, cudaStream_t);
template void launchCombineTrellisStages<64>(const float*, const float*, float*, int*, int, cudaStream_t);

// Exported C symbol – called by cuda_driver.py via ctypes
extern "C" {
    void launch_combine_kernel(
        void* d_left,
        void* d_right,
        void* d_out,
        void* d_argmin,
        int M,
        int N
    ) {
        if (M == 8) {
            launchCombineTrellisStages< 8>(
                (const float*)d_left, (const float*)d_right,
                (float*)d_out, (int*)d_argmin, N, 0);
        } else if (M == 16) {
            launchCombineTrellisStages<16>(
                (const float*)d_left, (const float*)d_right,
                (float*)d_out, (int*)d_argmin, N, 0);
        } else if (M == 32) {
            launchCombineTrellisStages<32>(
                (const float*)d_left, (const float*)d_right,
                (float*)d_out, (int*)d_argmin, N, 0);
        } else if (M == 64) {
            launchCombineTrellisStages<64>(
                (const float*)d_left, (const float*)d_right,
                (float*)d_out, (int*)d_argmin, N, 0);
        } else {
            fprintf(stderr, "launch_combine_kernel: unsupported M=%d (must be 8, 16, 32, or 64)\n", M);
        }
    }
}
