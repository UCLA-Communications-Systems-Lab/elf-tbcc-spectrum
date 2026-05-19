#include <cuda_runtime.h>
#include <stdint.h>

template <typename T>
__global__ void accumulate_to_spectrum(
    const T* buffer, int buffer_dim0, int buffer_dim1, 
    int state_idx, 
    T* spectrum
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int max_X = buffer_dim1;

    if (x < max_X) {
        atomicAdd((unsigned long long*)&spectrum[x], buffer[state_idx * max_X + x]);
    }
}

// standard cuda implementation of foldshift kernel (from distance_spectrum.cu)
// added a couple optimizations 
// (removed dead branch condition checking, minimized global memory touches, etc.)
template <typename T>
__global__ void numba_sharedMem_trellisStep_foldshift(
    const T* __restrict__ A_in, int A_dim0, int A_dim1,
    const uint8_t* __restrict__ W_in, int W_dim0, int W_dim1,
    const uint32_t* __restrict__ D_in, int D_dim0, int D_dim1,
    T* __restrict__ out, int out_dim0, int out_dim1, int curr_max_weight
) {
    uint32_t num_states = A_dim0;

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    uint32_t bs_x = blockDim.x;
    uint32_t bs_y = blockDim.y;

    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;

    uint32_t mid_y = num_states / 2;

    // Load W into shared memory
    // we process 64 states for each thread block
    // storage requirement: 64*2*1 = 128 bytes
    __shared__ uint8_t shared_W[64];

    // initialize shared_W to zero
    shared_W[ty] = 0;
    shared_W[ty + bs_y] = 0;

    // set W
    if (y < mid_y && z < W_dim1) {
        shared_W[ty] = W_in[y * W_dim1 + z];
        shared_W[ty + bs_y] = W_in[(y + mid_y) * W_dim1 + z];
    }

    // Load D into shared memory
    // we process 64 states for each thread block
    // storage requirement: 64*2*4 = 512 bytes (32-bit integers)
    __shared__ uint32_t shared_D[64];

    shared_D[ty] = 0;
    shared_D[ty + bs_y] = 0;

    if (y < mid_y && z < D_dim1) {
        shared_D[ty] = D_in[y * D_dim1 + z];
        shared_D[ty + bs_y] = D_in[(y + mid_y) * D_dim1 + z];
    }

    uint32_t num_blk_iters = (curr_max_weight + bs_x - 1) / bs_x;

    __shared__ T shared_A[64][32];
    __shared__ T shared_out[64][34];

    // implement per-thread carry register for overflow columns (minimize loads from global)
    T carry = 0;
    for (int i_blkiter = 0; i_blkiter < num_blk_iters; ++i_blkiter) {
        int x_id = x + i_blkiter * bs_x;

        // Load A into shared memory
        // storage requirement: 64*32*8 = 16,384 or 64*32*16 = 32,768
        shared_A[ty][tx] = 0;
        shared_A[ty + bs_y][tx] = 0;

        if (y < mid_y && x_id < curr_max_weight) {
            shared_A[ty][tx] = A_in[y * A_dim1 + x_id];
            shared_A[ty + bs_y][tx] = A_in[(y + mid_y) * A_dim1 + x_id];
        }

        // create smem output
        shared_out[ty][tx] = 0;
        shared_out[ty + bs_y][tx] = 0;

        if (tx < 2) {
            shared_out[ty][tx + bs_x] = 0;
            shared_out[ty + bs_y][tx + bs_x] = 0;
        }

        __syncthreads();

        // inject carry overflow
        if (tx < 2) {
            shared_out[2 * ty + z][tx] = carry;
        }

        __syncthreads();

        uint32_t dst_state_offset = shared_D[0] - z;

        if (y < mid_y && x_id < A_dim1) {
            uint8_t shift_amt_0 = shared_W[ty];
            uint32_t dst_state_0 = shared_D[ty] - dst_state_offset;
            int shifted_x_0 = tx + shift_amt_0;
            shared_out[dst_state_0][shifted_x_0] += shared_A[ty][tx];
        }

        __syncthreads();

        if (y < mid_y && x_id < A_dim1) {
            uint8_t shift_amt_1 = shared_W[ty + bs_y];
            uint32_t dst_state_1 = shared_D[ty + bs_y] - dst_state_offset;
            int shifted_x_1 = tx + shift_amt_1;
            shared_out[dst_state_1][shifted_x_1] += shared_A[ty + bs_y][tx];
        }

        __syncthreads();

        if (tx < 2) {
            carry = shared_out[2 * ty + z][tx + bs_x];
        }

        if (x_id < curr_max_weight + 2) {
            out[(2 * y + z) * out_dim1 + x_id] = shared_out[2 * ty + z][tx];
        }

        __syncthreads();
    }

    // flush carry 
    if (tx < 2) {
        int carry_x = num_blk_iters * bs_x + tx;
        if (carry_x < curr_max_weight + 2 && carry != 0) {
            out[(2 * y + z) * out_dim1 + carry_x] = carry;
        }
    }
}

// launch wrapper so we can have python driver 
extern "C" void launchFoldshiftPipeline (
    int starting_state,
    int num_states, int initial_max_weight,
    const uint8_t* d_W, int W_dim0, int W_dim1,
    const uint32_t* d_D, int D_dim0, int D_dim1,
    int num_trellis_stages, int max_shift_per_stage, int max_X,
    int basis_state, uint64_t* d_spectrum
) {
    int curr_max_weight = initial_max_weight;

    uint64_t* d_buffer_a;
    uint64_t* d_buffer_b;
    size_t allocation_size = (size_t)sizeof(uint64_t) * (size_t)max_X * (size_t)(num_states);

    cudaMalloc(&d_buffer_a, allocation_size);
    cudaMalloc(&d_buffer_b, allocation_size);

    cudaMemset(d_buffer_a, 0, allocation_size);
    cudaMemset(d_buffer_b, 0, allocation_size);

    uint64_t one_value = 1;
    cudaMemcpy(&d_buffer_a[starting_state * max_X], &one_value, sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t* d_in = d_buffer_a;
    uint64_t* d_out = d_buffer_b;
 
    for (int stage = 0; stage < num_trellis_stages; ++stage) {
        cudaMemset(d_out, 0, sizeof(uint64_t) * num_states * max_X);
 
        dim3 block(32, 32, 1);
        dim3 grid(1, (num_states / 2 + 31) / 32, W_dim1);
 
        numba_sharedMem_trellisStep_foldshift<uint64_t><<<grid, block>>>(
            d_in,  num_states, max_X,
            d_W,   W_dim0, W_dim1,
            d_D,   D_dim0, D_dim1,
            d_out, num_states, max_X,
            curr_max_weight
        );
        curr_max_weight += max_shift_per_stage;
 
        uint64_t* tmp = d_in; d_in = d_out; d_out = tmp;
    }
 
    int threads = 256;
    int blocks_1d = (max_X + threads - 1) / threads;
    accumulate_to_spectrum<uint64_t><<<blocks_1d, threads>>>(
        d_in, num_states, max_X, basis_state, d_spectrum
    );
    cudaDeviceSynchronize();

    cudaFree(d_buffer_a);
    cudaFree(d_buffer_b);
}

/*
 * Kernel: numba_trellisStep_conv
 * Flattens:
 * A_in: 2D array [A_dim0][A_dim1] -> A_in[y * A_dim1 + j]
 * W_in: 3D array [W_dim0][W_dim1][W_dim2] -> W_in[z * (W_dim1 * W_dim2) + y * W_dim2 + x_minus_j]
 *       Note: Dimensions of Python were [input] x [num_states] x [max weight]
 * D_in: 2D array [D_dim0][D_dim1] -> D_in[y * D_dim1 + z]
 * out: 2D array [out_dim0][out_dim1] -> out[end_state * out_dim1 + x]
 */
template <typename T>
__global__ void numba_trellisStep_conv(
    const T* A_in, int A_dim0, int A_dim1,
    const uint8_t* W_in, int W_dim0, int W_dim1, int W_dim2,
    const uint32_t* D_in, int D_dim0, int D_dim1,
    T* out, int out_dim0, int out_dim1
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z < W_dim0 && y < W_dim1 && x < (W_dim2 + A_dim1 - 1)) {
        T tmp_sum = 0;
        for (int j = 0; j < A_dim1; ++j) {
            int x_minus_j = x - j;
            if (x_minus_j >= 0 && x_minus_j < W_dim2) {
                // Equivalent to: tmp_sum += W_in[z, y, x_minus_j] * A_in[y, j]
                tmp_sum += ((T)W_in[z * W_dim1 * W_dim2 + y * W_dim2 + x_minus_j]) * A_in[y * A_dim1 + j];
            }
        }

        // Sum over all possible inputs
        uint32_t end_state = D_in[y * D_dim1 + z];
        atomicAdd(&out[end_state * out_dim1 + x], tmp_sum);
    }
}

/*
 * Kernel: numba_sharedMem_trellisStep_shift
 * Flattens:
 * A_in: 2D array [A_dim0][A_dim1] -> A_in[y * A_dim1 + x]
 * W_in: 2D array [W_dim0][W_dim1] -> W_in[y * W_dim1 + z]
 * D_in: 2D array [D_dim0][D_dim1] -> D_in[y * D_dim1 + z]
 * out: 2D array [out_dim0][out_dim1] -> out[dst_state * out_dim1 + shifted_x]
 */
template <typename T>
__global__ void numba_sharedMem_trellisStep_shift(
    const T* A_in, int A_dim0, int A_dim1,
    const uint8_t* W_in, int W_dim0, int W_dim1,
    const uint32_t* D_in, int D_dim0, int D_dim1,
    T* out, int out_dim0, int out_dim1
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    __shared__ uint8_t shared_W[32][1];
    if (y < W_dim0 && z < W_dim1) {
        shared_W[ty][tz] = W_in[y * W_dim1 + z];
    }
    __syncthreads();

    __shared__ uint32_t shared_D[32][1];
    if (y < D_dim0 && z < D_dim1) {
        shared_D[ty][tz] = D_in[y * D_dim1 + z];
    }
    __syncthreads();

    __shared__ T shared_A[32][32];
    if (y < A_dim0 && x < A_dim1) {
        shared_A[ty][tx] = A_in[y * A_dim1 + x];
    }
    __syncthreads();

    uint8_t shift_amt = shared_W[ty][tz];
    uint32_t dst_state = shared_D[ty][tz];

    int shifted_x = x + shift_amt;
    if (y < A_dim0 && x < A_dim1) {
        atomicAdd(&out[dst_state * out_dim1 + shifted_x], shared_A[ty][tx]);
    }
}