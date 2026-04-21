#include <cuda_runtime.h>
#include <stdint.h>

/*
 * Kernel: accumulate_to_spectrum
 * Flattens:
 * buffer: 2D array [buffer_dim0][buffer_dim1], accessed as buffer[state_idx * buffer_dim1 + x]
 * spectrum: 1D array [buffer_dim1], accessed as spectrum[x]
 */
template <typename T>
__global__ void accumulate_to_spectrum(
    const T* buffer, int buffer_dim0, int buffer_dim1, 
    int state_idx, 
    T* spectrum
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int max_X = buffer_dim1;

    if (x < max_X) {
        atomicAdd(&spectrum[x], buffer[state_idx * max_X + x]);
    }
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

/*
 * Kernel: numba_sharedMem_trellisStep_foldshift
 * Flattens:
 * A_in: 2D array [A_dim0][A_dim1] -> A_in[y * A_dim1 + x]
 * W_in: 2D array [W_dim0][W_dim1] -> W_in[y * W_dim1 + z]
 * D_in: 2D array [D_dim0][D_dim1] -> D_in[y * D_dim1 + z]
 * out: 2D array [out_dim0][out_dim1] -> out[(2 * y + z) * out_dim1 + x]
 */
template <typename T>
__global__ void numba_sharedMem_trellisStep_foldshift(
    const T* A_in, int A_dim0, int A_dim1,
    const uint8_t* W_in, int W_dim0, int W_dim1,
    const uint32_t* D_in, int D_dim0, int D_dim1,
    T* out, int out_dim0, int out_dim1
) {
    int num_states = A_dim0;
    int curr_max_weight = A_dim1;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int bs_x = blockDim.x;
    int bs_y = blockDim.y;
    // bs_z is unused as ty varies depending on blockDim.y

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // tz is unused

    int mid_y = num_states / 2;

    __shared__ uint8_t shared_W[64];
    if (ty < bs_y) {
        shared_W[ty] = 0;
        shared_W[ty + bs_y] = 0;
    }
    if (y < mid_y && ty < bs_y && z < W_dim1) {
        shared_W[ty] = W_in[y * W_dim1 + z];
        shared_W[ty + bs_y] = W_in[(y + mid_y) * W_dim1 + z];
    }

    __shared__ uint32_t shared_D[64];
    if (ty < bs_y) {
        shared_D[ty] = 0;
        shared_D[ty + bs_y] = 0;
    }
    if (y < mid_y && ty < bs_y && z < D_dim1) {
        shared_D[ty] = D_in[y * D_dim1 + z];
        shared_D[ty + bs_y] = D_in[(y + mid_y) * D_dim1 + z];
    }

    int num_blk_iters = (curr_max_weight + bs_x - 1) / bs_x;

    for (int i_blkiter = 0; i_blkiter < num_blk_iters; ++i_blkiter) {
        int x_id = x + i_blkiter * bs_x;

        __shared__ T shared_A[64][32];
        if (ty < bs_y && tx < bs_x) {
            shared_A[ty][tx] = 0;
            shared_A[ty + bs_y][tx] = 0;
        }
        if (y < mid_y && ty < bs_y && x_id < curr_max_weight) {
            shared_A[ty][tx] = A_in[y * A_dim1 + x_id];
            shared_A[ty + bs_y][tx] = A_in[(y + mid_y) * A_dim1 + x_id];
        }

        __shared__ T shared_out[64][34];
        if (ty < bs_y && tx < bs_x) {
            shared_out[ty][tx] = 0;
            shared_out[ty + bs_y][tx] = 0;
        }
        if (ty < bs_y && tx < 2) {
            shared_out[ty][tx + bs_x] = 0;
            shared_out[ty + bs_y][tx + bs_x] = 0;
        }

        __syncthreads();

        uint32_t dst_state_offset = shared_D[0] - z;

        if (y < mid_y && ty < bs_y && x_id < A_dim1) {
            uint8_t shift_amt_0 = shared_W[ty];
            uint32_t dst_state_0 = shared_D[ty] - dst_state_offset;
            int shifted_x_0 = tx + shift_amt_0;
            shared_out[dst_state_0][shifted_x_0] += shared_A[ty][tx];
        }

        __syncthreads();

        if (y < mid_y && ty < bs_y && x_id < A_dim1) {
            uint8_t shift_amt_1 = shared_W[ty + bs_y];
            uint32_t dst_state_1 = shared_D[ty + bs_y] - dst_state_offset;
            int shifted_x_1 = tx + shift_amt_1;
            shared_out[dst_state_1][shifted_x_1] += shared_A[ty + bs_y][tx];
        }

        __syncthreads();

        if (x_id < curr_max_weight + 2) {
            out[(2 * y + z) * out_dim1 + x_id] += shared_out[2 * ty + z][tx];
            if (tx < 2 && (x_id + bs_x) < (curr_max_weight + 2)) {
                out[(2 * y + z) * out_dim1 + (x_id + bs_x)] += shared_out[2 * ty + z][tx + bs_x];
            }
        }

        __syncthreads();
    }
}
