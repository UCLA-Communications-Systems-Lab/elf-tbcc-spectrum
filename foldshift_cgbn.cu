#include <cuda_runtime.h>
#include <stdint.h>
#include <cgbn/cgbn.h>

// threads per instance (4, 8, 16, or 32)
// number of threads that handle each integer (multiple threads are required for larger ints)
#define TPI 8
#define BITS 256
#define INSTANCES_PER_WARP (32 / TPI)

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef cgbn_mem_t<BITS> bn_mem_t; // storage type (for both global & shared memory)

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

// foldshift kernel with CGBN
__global__ void cgbn_sharedMem_trellisStep_foldshift(
    bn_mem_t* __restrict__ A_in, int A_dim0, int A_dim1,
    const uint8_t* __restrict__ W_in, int W_dim0, int W_dim1,
    const uint32_t* __restrict__ D_in, int D_dim0, int D_dim1,
    bn_mem_t* __restrict__ out, int out_dim0, int out_dim1, int curr_max_weight
) {
    context_t bn_context;
    env_t bn_env(bn_context);
    typedef typename env_t::cgbn_t bn_t;

    uint32_t num_states = A_dim0;

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    uint32_t bs_x = blockDim.x;
    uint32_t bs_y = blockDim.y;

    uint32_t block_instances = bs_x / TPI;

    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;

    uint32_t instance_x = tx / TPI;

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

    uint32_t num_blk_iters = (curr_max_weight + block_instances - 1) / block_instances;
    
    __shared__ bn_mem_t shared_A[64][INSTANCES_PER_WARP];
    __shared__ bn_mem_t shared_out[64][INSTANCES_PER_WARP + 2];

    // implement per-thread carry register for overflow columns (minimize loads from global)
    bn_t carry;
    cgbn_set_ui32(bn_env, carry, 0);

    for (int i_blkiter = 0; i_blkiter < num_blk_iters; ++i_blkiter) {
        int x_id = instance_x + i_blkiter * block_instances;

        // Load A into shared memory
        // storage requirement: 64*32*8 = 16,384 or 64*32*16 = 32,768
        bn_t a_val;
        if (y < mid_y && x_id < curr_max_weight) {
            cgbn_load(bn_env, a_val, &A_in[y * A_dim1 + x_id]);
        } else {
            cgbn_set_ui32(bn_env, a_val, 0);
        }
        cgbn_store(bn_env, &shared_A[ty][instance_x], a_val);

        // shared_A[ty + bs_y][tx] = A_in[(y + mid_y) * A_dim1 + x_id];
        if (y < mid_y && x_id < curr_max_weight) {
            cgbn_load(bn_env, a_val, &A_in[(y + mid_y) * A_dim1 + x_id]);
        } else {
            cgbn_set_ui32(bn_env, a_val, 0);
        }
        cgbn_store(bn_env, &shared_A[ty + bs_y][instance_x], a_val);

        bn_t zero;
        cgbn_set_ui32(bn_env, zero, 0);

        // initialize shared out with zeros
        cgbn_store(bn_env, &shared_out[ty][instance_x], zero);
        cgbn_store(bn_env, &shared_out[ty + bs_y][instance_x], zero);

        if (instance_x < 2) {
            cgbn_store(bn_env, &shared_out[ty][instance_x + block_instances], zero);
            cgbn_store(bn_env, &shared_out[ty + bs_y][instance_x + block_instances], zero);
        }

        __syncthreads();

        // inject carry overflow
        if (instance_x < 2) {
            cgbn_store(bn_env, &shared_out[2 * ty + z][instance_x], carry);
        }

        __syncthreads();

        uint32_t dst_state_offset = shared_D[0] - z;

        bn_t a_elem, out_elem;
        if (y < mid_y && x_id < A_dim1) {
            uint8_t shift_amt_0 = shared_W[ty];
            uint32_t dst_state_0 = shared_D[ty] - dst_state_offset;
            int shifted_x_0 = instance_x + shift_amt_0;

            // shared_out[dst_state_0][shifted_x_0] += shared_A[ty][tx];
            cgbn_load(bn_env, a_elem, &shared_A[ty][instance_x]);
            cgbn_load(bn_env, out_elem, &shared_out[dst_state_0][shifted_x_0]);

            cgbn_add(bn_env, out_elem, out_elem, a_elem);
            cgbn_store(bn_env, &shared_out[dst_state_0][shifted_x_0], out_elem);
        }

        __syncthreads();

        if (y < mid_y && x_id < A_dim1) {
            uint8_t shift_amt_1 = shared_W[ty + bs_y];
            uint32_t dst_state_1 = shared_D[ty + bs_y] - dst_state_offset;
            int shifted_x_1 = instance_x + shift_amt_1;

            // shared_out[dst_state_1][shifted_x_1] += shared_A[ty + bs_y][tx];
            cgbn_load(bn_env, a_elem, &shared_A[ty + bs_y][instance_x]);
            cgbn_load(bn_env, out_elem, &shared_out[dst_state_1][shifted_x_1]);

            cgbn_add(bn_env, out_elem, out_elem, a_elem);
            cgbn_store(bn_env, &shared_out[dst_state_1][shifted_x_1], out_elem);
        }

        __syncthreads();

        if (instance_x < 2) {
            // carry = shared_out[2 * ty + z][tx + bs_x];
            cgbn_load(bn_env, carry, &shared_out[2 * ty + z][instance_x + block_instances]);
        }

        if (x_id < curr_max_weight + 2) {
            // out[(2 * y + z) * out_dim1 + x_id] = shared_out[2 * ty + z][tx];
            bn_t result;
            cgbn_load(bn_env, result, &shared_out[2 * ty + z][instance_x]);
            cgbn_store(bn_env, &out[(2 * y + z) * out_dim1 + x_id], result);
        }

        __syncthreads();
    }

    // flush carry 
    if (instance_x < 2) {
        int carry_x = num_blk_iters * block_instances + instance_x;
        if (carry_x < curr_max_weight + 2 && cgbn_compare_ui32(bn_env, carry, 0) != 0) {
            // out[(2 * y + z) * out_dim1 + carry_x] = carry;
            cgbn_store(bn_env, &out[(2 * y + z) * out_dim1 + carry_x], carry);
        }
    }
}

// launch wrapper so we can have python driver 
extern "C" void launchFoldshiftPipeline (
    uint64_t* d_buffer_a, uint64_t* d_buffer_b,
    int num_states, int initial_max_weight,
    const uint8_t* d_W, int W_dim0, int W_dim1,
    const uint32_t* d_D, int D_dim0, int D_dim1,
    int num_trellis_stages, int max_shift_per_stage, int max_X,
    int basis_state, uint64_t* d_spectrum
) {
    uint64_t* d_in = d_buffer_a;
    uint64_t* d_out = d_buffer_b;
    int curr_max_weight = initial_max_weight;
 
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
}