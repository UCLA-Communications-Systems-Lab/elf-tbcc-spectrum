#include <cuda_runtime.h>
#include <stdint.h>
#include <gmp.h>
#include <cgbn/cgbn.h>

// threads per instance (4, 8, 16, or 32)
// number of threads that handle each integer (multiple threads are required for larger ints)
#define TPI 4
#define BITS 128
#define INSTANCES_PER_WARP (32 / TPI)
#define MAX_BRANCH_WEIGHT 3

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef cgbn_mem_t<BITS> bn_mem_t; // storage type (for both global & shared memory)

// getters to help python execution
#pragma region
extern "C" int getCGBNBits() {
    return BITS;
}

extern "C" int getCGBNTPI() {
    return TPI;
}

extern "C" int getCGBNMemSize() {
    return sizeof(bn_mem_t);
}

extern "C" int getCGBNLimbs() {
    return BITS / 32;
}
#pragma endregion

__global__ void cgbn_accumulate_to_spectrum (
    bn_mem_t* buffer, int buffer_dim0, int buffer_dim1, 
    int state_idx, 
    bn_mem_t* spectrum
) {
    context_t bn_context;
    env_t bn_env(bn_context);
    typedef typename env_t::cgbn_t bn_t;

    int x = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    int max_X = buffer_dim1;

    bn_t spectrum_val, buffer_val;
    if (x < max_X) {
        cgbn_load(bn_env, spectrum_val, &spectrum[x]);
        cgbn_load(bn_env, buffer_val, &buffer[state_idx * max_X + x]);
        cgbn_add(bn_env, spectrum_val, spectrum_val, buffer_val);
        cgbn_store(bn_env, &spectrum[x], spectrum_val);
    }
}

// foldshift kernel with CGBN
__global__ void cgbn_sharedMem_trellisStep_foldshift(
    bn_mem_t* __restrict__ A_in, int A_dim0, int A_dim1,
    const uint8_t* __restrict__ W_in, int W_dim0, int W_dim1,
    const uint32_t* __restrict__ D_in, int D_dim0, int D_dim1,
    bn_mem_t* __restrict__ out, int out_dim0, int out_dim1, int curr_max_weight,
    int max_shift_per_stage
) {
    context_t bn_context;
    env_t bn_env(bn_context);
    typedef typename env_t::cgbn_t bn_t;

    uint32_t num_states = A_dim0;

    uint32_t block_y = blockIdx.x * gridDim.y + blockIdx.y;
    uint32_t y = block_y * blockDim.y + threadIdx.y;
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
    __shared__ bn_mem_t shared_out[64][INSTANCES_PER_WARP + MAX_BRANCH_WEIGHT];

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

        if (instance_x < max_shift_per_stage) {
            cgbn_store(bn_env, &shared_out[ty][instance_x + block_instances], zero);
            cgbn_store(bn_env, &shared_out[ty + bs_y][instance_x + block_instances], zero);
        }

        __syncthreads();

        // inject carry overflow
        if (instance_x < max_shift_per_stage) {
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

        if (instance_x < max_shift_per_stage) {
            // carry = shared_out[2 * ty + z][tx + bs_x];
            cgbn_load(bn_env, carry, &shared_out[2 * ty + z][instance_x + block_instances]);
        }

        if (x_id < curr_max_weight + max_shift_per_stage) {
            // out[(2 * y + z) * out_dim1 + x_id] = shared_out[2 * ty + z][tx];
            bn_t result;
            cgbn_load(bn_env, result, &shared_out[2 * ty + z][instance_x]);
            cgbn_store(bn_env, &out[(2 * y + z) * out_dim1 + x_id], result);
        }

        __syncthreads();
    }

    // flush carry 
    if (instance_x < max_shift_per_stage) {
        int carry_x = num_blk_iters * block_instances + instance_x;
        if (carry_x < curr_max_weight + max_shift_per_stage && cgbn_compare_ui32(bn_env, carry, 0) != 0) {
            // out[(2 * y + z) * out_dim1 + carry_x] = carry;
            cgbn_store(bn_env, &out[(2 * y + z) * out_dim1 + carry_x], carry);
        }
    }
}

// launch wrapper so we can have python driver 
extern "C" void launchCGBNPipeline (
    int starting_state,
    int num_states, int initial_max_weight,
    const uint8_t* d_W, int W_dim0, int W_dim1,
    const uint32_t* d_D, int D_dim0, int D_dim1,
    int num_trellis_stages, int max_shift_per_stage, int max_X,
    int basis_state, bn_mem_t* d_spectrum
) {
    if (max_shift_per_stage < 2 || max_shift_per_stage > MAX_BRANCH_WEIGHT) {
        return;
    }
    int curr_max_weight = initial_max_weight;

    bn_mem_t* d_bn_buffer_a;
    bn_mem_t* d_bn_buffer_b;
    size_t allocation_size = (size_t)sizeof(bn_mem_t) * (size_t)num_states * (size_t)max_X;

    cudaError_t err;
    printf("[CGBN] allocating two buffers of %.2f GB each\n", allocation_size / 1e9);

    err = cudaMallocManaged(&d_bn_buffer_a, allocation_size);
    if (err != cudaSuccess) {
        printf("[CGBN] cudaMalloc failed to allocate buffer_a, err: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemset(d_bn_buffer_a, 0, allocation_size);
    uint32_t one_value = 1;
    cudaMemcpy(&d_bn_buffer_a[starting_state * max_X]._limbs[0], &one_value, sizeof(uint32_t), cudaMemcpyHostToDevice);

    err = cudaMallocManaged(&d_bn_buffer_b, allocation_size);
    if (err != cudaSuccess) {
        printf("[CGBN] cudaMalloc failed to allocate buffer_a, err: %s\n", cudaGetErrorString(err));
        cudaFree(d_bn_buffer_a);
        return;
    }

    // for ping-pong enabling
    bn_mem_t* d_bn_in = d_bn_buffer_a;
    bn_mem_t* d_bn_out = d_bn_buffer_b;

    int total_blocks_y = (num_states / 2 + 31) / 32;
    int max_cuda_grid_dim = 65535;

    int grid_x = (total_blocks_y + max_cuda_grid_dim - 1) / max_cuda_grid_dim;
    int grid_y = (total_blocks_y + grid_x - 1) / grid_x;

    for (int stage = 0; stage < num_trellis_stages; ++stage) {
        cudaMemset(d_bn_out, 0, sizeof(bn_mem_t) * num_states * max_X);
 
        dim3 block(32, 32, 1);
        dim3 grid(grid_x, grid_y, W_dim1);
 
        cgbn_sharedMem_trellisStep_foldshift<<<grid, block>>>(
            d_bn_in, num_states, max_X,
            d_W, W_dim0, W_dim1,
            d_D, D_dim0, D_dim1,
            d_bn_out, num_states, max_X,
            curr_max_weight, max_shift_per_stage
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[CGBN] foldshift kernel failed, err %s\n", cudaGetErrorString(err));
            cudaFree(d_bn_buffer_a);
            cudaFree(d_bn_buffer_b);
            return;
        }

        curr_max_weight += max_shift_per_stage;
        // alternate buffers
        bn_mem_t* tmp = d_bn_in; d_bn_in = d_bn_out; d_bn_out = tmp;
    }
 
    int accumulate_threads = 256;
    int accumulate_blocks = ((max_X * TPI) + accumulate_threads - 1) / accumulate_threads;
    cgbn_accumulate_to_spectrum<<<accumulate_blocks, accumulate_threads>>>(
        d_bn_in, num_states, max_X, basis_state, d_spectrum
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CGBN] distance spectrum accumulate kernel failed, err: %s\n", cudaGetErrorString(err));
        cudaFree(d_bn_buffer_a);
        cudaFree(d_bn_buffer_b);
        return;
    }

    cudaFree(d_bn_buffer_a);
    cudaFree(d_bn_buffer_b);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[CGBN] device synchronization failed, err: %s\n", cudaGetErrorString(err));
        return;
    }
}
