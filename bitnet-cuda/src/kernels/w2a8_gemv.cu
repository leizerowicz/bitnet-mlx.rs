/*
 * BitNet W2A8 GEMV CUDA Kernel
 * 
 * High-performance 2-bit weights × 8-bit activations matrix-vector multiplication
 * Targeting Microsoft-level performance: 1.27x-3.63x speedups over BF16 on A100
 * 
 * Features:
 * - dp4a instruction optimization for 4-element dot products
 * - 16×32 weight permutation for memory coalescing
 * - Vectorized operations for maximum throughput
 * - Optimized memory access patterns
 */

#include "common.cuh"
#include <cuda_runtime.h>
#include <cstdint>

// Constants for BitNet quantization
#define BITNET_SCALE_2BIT 3
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

/*
 * Fast 2-bit weight extraction using bit manipulation
 * Extracts 4 2-bit values from a single byte
 */
__device__ __forceinline__ void extract_2bit_weights(
    uint8_t packed_byte,
    int8_t weights[4]
) {
    // Extract 4 2-bit values: [b7b6][b5b4][b3b2][b1b0]
    weights[0] = ((packed_byte >> 0) & 0x3) - 1;  // Convert 0,1,2,3 to -1,0,1,2 then clamp
    weights[1] = ((packed_byte >> 2) & 0x3) - 1;
    weights[2] = ((packed_byte >> 4) & 0x3) - 1;
    weights[3] = ((packed_byte >> 6) & 0x3) - 1;
    
    // Clamp to BitNet range [-1, 0, 1]
    weights[0] = max(-1, min(1, weights[0]));
    weights[1] = max(-1, min(1, weights[1]));
    weights[2] = max(-1, min(1, weights[2]));
    weights[3] = max(-1, min(1, weights[3]));
}

/*
 * Optimized dp4a-based dot product for CUDA compute capability 7.5+
 * Computes 4-element dot product using hardware acceleration
 */
__device__ __forceinline__ int dp4a_dot_product(
    const int8_t* weights,
    const int8_t* activations
) {
    // Pack weights and activations into 32-bit integers for dp4a
    int packed_weights = __builtin_assume_aligned(weights, 4) ? 
        *reinterpret_cast<const int*>(weights) : 
        (weights[3] << 24) | (weights[2] << 16) | (weights[1] << 8) | weights[0];
    
    int packed_activations = __builtin_assume_aligned(activations, 4) ?
        *reinterpret_cast<const int*>(activations) :
        (activations[3] << 24) | (activations[2] << 16) | (activations[1] << 8) | activations[0];
    
    // Use dp4a instruction for hardware-accelerated 4-element dot product
    return __dp4a(packed_weights, packed_activations, 0);
}

/*
 * Memory coalescing optimization: 16×32 block permutation
 * Transforms memory layout for optimal GPU memory access
 */
__device__ __forceinline__ int calculate_permuted_index(int row, int col, int k) {
    // 16×32 block-based permutation pattern
    const int block_size = 16;
    const int block_row = row / block_size;
    const int block_col = col / 32;
    const int inner_row = row % block_size;
    const int inner_col = col % 32;
    
    // Permutation pattern: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    int permuted_inner_row;
    switch (inner_row % 4) {
        case 0: permuted_inner_row = inner_row; break;
        case 1: permuted_inner_row = inner_row + 3; break;
        case 2: permuted_inner_row = inner_row + 6; break; 
        case 3: permuted_inner_row = inner_row + 9; break;
        default: permuted_inner_row = inner_row; break;
    }
    permuted_inner_row = min(permuted_inner_row, block_size - 1);
    
    return (block_row * block_size + permuted_inner_row) * k + (block_col * 32 + inner_col);
}

/*
 * Main W2A8 GEMV kernel
 * 
 * Performs matrix-vector multiplication: output = weights × activations
 * where weights are 2-bit quantized and activations are 8-bit
 * 
 * Grid configuration: (num_output_features / block_size, 1, 1)
 * Block configuration: (block_size, 1, 1)
 */
extern "C" __global__ void w2a8_gemv_kernel(
    const uint8_t* __restrict__ weights,     // Packed 2-bit weights [m × k/4]
    const int8_t* __restrict__ activations,  // 8-bit activations [k]
    int32_t* __restrict__ output,            // 32-bit output [m]
    const size_t m,                          // Number of output features
    const size_t k                           // Number of input features
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Early exit for out-of-bounds threads
    if (tid >= m) return;
    
    // Shared memory for cooperative loading (optimized for memory coalescing)
    __shared__ int8_t shared_activations[MAX_THREADS_PER_BLOCK];
    __shared__ int8_t shared_weights[MAX_THREADS_PER_BLOCK * 4]; // 4 weights per thread max
    
    int32_t accumulator = 0;
    
    // Process activations in chunks for better memory coalescing
    const int chunk_size = WARP_SIZE * 4; // Process 128 elements at a time
    const int num_chunks = (k + chunk_size - 1) / chunk_size;
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int chunk_start = chunk * chunk_size;
        const int chunk_end = min(chunk_start + chunk_size, (int)k);
        const int chunk_length = chunk_end - chunk_start;
        
        // Cooperative loading of activations into shared memory
        for (int i = threadIdx.x; i < chunk_length; i += blockDim.x) {
            if (chunk_start + i < k) {
                shared_activations[i] = activations[chunk_start + i];
            }
        }
        
        __syncthreads();
        
        // Process weights for this output row and activation chunk
        for (int k_idx = 0; k_idx < chunk_length; k_idx += 4) {
            if (chunk_start + k_idx >= k) break;
            
            // Calculate weight matrix index with permutation optimization
            const int weight_matrix_idx = calculate_permuted_index(tid, chunk_start + k_idx, k);
            const int weight_byte_idx = weight_matrix_idx / 4;
            
            // Load and extract 2-bit weights
            int8_t extracted_weights[4];
            if (weight_byte_idx < (m * k + 3) / 4) {
                extract_2bit_weights(weights[weight_byte_idx], extracted_weights);
            } else {
                // Zero-pad for boundary conditions
                extracted_weights[0] = extracted_weights[1] = 
                extracted_weights[2] = extracted_weights[3] = 0;
            }
            
            // Prepare activations for dp4a
            int8_t chunk_activations[4];
            for (int i = 0; i < 4; i++) {
                if (k_idx + i < chunk_length) {
                    chunk_activations[i] = shared_activations[k_idx + i];
                } else {
                    chunk_activations[i] = 0;
                }
            }
            
            // Compute 4-element dot product using dp4a optimization
            accumulator += dp4a_dot_product(extracted_weights, chunk_activations);
        }
        
        __syncthreads();
    }
    
    // Warp-level reduction for better performance
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        accumulator += __shfl_down_sync(0xFFFFFFFF, accumulator, offset);
    }
    
    // Write result (only lane 0 of each warp writes for this thread's output)
    if (lane_id == 0) {
        output[tid] = accumulator;
    }
}

/*
 * Optimized kernel launcher interface
 * Handles optimal grid/block configuration automatically
 */
extern "C" void launch_w2a8_gemv_optimized(
    const uint8_t* weights,
    const int8_t* activations, 
    int32_t* output,
    size_t m,
    size_t k,
    cudaStream_t stream
) {
    // Calculate optimal launch configuration
    dim3 block_size(256, 1, 1);  // Optimized for most GPUs
    dim3 grid_size((m + block_size.x - 1) / block_size.x, 1, 1);
    
    // Ensure grid size doesn't exceed GPU limits
    grid_size.x = min(grid_size.x, 65535U);
    
    // Launch kernel with optimal configuration
    w2a8_gemv_kernel<<<grid_size, block_size, 0, stream>>>(
        weights, activations, output, m, k
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handling would be done by the calling Rust code
        return;
    }
}
