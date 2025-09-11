/*
 * Common CUDA utilities and definitions for BitNet kernels
 */

#ifndef BITNET_CUDA_COMMON_CUH
#define BITNET_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cstdint>

// Common constants
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_SHARED_MEMORY_BYTES 49152  // 48KB typical limit

// BitNet quantization constants
#define BITNET_2BIT_MIN -1
#define BITNET_2BIT_MAX 1
#define BITNET_8BIT_MIN -128
#define BITNET_8BIT_MAX 127

// Utility macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            return; \
        } \
    } while(0)

#define DIVUP(a, b) (((a) + (b) - 1) / (b))

// Device utility functions
__device__ __forceinline__ int min(int a, int b) {
    return a < b ? a : b;
}

__device__ __forceinline__ int max(int a, int b) {
    return a > b ? a : b;
}

__device__ __forceinline__ float min(float a, float b) {
    return a < b ? a : b;
}

__device__ __forceinline__ float max(float a, float b) {
    return a > b ? a : b;
}

// Warp-level reduction utilities
__device__ __forceinline__ int warp_reduce_sum(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

#endif // BITNET_CUDA_COMMON_CUH
