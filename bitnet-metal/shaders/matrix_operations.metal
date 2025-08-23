#include <metal_stdlib>
using namespace metal;

/// Optimized matrix multiplication kernel for BitNet
/// Uses tiled approach with shared memory for maximum performance
kernel void bitnet_matmul_optimized(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    threadgroup float* shared_A [[threadgroup(0)]],
    threadgroup float* shared_B [[threadgroup(1)]]
) {
    const uint TILE_SIZE = 16;
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    
    // Process tiles
    for (uint tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load A tile into shared memory
        uint a_row = row;
        uint a_col = tile * TILE_SIZE + lid.x;
        if (a_row < M && a_col < K) {
            shared_A[lid.y * TILE_SIZE + lid.x] = A[a_row * K + a_col];
        } else {
            shared_A[lid.y * TILE_SIZE + lid.x] = 0.0;
        }
        
        // Load B tile into shared memory
        uint b_row = tile * TILE_SIZE + lid.y;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            shared_B[lid.y * TILE_SIZE + lid.x] = B[b_row * N + b_col];
        } else {
            shared_B[lid.y * TILE_SIZE + lid.x] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial sum for this tile
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[lid.y * TILE_SIZE + k] * shared_B[k * TILE_SIZE + lid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store result
    C[row * N + col] = sum;
}

/// Element-wise addition with broadcasting support
kernel void bitnet_add_broadcasted(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint4& shape_a [[buffer(3)]],
    constant uint4& shape_b [[buffer(4)]],
    constant uint4& shape_out [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= shape_out.x * shape_out.y * shape_out.z * shape_out.w) return;
    
    // Compute output indices
    uint w = gid % shape_out.w;
    uint z = (gid / shape_out.w) % shape_out.z;
    uint y = (gid / (shape_out.w * shape_out.z)) % shape_out.y;
    uint x = gid / (shape_out.w * shape_out.z * shape_out.y);
    
    // Compute input indices with broadcasting
    uint a_x = (shape_a.x == 1) ? 0 : x;
    uint a_y = (shape_a.y == 1) ? 0 : y;
    uint a_z = (shape_a.z == 1) ? 0 : z;
    uint a_w = (shape_a.w == 1) ? 0 : w;
    
    uint b_x = (shape_b.x == 1) ? 0 : x;
    uint b_y = (shape_b.y == 1) ? 0 : y;
    uint b_z = (shape_b.z == 1) ? 0 : z;
    uint b_w = (shape_b.w == 1) ? 0 : w;
    
    uint a_idx = a_x * shape_a.y * shape_a.z * shape_a.w + 
                 a_y * shape_a.z * shape_a.w + 
                 a_z * shape_a.w + a_w;
    
    uint b_idx = b_x * shape_b.y * shape_b.z * shape_b.w + 
                 b_y * shape_b.z * shape_b.w + 
                 b_z * shape_b.w + b_w;
    
    output[gid] = a[a_idx] + b[b_idx];
}

/// Element-wise multiplication with broadcasting support
kernel void bitnet_mul_broadcasted(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint4& shape_a [[buffer(3)]],
    constant uint4& shape_b [[buffer(4)]],
    constant uint4& shape_out [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= shape_out.x * shape_out.y * shape_out.z * shape_out.w) return;
    
    // Compute output indices
    uint w = gid % shape_out.w;
    uint z = (gid / shape_out.w) % shape_out.z;
    uint y = (gid / (shape_out.w * shape_out.z)) % shape_out.y;
    uint x = gid / (shape_out.w * shape_out.z * shape_out.y);
    
    // Compute input indices with broadcasting
    uint a_x = (shape_a.x == 1) ? 0 : x;
    uint a_y = (shape_a.y == 1) ? 0 : y;
    uint a_z = (shape_a.z == 1) ? 0 : z;
    uint a_w = (shape_a.w == 1) ? 0 : w;
    
    uint b_x = (shape_b.x == 1) ? 0 : x;
    uint b_y = (shape_b.y == 1) ? 0 : y;
    uint b_z = (shape_b.z == 1) ? 0 : z;
    uint b_w = (shape_b.w == 1) ? 0 : w;
    
    uint a_idx = a_x * shape_a.y * shape_a.z * shape_a.w + 
                 a_y * shape_a.z * shape_a.w + 
                 a_z * shape_a.w + a_w;
    
    uint b_idx = b_x * shape_b.y * shape_b.z * shape_b.w + 
                 b_y * shape_b.z * shape_b.w + 
                 b_z * shape_b.w + b_w;
    
    output[gid] = a[a_idx] * b[b_idx];
}

/// Batched matrix multiplication for efficient training/inference
kernel void bitnet_batched_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch = gid.z;
    uint row = gid.y;
    uint col = gid.x;
    
    if (batch >= batch_size || row >= M || col >= N) return;
    
    uint batch_offset_a = batch * M * K;
    uint batch_offset_b = batch * K * N;
    uint batch_offset_c = batch * M * N;
    
    float sum = 0.0;
    
    for (uint k = 0; k < K; ++k) {
        sum += A[batch_offset_a + row * K + k] * B[batch_offset_b + k * N + col];
    }
    
    C[batch_offset_c + row * N + col] = sum;
}

/// Sum reduction along specified axis
kernel void bitnet_sum_reduction(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint4& input_shape [[buffer(2)]],
    constant uint& axis [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    uint tid = gid.x % 256;  // Thread ID within threadgroup
    
    // Compute which reduction we're working on
    uint reduction_idx = gid.x / 256;
    
    // Load data for reduction
    float sum = 0.0;
    uint stride = 1;
    uint total_size = input_shape.x * input_shape.y * input_shape.z * input_shape.w;
    
    // Calculate which elements this thread should sum
    for (uint i = tid; i < total_size; i += 256) {
        sum += input[i];
    }
    
    shared_data[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce within threadgroup
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        output[reduction_idx] = shared_data[0];
    }
}

/// Mean reduction along specified axis
kernel void bitnet_mean_reduction(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint4& input_shape [[buffer(2)]],
    constant uint& axis [[buffer(3)]],
    constant uint& reduction_size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    uint tid = gid.x % 256;  // Thread ID within threadgroup
    uint reduction_idx = gid.x / 256;
    
    // Load data for reduction
    float sum = 0.0;
    uint total_size = input_shape.x * input_shape.y * input_shape.z * input_shape.w;
    
    for (uint i = tid; i < total_size; i += 256) {
        sum += input[i];
    }
    
    shared_data[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce within threadgroup
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        output[reduction_idx] = shared_data[0] / float(reduction_size);
    }
}
