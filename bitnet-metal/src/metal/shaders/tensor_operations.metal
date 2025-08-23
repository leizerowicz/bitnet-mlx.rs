#include <metal_stdlib>
using namespace metal;

/// Element-wise addition kernel
kernel void elementwise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = a[gid] + b[gid];
}

/// Element-wise multiplication kernel
kernel void elementwise_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = a[gid] * b[gid];
}

/// Element-wise subtraction kernel
kernel void elementwise_sub(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = a[gid] - b[gid];
}

/// Element-wise division kernel
kernel void elementwise_div(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = a[gid] / b[gid];
}

/// Optimized matrix multiplication kernel with shared memory
kernel void matrix_multiply_optimized(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    threadgroup float* shared_A [[threadgroup(0)]],
    threadgroup float* shared_B [[threadgroup(1)]]
) {
    uint M = dims.x;  // rows of A
    uint K = dims.y;  // cols of A / rows of B  
    uint N = dims.z;  // cols of B
    
    uint row = gid.y;
    uint col = gid.x;
    
    const uint tile_size = 16;
    
    float sum = 0.0f;
    
    for (uint tile = 0; tile < (K + tile_size - 1) / tile_size; tile++) {
        // Load tiles into shared memory
        uint shared_row = lid.y;
        uint shared_col = lid.x;
        
        // Load A tile
        uint a_row = row;
        uint a_col = tile * tile_size + shared_col;
        if (a_row < M && a_col < K) {
            shared_A[shared_row * tile_size + shared_col] = A[a_row * K + a_col];
        } else {
            shared_A[shared_row * tile_size + shared_col] = 0.0f;
        }
        
        // Load B tile
        uint b_row = tile * tile_size + shared_row;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            shared_B[shared_row * tile_size + shared_col] = B[b_row * N + b_col];
        } else {
            shared_B[shared_row * tile_size + shared_col] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial sum
        for (uint k = 0; k < tile_size; k++) {
            sum += shared_A[shared_row * tile_size + k] * shared_B[k * tile_size + shared_col];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/// Reduction sum kernel
kernel void reduction_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    uint tid = gid;
    uint local_id = gid % 256;
    
    // Load data into shared memory
    shared_data[local_id] = (tid < count) ? input[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Perform reduction in shared memory
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_data[local_id] += shared_data[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (local_id == 0) {
        output[gid / 256] = shared_data[0];
    }
}

/// ReLU activation kernel
kernel void relu_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = max(0.0f, input[gid]);
}

/// GELU activation kernel
kernel void gelu_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    const float sqrt_2_over_pi = 0.7978845608f;
    const float a = 0.044715f;
    
    float x = input[gid];
    float tanh_arg = sqrt_2_over_pi * (x + a * x * x * x);
    output[gid] = 0.5f * x * (1.0f + tanh(tanh_arg));
}

/// Sigmoid activation kernel  
kernel void sigmoid_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = 1.0f / (1.0f + exp(-input[gid]));
}

/// Softmax kernel (requires two passes - this is the first pass for max finding)
kernel void softmax_max(
    device const float* input [[buffer(0)]],
    device float* max_values [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    threadgroup float* shared_max [[threadgroup(0)]]
) {
    uint batch_size = dims.x;
    uint feature_size = dims.y;
    
    uint batch_idx = gid.y;
    uint feature_idx = gid.x;
    uint local_id = feature_idx % 256;
    
    if (batch_idx >= batch_size) return;
    
    // Load maximum for this thread
    float local_max = -INFINITY;
    for (uint i = feature_idx; i < feature_size; i += 256) {
        if (i < feature_size) {
            local_max = max(local_max, input[batch_idx * feature_size + i]);
        }
    }
    
    shared_max[local_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Find maximum across threadgroup
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_max[local_id] = max(shared_max[local_id], shared_max[local_id + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (local_id == 0) {
        max_values[batch_idx] = shared_max[0];
    }
}

/// Softmax exponential and sum kernel (second pass)
kernel void softmax_exp_sum(
    device const float* input [[buffer(0)]],
    device const float* max_values [[buffer(1)]],
    device float* output [[buffer(2)]],
    device float* sum_values [[buffer(3)]],
    constant uint2& dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    threadgroup float* shared_sum [[threadgroup(0)]]
) {
    uint batch_size = dims.x;
    uint feature_size = dims.y;
    
    uint batch_idx = gid.y;
    uint feature_idx = gid.x;
    uint local_id = feature_idx % 256;
    
    if (batch_idx >= batch_size) return;
    
    float max_val = max_values[batch_idx];
    float local_sum = 0.0f;
    
    // Compute exponentials and accumulate sum
    for (uint i = feature_idx; i < feature_size; i += 256) {
        if (i < feature_size) {
            float exp_val = exp(input[batch_idx * feature_size + i] - max_val);
            output[batch_idx * feature_size + i] = exp_val;
            local_sum += exp_val;
        }
    }
    
    shared_sum[local_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Sum across threadgroup
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_sum[local_id] += shared_sum[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (local_id == 0) {
        sum_values[batch_idx] = shared_sum[0];
    }
}

/// Softmax normalization kernel (third pass)
kernel void softmax_normalize(
    device float* output [[buffer(0)]],
    device const float* sum_values [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_size = dims.x;
    uint feature_size = dims.y;
    
    uint batch_idx = gid.y;
    uint feature_idx = gid.x;
    
    if (batch_idx >= batch_size || feature_idx >= feature_size) return;
    
    float sum_val = sum_values[batch_idx];
    output[batch_idx * feature_size + feature_idx] /= sum_val;
}
