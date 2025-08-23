#include <metal_stdlib>
using namespace metal;

/// BitLinear forward pass kernel with optimized matrix multiplication
/// Performs quantized matrix multiplication for BitLinear layers
kernel void bitlinear_forward(
    device const int8_t* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* weight_scale [[buffer(3)]],
    device const float* input_scale [[buffer(4)]],
    constant uint& input_size [[buffer(5)]],
    constant uint& output_size [[buffer(6)]],
    uint2 index [[thread_position_in_grid]]
) {
    uint out_idx = index.x;
    uint batch_idx = index.y;
    
    if (out_idx >= output_size) return;
    
    // Compute dot product for this output element
    float sum = 0.0f;
    for (uint i = 0; i < input_size; i++) {
        int8_t w = weights[out_idx * input_size + i];
        float x = input[batch_idx * input_size + i];
        sum += float(w) * x;
    }
    
    // Apply scaling
    float result = sum * weight_scale[0] * input_scale[batch_idx];
    output[batch_idx * output_size + out_idx] = result;
}

/// Optimized BitLinear forward pass with tiling for large matrices
kernel void bitlinear_forward_tiled(
    device const int8_t* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* weight_scale [[buffer(3)]],
    device const float* input_scale [[buffer(4)]],
    constant uint& input_size [[buffer(5)]],
    constant uint& output_size [[buffer(6)]],
    constant uint& tile_size [[buffer(7)]],
    uint2 index [[thread_position_in_grid]],
    threadgroup float* shared_input [[threadgroup(0)]],
    threadgroup int8_t* shared_weights [[threadgroup(1)]]
) {
    uint out_idx = index.x;
    uint batch_idx = index.y;
    uint local_id = index.x % tile_size;
    
    if (out_idx >= output_size) return;
    
    float sum = 0.0f;
    
    // Process input in tiles to optimize memory access
    for (uint tile_start = 0; tile_start < input_size; tile_start += tile_size) {
        // Load tile of input into shared memory
        if (local_id < tile_size && tile_start + local_id < input_size) {
            shared_input[local_id] = input[batch_idx * input_size + tile_start + local_id];
        } else {
            shared_input[local_id] = 0.0f;
        }
        
        // Load tile of weights into shared memory
        if (local_id < tile_size && tile_start + local_id < input_size) {
            shared_weights[local_id] = weights[out_idx * input_size + tile_start + local_id];
        } else {
            shared_weights[local_id] = 0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint i = 0; i < tile_size && tile_start + i < input_size; i++) {
            sum += float(shared_weights[i]) * shared_input[i];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Apply scaling and store result
    float result = sum * weight_scale[0] * input_scale[batch_idx];
    output[batch_idx * output_size + out_idx] = result;
}

/// BitLinear activation quantization kernel
/// Quantizes activations using sign activation with mean absolute scaling
kernel void bitlinear_activation_quant(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    threadgroup float* shared_sum [[threadgroup(0)]]
) {
    uint local_id = index % 256;
    
    // Compute mean absolute value for scale factor
    float local_sum = 0.0f;
    for (uint i = index; i < size; i += 256) {
        local_sum += abs(input[i]);
    }
    
    shared_sum[local_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to compute total sum
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_sum[local_id] += shared_sum[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute and store scale factor
    if (local_id == 0 && index / 256 == 0) {
        scale[0] = shared_sum[0] / float(size);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Apply quantization with computed scale
    if (index < size) {
        float s = scale[0];
        if (s > 1e-8f) {
            output[index] = (input[index] >= 0.0f) ? s : -s;
        } else {
            output[index] = 0.0f;
        }
    }
}
    
    if (out_idx >= output_size) return;
    
    float sum = 0.0f;
    
    // Process input in tiles for better cache efficiency
    for (uint tile_start = 0; tile_start < input_size; tile_start += tile_size) {
        uint tile_end = min(tile_start + tile_size, input_size);
        
        // Load input tile into shared memory
        if (local_id < (tile_end - tile_start)) {
            shared_input[local_id] = input[batch_idx * input_size + tile_start + local_id];
            shared_weights[local_id] = weights[out_idx * input_size + tile_start + local_id];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint i = 0; i < (tile_end - tile_start); i++) {
            sum += float(shared_weights[i]) * shared_input[i];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Apply scaling and store result
    float result = sum * weight_scale[0] * input_scale[batch_idx];
    output[batch_idx * output_size + out_idx] = result;
}

/// BitLinear activation quantization kernel
/// Quantizes activations using absmean scaling before BitLinear layer
kernel void bitlinear_activation_quant(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& feature_size [[buffer(4)]],
    uint2 index [[thread_position_in_grid]],
    threadgroup float* shared_sum [[threadgroup(0)]]
) {
    uint batch_idx = index.y;
    uint feature_idx = index.x;
    uint local_id = feature_idx % 256;
    
    if (batch_idx >= batch_size || feature_idx >= feature_size) return;
    
    // Compute sum of absolute values for this batch
    shared_sum[local_id] = 0.0f;
    for (uint i = feature_idx; i < feature_size; i += 256) {
        if (i < feature_size) {
            shared_sum[local_id] += abs(input[batch_idx * feature_size + i]);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction to compute total sum
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_sum[local_id] += shared_sum[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute and store scale (mean absolute value)
    if (local_id == 0) {
        scale[batch_idx] = shared_sum[0] / float(feature_size);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Apply quantization with scaling
    float current_scale = scale[batch_idx];
    if (current_scale > 0.0f) {
        float value = input[batch_idx * feature_size + feature_idx];
        output[batch_idx * feature_size + feature_idx] = sign(value) * current_scale;
    } else {
        output[batch_idx * feature_size + feature_idx] = 0.0f;
    }
}

/// BitLinear weight preparation kernel
/// Prepares weights for efficient BitLinear computation
kernel void bitlinear_weight_prep(
    device const float* fp_weights [[buffer(0)]],
    device int8_t* quant_weights [[buffer(1)]],
    device float* weight_scale [[buffer(2)]],
    constant uint& weight_size [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    threadgroup float* shared_sum [[threadgroup(0)]]
) {
    uint local_id = index % 256;
    
    // Compute sum of absolute values for absmean scaling
    shared_sum[local_id] = 0.0f;
    for (uint i = index; i < weight_size; i += 256) {
        if (i < weight_size) {
            shared_sum[local_id] += abs(fp_weights[i]);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction to compute total sum
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_sum[local_id] += shared_sum[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute and store absmean scale
    if (local_id == 0 && index / 256 == 0) {
        weight_scale[0] = shared_sum[0] / float(weight_size);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Quantize weights using sign function
    if (index < weight_size) {
        float weight = fp_weights[index];
        quant_weights[index] = (weight >= 0.0f) ? 1 : -1;
    }
}

/// Batched BitLinear forward pass for multiple samples
kernel void bitlinear_batched_forward(
    device const int8_t* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* weight_scale [[buffer(3)]],
    device const float* input_scales [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    constant uint& input_size [[buffer(6)]],
    constant uint& output_size [[buffer(7)]],
    uint3 index [[thread_position_in_grid]]
) {
    uint batch_idx = index.z;
    uint out_idx = index.y;
    uint in_idx = index.x;
    
    if (batch_idx >= batch_size || out_idx >= output_size || in_idx >= input_size) return;
    
    // Use atomic operations to accumulate results
    float partial_sum = float(weights[out_idx * input_size + in_idx]) * 
                       input[batch_idx * input_size + in_idx];
    
    // Apply scaling
    partial_sum *= weight_scale[0] * input_scales[batch_idx];
    
    // Atomic add to accumulate partial results
    atomic_fetch_add_explicit(
        (device atomic<float>*)&output[batch_idx * output_size + out_idx],
        partial_sum,
        memory_order_relaxed
    );
}

/// BitLinear backward pass for gradient computation
kernel void bitlinear_backward(
    device const float* grad_output [[buffer(0)]],
    device const int8_t* weights [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* grad_input [[buffer(3)]],
    device float* grad_weights [[buffer(4)]],
    device const float* weight_scale [[buffer(5)]],
    device const float* input_scale [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    constant uint& input_size [[buffer(8)]],
    constant uint& output_size [[buffer(9)]],
    uint3 index [[thread_position_in_grid]]
) {
    uint batch_idx = index.z;
    uint out_idx = index.y;
    uint in_idx = index.x;
    
    if (batch_idx >= batch_size || out_idx >= output_size || in_idx >= input_size) return;
    
    float grad_out = grad_output[batch_idx * output_size + out_idx];
    float w = float(weights[out_idx * input_size + in_idx]);
    float x = input[batch_idx * input_size + in_idx];
    float w_scale = weight_scale[0];
    float x_scale = input_scale[batch_idx];
    
    // Gradient w.r.t. input
    float grad_x = grad_out * w * w_scale * x_scale;
    atomic_fetch_add_explicit(
        (device atomic<float>*)&grad_input[batch_idx * input_size + in_idx],
        grad_x,
        memory_order_relaxed
    );
    
    // Gradient w.r.t. weights (accumulated across batch)
    float grad_w = grad_out * x * w_scale * x_scale;
    atomic_fetch_add_explicit(
        (device atomic<float>*)&grad_weights[out_idx * input_size + in_idx],
        grad_w,
        memory_order_relaxed
    );
}

/// BitLinear layer normalization kernel
/// Applies layer normalization before BitLinear transformation
kernel void bitlinear_layer_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& feature_size [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint2 index [[thread_position_in_grid]],
    threadgroup float* shared_mean [[threadgroup(0)]],
    threadgroup float* shared_var [[threadgroup(1)]]
) {
    uint batch_idx = index.y;
    uint feature_idx = index.x;
    uint local_id = feature_idx % 256;
    
    if (batch_idx >= batch_size || feature_idx >= feature_size) return;
    
    // Compute mean
    shared_mean[local_id] = 0.0f;
    for (uint i = feature_idx; i < feature_size; i += 256) {
        if (i < feature_size) {
            shared_mean[local_id] += input[batch_idx * feature_size + i];
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction for mean
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_mean[local_id] += shared_mean[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean = shared_mean[0] / float(feature_size);
    
    // Compute variance
    shared_var[local_id] = 0.0f;
    for (uint i = feature_idx; i < feature_size; i += 256) {
        if (i < feature_size) {
            float diff = input[batch_idx * feature_size + i] - mean;
            shared_var[local_id] += diff * diff;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction for variance
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_var[local_id] += shared_var[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float variance = shared_var[0] / float(feature_size);
    float std_dev = sqrt(variance + eps);
    
    // Apply normalization
    float normalized = (input[batch_idx * feature_size + feature_idx] - mean) / std_dev;
    output[batch_idx * feature_size + feature_idx] = normalized * gamma[feature_idx] + beta[feature_idx];
}

/// BitLinear residual connection kernel
/// Adds residual connection to BitLinear output
kernel void bitlinear_residual(
    device const float* bitlinear_output [[buffer(0)]],
    device const float* residual_input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant float& residual_scale [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= size) return;
    
    output[index] = bitlinear_output[index] + residual_scale * residual_input[index];
}

/// BitLinear attention mechanism kernel
/// Optimized attention computation for BitLinear transformers
kernel void bitlinear_attention(
    device const int8_t* q_weights [[buffer(0)]],
    device const int8_t* k_weights [[buffer(1)]],
    device const int8_t* v_weights [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* attention_output [[buffer(4)]],
    device const float* q_scale [[buffer(5)]],
    device const float* k_scale [[buffer(6)]],
    device const float* v_scale [[buffer(7)]],
    device const float* input_scale [[buffer(8)]],
    constant uint& seq_length [[buffer(9)]],
    constant uint& hidden_size [[buffer(10)]],
    constant uint& head_size [[buffer(11)]],
    uint3 index [[thread_position_in_grid]]
) {
    uint seq_idx = index.z;
    uint head_idx = index.y;
    uint dim_idx = index.x;
    
    if (seq_idx >= seq_length || head_idx >= (hidden_size / head_size) || dim_idx >= head_size) return;
    
    uint head_offset = head_idx * head_size;
    
    // Compute Q, K, V using BitLinear operations
    float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
    
    for (uint i = 0; i < hidden_size; i++) {
        float x = input[seq_idx * hidden_size + i];
        
        q_sum += float(q_weights[(head_offset + dim_idx) * hidden_size + i]) * x;
        k_sum += float(k_weights[(head_offset + dim_idx) * hidden_size + i]) * x;
        v_sum += float(v_weights[(head_offset + dim_idx) * hidden_size + i]) * x;
    }
    
    float q = q_sum * q_scale[0] * input_scale[seq_idx];
    float k = k_sum * k_scale[0] * input_scale[seq_idx];
    float v = v_sum * v_scale[0] * input_scale[seq_idx];
    
    // Store intermediate Q, K, V values (simplified attention)
    // In practice, this would be followed by attention score computation
    attention_output[seq_idx * hidden_size + head_offset + dim_idx] = v;
}

/// BitLinear performance profiling kernel
/// Measures execution time and throughput for BitLinear operations
kernel void bitlinear_profile(
    device const int8_t* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device float* timing_results [[buffer(3)]],
    device const float* weight_scale [[buffer(4)]],
    device const float* input_scale [[buffer(5)]],
    constant uint& input_size [[buffer(6)]],
    constant uint& output_size [[buffer(7)]],
    constant uint& iterations [[buffer(8)]],
    uint2 index [[thread_position_in_grid]]
) {
    uint out_idx = index.x;
    uint batch_idx = index.y;
    
    if (out_idx >= output_size) return;
    
    // Measure execution time for profiling
    uint64_t start_time = metal::get_timestamp();
    
    for (uint iter = 0; iter < iterations; iter++) {
        float sum = 0.0f;
        for (uint i = 0; i < input_size; i++) {
            int8_t w = weights[out_idx * input_size + i];
            float x = input[batch_idx * input_size + i];
            sum += float(w) * x;
        }
        
        float result = sum * weight_scale[0] * input_scale[batch_idx];
        output[batch_idx * output_size + out_idx] = result;
    }
    
    uint64_t end_time = metal::get_timestamp();
    
    // Store timing results for analysis
    if (out_idx == 0 && batch_idx == 0) {
        timing_results[0] = float(end_time - start_time);
        timing_results[1] = float(iterations * input_size * output_size); // Total operations
    }
}
