#include <metal_stdlib>
using namespace metal;

/// BitNet 1.58-bit quantization kernel
/// Quantizes floating-point values to {-1, 0, 1} representation
kernel void bitnet_158_quantize(
    device const float* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device const float* zero_point [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= size) return;
    
    float value = input[index];
    float scaled = value / scale[0] + zero_point[0];
    
    // BitNet 1.58-bit quantization: {-1, 0, 1}
    int8_t quantized;
    if (scaled <= -0.5) {
        quantized = -1;
    } else if (scaled >= 0.5) {
        quantized = 1;
    } else {
        quantized = 0;
    }
    
    output[index] = quantized;
}

/// BitNet 1.58-bit dequantization kernel
/// Converts {-1, 0, 1} values back to floating-point
kernel void bitnet_158_dequantize(
    device const int8_t* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device const float* zero_point [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= size) return;
    
    int8_t quantized = input[index];
    float dequantized = (float(quantized) - zero_point[0]) * scale[0];
    output[index] = dequantized;
}

/// Adaptive quantization with dynamic scale computation
kernel void bitnet_adaptive_quantize(
    device const float* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device float* computed_scale [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    uint local_id = index % 256;
    
    // Load data into shared memory for scale computation
    shared_data[local_id] = (index < size) ? abs(input[index]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute maximum absolute value for scaling
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_data[local_id] = max(shared_data[local_id], shared_data[local_id + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store computed scale
    if (local_id == 0 && index / 256 == 0) {
        computed_scale[0] = shared_data[0];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Quantize using computed scale
    if (index < size) {
        float scale = computed_scale[0];
        float normalized = (scale > 0.0f) ? input[index] / scale : 0.0f;
        
        int8_t quantized;
        if (normalized <= -0.5f) {
            quantized = -1;
        } else if (normalized >= 0.5f) {
            quantized = 1;
        } else {
            quantized = 0;
        }
        
        output[index] = quantized;
    }
}

/// Weight quantization with absmean scaling
kernel void bitnet_weight_quantize(
    device const float* weights [[buffer(0)]],
    device int8_t* quantized_weights [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    threadgroup float* shared_sum [[threadgroup(0)]]
) {
    uint local_id = index % 256;
    
    // Compute sum of absolute values for absmean scaling
    shared_sum[local_id] = (index < size) ? abs(weights[index]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction to compute sum
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_sum[local_id] += shared_sum[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute and store absmean scale
    if (local_id == 0 && index / 256 == 0) {
        scale[0] = shared_sum[0] / float(size);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Quantize weights using absmean scale
    if (index < size) {
        float weight_scale = scale[0];
        float normalized = (weight_scale > 0.0f) ? weights[index] / weight_scale : 0.0f;
        
        // Sign-based quantization for weights
        int8_t quantized = (normalized >= 0.0f) ? 1 : -1;
        quantized_weights[index] = quantized;
    }
}

/// Activation quantization with per-tensor scaling
kernel void bitnet_activation_quantize(
    device const float* activations [[buffer(0)]],
    device float* quantized_activations [[buffer(1)]],
    device float* scale [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    threadgroup float* shared_sum [[threadgroup(0)]]
) {
    uint local_id = index % 256;
    
    // Compute sum of absolute values for scaling
    shared_sum[local_id] = (index < size) ? abs(activations[index]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction to compute sum
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_sum[local_id] += shared_sum[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute and store scale (mean absolute value)
    if (local_id == 0 && index / 256 == 0) {
        scale[0] = shared_sum[0] / float(size);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Apply quantization with scaling
    if (index < size) {
        float activation_scale = scale[0];
        if (activation_scale > 0.0f) {
            float sign_val = (activations[index] >= 0.0f) ? 1.0f : -1.0f;
            quantized_activations[index] = sign_val * activation_scale;
        } else {
            quantized_activations[index] = 0.0f;
        }
    }
}

/// Multi-bit quantization kernel (2-bit, 4-bit, 8-bit)
kernel void bitnet_multibit_quantize(
    device const float* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device const float* zero_point [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    constant uint& bits [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= size) return;
    
    float value = input[index];
    float scaled = value / scale[0] + zero_point[0];
    
    // Compute quantization levels based on bit width
    int max_val = (1 << (bits - 1)) - 1;  // e.g., 3 for 3-bit, 7 for 4-bit
    int min_val = -max_val - 1;           // e.g., -4 for 3-bit, -8 for 4-bit
    
    // Clamp and round to nearest integer
    int quantized = int(round(scaled));
    quantized = max(min_val, min(max_val, quantized));
    
    output[index] = int8_t(quantized);
}

/// Batch quantization for multiple tensors
kernel void bitnet_batch_quantize(
    device const float* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zero_points [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& tensor_size [[buffer(5)]],
    uint3 index [[thread_position_in_grid]]
) {
    uint batch_idx = index.z;
    uint tensor_idx = index.y * index.x + index.x;
    
    if (batch_idx >= batch_size || tensor_idx >= tensor_size) return;
    
    uint global_idx = batch_idx * tensor_size + tensor_idx;
    
    float value = input[global_idx];
    float scale = scales[batch_idx];
    float zero_point = zero_points[batch_idx];
    
    float scaled = value / scale + zero_point;
    
    // BitNet 1.58-bit quantization
    int8_t quantized;
    if (scaled <= -0.5f) {
        quantized = -1;
    } else if (scaled >= 0.5f) {
        quantized = 1;
    } else {
        quantized = 0;
    }
    
    output[global_idx] = quantized;
}

/// Quantization error analysis kernel
kernel void bitnet_quantization_error(
    device const float* original [[buffer(0)]],
    device const float* dequantized [[buffer(1)]],
    device float* error_metrics [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    threadgroup float* shared_error [[threadgroup(0)]]
) {
    uint local_id = index % 256;
    
    // Compute squared error
    float error = 0.0f;
    if (index < size) {
        float diff = original[index] - dequantized[index];
        error = diff * diff;
    }
    
    shared_error[local_id] = error;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction to compute total squared error
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_error[local_id] += shared_error[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store error metrics (MSE, RMSE, etc.)
    if (local_id == 0 && index / 256 == 0) {
        float mse = shared_error[0] / float(size);
        error_metrics[0] = mse;           // Mean Squared Error
        error_metrics[1] = sqrt(mse);     // Root Mean Squared Error
    }
}

/// Straight-Through Estimator (STE) gradient computation
kernel void bitnet_ste_gradient(
    device const float* input_grad [[buffer(0)]],
    device const float* input_values [[buffer(1)]],
    device float* output_grad [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant float& clip_threshold [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= size) return;
    
    float value = input_values[index];
    float grad = input_grad[index];
    
    // STE: pass gradient through if value is within clipping range
    if (abs(value) <= clip_threshold) {
        output_grad[index] = grad;
    } else {
        output_grad[index] = 0.0f;  // Clip gradient for values outside range
    }
}

/// Quantization-aware training (QAT) forward pass
kernel void bitnet_qat_forward(
    device const float* input [[buffer(0)]],
    device float* fake_quantized [[buffer(1)]],
    device int8_t* true_quantized [[buffer(2)]],
    device float* scale [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    constant bool& training_mode [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= size) return;
    
    float value = input[index];
    float current_scale = scale[0];
    
    if (training_mode) {
        // Fake quantization for gradient flow
        float normalized = value / current_scale;
        
        // Quantize to {-1, 0, 1}
        float quantized_float;
        if (normalized <= -0.5f) {
            quantized_float = -1.0f;
        } else if (normalized >= 0.5f) {
            quantized_float = 1.0f;
        } else {
            quantized_float = 0.0f;
        }
        
        fake_quantized[index] = quantized_float * current_scale;
    } else {
        // True quantization for inference
        float normalized = value / current_scale;
        
        int8_t quantized;
        if (normalized <= -0.5f) {
            quantized = -1;
        } else if (normalized >= 0.5f) {
            quantized = 1;
        } else {
            quantized = 0;
        }
        
        true_quantized[index] = quantized;
        fake_quantized[index] = float(quantized) * current_scale;
    }
}
