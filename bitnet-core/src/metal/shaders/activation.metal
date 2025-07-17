#include <metal_stdlib>
using namespace metal;

// Activation functions for BitNet implementation
// Optimized Metal compute shaders for various activation functions

// Constants
constant float GELU_COEFF = 0.044715;
constant float SQRT_2_PI = 0.7978845608028654; // sqrt(2/π)
constant float SWISH_BETA = 1.0;

// ReLU activation function
kernel void relu_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    output[gid] = max(0.0, input[gid]);
}

// ReLU backward pass
kernel void relu_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    grad_input[gid] = (input[gid] > 0.0) ? grad_output[gid] : 0.0;
}

// GELU activation function (Gaussian Error Linear Unit)
kernel void gelu_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    float x = input[gid];
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    float x_cubed = x * x * x;
    float inner = SQRT_2_PI * (x + GELU_COEFF * x_cubed);
    float tanh_val = tanh(inner);
    output[gid] = 0.5 * x * (1.0 + tanh_val);
}

// GELU backward pass
kernel void gelu_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    float x = input[gid];
    float x_squared = x * x;
    float x_cubed = x_squared * x;
    
    // Compute derivative of GELU
    float inner = SQRT_2_PI * (x + GELU_COEFF * x_cubed);
    float tanh_val = tanh(inner);
    float sech_squared = 1.0 - tanh_val * tanh_val;
    
    float derivative = 0.5 * (1.0 + tanh_val) + 
                      0.5 * x * sech_squared * SQRT_2_PI * (1.0 + 3.0 * GELU_COEFF * x_squared);
    
    grad_input[gid] = grad_output[gid] * derivative;
}

// Swish/SiLU activation function
kernel void swish_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    float x = input[gid];
    // Swish(x) = x * sigmoid(βx)
    float sigmoid_val = 1.0 / (1.0 + exp(-SWISH_BETA * x));
    output[gid] = x * sigmoid_val;
}

// Swish backward pass
kernel void swish_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    float x = input[gid];
    float sigmoid_val = 1.0 / (1.0 + exp(-SWISH_BETA * x));
    
    // Derivative: sigmoid(βx) + x * sigmoid(βx) * (1 - sigmoid(βx)) * β
    float derivative = sigmoid_val + x * sigmoid_val * (1.0 - sigmoid_val) * SWISH_BETA;
    
    grad_input[gid] = grad_output[gid] * derivative;
}

// Sigmoid activation function
kernel void sigmoid_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    float x = input[gid];
    output[gid] = 1.0 / (1.0 + exp(-x));
}

// Sigmoid backward pass
kernel void sigmoid_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    float x = input[gid];
    float sigmoid_val = 1.0 / (1.0 + exp(-x));
    float derivative = sigmoid_val * (1.0 - sigmoid_val);
    
    grad_input[gid] = grad_output[gid] * derivative;
}

// Tanh activation function
kernel void tanh_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    output[gid] = tanh(input[gid]);
}

// Tanh backward pass
kernel void tanh_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    float tanh_val = tanh(input[gid]);
    float derivative = 1.0 - tanh_val * tanh_val;
    
    grad_input[gid] = grad_output[gid] * derivative;
}

// Leaky ReLU activation function
kernel void leaky_relu_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& negative_slope [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    float x = input[gid];
    output[gid] = (x > 0.0) ? x : negative_slope * x;
}

// Leaky ReLU backward pass
kernel void leaky_relu_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant float& negative_slope [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    float derivative = (input[gid] > 0.0) ? 1.0 : negative_slope;
    grad_input[gid] = grad_output[gid] * derivative;
}

// Softmax activation function (numerically stable)
kernel void softmax_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& feature_size [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint feature_idx = gid.x;
    
    if (batch_idx >= batch_size || feature_idx >= feature_size) {
        return;
    }
    
    // Find maximum for numerical stability
    float max_val = -INFINITY;
    uint offset = batch_idx * feature_size;
    
    for (uint i = 0; i < feature_size; i++) {
        max_val = max(max_val, input[offset + i]);
    }
    
    // Compute exponentials and sum
    float sum_exp = 0.0;
    for (uint i = 0; i < feature_size; i++) {
        sum_exp += exp(input[offset + i] - max_val);
    }
    
    // Compute softmax
    float exp_val = exp(input[offset + feature_idx] - max_val);
    output[offset + feature_idx] = exp_val / sum_exp;
}

// Softmax backward pass
kernel void softmax_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* softmax_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& feature_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint feature_idx = gid.x;
    
    if (batch_idx >= batch_size || feature_idx >= feature_size) {
        return;
    }
    
    uint offset = batch_idx * feature_size;
    float softmax_val = softmax_output[offset + feature_idx];
    
    // Compute sum of grad_output * softmax_output
    float sum_grad_softmax = 0.0;
    for (uint i = 0; i < feature_size; i++) {
        sum_grad_softmax += grad_output[offset + i] * softmax_output[offset + i];
    }
    
    // Compute gradient
    grad_input[offset + feature_idx] = softmax_val * (grad_output[offset + feature_idx] - sum_grad_softmax);
}

// Layer normalization
kernel void layer_norm_forward(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* mean [[buffer(4)]],
    device float* variance [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    constant uint& feature_size [[buffer(7)]],
    constant float& epsilon [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint feature_idx = gid.x;
    
    if (batch_idx >= batch_size || feature_idx >= feature_size) {
        return;
    }
    
    uint offset = batch_idx * feature_size;
    
    // Compute mean
    if (feature_idx == 0) {
        float sum = 0.0;
        for (uint i = 0; i < feature_size; i++) {
            sum += input[offset + i];
        }
        mean[batch_idx] = sum / float(feature_size);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Compute variance
    if (feature_idx == 0) {
        float mean_val = mean[batch_idx];
        float sum_sq_diff = 0.0;
        for (uint i = 0; i < feature_size; i++) {
            float diff = input[offset + i] - mean_val;
            sum_sq_diff += diff * diff;
        }
        variance[batch_idx] = sum_sq_diff / float(feature_size);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Normalize
    float mean_val = mean[batch_idx];
    float var_val = variance[batch_idx];
    float std_val = sqrt(var_val + epsilon);
    
    float normalized = (input[offset + feature_idx] - mean_val) / std_val;
    output[offset + feature_idx] = gamma[feature_idx] * normalized + beta[feature_idx];
}

// Fused activation functions for efficiency
kernel void fused_relu_dropout(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* dropout_mask [[buffer(2)]],
    constant float& dropout_prob [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    
    float relu_output = max(0.0, input[gid]);
    float scale = 1.0 / (1.0 - dropout_prob);
    output[gid] = relu_output * dropout_mask[gid] * scale;
}