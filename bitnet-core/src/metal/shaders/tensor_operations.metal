#include <metal_stdlib>
using namespace metal;

// Device functions for BitNet operations

// Ternary quantization function
device int8_t quantize_ternary(float value) {
    if (value > 0.5) return 1;
    else if (value < -0.5) return -1;
    else return 0;
}

// Activation functions
device float relu(float x) {
    return fmax(0.0, x);
}

device float tanh_activation(float x) {
    return tanh(x);
}

device float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

device float gelu(float x) {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI_F) * (x + 0.044715 * x * x * x)));
}

device float swish(float x) {
    return x * sigmoid(x);
}

// Kernel functions

// Basic tensor addition
kernel void tensor_add(device const float* input_a [[buffer(0)]],
                      device const float* input_b [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = input_a[id] + input_b[id];
}

// Basic tensor multiplication
kernel void tensor_mul(device const float* input_a [[buffer(0)]],
                      device const float* input_b [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = input_a[id] * input_b[id];
}

// BitNet ternary quantization
kernel void bitnet_quantize_ternary(device const float* input [[buffer(0)]],
                                   device int8_t* output [[buffer(1)]],
                                   uint id [[thread_position_in_grid]]) {
    output[id] = quantize_ternary(input[id]);
}

// BitNet dequantization
kernel void bitnet_dequantize_ternary(device const int8_t* input [[buffer(0)]],
                                     device float* output [[buffer(1)]],
                                     uint id [[thread_position_in_grid]]) {
    output[id] = (float)input[id];
}

// Matrix multiplication for BitNet
kernel void bitnet_matmul(device const float* matrix_a [[buffer(0)]],
                         device const float* matrix_b [[buffer(1)]],
                         device float* output [[buffer(2)]],
                         constant uint& rows_a [[buffer(3)]],
                         constant uint& cols_a [[buffer(4)]],
                         constant uint& cols_b [[buffer(5)]],
                         uint2 position [[thread_position_in_grid]]) {
    uint row = position.y;
    uint col = position.x;
    
    if (row >= rows_a || col >= cols_b) return;
    
    float sum = 0.0;
    for (uint k = 0; k < cols_a; k++) {
        sum += matrix_a[row * cols_a + k] * matrix_b[k * cols_b + col];
    }
    output[row * cols_b + col] = sum;
}

// BitLinear forward pass
kernel void bitlinear_forward(device const float* input [[buffer(0)]],
                             device const int8_t* weight [[buffer(1)]],
                             device const float* bias [[buffer(2)]],
                             device float* output [[buffer(3)]],
                             constant uint& input_size [[buffer(4)]],
                             constant uint& output_size [[buffer(5)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= output_size) return;
    
    float sum = 0.0;
    for (uint i = 0; i < input_size; i++) {
        sum += input[i] * (float)weight[id * input_size + i];
    }
    if (bias) {
        sum += bias[id];
    }
    output[id] = sum;
}

// Activation kernels
kernel void apply_relu(device const float* input [[buffer(0)]],
                      device float* output [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = relu(input[id]);
}

kernel void apply_tanh(device const float* input [[buffer(0)]],
                      device float* output [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = tanh_activation(input[id]);
}

kernel void apply_sigmoid(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         uint id [[thread_position_in_grid]]) {
    output[id] = sigmoid(input[id]);
}

kernel void apply_gelu(device const float* input [[buffer(0)]],
                      device float* output [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = gelu(input[id]);
}

kernel void apply_swish(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    output[id] = swish(input[id]);
}

// Reduction operations
kernel void reduce_sum(device const float* input [[buffer(0)]],
                      device float* output [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    // Simple implementation - could be optimized with shared memory
    float sum = 0.0;
    output[id] = input[id]; // Placeholder for actual reduction logic
}

kernel void reduce_mean(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    output[id] = input[id] / (float)size;
}

// Normalization kernels
kernel void layer_norm(device const float* input [[buffer(0)]],
                      device const float* gamma [[buffer(1)]],
                      device const float* beta [[buffer(2)]],
                      device float* output [[buffer(3)]],
                      constant uint& size [[buffer(4)]],
                      constant float& epsilon [[buffer(5)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= size) return;
    
    // Simple layer normalization - could be optimized
    float mean = 0.0;
    float variance = 0.0;
    
    // Calculate mean
    for (uint i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= (float)size;
    
    // Calculate variance
    for (uint i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)size;
    
    // Normalize
    float normalized = (input[id] - mean) / sqrt(variance + epsilon);
    output[id] = gamma[id] * normalized + beta[id];
}

// Batch normalization
kernel void batch_norm(device const float* input [[buffer(0)]],
                      device const float* gamma [[buffer(1)]],
                      device const float* beta [[buffer(2)]],
                      device const float* running_mean [[buffer(3)]],
                      device const float* running_var [[buffer(4)]],
                      device float* output [[buffer(5)]],
                      constant float& epsilon [[buffer(6)]],
                      uint id [[thread_position_in_grid]]) {
    float normalized = (input[id] - running_mean[id]) / sqrt(running_var[id] + epsilon);
    output[id] = gamma[id] * normalized + beta[id];
}

// Convolution kernel (simplified 2D)
kernel void conv2d(device const float* input [[buffer(0)]],
                  device const float* weight [[buffer(1)]],
                  device float* output [[buffer(2)]],
                  constant uint& input_height [[buffer(3)]],
                  constant uint& input_width [[buffer(4)]],
                  constant uint& kernel_height [[buffer(5)]],
                  constant uint& kernel_width [[buffer(6)]],
                  constant uint& output_height [[buffer(7)]],
                  constant uint& output_width [[buffer(8)]],
                  uint2 position [[thread_position_in_grid]]) {
    uint out_y = position.y;
    uint out_x = position.x;
    
    if (out_y >= output_height || out_x >= output_width) return;
    
    float sum = 0.0;
    for (uint ky = 0; ky < kernel_height; ky++) {
        for (uint kx = 0; kx < kernel_width; kx++) {
            uint in_y = out_y + ky;
            uint in_x = out_x + kx;
            
            if (in_y < input_height && in_x < input_width) {
                uint input_idx = in_y * input_width + in_x;
                uint weight_idx = ky * kernel_width + kx;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    output[out_y * output_width + out_x] = sum;
}
