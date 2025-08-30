//
// BitNet Inference Engine - Optimized Metal Compute Shaders
// Advanced BitLinear inference operations optimized for Apple Silicon GPU
//
// Phase 5 Day 8: GPU Optimization Implementation
//

#include <metal_stdlib>
using namespace metal;

struct InferenceParams {
    uint batch_size;
    uint input_dim;
    uint output_dim;
    uint quantization_bits;
};

// Enhanced inference parameters with memory and performance optimization
struct OptimizedInferenceParams {
    uint batch_size;
    uint input_dim;
    uint output_dim;
    uint quantization_bits;
    uint tile_size;          // For tiled processing
    float scale_factor;      // Global scale factor
    uint use_fast_math;      // Fast math optimizations flag
};

// Advanced RMSNorm parameters for layer normalization
struct RMSNormParams {
    uint dim;
    float eps;
    uint use_fused_ops;     // Enable fused operations
};

// BitLinear quantization parameters
struct QuantizationParams {
    float scale;
    float zero_point;
    uint bits;
    uint symmetric;         // Symmetric vs asymmetric quantization
};

//
// MARK: - Core BitLinear Inference Kernel
//
kernel void bitlinear_inference_optimized(
    device const float* weights [[buffer(0)]],
    device const float* inputs [[buffer(1)]],
    device float* outputs [[buffer(2)]],
    constant InferenceParams& params [[buffer(3)]],
    uint3 thread_position [[thread_position_in_grid]]
) {
    uint batch_idx = thread_position.x;
    uint output_idx = thread_position.y;
    
    if (batch_idx >= params.batch_size || output_idx >= params.output_dim) {
        return;
    }
    
    float sum = 0.0;
    
    // Optimized inner product with SIMD vectorization
    for (uint i = 0; i < params.input_dim; i += 4) {
        // Check bounds for vectorized access
        if (i + 3 < params.input_dim) {
            float4 input_vec = float4(
                inputs[batch_idx * params.input_dim + i],
                inputs[batch_idx * params.input_dim + i + 1],
                inputs[batch_idx * params.input_dim + i + 2],
                inputs[batch_idx * params.input_dim + i + 3]
            );
            
            float4 weight_vec = float4(
                weights[output_idx * params.input_dim + i],
                weights[output_idx * params.input_dim + i + 1],
                weights[output_idx * params.input_dim + i + 2],
                weights[output_idx * params.input_dim + i + 3]
            );
            
            sum += dot(input_vec, weight_vec);
        } else {
            // Handle remaining elements
            for (uint j = i; j < params.input_dim; j++) {
                sum += inputs[batch_idx * params.input_dim + j] * 
                       weights[output_idx * params.input_dim + j];
            }
        }
    }
    
    outputs[batch_idx * params.output_dim + output_idx] = sum;
}

//
// MARK: - Tiled BitLinear Inference for Large Matrices
//
kernel void bitlinear_inference_tiled(
    device const float* weights [[buffer(0)]],
    device const float* inputs [[buffer(1)]],
    device float* outputs [[buffer(2)]],
    constant OptimizedInferenceParams& params [[buffer(3)]],
    threadgroup float* shared_memory [[threadgroup(0)]],
    uint3 thread_position [[thread_position_in_grid]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    uint batch_idx = threadgroup_position.x * params.tile_size + thread_position_in_threadgroup.x;
    uint output_idx = threadgroup_position.y * params.tile_size + thread_position_in_threadgroup.y;
    
    if (batch_idx >= params.batch_size || output_idx >= params.output_dim) {
        return;
    }
    
    float sum = 0.0;
    
    // Tiled computation with shared memory
    for (uint tile = 0; tile < (params.input_dim + params.tile_size - 1) / params.tile_size; tile++) {
        uint tile_start = tile * params.tile_size;
        uint tile_end = min(tile_start + params.tile_size, params.input_dim);
        
        // Load tile into shared memory
        for (uint i = tile_start; i < tile_end; i++) {
            uint local_i = i - tile_start;
            
            // Coalesced memory access
            if (thread_position_in_threadgroup.x == 0) {
                shared_memory[local_i] = weights[output_idx * params.input_dim + i];
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint i = tile_start; i < tile_end; i++) {
            uint local_i = i - tile_start;
            sum += inputs[batch_idx * params.input_dim + i] * shared_memory[local_i];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Apply scale factor if using quantization
    if (params.scale_factor != 1.0) {
        sum *= params.scale_factor;
    }
    
    outputs[batch_idx * params.output_dim + output_idx] = sum;
}

//
// MARK: - Quantized BitLinear Inference
//
kernel void bitlinear_inference_quantized(
    device const uchar* quantized_weights [[buffer(0)]],  // Quantized weights
    device const float* inputs [[buffer(1)]],
    device float* outputs [[buffer(2)]],
    constant InferenceParams& params [[buffer(3)]],
    constant QuantizationParams& quant_params [[buffer(4)]],
    uint3 thread_position [[thread_position_in_grid]]
) {
    uint batch_idx = thread_position.x;
    uint output_idx = thread_position.y;
    
    if (batch_idx >= params.batch_size || output_idx >= params.output_dim) {
        return;
    }
    
    float sum = 0.0;
    
    // Process quantized weights efficiently
    for (uint i = 0; i < params.input_dim; i += 4) {
        if (i + 3 < params.input_dim) {
            // Load 4 quantized weights at once
            uchar4 quant_weights = *((device uchar4*)(quantized_weights + output_idx * params.input_dim + i));
            
            float4 input_vec = float4(
                inputs[batch_idx * params.input_dim + i],
                inputs[batch_idx * params.input_dim + i + 1],
                inputs[batch_idx * params.input_dim + i + 2],
                inputs[batch_idx * params.input_dim + i + 3]
            );
            
            // Dequantize weights on-the-fly
            float4 weight_vec = float4(
                (float(quant_weights.x) - quant_params.zero_point) * quant_params.scale,
                (float(quant_weights.y) - quant_params.zero_point) * quant_params.scale,
                (float(quant_weights.z) - quant_params.zero_point) * quant_params.scale,
                (float(quant_weights.w) - quant_params.zero_point) * quant_params.scale
            );
            
            sum += dot(input_vec, weight_vec);
        } else {
            // Handle remaining elements
            for (uint j = i; j < params.input_dim; j++) {
                uchar quant_weight = quantized_weights[output_idx * params.input_dim + j];
                float weight = (float(quant_weight) - quant_params.zero_point) * quant_params.scale;
                sum += inputs[batch_idx * params.input_dim + j] * weight;
            }
        }
    }
    
    outputs[batch_idx * params.output_dim + output_idx] = sum;
}

//
// MARK: - RMS Layer Normalization
//
kernel void rms_layer_norm(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint3 thread_position [[thread_position_in_grid]]
) {
    uint batch_idx = thread_position.x;
    uint elem_idx = thread_position.y;
    
    if (batch_idx >= params.dim || elem_idx >= params.dim) {
        return;
    }
    
    // Compute RMS across the feature dimension
    float sum_squares = 0.0;
    
    for (uint i = 0; i < params.dim; i++) {
        float val = input[batch_idx * params.dim + i];
        sum_squares += val * val;
    }
    
    float rms = sqrt(sum_squares / float(params.dim) + params.eps);
    
    // Normalize and scale
    float normalized_val = input[batch_idx * params.dim + elem_idx] / rms;
    
    if (scale != nullptr) {
        normalized_val *= scale[elem_idx];
    }
    
    output[batch_idx * params.dim + elem_idx] = normalized_val;
}

//
// MARK: - Batch Processing Helper
//
kernel void batch_copy_with_padding(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& src_size [[buffer(2)]],
    constant uint& dst_size [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint thread_position [[thread_position_in_grid]]
) {
    uint total_threads = batch_size * dst_size;
    
    if (thread_position >= total_threads) {
        return;
    }
    
    uint batch_idx = thread_position / dst_size;
    uint elem_idx = thread_position % dst_size;
    
    if (elem_idx < src_size) {
        dst[thread_position] = src[batch_idx * src_size + elem_idx];
    } else {
        dst[thread_position] = 0.0; // Zero padding
    }
}

//
// MARK: - Memory Transfer Optimization
//
kernel void async_memory_prefetch(
    device const float* src [[buffer(0)]],
    device float* prefetch_buffer [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint thread_position [[thread_position_in_grid]]
) {
    if (thread_position >= size) {
        return;
    }
    
    // Prefetch data with optimal memory access pattern
    prefetch_buffer[thread_position] = src[thread_position];
}
