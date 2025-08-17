//! Streaming Conversion for Large Tensors
//!
//! This module implements streaming conversion that processes large tensors in chunks
//! to minimize memory usage and enable processing of tensors larger than available memory.

use crate::memory::conversion::{
    ConversionResult, ConversionError, ConversionContext, Converter,
    config::StreamingConfig
};
use crate::memory::tensor::{BitNetTensor, BitNetDType};
use crate::memory::HybridMemoryPool;
use std::sync::Arc;
use std::thread;
use crossbeam_channel::bounded;

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

/// Streaming converter for processing large tensors in chunks
pub struct StreamingConverter {
    pub(crate) config: StreamingConfig,
}

impl StreamingConverter {
    /// Creates a new streaming converter with the given configuration
    pub fn new(config: StreamingConfig) -> ConversionResult<Self> {
        config.validate().map_err(|e| ConversionError::ConfigError { reason: e })?;
        
        Ok(Self { config })
    }

    /// Creates a streaming converter with default configuration
    pub fn default() -> ConversionResult<Self> {
        Self::new(StreamingConfig::default())
    }

    /// Performs streaming conversion of a large tensor
    pub fn stream_convert(
        &self,
        source: &BitNetTensor,
        target_dtype: BitNetDType,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<BitNetTensor> {
        let source_dtype = source.dtype();
        let shape = source.shape();
        let device = source.device();
        
        #[cfg(feature = "tracing")]
        info!("Starting streaming conversion from {} to {} for tensor shape {:?}", 
              source_dtype, target_dtype, shape);

        // Check if streaming is actually needed
        let total_size = source.size_bytes();
        if total_size < self.config.streaming_threshold {
            #[cfg(feature = "tracing")]
            debug!("Tensor size {} bytes is below streaming threshold, using standard conversion", total_size);
            return self.standard_convert(source, target_dtype, pool);
        }

        // Calculate chunk parameters
        let element_count: usize = shape.iter().product();
        let chunk_info = self.calculate_chunk_parameters(&shape, source_dtype, target_dtype)?;
        
        #[cfg(feature = "tracing")]
        debug!("Streaming conversion: {} chunks of {} elements each", 
               chunk_info.num_chunks, chunk_info.elements_per_chunk);

        // Create target tensor
        let target_tensor = BitNetTensor::zeros(&shape, target_dtype, &device, pool)
            .map_err(|e| ConversionError::InternalError { reason: e.to_string() })?;

        // Process chunks
        if self.config.parallel_chunks > 1 {
            self.parallel_stream_convert(source, &target_tensor, &chunk_info, pool)
        } else {
            self.sequential_stream_convert(source, &target_tensor, &chunk_info, pool)
        }?;

        #[cfg(feature = "tracing")]
        info!("Streaming conversion completed successfully");

        Ok(target_tensor)
    }

    /// Performs standard conversion for small tensors
    fn standard_convert(
        &self,
        source: &BitNetTensor,
        target_dtype: BitNetDType,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<BitNetTensor> {
        let shape = source.shape();
        let device = source.device();
        
        let target_tensor = BitNetTensor::zeros(&shape, target_dtype, &device, pool)
            .map_err(|e| ConversionError::InternalError { reason: e.to_string() })?;

        // Perform element-wise conversion
        self.convert_elements(
            source, &target_tensor, 
            0, shape.iter().product(), 
            &ConversionParams::new(source.dtype(), target_dtype)
        )?;

        Ok(target_tensor)
    }

    /// Performs sequential streaming conversion
    fn sequential_stream_convert(
        &self,
        source: &BitNetTensor,
        target: &BitNetTensor,
        chunk_info: &ChunkInfo,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<()> {
        let conversion_params = ConversionParams::new(source.dtype(), target.dtype());
        
        for chunk_idx in 0..chunk_info.num_chunks {
            let start_element = chunk_idx * chunk_info.elements_per_chunk;
            let end_element = std::cmp::min(
                start_element + chunk_info.elements_per_chunk,
                chunk_info.total_elements
            );
            
            #[cfg(feature = "tracing")]
            debug!("Processing chunk {} ({}-{})", chunk_idx, start_element, end_element);

            self.convert_elements(source, target, start_element, end_element, &conversion_params)?;
            
            // Optional: yield to other threads
            if chunk_idx % 10 == 0 {
                thread::yield_now();
            }
        }

        Ok(())
    }

    /// Performs parallel streaming conversion
    fn parallel_stream_convert(
        &self,
        source: &BitNetTensor,
        target: &BitNetTensor,
        chunk_info: &ChunkInfo,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<()> {
        let num_workers = std::cmp::min(self.config.parallel_chunks, chunk_info.num_chunks);
        let (task_sender, task_receiver) = bounded::<ChunkTask>(num_workers * 2);
        let (result_sender, result_receiver) = bounded::<ConversionResult<()>>(chunk_info.num_chunks);

        #[cfg(feature = "tracing")]
        debug!("Starting parallel streaming with {} workers", num_workers);

        // Spawn worker threads
        let mut workers = Vec::new();
        for worker_id in 0..num_workers {
            let task_rx = task_receiver.clone();
            let result_tx = result_sender.clone();
            let source_clone = source.clone();
            let target_clone = target.clone();
            let conversion_params = ConversionParams::new(source.dtype(), target.dtype());

            let worker = thread::spawn(move || {
                #[cfg(feature = "tracing")]
                debug!("Worker {} started", worker_id);

                while let Ok(task) = task_rx.recv() {
                    let result = Self::convert_elements_static(
                        &source_clone, &target_clone,
                        task.start_element, task.end_element,
                        &conversion_params
                    );
                    
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }

                #[cfg(feature = "tracing")]
                debug!("Worker {} finished", worker_id);
            });
            
            workers.push(worker);
        }

        // Send tasks to workers
        for chunk_idx in 0..chunk_info.num_chunks {
            let start_element = chunk_idx * chunk_info.elements_per_chunk;
            let end_element = std::cmp::min(
                start_element + chunk_info.elements_per_chunk,
                chunk_info.total_elements
            );

            let task = ChunkTask {
                chunk_id: chunk_idx,
                start_element,
                end_element,
            };

            task_sender.send(task).map_err(|_| ConversionError::StreamingError {
                reason: "Failed to send task to worker".to_string()
            })?;
        }

        // Close task channel
        drop(task_sender);

        // Collect results
        let mut errors = Vec::new();
        for _ in 0..chunk_info.num_chunks {
            match result_receiver.recv() {
                Ok(Ok(())) => {}, // Success
                Ok(Err(e)) => errors.push(e),
                Err(_) => errors.push(ConversionError::StreamingError {
                    reason: "Failed to receive result from worker".to_string()
                }),
            }
        }

        // Wait for all workers to finish
        for worker in workers {
            let _ = worker.join();
        }

        // Return first error if any occurred
        if let Some(error) = errors.into_iter().next() {
            return Err(error);
        }

        Ok(())
    }

    /// Converts elements in a specific range
    fn convert_elements(
        &self,
        source: &BitNetTensor,
        target: &BitNetTensor,
        start_element: usize,
        end_element: usize,
        params: &ConversionParams,
    ) -> ConversionResult<()> {
        Self::convert_elements_static(source, target, start_element, end_element, params)
    }

    /// Static version of convert_elements for use in worker threads
    fn convert_elements_static(
        source: &BitNetTensor,
        target: &BitNetTensor,
        start_element: usize,
        end_element: usize,
        params: &ConversionParams,
    ) -> ConversionResult<()> {
        let source_bytes_per_element = params.source_dtype.bits_per_element() / 8;
        let target_bytes_per_element = params.target_dtype.bits_per_element() / 8;
        
        let start_byte_offset = start_element * source_bytes_per_element;
        let target_start_byte_offset = start_element * target_bytes_per_element;

        unsafe {
            let source_ptr = source.data.memory_handle.as_ptr().add(start_byte_offset);
            let target_ptr = target.data.memory_handle.as_ptr().add(target_start_byte_offset) as *mut u8;

            // Perform the actual conversion based on data types
            match (params.source_dtype, params.target_dtype) {
                // F32 to F16 conversion
                (BitNetDType::F32, BitNetDType::F16) => {
                    Self::convert_f32_to_f16(source_ptr, target_ptr, end_element - start_element)?;
                }
                // F16 to F32 conversion
                (BitNetDType::F16, BitNetDType::F32) => {
                    Self::convert_f16_to_f32(source_ptr, target_ptr, end_element - start_element)?;
                }
                // F32 to I8 quantization
                (BitNetDType::F32, BitNetDType::I8) => {
                    Self::convert_f32_to_i8(source_ptr, target_ptr, end_element - start_element)?;
                }
                // I8 to F32 dequantization
                (BitNetDType::I8, BitNetDType::F32) => {
                    Self::convert_i8_to_f32(source_ptr, target_ptr, end_element - start_element)?;
                }
                // F32 to BitNet 1.58b quantization
                (BitNetDType::F32, BitNetDType::BitNet158) => {
                    Self::convert_f32_to_bitnet158(source_ptr, target_ptr, end_element - start_element)?;
                }
                // BitNet 1.58b to F32 dequantization
                (BitNetDType::BitNet158, BitNetDType::F32) => {
                    Self::convert_bitnet158_to_f32(source_ptr, target_ptr, end_element - start_element)?;
                }
                // Add more conversion cases as needed
                _ => {
                    return Err(ConversionError::UnsupportedConversion {
                        from: params.source_dtype,
                        to: params.target_dtype,
                    });
                }
            }
        }

        Ok(())
    }

    /// Calculates optimal chunk parameters for streaming
    fn calculate_chunk_parameters(
        &self,
        shape: &[usize],
        source_dtype: BitNetDType,
        target_dtype: BitNetDType,
    ) -> ConversionResult<ChunkInfo> {
        let total_elements: usize = shape.iter().product();
        let source_bytes_per_element = source_dtype.bits_per_element() / 8;
        let target_bytes_per_element = target_dtype.bits_per_element() / 8;
        
        // Calculate elements per chunk based on memory constraints
        let max_bytes_per_chunk = std::cmp::min(self.config.chunk_size, self.config.buffer_size / 2);
        let elements_per_chunk = max_bytes_per_chunk / std::cmp::max(source_bytes_per_element, target_bytes_per_element);
        let elements_per_chunk = std::cmp::max(1, elements_per_chunk);
        
        let num_chunks = (total_elements + elements_per_chunk - 1) / elements_per_chunk;

        Ok(ChunkInfo {
            total_elements,
            elements_per_chunk,
            num_chunks,
            source_bytes_per_element,
            target_bytes_per_element,
        })
    }

    // Conversion implementations for different data type pairs

    unsafe fn convert_f32_to_f16(
        source_ptr: *const u8,
        target_ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let source = std::slice::from_raw_parts(source_ptr as *const f32, element_count);
        let target = std::slice::from_raw_parts_mut(target_ptr as *mut u16, element_count);

        for i in 0..element_count {
            // Convert f32 to f16 (simplified - in practice would use proper IEEE 754 conversion)
            let f32_val = source[i];
            let f16_bits = if f32_val.is_nan() {
                0x7E00u16 // NaN
            } else if f32_val.is_infinite() {
                if f32_val.is_sign_positive() { 0x7C00u16 } else { 0xFC00u16 }
            } else {
                // Simplified conversion - truncate mantissa
                let bits = f32_val.to_bits();
                let sign = (bits >> 16) & 0x8000;
                let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
                let mantissa = (bits >> 13) & 0x3FF;
                
                if exp <= 0 {
                    sign as u16 // Underflow to zero
                } else if exp >= 31 {
                    sign as u16 | 0x7C00 // Overflow to infinity
                } else {
                    sign as u16 | ((exp as u16) << 10) | mantissa as u16
                }
            };
            target[i] = f16_bits;
        }

        Ok(())
    }

    unsafe fn convert_f16_to_f32(
        source_ptr: *const u8,
        target_ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let source = std::slice::from_raw_parts(source_ptr as *const u16, element_count);
        let target = std::slice::from_raw_parts_mut(target_ptr as *mut f32, element_count);

        for i in 0..element_count {
            let f16_bits = source[i];
            let sign = (f16_bits & 0x8000) as u32;
            let exp = ((f16_bits >> 10) & 0x1F) as i32;
            let mantissa = (f16_bits & 0x3FF) as u32;

            let f32_bits = if exp == 0 {
                if mantissa == 0 {
                    sign << 16 // Zero
                } else {
                    // Denormalized number
                    let exp_adj = 127 - 15 - 10;
                    sign << 16 | ((exp_adj as u32) << 23) | (mantissa << 13)
                }
            } else if exp == 31 {
                if mantissa == 0 {
                    sign << 16 | 0x7F800000 // Infinity
                } else {
                    sign << 16 | 0x7FC00000 // NaN
                }
            } else {
                // Normalized number
                let exp_adj = exp - 15 + 127;
                sign << 16 | ((exp_adj as u32) << 23) | (mantissa << 13)
            };

            target[i] = f32::from_bits(f32_bits);
        }

        Ok(())
    }

    unsafe fn convert_f32_to_i8(
        source_ptr: *const u8,
        target_ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let source = std::slice::from_raw_parts(source_ptr as *const f32, element_count);
        let target = std::slice::from_raw_parts_mut(target_ptr as *mut i8, element_count);

        for i in 0..element_count {
            let f32_val = source[i];
            // Simple quantization: clamp to [-128, 127] and round
            let quantized = if f32_val.is_nan() {
                0i8
            } else {
                let clamped = f32_val.clamp(-128.0, 127.0);
                clamped.round() as i8
            };
            target[i] = quantized;
        }

        Ok(())
    }

    unsafe fn convert_i8_to_f32(
        source_ptr: *const u8,
        target_ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let source = std::slice::from_raw_parts(source_ptr as *const i8, element_count);
        let target = std::slice::from_raw_parts_mut(target_ptr as *mut f32, element_count);

        for i in 0..element_count {
            target[i] = source[i] as f32;
        }

        Ok(())
    }

    unsafe fn convert_f32_to_bitnet158(
        source_ptr: *const u8,
        target_ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let source = std::slice::from_raw_parts(source_ptr as *const f32, element_count);
        
        // BitNet 1.58b uses 2 bits per element, packed 4 elements per byte
        let byte_count = (element_count + 3) / 4;
        let target = std::slice::from_raw_parts_mut(target_ptr, byte_count);

        for i in 0..element_count {
            let f32_val = source[i];
            // Quantize to {-1, 0, +1}
            let quantized = if f32_val > 0.5 {
                1u8 // +1 -> 01
            } else if f32_val < -0.5 {
                3u8 // -1 -> 11
            } else {
                0u8 // 0 -> 00
            };

            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            
            if bit_offset == 0 {
                target[byte_idx] = quantized;
            } else {
                target[byte_idx] |= quantized << bit_offset;
            }
        }

        Ok(())
    }

    unsafe fn convert_bitnet158_to_f32(
        source_ptr: *const u8,
        target_ptr: *mut u8,
        element_count: usize,
    ) -> ConversionResult<()> {
        let byte_count = (element_count + 3) / 4;
        let source = std::slice::from_raw_parts(source_ptr, byte_count);
        let target = std::slice::from_raw_parts_mut(target_ptr as *mut f32, element_count);

        for i in 0..element_count {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let quantized = (source[byte_idx] >> bit_offset) & 0x3;

            let f32_val = match quantized {
                0 => 0.0f32,   // 00 -> 0
                1 => 1.0f32,   // 01 -> +1
                3 => -1.0f32,  // 11 -> -1
                _ => 0.0f32,   // Invalid, default to 0
            };

            target[i] = f32_val;
        }

        Ok(())
    }
}

impl Converter for StreamingConverter {
    fn convert(
        &self,
        source: &BitNetTensor,
        context: &ConversionContext,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<BitNetTensor> {
        self.stream_convert(source, context.target_dtype, pool)
    }

    fn supports(&self, context: &ConversionContext) -> bool {
        // Streaming converter supports most conversions
        !context.is_zero_copy_compatible() && 
        std::mem::discriminant(&context.source_device) == std::mem::discriminant(&context.target_device)
    }

    fn estimate_time_ms(&self, context: &ConversionContext) -> u64 {
        let element_count: usize = context.shape.iter().product();
        let size_bytes = context.source_dtype.bytes_for_elements(element_count);
        
        // Estimate based on processing speed (~1 GB/s for conversions)
        let base_time = (size_bytes as f64) / (1024.0 * 1024.0 * 1024.0) * 1000.0;
        
        // Add overhead for chunking and parallel processing
        let overhead_factor = if size_bytes > self.config.streaming_threshold {
            1.2 // 20% overhead for streaming
        } else {
            1.0
        };
        
        (base_time * overhead_factor) as u64
    }
}

/// Information about chunk processing
#[derive(Debug, Clone)]
struct ChunkInfo {
    total_elements: usize,
    elements_per_chunk: usize,
    num_chunks: usize,
    source_bytes_per_element: usize,
    target_bytes_per_element: usize,
}

/// Task for processing a chunk
#[derive(Debug, Clone)]
struct ChunkTask {
    chunk_id: usize,
    start_element: usize,
    end_element: usize,
}

/// Parameters for conversion operations
#[derive(Debug, Clone)]
struct ConversionParams {
    source_dtype: BitNetDType,
    target_dtype: BitNetDType,
}

impl ConversionParams {
    fn new(source_dtype: BitNetDType, target_dtype: BitNetDType) -> Self {
        Self {
            source_dtype,
            target_dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;
    use crate::memory::HybridMemoryPool;

    #[test]
    fn test_streaming_converter_creation() {
        let config = StreamingConfig::default();
        let converter = StreamingConverter::new(config).unwrap();
        assert_eq!(converter.config.chunk_size, 1024 * 1024);
    }

    #[test]
    fn test_chunk_calculation() {
        let config = StreamingConfig::default();
        let converter = StreamingConverter::new(config).unwrap();
        
        let shape = vec![1000, 1000];
        let chunk_info = converter.calculate_chunk_parameters(
            &shape, 
            BitNetDType::F32, 
            BitNetDType::F16
        ).unwrap();
        
        assert_eq!(chunk_info.total_elements, 1_000_000);
        assert!(chunk_info.num_chunks > 1);
        assert!(chunk_info.elements_per_chunk > 0);
    }

    #[test]
    fn test_small_tensor_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = StreamingConfig::default();
        let converter = StreamingConverter::new(config).unwrap();

        let source = BitNetTensor::ones(&[10, 10], BitNetDType::F32, &device, &pool).unwrap();
        let result = converter.stream_convert(&source, BitNetDType::F16, &pool).unwrap();

        assert_eq!(result.dtype(), BitNetDType::F16);
        assert_eq!(result.shape(), vec![10, 10]);
    }

    #[test]
    fn test_converter_trait_implementation() {
        let device = get_cpu_device();
        let config = StreamingConfig::default();
        let converter = StreamingConverter::new(config).unwrap();

        let context = ConversionContext::new(
            BitNetDType::F32,
            BitNetDType::I8,
            device.clone(),
            device.clone(),
            vec![1000, 1000],
        );

        assert!(converter.supports(&context));
        let time_estimate = converter.estimate_time_ms(&context);
        assert!(time_estimate > 0);
    }

    #[test]
    fn test_unsupported_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = StreamingConfig::default();
        let converter = StreamingConverter::new(config).unwrap();

        let source = BitNetTensor::zeros(&[10, 10], BitNetDType::F32, &device, &pool).unwrap();
        
        // Test with an unsupported conversion (this would need to be implemented)
        // For now, most conversions are supported, so this test checks the error path
        let result = converter.stream_convert(&source, BitNetDType::I4, &pool);
        assert!(result.is_err());
    }
}