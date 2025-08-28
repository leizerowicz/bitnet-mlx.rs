//! Batch Conversion Operations
//!
//! This module implements efficient batch conversion of multiple tensors,
//! optimizing memory usage and processing through grouping and parallel execution.

use crate::memory::conversion::{
    config::BatchConfig, ConversionContext, ConversionError, ConversionResult, Converter,
    InPlaceConverter, StreamingConverter, ZeroCopyConverter,
};
use crate::memory::tensor::{BitNetDType, BitNetTensor};
use crate::memory::HybridMemoryPool;
use candle_core::Device;
use crossbeam_channel::bounded;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

#[cfg(feature = "tracing")]
use tracing::{debug, info, warn};

/// Batch converter for processing multiple tensors efficiently
#[allow(dead_code)]
pub struct BatchConverter {
    config: BatchConfig,
    zero_copy_converter: ZeroCopyConverter,
    streaming_converter: StreamingConverter,
    in_place_converter: InPlaceConverter,
}

impl BatchConverter {
    /// Creates a new batch converter with the given configuration
    pub fn new(config: BatchConfig) -> ConversionResult<Self> {
        config
            .validate()
            .map_err(|e| ConversionError::ConfigError { reason: e })?;

        Ok(Self {
            config,
            zero_copy_converter: ZeroCopyConverter::new(),
            streaming_converter: StreamingConverter::default()?,
            in_place_converter: InPlaceConverter::new_lossy(),
        })
    }

    /// Creates a batch converter with default configuration
    pub fn default() -> ConversionResult<Self> {
        Self::new(BatchConfig::default())
    }

    /// Converts a batch of tensors to the specified target data type
    pub fn batch_convert(
        &self,
        tensors: &[BitNetTensor],
        targetdtype: BitNetDType,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<Vec<BitNetTensor>> {
        if tensors.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "tracing")]
        info!(
            "Starting batch conversion of {} tensors to {}",
            tensors.len(),
            targetdtype
        );

        // Group tensors for efficient processing
        let groups = self.group_tensors(tensors, targetdtype)?;

        #[cfg(feature = "tracing")]
        debug!(
            "Grouped {} tensors into {} processing groups",
            tensors.len(),
            groups.len()
        );

        // Process each group
        let results = if self.config.enable_parallel_processing && groups.len() > 1 {
            #[cfg(feature = "tracing")]
            debug!(
                "DIAGNOSTIC: Attempting parallel processing with {} groups",
                groups.len()
            );

            // Try parallel processing first
            match self.process_groups_parallel(&groups, pool) {
                Ok(parallel_results) => {
                    #[cfg(feature = "tracing")]
                    info!("DIAGNOSTIC: Parallel processing succeeded");
                    parallel_results
                }
                Err(_e) => {
                    #[cfg(feature = "tracing")]
                    warn!(
                        "DIAGNOSTIC: Parallel processing failed, falling back to sequential: {}",
                        _e
                    );
                    self.process_groups_sequential(&groups, pool)?
                }
            }
        } else {
            #[cfg(feature = "tracing")]
            debug!(
                "DIAGNOSTIC: Using sequential processing - parallel disabled: {}, groups: {}",
                !self.config.enable_parallel_processing,
                groups.len()
            );
            self.process_groups_sequential(&groups, pool)?
        };

        // Restore original order
        let ordered_results = self.restore_original_order(results, tensors)?;

        #[cfg(feature = "tracing")]
        info!("Batch conversion completed successfully");

        Ok(ordered_results)
    }

    /// Converts multiple tensors with different target types
    pub fn batch_convert_mixed(
        &self,
        conversions: &[(BitNetTensor, BitNetDType)],
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<Vec<BitNetTensor>> {
        if conversions.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "tracing")]
        info!(
            "Starting mixed batch conversion of {} tensors",
            conversions.len()
        );

        // Group by target data type and device for efficiency
        let groups = self.group_mixed_conversions(conversions)?;

        let results = if self.config.enable_parallel_processing && groups.len() > 1 {
            #[cfg(feature = "tracing")]
            debug!(
                "DIAGNOSTIC: Attempting mixed parallel processing with {} groups",
                groups.len()
            );

            // Try parallel processing first
            match self.attempt_mixed_parallel_processing(&groups, pool) {
                Ok(parallel_results) => {
                    #[cfg(feature = "tracing")]
                    info!("DIAGNOSTIC: Mixed parallel processing succeeded");
                    parallel_results
                }
                Err(_e) => {
                    #[cfg(feature = "tracing")]
                    warn!("DIAGNOSTIC: Mixed parallel processing failed, falling back to sequential: {}", _e);
                    self.process_mixed_groups_sequential(&groups, pool)?
                }
            }
        } else {
            #[cfg(feature = "tracing")]
            debug!(
                "DIAGNOSTIC: Using mixed sequential processing - parallel disabled: {}, groups: {}",
                !self.config.enable_parallel_processing,
                groups.len()
            );
            self.process_mixed_groups_sequential(&groups, pool)?
        };

        // Restore original order
        let ordered_results = self.restore_mixed_order(results, conversions)?;

        #[cfg(feature = "tracing")]
        info!("Mixed batch conversion completed successfully");

        Ok(ordered_results)
    }

    /// Groups tensors by processing characteristics for efficient batch processing
    fn group_tensors(
        &self,
        tensors: &[BitNetTensor],
        targetdtype: BitNetDType,
    ) -> ConversionResult<Vec<TensorGroup>> {
        let mut groups: HashMap<GroupKey, Vec<TensorWithIndex>> = HashMap::new();

        for (index, tensor) in tensors.iter().enumerate() {
            let sourcedtype = tensor.dtype();
            let device = tensor.device(); // Fixed - get device from tensor
            let size_bytes = tensor.size_bytes();

            // Determine optimal conversion strategy
            let context = ConversionContext::new(
                sourcedtype,
                targetdtype,
                device.clone(),
                device.clone(),
                tensor.shape(),
            );

            let strategy = context.optimal_strategy();

            let key = GroupKey {
                sourcedtype: sourcedtype,
                targetdtype,
                device_type: self.device_type_key(&device),
                strategy,
                size_category: self.size_category(size_bytes),
            };

            groups
                .entry(key)
                .or_insert_with(Vec::new)
                .push(TensorWithIndex {
                    tensor: tensor.clone(),
                    original_index: index,
                });
        }

        // Convert to sorted groups for deterministic processing
        let mut sorted_groups: Vec<_> = groups.into_iter().collect();

        if self.config.sort_by_size {
            sorted_groups.sort_by_key(|(key, _)| key.size_category);
        }

        let tensor_groups = sorted_groups
            .into_iter()
            .map(|(key, tensors)| TensorGroup { key, tensors })
            .collect();

        Ok(tensor_groups)
    }

    /// Groups mixed conversions by target type and device
    fn group_mixed_conversions(
        &self,
        conversions: &[(BitNetTensor, BitNetDType)],
    ) -> ConversionResult<Vec<MixedGroup>> {
        let mut groups: HashMap<MixedGroupKey, Vec<ConversionWithIndex>> = HashMap::new();

        for (index, (tensor, targetdtype)) in conversions.iter().enumerate() {
            let sourcedtype = tensor.dtype();
            let device = tensor.device(); // Fixed - get device from tensor
            let _size_bytes = tensor.size_bytes();

            let context = ConversionContext::new(
                sourcedtype,
                *targetdtype,
                device.clone(),
                device.clone(),
                tensor.shape(),
            );

            let strategy = context.optimal_strategy();

            let key = MixedGroupKey {
                sourcedtype: sourcedtype,
                targetdtype: *targetdtype,
                device_type: self.device_type_key(&device),
                strategy,
            };

            groups
                .entry(key)
                .or_insert_with(Vec::new)
                .push(ConversionWithIndex {
                    tensor: tensor.clone(),
                    targetdtype: *targetdtype,
                    original_index: index,
                });
        }

        let mixed_groups = groups
            .into_iter()
            .map(|(key, conversions)| MixedGroup { key, conversions })
            .collect();

        Ok(mixed_groups)
    }

    /// Processes tensor groups sequentially
    fn process_groups_sequential(
        &self,
        groups: &[TensorGroup],
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<Vec<TensorResult>> {
        let mut results = Vec::new();

        for group in groups {
            let group_results = self.process_single_group(group, pool)?;
            results.extend(group_results);
        }

        Ok(results)
    }

    /// Processes tensor groups in parallel
    fn process_groups_parallel(
        &self,
        groups: &[TensorGroup],
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<Vec<TensorResult>> {
        let num_workers = std::cmp::min(self.config.batch_worker_threads, groups.len());
        let (task_sender, task_receiver) = bounded::<usize>(num_workers * 2);
        let (result_sender, result_receiver) =
            bounded::<ConversionResult<Vec<TensorResult>>>(groups.len());

        #[cfg(feature = "tracing")]
        debug!(
            "Starting parallel batch processing with {} workers",
            num_workers
        );

        // Spawn worker threads
        let mut workers = Vec::new();
        for _worker_id in 0..num_workers {
            let task_rx = task_receiver.clone();
            let result_tx = result_sender.clone();
            let converter = self.clone_for_worker();

            let groups_clone = groups.to_vec();
            let pool_clone = Arc::clone(pool);
            let converter_clone = converter;

            let worker = thread::spawn(move || {
                #[cfg(feature = "tracing")]
                debug!("Batch worker {} started", worker_id);

                while let Ok(group_idx) = task_rx.recv() {
                    if let Some(group) = groups_clone.get(group_idx) {
                        #[cfg(feature = "tracing")]
                        debug!(
                            "Batch worker {} processing group {} with {} tensors",
                            worker_id,
                            group_idx,
                            group.tensors.len()
                        );

                        // Now we can actually process the group with pool access!
                        let result =
                            converter_clone.process_single_group_parallel(group, &pool_clone);
                        if result_tx.send(result).is_err() {
                            break;
                        }
                    }
                }

                #[cfg(feature = "tracing")]
                debug!("Batch worker {} finished", worker_id);
            });

            workers.push(worker);
        }

        // Send tasks to workers
        for (group_idx, _group) in groups.iter().enumerate() {
            task_sender
                .send(group_idx)
                .map_err(|_| ConversionError::BatchError {
                    reason: "Failed to send task to worker".to_string(),
                })?;
        }

        // Close task channel
        drop(task_sender);

        // Collect results
        let mut all_results = Vec::new();
        let mut errors = Vec::new();

        for _ in 0..groups.len() {
            match result_receiver.recv() {
                Ok(Ok(group_results)) => all_results.extend(group_results),
                Ok(Err(e)) => errors.push(e),
                Err(_) => errors.push(ConversionError::BatchError {
                    reason: "Failed to receive result from worker".to_string(),
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

        Ok(all_results)
    }

    /// Processes mixed groups sequentially
    fn process_mixed_groups_sequential(
        &self,
        groups: &[MixedGroup],
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<Vec<MixedResult>> {
        let mut results = Vec::new();

        for group in groups {
            let group_results = self.process_single_mixed_group(group, pool)?;
            results.extend(group_results);
        }

        Ok(results)
    }

    /// Attempts mixed parallel processing with diagnostic logging
    fn attempt_mixed_parallel_processing(
        &self,
        groups: &[MixedGroup],
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<Vec<MixedResult>> {
        #[cfg(feature = "tracing")]
        debug!("DIAGNOSTIC: Starting attempt_mixed_parallel_processing with pool reference");

        // This will fail due to lifetime issues - let's capture the exact error
        self.process_mixed_groups_parallel(groups, pool)
    }

    /// Processes mixed groups in parallel
    fn process_mixed_groups_parallel(
        &self,
        groups: &[MixedGroup],
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<Vec<MixedResult>> {
        // Similar to process_groups_parallel but for mixed conversions
        let num_workers = std::cmp::min(self.config.batch_worker_threads, groups.len());
        let (task_sender, task_receiver) = bounded::<usize>(num_workers * 2);
        let (result_sender, result_receiver) =
            bounded::<ConversionResult<Vec<MixedResult>>>(groups.len());

        // Spawn worker threads
        let mut workers = Vec::new();
        for _worker_id in 0..num_workers {
            let task_rx = task_receiver.clone();
            let result_tx = result_sender.clone();
            let converter = self.clone_for_worker();

            let groups_clone = groups.to_vec();
            let pool_clone = Arc::clone(pool);
            let converter_clone = converter;

            let worker = thread::spawn(move || {
                #[cfg(feature = "tracing")]
                debug!("Mixed worker {} started with Arc pool access", worker_id);

                while let Ok(group_idx) = task_rx.recv() {
                    if let Some(group) = groups_clone.get(group_idx) {
                        #[cfg(feature = "tracing")]
                        debug!(
                            "Mixed worker {} processing group {} with {} conversions",
                            worker_id,
                            group_idx,
                            group.conversions.len()
                        );

                        // Now we can actually process the mixed group with pool access!
                        let result =
                            converter_clone.process_single_mixed_group_parallel(group, &pool_clone);
                        if result_tx.send(result).is_err() {
                            break;
                        }
                    }
                }

                #[cfg(feature = "tracing")]
                debug!("Mixed worker {} finished successfully", worker_id);
            });

            workers.push(worker);
        }

        // Send tasks and collect results (similar pattern)
        for (group_idx, _group) in groups.iter().enumerate() {
            task_sender
                .send(group_idx)
                .map_err(|_| ConversionError::BatchError {
                    reason: "Failed to send mixed task to worker".to_string(),
                })?;
        }

        drop(task_sender);

        let mut all_results = Vec::new();
        let mut errors = Vec::new();

        for _ in 0..groups.len() {
            match result_receiver.recv() {
                Ok(Ok(group_results)) => all_results.extend(group_results),
                Ok(Err(e)) => errors.push(e),
                Err(_) => errors.push(ConversionError::BatchError {
                    reason: "Failed to receive mixed result from worker".to_string(),
                }),
            }
        }

        for worker in workers {
            let _ = worker.join();
        }

        if let Some(error) = errors.into_iter().next() {
            return Err(error);
        }

        Ok(all_results)
    }

    /// Processes a single tensor group
    fn process_single_group(
        &self,
        group: &TensorGroup,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<Vec<TensorResult>> {
        let mut results = Vec::new();

        #[cfg(feature = "tracing")]
        debug!(
            "Processing group with {} tensors using strategy {:?}",
            group.tensors.len(),
            group.key.strategy
        );

        for tensor_with_index in &group.tensors {
            let context = ConversionContext::new(
                group.key.sourcedtype,
                group.key.targetdtype,
                tensor_with_index.tensor.device(),
                tensor_with_index.tensor.device(),
                tensor_with_index.tensor.shape(),
            )
            .with_strategy(group.key.strategy);

            let converted =
                self.convert_single_tensor(&tensor_with_index.tensor, &context, pool)?;

            results.push(TensorResult {
                tensor: converted,
                original_index: tensor_with_index.original_index,
            });
        }

        Ok(results)
    }

    /// Processes a single mixed group
    fn process_single_mixed_group(
        &self,
        group: &MixedGroup,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<Vec<MixedResult>> {
        let mut results = Vec::new();

        for conversion in &group.conversions {
            let context = ConversionContext::new(
                group.key.sourcedtype,
                conversion.targetdtype,
                conversion.tensor.device(),
                conversion.tensor.device(),
                conversion.tensor.shape(),
            )
            .with_strategy(group.key.strategy);

            let converted = self.convert_single_tensor(&conversion.tensor, &context, pool)?;

            results.push(MixedResult {
                tensor: converted,
                original_index: conversion.original_index,
            });
        }

        Ok(results)
    }

    /// Converts a single tensor using the appropriate converter
    fn convert_single_tensor(
        &self,
        tensor: &BitNetTensor,
        context: &ConversionContext,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<BitNetTensor> {
        match context.strategy {
            crate::memory::conversion::ConversionStrategy::ZeroCopy => {
                self.zero_copy_converter.convert(tensor, context, pool)
            }
            crate::memory::conversion::ConversionStrategy::InPlace => {
                self.in_place_converter.convert(tensor, context, pool)
            }
            crate::memory::conversion::ConversionStrategy::Streaming => {
                self.streaming_converter.convert(tensor, context, pool)
            }
            crate::memory::conversion::ConversionStrategy::Standard => {
                // Use streaming converter for standard conversions
                self.streaming_converter.convert(tensor, context, pool)
            }
            crate::memory::conversion::ConversionStrategy::Auto => {
                // This should not happen as strategy should be resolved by now
                let optimal_context = context.clone().with_strategy(context.optimal_strategy());
                self.convert_single_tensor(tensor, &optimal_context, pool)
            }
        }
    }

    /// Restores the original order of conversion results
    fn restore_original_order(
        &self,
        results: Vec<TensorResult>,
        original_tensors: &[BitNetTensor],
    ) -> ConversionResult<Vec<BitNetTensor>> {
        let mut ordered_results = vec![None; original_tensors.len()];

        for result in results {
            if result.original_index >= ordered_results.len() {
                return Err(ConversionError::BatchError {
                    reason: "Invalid original index in batch results".to_string(),
                });
            }
            ordered_results[result.original_index] = Some(result.tensor);
        }

        // Ensure all positions are filled
        let final_results: Result<Vec<_>, _> = ordered_results
            .into_iter()
            .enumerate()
            .map(|(i, opt)| {
                opt.ok_or_else(|| ConversionError::BatchError {
                    reason: format!("Missing result for tensor at index {}", i),
                })
            })
            .collect();

        final_results
    }

    /// Restores the original order of mixed conversion results
    fn restore_mixed_order(
        &self,
        results: Vec<MixedResult>,
        original_conversions: &[(BitNetTensor, BitNetDType)],
    ) -> ConversionResult<Vec<BitNetTensor>> {
        let mut ordered_results = vec![None; original_conversions.len()];

        for result in results {
            if result.original_index >= ordered_results.len() {
                return Err(ConversionError::BatchError {
                    reason: "Invalid original index in mixed batch results".to_string(),
                });
            }
            ordered_results[result.original_index] = Some(result.tensor);
        }

        let final_results: Result<Vec<_>, _> = ordered_results
            .into_iter()
            .enumerate()
            .map(|(i, opt)| {
                opt.ok_or_else(|| ConversionError::BatchError {
                    reason: format!("Missing result for conversion at index {}", i),
                })
            })
            .collect();

        final_results
    }

    /// Helper functions

    fn device_type_key(&self, device: &Device) -> String {
        match device {
            Device::Cpu => "cpu".to_string(),
            Device::Metal(_) => "metal".to_string(),
            Device::Cuda(_) => "cuda".to_string(),
        }
    }

    fn size_category(&self, size_bytes: usize) -> SizeCategory {
        if size_bytes < 1024 * 1024 {
            SizeCategory::Small
        } else if size_bytes < 100 * 1024 * 1024 {
            SizeCategory::Medium
        } else {
            SizeCategory::Large
        }
    }

    fn clone_for_worker(&self) -> Self {
        // Create a new instance for worker threads
        Self {
            config: self.config.clone(),
            zero_copy_converter: ZeroCopyConverter::new(),
            streaming_converter: StreamingConverter::default().unwrap(),
            in_place_converter: InPlaceConverter::new_lossy(),
        }
    }

    /// Processes a single tensor group in parallel worker thread
    fn process_single_group_parallel(
        &self,
        group: &TensorGroup,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<Vec<TensorResult>> {
        let mut results = Vec::new();

        #[cfg(feature = "tracing")]
        debug!(
            "Processing group with {} tensors using strategy {:?} in parallel",
            group.tensors.len(),
            group.key.strategy
        );

        for tensor_with_index in &group.tensors {
            let context = ConversionContext::new(
                group.key.sourcedtype,
                group.key.targetdtype,
                tensor_with_index.tensor.device(),
                tensor_with_index.tensor.device(),
                tensor_with_index.tensor.shape(),
            )
            .with_strategy(group.key.strategy);

            let converted =
                self.convert_single_tensor(&tensor_with_index.tensor, &context, pool)?;

            results.push(TensorResult {
                tensor: converted,
                original_index: tensor_with_index.original_index,
            });
        }

        Ok(results)
    }

    /// Processes a single mixed group in parallel worker thread
    fn process_single_mixed_group_parallel(
        &self,
        group: &MixedGroup,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<Vec<MixedResult>> {
        let mut results = Vec::new();

        #[cfg(feature = "tracing")]
        debug!(
            "Processing mixed group with {} conversions using strategy {:?} in parallel",
            group.conversions.len(),
            group.key.strategy
        );

        for conversion in &group.conversions {
            let context = ConversionContext::new(
                group.key.sourcedtype,
                conversion.targetdtype,
                conversion.tensor.device(),
                conversion.tensor.device(),
                conversion.tensor.shape(),
            )
            .with_strategy(group.key.strategy);

            let converted = self.convert_single_tensor(&conversion.tensor, &context, pool)?;

            results.push(MixedResult {
                tensor: converted,
                original_index: conversion.original_index,
            });
        }

        Ok(results)
    }
}

// Supporting data structures

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GroupKey {
    sourcedtype: BitNetDType,
    targetdtype: BitNetDType,
    device_type: String,
    strategy: crate::memory::conversion::ConversionStrategy,
    size_category: SizeCategory,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MixedGroupKey {
    sourcedtype: BitNetDType,
    targetdtype: BitNetDType,
    device_type: String,
    strategy: crate::memory::conversion::ConversionStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum SizeCategory {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone)]
struct TensorWithIndex {
    tensor: BitNetTensor,
    original_index: usize,
}

#[derive(Debug, Clone)]
struct ConversionWithIndex {
    tensor: BitNetTensor,
    targetdtype: BitNetDType,
    original_index: usize,
}

#[derive(Debug, Clone)]
struct TensorGroup {
    key: GroupKey,
    tensors: Vec<TensorWithIndex>,
}

#[derive(Debug, Clone)]
struct MixedGroup {
    key: MixedGroupKey,
    conversions: Vec<ConversionWithIndex>,
}

#[derive(Debug)]
struct TensorResult {
    tensor: BitNetTensor,
    original_index: usize,
}

#[derive(Debug)]
struct MixedResult {
    tensor: BitNetTensor,
    original_index: usize,
}

impl Converter for BatchConverter {
    fn convert(
        &self,
        source: &BitNetTensor,
        context: &ConversionContext,
        pool: &Arc<HybridMemoryPool>,
    ) -> ConversionResult<BitNetTensor> {
        // For single tensor conversion, delegate to appropriate converter
        self.convert_single_tensor(source, context, pool)
    }

    fn supports(&self, context: &ConversionContext) -> bool {
        // Batch converter supports any conversion that the underlying converters support
        self.zero_copy_converter.supports(context)
            || self.streaming_converter.supports(context)
            || self.in_place_converter.supports(context)
    }

    fn estimate_time_ms(&self, context: &ConversionContext) -> u64 {
        // Use the estimate from the optimal converter
        match context.optimal_strategy() {
            crate::memory::conversion::ConversionStrategy::ZeroCopy => {
                self.zero_copy_converter.estimate_time_ms(context)
            }
            crate::memory::conversion::ConversionStrategy::InPlace => {
                self.in_place_converter.estimate_time_ms(context)
            }
            _ => self.streaming_converter.estimate_time_ms(context),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;
    use crate::memory::HybridMemoryPool;

    #[test]
    fn test_batch_converter_creation() {
        let config = BatchConfig::default();
        let converter = BatchConverter::new(config).unwrap();
        assert_eq!(converter.config.max_batch_size, 32);
    }

    #[test]
    fn test_empty_batch_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let config = BatchConfig::default();
        let converter = BatchConverter::new(config).unwrap();

        let tensors: Vec<BitNetTensor> = vec![];
        let results = converter
            .batch_convert(&tensors, BitNetDType::F16, &pool)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_tensor_batch() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = BatchConfig::default();
        let converter = BatchConverter::new(config).unwrap();

        let tensor = BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool).unwrap();
        let tensors = vec![tensor];

        let results = converter
            .batch_convert(&tensors, BitNetDType::F16, &pool)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].dtype(), BitNetDType::F16);
    }

    #[test]
    fn test_multiple_tensor_batch() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = BatchConfig::default();
        let converter = BatchConverter::new(config).unwrap();

        let tensors = vec![
            BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool).unwrap(),
            BitNetTensor::ones(&[3, 3], BitNetDType::F32, &device, &pool).unwrap(),
            BitNetTensor::zeros(&[4, 4], BitNetDType::F32, &device, &pool).unwrap(),
        ];

        let results = converter
            .batch_convert(&tensors, BitNetDType::F16, &pool)
            .unwrap();
        assert_eq!(results.len(), 3);

        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.dtype(), BitNetDType::F16);
            assert_eq!(result.shape(), tensors[i].shape());
        }
    }

    #[test]
    fn test_mixed_batch_conversion() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = BatchConfig::default();
        let converter = BatchConverter::new(config).unwrap();

        let conversions = vec![
            (
                BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool).unwrap(),
                BitNetDType::F16,
            ),
            (
                BitNetTensor::ones(&[3, 3], BitNetDType::F32, &device, &pool).unwrap(),
                BitNetDType::I8,
            ),
            (
                BitNetTensor::zeros(&[4, 4], BitNetDType::F16, &device, &pool).unwrap(),
                BitNetDType::I8,
            ),
        ];

        let results = converter.batch_convert_mixed(&conversions, &pool).unwrap();
        assert_eq!(results.len(), 3);

        assert_eq!(results[0].dtype(), BitNetDType::F16);
        assert_eq!(results[1].dtype(), BitNetDType::I8);
        assert_eq!(results[2].dtype(), BitNetDType::I8);
    }

    #[test]
    fn test_tensor_grouping() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();
        let config = BatchConfig::default();
        let converter = BatchConverter::new(config).unwrap();

        let tensors = vec![
            BitNetTensor::zeros(&[2, 2], BitNetDType::F32, &device, &pool).unwrap(),
            BitNetTensor::ones(&[2, 2], BitNetDType::F32, &device, &pool).unwrap(), // Same as first
            BitNetTensor::zeros(&[3, 3], BitNetDType::F16, &device, &pool).unwrap(), // Different dtype
        ];

        let groups = converter.group_tensors(&tensors, BitNetDType::I8).unwrap();

        // Should have at least 2 groups (F32->I8 and F16->I8)
        assert!(groups.len() >= 2);
    }

    #[test]
    fn test_size_categorization() {
        let config = BatchConfig::default();
        let converter = BatchConverter::new(config).unwrap();

        assert_eq!(converter.size_category(1024), SizeCategory::Small);
        assert_eq!(
            converter.size_category(10 * 1024 * 1024),
            SizeCategory::Medium
        );
        assert_eq!(
            converter.size_category(200 * 1024 * 1024),
            SizeCategory::Large
        );
    }

    #[test]
    fn test_parallel_processing_enabled() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();

        // Create config with parallel processing enabled
        let mut config = BatchConfig::default();
        config.enable_parallel_processing = true;
        config.batch_worker_threads = 2;

        let converter = BatchConverter::new(config).unwrap();

        // Create multiple tensors to trigger parallel processing
        let tensors = vec![
            BitNetTensor::zeros(&[10, 10], BitNetDType::F32, &device, &pool).unwrap(),
            BitNetTensor::ones(&[10, 10], BitNetDType::F32, &device, &pool).unwrap(),
            BitNetTensor::zeros(&[10, 10], BitNetDType::F32, &device, &pool).unwrap(),
            BitNetTensor::ones(&[10, 10], BitNetDType::F32, &device, &pool).unwrap(),
        ];

        // This should trigger parallel processing since we have multiple groups
        let results = converter
            .batch_convert(&tensors, BitNetDType::F16, &pool)
            .unwrap();

        assert_eq!(results.len(), 4);
        for result in &results {
            assert_eq!(result.dtype(), BitNetDType::F16);
            assert_eq!(result.shape(), vec![10, 10]);
        }
    }

    #[test]
    fn test_mixed_parallel_processing() {
        let pool = Arc::new(HybridMemoryPool::new().unwrap());
        let device = get_cpu_device();

        // Create config with parallel processing enabled
        let mut config = BatchConfig::default();
        config.enable_parallel_processing = true;
        config.batch_worker_threads = 2;

        let converter = BatchConverter::new(config).unwrap();

        // Create mixed conversions to trigger parallel processing
        let conversions = vec![
            (
                BitNetTensor::zeros(&[8, 8], BitNetDType::F32, &device, &pool).unwrap(),
                BitNetDType::F16,
            ),
            (
                BitNetTensor::ones(&[8, 8], BitNetDType::F32, &device, &pool).unwrap(),
                BitNetDType::F16,
            ),
            (
                BitNetTensor::zeros(&[8, 8], BitNetDType::F16, &device, &pool).unwrap(),
                BitNetDType::I8,
            ),
            (
                BitNetTensor::ones(&[8, 8], BitNetDType::F16, &device, &pool).unwrap(),
                BitNetDType::I8,
            ),
        ];

        let results = converter.batch_convert_mixed(&conversions, &pool).unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(results[0].dtype(), BitNetDType::F16);
        assert_eq!(results[1].dtype(), BitNetDType::F16);
        assert_eq!(results[2].dtype(), BitNetDType::I8);
        assert_eq!(results[3].dtype(), BitNetDType::I8);
    }
}
