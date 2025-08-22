//! Calibration dataset implementation
//!
//! This module provides the main CalibrationDataset implementation with support
//! for loading data, streaming large datasets, and collecting activation statistics
//! for quantization parameter optimization.

use crate::calibration::error::{CalibrationError, CalibrationResult};
use crate::calibration::config::CalibrationConfig;
use crate::calibration::statistics::{StatisticsCollector, LayerStatistics, StatisticsUpdate};
use crate::calibration::sampling::{SamplerFactory, RepresentativeSampler, SamplingMetadata};
use crate::calibration::streaming::StreamingProcessor;
use candle_core::{Tensor, Device};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use serde::{Deserialize, Serialize};

/// Main calibration dataset for quantization parameter optimization
pub struct CalibrationDataset {
    /// Configuration
    config: CalibrationConfig,
    /// Statistics collector
    statistics_collector: StatisticsCollector,
    /// Representative sampler
    sampler: Box<dyn RepresentativeSampler>,
    /// Streaming processor (if enabled)
    streaming_processor: Option<StreamingProcessor>,
    /// Loaded data samples
    samples: Vec<Tensor>,
    /// Sample metadata
    sample_metadata: Vec<SampleMetadata>,
    /// Device for tensor operations
    device: Device,
    /// Processing state
    state: DatasetState,
    /// Performance metrics
    metrics: ProcessingMetrics,
}

/// Metadata for individual samples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleMetadata {
    /// Sample identifier
    pub id: String,
    /// Sample source path
    pub source_path: Option<PathBuf>,
    /// Sample shape information
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
    /// Sample weight for importance sampling
    pub weight: f32,
    /// Processing timestamp
    pub processed_at: SystemTime,
    /// Additional custom metadata
    pub custom_fields: HashMap<String, String>,
}

/// Dataset processing state
#[derive(Debug, Clone, PartialEq)]
pub enum DatasetState {
    /// Dataset is being initialized
    Initializing,
    /// Dataset is loading data
    Loading,
    /// Dataset is ready for processing
    Ready,
    /// Dataset is processing samples
    Processing,
    /// Dataset processing is completed
    Complete,
    /// Dataset encountered an error
    Error(String),
}

/// Performance and processing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Total samples loaded
    pub total_samples: usize,
    /// Samples processed
    pub processed_samples: usize,
    /// Loading time (seconds)
    pub loading_time: f64,
    /// Processing time (seconds)
    pub processing_time: f64,
    /// Memory usage peak (bytes)
    pub peak_memory_usage: usize,
    /// Throughput (samples/second)
    pub throughput: f32,
    /// Error count
    pub error_count: usize,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

/// Iterator for processing dataset batches
pub struct DatasetIterator<'a> {
    dataset: &'a CalibrationDataset,
    current_index: usize,
    batch_size: usize,
    selected_indices: Vec<usize>,
}

/// Batch processor for handling dataset batches
pub struct BatchProcessor {
    /// Batch size
    batch_size: usize,
    /// Processing function
    process_fn: Box<dyn Fn(&[Tensor]) -> CalibrationResult<()> + Send + Sync>,
}

impl CalibrationDataset {
    /// Create a new calibration dataset with the given configuration
    pub fn new(config: CalibrationConfig) -> CalibrationResult<Self> {
        let device = Self::select_device(&config)?;
        let statistics_collector = StatisticsCollector::new();
        let sampler = SamplerFactory::create(&config.sampling_strategy);
        
        let streaming_processor = if config.streaming_enabled {
            Some(StreamingProcessor::new(config.streaming_config.clone()))
        } else {
            None
        };

        Ok(Self {
            config,
            statistics_collector,
            sampler,
            streaming_processor,
            samples: Vec::new(),
            sample_metadata: Vec::new(),
            device,
            state: DatasetState::Initializing,
            metrics: ProcessingMetrics::new(),
        })
    }

    /// Load calibration data from a file path
    pub fn load_from_path<P: AsRef<Path>>(&mut self, path: P) -> CalibrationResult<()> {
        let path = path.as_ref();
        self.state = DatasetState::Loading;
        let start_time = std::time::Instant::now();

        if !path.exists() {
            return Err(CalibrationError::invalid_path(path.to_string_lossy()));
        }

        // Determine file type and load accordingly
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("npz") | Some("npy") => self.load_numpy_data(path)?,
            Some("json") => self.load_json_data(path)?,
            Some("bin") => self.load_binary_data(path)?,
            Some("pt") | Some("pth") => self.load_pytorch_data(path)?,
            _ => {
                // Try to load as directory of files
                if path.is_dir() {
                    self.load_directory_data(path)?;
                } else {
                    return Err(CalibrationError::dataset(format!(
                        "Unsupported file format: {:?}",
                        path.extension()
                    )));
                }
            }
        }

        self.metrics.loading_time = start_time.elapsed().as_secs_f64();
        self.metrics.total_samples = self.samples.len();
        self.state = DatasetState::Ready;
        
        Ok(())
    }

    /// Load data from multiple paths
    pub fn load_from_paths<P: AsRef<Path>>(&mut self, paths: &[P]) -> CalibrationResult<()> {
        for path in paths {
            self.load_from_path(path)?;
        }
        Ok(())
    }

    /// Load tensors directly
    pub fn load_tensors(&mut self, tensors: Vec<Tensor>) -> CalibrationResult<()> {
        self.state = DatasetState::Loading;
        let start_time = std::time::Instant::now();

        for (idx, tensor) in tensors.into_iter().enumerate() {
            // Create metadata for the tensor
            let metadata = SampleMetadata {
                id: format!("tensor_{idx}"),
                source_path: None,
                shape: tensor.shape().dims().to_vec(),
                dtype: "f32".to_string(), // Assume f32 for simplicity
                weight: 1.0,
                processed_at: SystemTime::now(),
                custom_fields: HashMap::new(),
            };

            self.samples.push(tensor);
            self.sample_metadata.push(metadata);
        }

        self.metrics.loading_time = start_time.elapsed().as_secs_f64();
        self.metrics.total_samples = self.samples.len();
        self.state = DatasetState::Ready;

        Ok(())
    }

    /// Collect activation statistics from the loaded dataset
    pub fn collect_statistics(&mut self) -> CalibrationResult<HashMap<String, LayerStatistics>> {
        if self.state != DatasetState::Ready {
            return Err(CalibrationError::dataset("Dataset is not ready for processing"));
        }

        self.state = DatasetState::Processing;
        let start_time = std::time::Instant::now();

        // Select representative samples using the configured sampling strategy
        let total_samples = self.samples.len();
        let target_samples = self.config.max_samples.min(total_samples);
        
        let sampling_metadata = self.create_sampling_metadata()?;
        let selected_indices = self.sampler.sample_indices(
            total_samples, 
            target_samples,
            Some(&sampling_metadata)
        )?;

        // Process selected samples
        for &idx in &selected_indices {
            if idx < self.samples.len() {
                let tensor = &self.samples[idx];
                let sample_id = &self.sample_metadata[idx].id;
                
                // For calibration, we typically collect statistics on layer outputs
                // This is a placeholder - in practice, you'd run the tensor through your model
                self.statistics_collector.update(sample_id, tensor)?;
                
                self.metrics.processed_samples += 1;
            }
        }

        self.metrics.processing_time = start_time.elapsed().as_secs_f64();
        self.metrics.throughput = self.metrics.processed_samples as f32 / self.metrics.processing_time as f32;
        self.state = DatasetState::Complete;

        Ok(self.statistics_collector.get_all_statistics().clone())
    }

    /// Create an iterator over the dataset
    pub fn iter(&mut self) -> CalibrationResult<DatasetIterator> {
        if self.state != DatasetState::Ready && self.state != DatasetState::Complete {
            return Err(CalibrationError::dataset("Dataset is not ready for iteration"));
        }

        // Select samples for iteration
        let total_samples = self.samples.len();
        let target_samples = self.config.max_samples.min(total_samples);
        
        let sampling_metadata = self.create_sampling_metadata()?;
        let selected_indices = self.sampler.sample_indices(
            total_samples,
            target_samples,
            Some(&sampling_metadata)
        )?;

        Ok(DatasetIterator {
            dataset: self,
            current_index: 0,
            batch_size: self.config.batch_size,
            selected_indices,
        })
    }

    /// Get dataset statistics
    pub fn get_metrics(&self) -> &ProcessingMetrics {
        &self.metrics
    }

    /// Get current state
    pub fn get_state(&self) -> &DatasetState {
        &self.state
    }

    /// Get configuration
    pub fn get_config(&self) -> &CalibrationConfig {
        &self.config
    }

    /// Get sample count
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Reset dataset state
    pub fn reset(&mut self) {
        self.samples.clear();
        self.sample_metadata.clear();
        self.statistics_collector.reset();
        self.sampler.reset();
        self.state = DatasetState::Initializing;
        self.metrics = ProcessingMetrics::new();
    }

    /// Select appropriate device based on configuration
    fn select_device(config: &CalibrationConfig) -> CalibrationResult<Device> {
        if config.device.auto_select {
            // Auto-select best available device
            match Device::cuda_if_available(0) {
                Ok(cuda_device) => Ok(cuda_device),
                Err(_) => {
                    #[cfg(target_os = "macos")]
                    {
                        match Device::new_metal(0) {
                            Ok(metal_device) => Ok(metal_device),
                            Err(_) => Ok(Device::Cpu),
                        }
                    }
                    #[cfg(not(target_os = "macos"))]
                    Ok(Device::Cpu)
                }
            }
        } else {
            match config.device.preferred_device {
                crate::calibration::config::PreferredDevice::Cpu => Ok(Device::Cpu),
                crate::calibration::config::PreferredDevice::Gpu => {
                    Device::cuda_if_available(0).or_else(|_| Ok(Device::Cpu))
                },
                crate::calibration::config::PreferredDevice::Metal => {
                    #[cfg(target_os = "macos")]
                    {
                        Device::new_metal(0).or_else(|_| Ok(Device::Cpu))
                    }
                    #[cfg(not(target_os = "macos"))]
                    Ok(Device::Cpu)
                },
                crate::calibration::config::PreferredDevice::Auto => Ok(Device::Cpu),
            }
        }
    }

    /// Create sampling metadata from current dataset
    fn create_sampling_metadata(&self) -> CalibrationResult<SamplingMetadata> {
        let mut importance_scores = Vec::new();
        let mut clusters = Vec::new();
        let mut sample_weights = Vec::new();

        for (idx, metadata) in self.sample_metadata.iter().enumerate() {
            // Calculate importance score based on tensor properties
            let tensor = &self.samples[idx];
            let importance = self.calculate_tensor_importance(tensor)?;
            importance_scores.push(importance);
            
            // Simple clustering based on tensor shape
            let cluster_id = self.assign_cluster(&metadata.shape);
            clusters.push(cluster_id);
            
            sample_weights.push(metadata.weight);
        }

        Ok(SamplingMetadata {
            importance_scores,
            clusters,
            activation_stats: Vec::new(), // Would be populated in real use
            sample_weights,
            extra_metadata: HashMap::new(),
        })
    }

    /// Calculate tensor importance for sampling
    fn calculate_tensor_importance(&self, tensor: &Tensor) -> CalibrationResult<f32> {
        // Simple importance calculation based on tensor statistics
        let flattened = tensor.flatten_all()
            .map_err(|e| CalibrationError::statistics(format!("Failed to flatten tensor: {e}")))?;
        
        let values: Vec<f32> = flattened.to_vec1()
            .map_err(|e| CalibrationError::statistics(format!("Failed to convert tensor: {e}")))?;
        
        if values.is_empty() {
            return Ok(0.0);
        }

        // Use variance as importance measure
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        let variance: f32 = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        
        Ok(variance.sqrt()) // Standard deviation as importance
    }

    /// Assign cluster based on tensor shape
    fn assign_cluster(&self, shape: &[usize]) -> usize {
        // Simple clustering based on number of dimensions and total size
        let num_dims = shape.len();
        let total_size: usize = shape.iter().product();
        
        match (num_dims, total_size) {
            (1, _) => 0,           // 1D tensors
            (2, size) if size < 1000 => 1,    // Small 2D tensors
            (2, _) => 2,           // Large 2D tensors
            (3, _) => 3,           // 3D tensors
            (4, _) => 4,           // 4D tensors
            _ => 5,                // Other shapes
        }
    }

    // Placeholder methods for different data loading formats
    fn load_numpy_data(&mut self, _path: &Path) -> CalibrationResult<()> {
        // Placeholder for NumPy data loading
        // In practice, you'd use a library like numpy-rs or ndarray
        Err(CalibrationError::dataset("NumPy loading not implemented yet"))
    }

    fn load_json_data(&mut self, _path: &Path) -> CalibrationResult<()> {
        // Placeholder for JSON data loading
        Err(CalibrationError::dataset("JSON loading not implemented yet"))
    }

    fn load_binary_data(&mut self, _path: &Path) -> CalibrationResult<()> {
        // Placeholder for binary data loading
        Err(CalibrationError::dataset("Binary loading not implemented yet"))
    }

    fn load_pytorch_data(&mut self, _path: &Path) -> CalibrationResult<()> {
        // Placeholder for PyTorch data loading
        Err(CalibrationError::dataset("PyTorch loading not implemented yet"))
    }

    fn load_directory_data(&mut self, _path: &Path) -> CalibrationResult<()> {
        // Placeholder for directory data loading
        Err(CalibrationError::dataset("Directory loading not implemented yet"))
    }
}

impl<'a> DatasetIterator<'a> {
    /// Get next batch of tensors
    pub fn next_batch(&mut self) -> Option<Vec<&Tensor>> {
        if self.current_index >= self.selected_indices.len() {
            return None;
        }

        let end_index = (self.current_index + self.batch_size).min(self.selected_indices.len());
        let batch_indices = &self.selected_indices[self.current_index..end_index];
        
        let batch: Vec<&Tensor> = batch_indices
            .iter()
            .filter_map(|&idx| self.dataset.samples.get(idx))
            .collect();

        self.current_index = end_index;
        
        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    /// Check if more batches are available
    pub fn has_next(&self) -> bool {
        self.current_index < self.selected_indices.len()
    }

    /// Get progress (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        if self.selected_indices.is_empty() {
            1.0
        } else {
            self.current_index as f32 / self.selected_indices.len() as f32
        }
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
    }
}

impl Iterator for DatasetIterator<'_> {
    type Item = Vec<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch().map(|batch| {
            batch.into_iter().cloned().collect()
        })
    }
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new<F>(batch_size: usize, process_fn: F) -> Self 
    where
        F: Fn(&[Tensor]) -> CalibrationResult<()> + Send + Sync + 'static,
    {
        Self {
            batch_size,
            process_fn: Box::new(process_fn),
        }
    }

    /// Process a dataset using this batch processor
    pub fn process_dataset(&self, dataset: &mut CalibrationDataset) -> CalibrationResult<()> {
        let mut iterator = dataset.iter()?;
        
        while let Some(batch) = iterator.next_batch() {
            let owned_batch: Vec<Tensor> = batch.into_iter().cloned().collect();
            (self.process_fn)(&owned_batch)?;
        }
        
        Ok(())
    }
}

impl ProcessingMetrics {
    /// Create new processing metrics
    fn new() -> Self {
        Self {
            total_samples: 0,
            processed_samples: 0,
            loading_time: 0.0,
            processing_time: 0.0,
            peak_memory_usage: 0,
            throughput: 0.0,
            error_count: 0,
            last_updated: SystemTime::now(),
        }
    }

    /// Update metrics
    pub fn update(&mut self) {
        self.last_updated = SystemTime::now();
        if self.processing_time > 0.0 {
            self.throughput = self.processed_samples as f32 / self.processing_time as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::config::CalibrationConfigBuilder;
    use candle_core::Device;

    fn create_test_tensors(count: usize) -> Vec<Tensor> {
        let device = Device::Cpu;
        (0..count)
            .map(|i| {
                let data: Vec<f32> = (0..10).map(|j| (i * 10 + j) as f32 / 100.0).collect();
                Tensor::from_vec(data, &[2, 5], &device).unwrap()
            })
            .collect()
    }

    #[test]
    fn test_dataset_creation() -> CalibrationResult<()> {
        let config = CalibrationConfigBuilder::new()
            .batch_size(4)
            .max_samples(100)
            .build()?;
        
        let dataset = CalibrationDataset::new(config)?;
        assert_eq!(dataset.get_state(), &DatasetState::Initializing);
        assert_eq!(dataset.len(), 0);
        
        Ok(())
    }

    #[test]
    fn test_tensor_loading() -> CalibrationResult<()> {
        let config = CalibrationConfigBuilder::new().build()?;
        let mut dataset = CalibrationDataset::new(config)?;
        
        let tensors = create_test_tensors(5);
        dataset.load_tensors(tensors)?;
        
        assert_eq!(dataset.len(), 5);
        assert_eq!(dataset.get_state(), &DatasetState::Ready);
        
        Ok(())
    }

    #[test]
    fn test_statistics_collection() -> CalibrationResult<()> {
        let config = CalibrationConfigBuilder::new()
            .max_samples(3)
            .build()?;
        let mut dataset = CalibrationDataset::new(config)?;
        
        let tensors = create_test_tensors(5);
        dataset.load_tensors(tensors)?;
        
        let stats = dataset.collect_statistics()?;
        assert!(!stats.is_empty());
        assert_eq!(dataset.get_state(), &DatasetState::Complete);
        
        Ok(())
    }

    #[test]
    fn test_dataset_iteration() -> CalibrationResult<()> {
        let config = CalibrationConfigBuilder::new()
            .batch_size(2)
            .max_samples(4)
            .build()?;
        let mut dataset = CalibrationDataset::new(config)?;
        
        let tensors = create_test_tensors(6);
        dataset.load_tensors(tensors)?;
        
        let mut iterator = dataset.iter()?;
        let mut batch_count = 0;
        
        while let Some(batch) = iterator.next_batch() {
            assert!(!batch.is_empty());
            assert!(batch.len() <= 2); // Batch size
            batch_count += 1;
        }
        
        assert!(batch_count > 0);
        
        Ok(())
    }

    #[test]
    fn test_batch_processor() -> CalibrationResult<()> {
        let config = CalibrationConfigBuilder::new().build()?;
        let mut dataset = CalibrationDataset::new(config)?;
        
        let tensors = create_test_tensors(3);
        dataset.load_tensors(tensors)?;
        
        let processed_batches = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = processed_batches.clone();
        let processor = BatchProcessor::new(2, move |batch| {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            assert!(!batch.is_empty());
            Ok(())
        });
        
        processor.process_dataset(&mut dataset)?;
        
        Ok(())
    }

    #[test]
    fn test_sampling_metadata_creation() -> CalibrationResult<()> {
        let config = CalibrationConfigBuilder::new().build()?;
        let mut dataset = CalibrationDataset::new(config)?;
        
        let tensors = create_test_tensors(3);
        dataset.load_tensors(tensors)?;
        
        let metadata = dataset.create_sampling_metadata()?;
        assert_eq!(metadata.importance_scores.len(), 3);
        assert_eq!(metadata.clusters.len(), 3);
        assert_eq!(metadata.sample_weights.len(), 3);
        
        Ok(())
    }
}
