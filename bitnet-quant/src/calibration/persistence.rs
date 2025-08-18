//! Persistence layer for calibration statistics
//!
//! This module provides functionality to save and load calibration statistics,
//! histograms, and quantization parameters for reuse across different runs.

use crate::calibration::error::{CalibrationError, CalibrationResult};
use crate::calibration::config::{PersistenceConfig, StorageFormat};
use crate::calibration::statistics::LayerStatistics;
use crate::calibration::histogram::ActivationHistogram;
use crate::calibration::{CalibrationSummary, QuantizationParameters, CalibrationMetadata};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs::{File, create_dir_all};
use std::io::{BufReader, BufWriter, Read, Write};
use std::time::SystemTime;

/// Main persistence interface for calibration data
#[derive(Debug)]
pub struct CalibrationCache {
    /// Configuration
    config: PersistenceConfig,
    /// In-memory cache
    memory_cache: HashMap<String, CacheEntry>,
    /// Cache directory
    cache_directory: PathBuf,
    /// Cache metrics
    metrics: CacheMetrics,
}

/// Individual cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Entry identifier
    pub id: String,
    /// Entry version for compatibility
    pub version: String,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last accessed timestamp
    pub last_accessed: SystemTime,
    /// Entry data
    pub data: CacheData,
    /// Data checksum (if enabled)
    pub checksum: Option<String>,
    /// Compression information
    pub compression_info: Option<CompressionInfo>,
}

/// Types of data that can be cached
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheData {
    /// Layer statistics
    LayerStatistics(LayerStatistics),
    /// Activation histogram
    Histogram(ActivationHistogram),
    /// Quantization parameters
    QuantizationParams(QuantizationParameters),
    /// Complete calibration results
    CalibrationResults(CalibrationSummary),
    /// Custom data
    Custom(serde_json::Value),
}

/// Compression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Compression algorithm used
    pub algorithm: String,
}

/// Cache performance metrics
#[derive(Debug, Clone)]
pub struct CacheMetrics {
    /// Cache hits
    pub hits: usize,
    /// Cache misses
    pub misses: usize,
    /// Total entries
    pub total_entries: usize,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Disk usage (bytes)
    pub disk_usage: usize,
    /// Average access time (ms)
    pub avg_access_time: f32,
}

/// Statistics persistence interface
pub trait StatisticsPersistence {
    /// Save statistics to storage
    fn save_statistics(
        &mut self,
        key: &str,
        statistics: &LayerStatistics,
    ) -> CalibrationResult<()>;

    /// Load statistics from storage
    fn load_statistics(&mut self, key: &str) -> CalibrationResult<Option<LayerStatistics>>;

    /// Save histogram to storage
    fn save_histogram(
        &mut self,
        key: &str,
        histogram: &ActivationHistogram,
    ) -> CalibrationResult<()>;

    /// Load histogram from storage
    fn load_histogram(&mut self, key: &str) -> CalibrationResult<Option<ActivationHistogram>>;

    /// Save quantization parameters
    fn save_quantization_params(
        &mut self,
        key: &str,
        params: &QuantizationParameters,
    ) -> CalibrationResult<()>;

    /// Load quantization parameters
    fn load_quantization_params(&mut self, key: &str) -> CalibrationResult<Option<QuantizationParameters>>;

    /// Check if entry exists
    fn exists(&self, key: &str) -> bool;

    /// Remove entry
    fn remove(&mut self, key: &str) -> CalibrationResult<bool>;

    /// Clear all entries
    fn clear(&mut self) -> CalibrationResult<()>;

    /// Get cache metrics
    fn get_metrics(&self) -> &CacheMetrics;
}

impl CalibrationCache {
    /// Create a new calibration cache
    pub fn new(config: PersistenceConfig) -> CalibrationResult<Self> {
        let cache_directory = config.save_directory.clone().unwrap_or_else(|| {
            std::env::temp_dir().join("bitnet_calibration_cache")
        });

        // Create cache directory if it doesn't exist
        if !cache_directory.exists() {
            create_dir_all(&cache_directory)
                .map_err(|e| CalibrationError::persistence(format!("Failed to create cache directory: {}", e)))?;
        }

        Ok(Self {
            config,
            memory_cache: HashMap::new(),
            cache_directory,
            metrics: CacheMetrics::new(),
        })
    }

    /// Save calibration results to cache
    pub fn save_calibration_results(
        &mut self,
        key: &str,
        results: &CalibrationSummary,
    ) -> CalibrationResult<()> {
        let entry = CacheEntry::new(
            key.to_string(),
            CacheData::CalibrationResults(results.clone()),
        );

        self.save_entry(key, entry)
    }

    /// Load calibration results from cache
    pub fn load_calibration_results(&mut self, key: &str) -> CalibrationResult<Option<CalibrationSummary>> {
        if let Some(entry) = self.load_entry(key)? {
            match entry.data {
                CacheData::CalibrationResults(results) => Ok(Some(results)),
                _ => Err(CalibrationError::persistence("Invalid cache data type")),
            }
        } else {
            Ok(None)
        }
    }

    /// Save entry to cache
    fn save_entry(&mut self, key: &str, mut entry: CacheEntry) -> CalibrationResult<()> {
        let start_time = std::time::Instant::now();

        // Calculate checksum if enabled
        if self.config.enable_checksums {
            entry.checksum = Some(self.calculate_checksum(&entry.data)?);
        }

        // Compress if enabled
        let serialized_data = if self.config.compression_enabled {
            let compressed = self.compress_data(&entry)?;
            entry.compression_info = Some(compressed.1);
            compressed.0
        } else {
            self.serialize_data(&entry)?
        };

        // Save to disk if auto-save is enabled
        if self.config.auto_save {
            let file_path = self.get_cache_file_path(key);
            let mut file = BufWriter::new(
                File::create(&file_path)
                    .map_err(|e| CalibrationError::persistence(format!("Failed to create cache file: {}", e)))?
            );
            
            file.write_all(&serialized_data)
                .map_err(|e| CalibrationError::persistence(format!("Failed to write cache file: {}", e)))?;
        }

        // Update in-memory cache
        if self.memory_cache.len() < self.config.cache_size {
            self.memory_cache.insert(key.to_string(), entry);
        }

        // Update metrics
        self.metrics.total_entries += 1;
        let access_time = start_time.elapsed().as_millis() as f32;
        self.update_avg_access_time(access_time);

        Ok(())
    }

    /// Load entry from cache
    fn load_entry(&mut self, key: &str) -> CalibrationResult<Option<CacheEntry>> {
        let start_time = std::time::Instant::now();

        // Check memory cache first
        if let Some(entry) = self.memory_cache.get(key) {
            let mut entry = entry.clone();
            entry.last_accessed = SystemTime::now();
            self.metrics.hits += 1;
            return Ok(Some(entry));
        }

        // Check disk cache
        let file_path = self.get_cache_file_path(key);
        if file_path.exists() {
            let mut file = BufReader::new(
                File::open(&file_path)
                    .map_err(|e| CalibrationError::persistence(format!("Failed to open cache file: {}", e)))?
            );

            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)
                .map_err(|e| CalibrationError::persistence(format!("Failed to read cache file: {}", e)))?;

            let mut entry = if self.config.compression_enabled {
                self.decompress_data(&buffer)?
            } else {
                self.deserialize_data(&buffer)?
            };

            // Verify checksum if enabled
            if self.config.enable_checksums {
                if let Some(stored_checksum) = &entry.checksum {
                    let calculated_checksum = self.calculate_checksum(&entry.data)?;
                    if stored_checksum != &calculated_checksum {
                        return Err(CalibrationError::persistence("Checksum verification failed"));
                    }
                }
            }

            entry.last_accessed = SystemTime::now();

            // Add to memory cache if there's space
            if self.memory_cache.len() < self.config.cache_size {
                self.memory_cache.insert(key.to_string(), entry.clone());
            }

            self.metrics.hits += 1;
            let access_time = start_time.elapsed().as_millis() as f32;
            self.update_avg_access_time(access_time);

            Ok(Some(entry))
        } else {
            self.metrics.misses += 1;
            Ok(None)
        }
    }

    /// Get cache file path for a key
    fn get_cache_file_path(&self, key: &str) -> PathBuf {
        let safe_key = key.replace('/', "_").replace('\\', "_");
        let extension = match self.config.storage_format {
            StorageFormat::Json => "json",
            StorageFormat::Bincode => "bin",
            StorageFormat::MessagePack => "msgpack",
            StorageFormat::Parquet => "parquet",
        };
        
        self.cache_directory.join(format!("{}.{}", safe_key, extension))
    }

    /// Serialize data based on storage format
    fn serialize_data(&self, entry: &CacheEntry) -> CalibrationResult<Vec<u8>> {
        match self.config.storage_format {
            StorageFormat::Json => {
                serde_json::to_vec(entry)
                    .map_err(|e| CalibrationError::persistence(format!("JSON serialization failed: {}", e)))
            },
            StorageFormat::Bincode => {
                bincode::serialize(entry)
                    .map_err(|e| CalibrationError::persistence(format!("Bincode serialization failed: {}", e)))
            },
            StorageFormat::MessagePack => {
                rmp_serde::to_vec(entry)
                    .map_err(|e| CalibrationError::persistence(format!("MessagePack serialization failed: {}", e)))
            },
            StorageFormat::Parquet => {
                // Parquet serialization would require more complex implementation
                Err(CalibrationError::persistence("Parquet format not implemented"))
            },
        }
    }

    /// Deserialize data based on storage format
    fn deserialize_data(&self, data: &[u8]) -> CalibrationResult<CacheEntry> {
        match self.config.storage_format {
            StorageFormat::Json => {
                serde_json::from_slice(data)
                    .map_err(|e| CalibrationError::persistence(format!("JSON deserialization failed: {}", e)))
            },
            StorageFormat::Bincode => {
                bincode::deserialize(data)
                    .map_err(|e| CalibrationError::persistence(format!("Bincode deserialization failed: {}", e)))
            },
            StorageFormat::MessagePack => {
                rmp_serde::from_slice(data)
                    .map_err(|e| CalibrationError::persistence(format!("MessagePack deserialization failed: {}", e)))
            },
            StorageFormat::Parquet => {
                Err(CalibrationError::persistence("Parquet format not implemented"))
            },
        }
    }

    /// Compress data
    fn compress_data(&self, entry: &CacheEntry) -> CalibrationResult<(Vec<u8>, CompressionInfo)> {
        let serialized = self.serialize_data(entry)?;
        let original_size = serialized.len();
        
        // Simple compression using flate2
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(self.config.compression_level));
        encoder.write_all(&serialized)
            .map_err(|e| CalibrationError::persistence(format!("Compression failed: {}", e)))?;
        
        let compressed = encoder.finish()
            .map_err(|e| CalibrationError::persistence(format!("Compression finalization failed: {}", e)))?;
        
        let compressed_size = compressed.len();
        let compression_ratio = compressed_size as f32 / original_size as f32;
        
        let compression_info = CompressionInfo {
            original_size,
            compressed_size,
            compression_ratio,
            algorithm: "gzip".to_string(),
        };
        
        Ok((compressed, compression_info))
    }

    /// Decompress data
    fn decompress_data(&self, compressed: &[u8]) -> CalibrationResult<CacheEntry> {
        use std::io::Read;
        let mut decoder = flate2::read::GzDecoder::new(compressed);
        let mut decompressed = Vec::new();
        
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| CalibrationError::persistence(format!("Decompression failed: {}", e)))?;
        
        self.deserialize_data(&decompressed)
    }

    /// Calculate checksum for data integrity
    fn calculate_checksum(&self, data: &CacheData) -> CalibrationResult<String> {
        let serialized = serde_json::to_vec(data)
            .map_err(|e| CalibrationError::persistence(format!("Checksum calculation failed: {}", e)))?;
        
        let checksum = md5::compute(&serialized);
        Ok(format!("{:x}", checksum))
    }

    /// Update average access time
    fn update_avg_access_time(&mut self, new_time: f32) {
        let total_accesses = self.metrics.hits + self.metrics.misses;
        if total_accesses > 0 {
            self.metrics.avg_access_time = 
                (self.metrics.avg_access_time * (total_accesses - 1) as f32 + new_time) / total_accesses as f32;
        } else {
            self.metrics.avg_access_time = new_time;
        }
    }

    /// Get cache hit ratio
    pub fn get_hit_ratio(&self) -> f32 {
        let total = self.metrics.hits + self.metrics.misses;
        if total > 0 {
            self.metrics.hits as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Cleanup old cache entries
    pub fn cleanup_old_entries(&mut self, max_age_days: u64) -> CalibrationResult<usize> {
        let cutoff_date = SystemTime::now() - std::time::Duration::from_secs(max_age_days as u64 * 24 * 60 * 60);
        let mut removed_count = 0;

        // Clean memory cache
        self.memory_cache.retain(|_, entry| {
            if entry.last_accessed < cutoff_date {
                removed_count += 1;
                false
            } else {
                true
            }
        });

        // Clean disk cache
        if let Ok(entries) = std::fs::read_dir(&self.cache_directory) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        if modified < cutoff_date {
                            if std::fs::remove_file(entry.path()).is_ok() {
                                removed_count += 1;
                            }
                        }
                    }
                }
            }
        }

        self.metrics.total_entries = self.metrics.total_entries.saturating_sub(removed_count);
        Ok(removed_count)
    }
}

impl StatisticsPersistence for CalibrationCache {
    fn save_statistics(&mut self, key: &str, statistics: &LayerStatistics) -> CalibrationResult<()> {
        let entry = CacheEntry::new(
            key.to_string(),
            CacheData::LayerStatistics(statistics.clone()),
        );
        self.save_entry(key, entry)
    }

    fn load_statistics(&mut self, key: &str) -> CalibrationResult<Option<LayerStatistics>> {
        if let Some(entry) = self.load_entry(key)? {
            match entry.data {
                CacheData::LayerStatistics(stats) => Ok(Some(stats)),
                _ => Err(CalibrationError::persistence("Invalid cache data type")),
            }
        } else {
            Ok(None)
        }
    }

    fn save_histogram(&mut self, key: &str, histogram: &ActivationHistogram) -> CalibrationResult<()> {
        let entry = CacheEntry::new(
            key.to_string(),
            CacheData::Histogram(histogram.clone()),
        );
        self.save_entry(key, entry)
    }

    fn load_histogram(&mut self, key: &str) -> CalibrationResult<Option<ActivationHistogram>> {
        if let Some(entry) = self.load_entry(key)? {
            match entry.data {
                CacheData::Histogram(hist) => Ok(Some(hist)),
                _ => Err(CalibrationError::persistence("Invalid cache data type")),
            }
        } else {
            Ok(None)
        }
    }

    fn save_quantization_params(&mut self, key: &str, params: &QuantizationParameters) -> CalibrationResult<()> {
        let entry = CacheEntry::new(
            key.to_string(),
            CacheData::QuantizationParams(params.clone()),
        );
        self.save_entry(key, entry)
    }

    fn load_quantization_params(&mut self, key: &str) -> CalibrationResult<Option<QuantizationParameters>> {
        if let Some(entry) = self.load_entry(key)? {
            match entry.data {
                CacheData::QuantizationParams(params) => Ok(Some(params)),
                _ => Err(CalibrationError::persistence("Invalid cache data type")),
            }
        } else {
            Ok(None)
        }
    }

    fn exists(&self, key: &str) -> bool {
        self.memory_cache.contains_key(key) || self.get_cache_file_path(key).exists()
    }

    fn remove(&mut self, key: &str) -> CalibrationResult<bool> {
        let mut removed = false;

        // Remove from memory cache
        if self.memory_cache.remove(key).is_some() {
            removed = true;
        }

        // Remove from disk cache
        let file_path = self.get_cache_file_path(key);
        if file_path.exists() {
            std::fs::remove_file(&file_path)
                .map_err(|e| CalibrationError::persistence(format!("Failed to remove cache file: {}", e)))?;
            removed = true;
        }

        if removed {
            self.metrics.total_entries = self.metrics.total_entries.saturating_sub(1);
        }

        Ok(removed)
    }

    fn clear(&mut self) -> CalibrationResult<()> {
        // Clear memory cache
        self.memory_cache.clear();

        // Clear disk cache
        if let Ok(entries) = std::fs::read_dir(&self.cache_directory) {
            for entry in entries.flatten() {
                if let Err(e) = std::fs::remove_file(entry.path()) {
                    return Err(CalibrationError::persistence(format!("Failed to remove cache file: {}", e)));
                }
            }
        }

        self.metrics = CacheMetrics::new();
        Ok(())
    }

    fn get_metrics(&self) -> &CacheMetrics {
        &self.metrics
    }
}

impl CacheEntry {
    /// Create a new cache entry
    fn new(id: String, data: CacheData) -> Self {
        let now = SystemTime::now();
        Self {
            id,
            version: "1.0".to_string(),
            created_at: now,
            last_accessed: now,
            data,
            checksum: None,
            compression_info: None,
        }
    }
}

impl CacheMetrics {
    /// Create new cache metrics
    fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            total_entries: 0,
            memory_usage: 0,
            disk_usage: 0,
            avg_access_time: 0.0,
        }
    }
}

// Add necessary dependencies to Cargo.toml
// flate2 = "1.0"
// md5 = "0.7"
// bincode = "1.3"
// rmp-serde = "1.1"

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::config::PersistenceConfig;
    use tempfile::tempdir;

    #[test]
    fn test_cache_creation() -> CalibrationResult<()> {
        let temp_dir = tempdir().unwrap();
        let config = PersistenceConfig {
            auto_save: true,
            save_directory: Some(temp_dir.path().to_path_buf()),
            storage_format: StorageFormat::Json,
            compression_enabled: false,
            compression_level: 6,
            cache_size: 10,
            enable_checksums: true,
        };

        let cache = CalibrationCache::new(config)?;
        assert_eq!(cache.get_metrics().total_entries, 0);
        
        Ok(())
    }

    #[test]
    fn test_cache_save_load_statistics() -> CalibrationResult<()> {
        let temp_dir = tempdir().unwrap();
        let config = PersistenceConfig {
            auto_save: true,
            save_directory: Some(temp_dir.path().to_path_buf()),
            storage_format: StorageFormat::Json,
            compression_enabled: false,
            compression_level: 6,
            cache_size: 10,
            enable_checksums: true,
        };

        let mut cache = CalibrationCache::new(config)?;

        // Create test statistics (simplified)
        use crate::calibration::statistics::{LayerStatistics, MinMaxStats, MomentStats, PercentileStats, OutlierStats, ShapeInfo, OutlierMethod};
        
        let stats = LayerStatistics {
            layer_name: "test_layer".to_string(),
            min_max: MinMaxStats {
                global_min: -1.0,
                global_max: 1.0,
                channel_min: vec![-1.0, -0.5],
                channel_max: vec![1.0, 0.5],
                ema_min: -0.9,
                ema_max: 0.9,
                ema_decay: 0.01,
            },
            moments: MomentStats {
                mean: 0.0,
                variance: 1.0,
                std_dev: 1.0,
                skewness: 0.0,
                kurtosis: 0.0,
                channel_means: vec![0.0, 0.1],
                channel_variances: vec![1.0, 0.9],
            },
            percentiles: PercentileStats {
                percentiles: vec![50.0],
                values: vec![0.0],
                iqr: 1.0,
                mad: 0.5,
            },
            outliers: OutlierStats {
                outlier_count: 0,
                outlier_ratio: 0.0,
                threshold: 3.0,
                method: OutlierMethod::StandardDeviation,
            },
            shape_info: ShapeInfo {
                dimensions: vec![2, 3, 4],
                num_elements: 24,
                num_channels: Some(3),
                sparsity_ratio: 0.1,
                dtype: "f32".to_string(),
            },
            update_count: 1,
            last_updated: SystemTime::now(),
        };

        // Save statistics
        cache.save_statistics("test_key", &stats)?;
        
        // Load statistics
        let loaded_stats = cache.load_statistics("test_key")?;
        assert!(loaded_stats.is_some());
        
        let loaded_stats = loaded_stats.unwrap();
        assert_eq!(loaded_stats.layer_name, "test_layer");
        assert_eq!(loaded_stats.shape_info.num_elements, 24);

        Ok(())
    }

    #[test]
    fn test_cache_hit_ratio() -> CalibrationResult<()> {
        let temp_dir = tempdir().unwrap();
        let config = PersistenceConfig {
            auto_save: false, // Don't save to disk for this test
            save_directory: Some(temp_dir.path().to_path_buf()),
            storage_format: StorageFormat::Json,
            compression_enabled: false,
            compression_level: 6,
            cache_size: 10,
            enable_checksums: false,
        };

        let mut cache = CalibrationCache::new(config)?;

        // Initially no hits or misses
        assert_eq!(cache.get_hit_ratio(), 0.0);

        // Try to load non-existent entry (miss)
        let result = cache.load_entry("non_existent")?;
        assert!(result.is_none());
        assert!(cache.get_hit_ratio() < 1.0);

        Ok(())
    }

    #[test]
    fn test_compression() -> CalibrationResult<()> {
        let temp_dir = tempdir().unwrap();
        let config = PersistenceConfig {
            auto_save: true,
            save_directory: Some(temp_dir.path().to_path_buf()),
            storage_format: StorageFormat::Json,
            compression_enabled: true,
            compression_level: 6,
            cache_size: 10,
            enable_checksums: true,
        };

        let mut cache = CalibrationCache::new(config)?;
        
        // Create a dummy entry with custom data
        let large_data = serde_json::json!({
            "data": vec![1.0; 1000], // Large array to test compression
            "metadata": "test"
        });

        let entry = CacheEntry::new(
            "test_compression".to_string(),
            CacheData::Custom(large_data),
        );

        let (compressed, compression_info) = cache.compress_data(&entry)?;
        
        assert!(compression_info.compressed_size < compression_info.original_size);
        assert!(compression_info.compression_ratio < 1.0);
        
        // Test decompression
        let decompressed = cache.decompress_data(&compressed)?;
        assert_eq!(decompressed.id, entry.id);

        Ok(())
    }
}
