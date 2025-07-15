//! Tensor Metadata
//!
//! This module provides metadata tracking for BitNet tensors, including
//! shape information, device placement, creation time, and lifecycle state.

use crate::memory::tensor::BitNetDType;
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Metadata associated with a BitNet tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// Unique identifier for this tensor
    pub id: u64,
    /// Shape of the tensor (dimensions)
    pub shape: Vec<usize>,
    /// Data type of the tensor
    pub dtype: BitNetDType,
    /// Device where the tensor is stored
    pub device: DeviceInfo,
    /// Size in bytes of the tensor data
    pub size_bytes: usize,
    /// Number of elements in the tensor
    pub element_count: usize,
    /// Creation timestamp (seconds since Unix epoch)
    pub created_at: u64,
    /// Last access timestamp (seconds since Unix epoch)
    pub last_accessed: u64,
    /// Reference count for this tensor
    pub ref_count: usize,
    /// Whether the tensor is currently being migrated
    pub is_migrating: bool,
    /// Optional name/label for debugging
    pub name: Option<String>,
    /// Additional tags for categorization
    pub tags: Vec<String>,
}

/// Device information that can be serialized
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceInfo {
    /// CPU device
    Cpu,
    /// Metal GPU device with optional device ID
    Metal(Option<String>),
    /// CUDA GPU device with device ID
    Cuda(String),
}

impl From<&Device> for DeviceInfo {
    fn from(device: &Device) -> Self {
        match device {
            Device::Cpu => DeviceInfo::Cpu,
            Device::Metal(metal_device) => {
                DeviceInfo::Metal(Some(format!("{:?}", metal_device.id())))
            }
            Device::Cuda(cuda_device) => {
                DeviceInfo::Cuda(format!("{:?}", cuda_device))
            }
        }
    }
}

impl TensorMetadata {
    /// Creates new tensor metadata
    pub fn new(
        id: u64,
        shape: Vec<usize>,
        dtype: BitNetDType,
        device: &Device,
        name: Option<String>,
    ) -> Self {
        let element_count = shape.iter().product();
        let size_bytes = dtype.bytes_for_elements(element_count);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id,
            shape,
            dtype,
            device: DeviceInfo::from(device),
            size_bytes,
            element_count,
            created_at: now,
            last_accessed: now,
            ref_count: 1,
            is_migrating: false,
            name,
            tags: Vec::new(),
        }
    }

    /// Updates the last accessed timestamp
    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Increments the reference count
    pub fn add_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Decrements the reference count and returns true if it reaches zero
    pub fn remove_ref(&mut self) -> bool {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
        self.ref_count == 0
    }

    /// Returns the current reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count
    }

    /// Sets the migration state
    pub fn set_migrating(&mut self, migrating: bool) {
        self.is_migrating = migrating;
    }

    /// Returns true if the tensor is currently being migrated
    pub fn is_migrating(&self) -> bool {
        self.is_migrating
    }

    /// Adds a tag to the tensor
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Removes a tag from the tensor
    pub fn remove_tag(&mut self, tag: &str) {
        self.tags.retain(|t| t != tag);
    }

    /// Returns true if the tensor has the specified tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(&tag.to_string())
    }

    /// Returns the age of the tensor in seconds
    pub fn age_seconds(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(self.created_at)
    }

    /// Returns the time since last access in seconds
    pub fn idle_time_seconds(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(self.last_accessed)
    }

    /// Returns the tensor rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns true if this is a scalar tensor (0 dimensions)
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// Returns true if this is a vector tensor (1 dimension)
    pub fn is_vector(&self) -> bool {
        self.shape.len() == 1
    }

    /// Returns true if this is a matrix tensor (2 dimensions)
    pub fn is_matrix(&self) -> bool {
        self.shape.len() == 2
    }

    /// Returns the memory efficiency compared to F32
    pub fn memory_efficiency(&self) -> f32 {
        self.dtype.memory_efficiency()
    }

    /// Returns a human-readable description of the tensor
    pub fn description(&self) -> String {
        let shape_str = if self.shape.is_empty() {
            "scalar".to_string()
        } else {
            format!("[{}]", self.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
        };

        let device_str = match &self.device {
            DeviceInfo::Cpu => "CPU".to_string(),
            DeviceInfo::Metal(Some(id)) => format!("Metal({})", id),
            DeviceInfo::Metal(None) => "Metal".to_string(),
            DeviceInfo::Cuda(id) => format!("CUDA({})", id),
        };

        let name_str = self.name.as_ref()
            .map(|n| format!("'{}' ", n))
            .unwrap_or_default();

        format!(
            "{}tensor {} {} on {} ({} elements, {} bytes, refs: {})",
            name_str,
            shape_str,
            self.dtype,
            device_str,
            self.element_count,
            self.size_bytes,
            self.ref_count
        )
    }

    /// Updates the metadata when the tensor is migrated to a new device
    pub fn update_device(&mut self, device: &Device) {
        self.device = DeviceInfo::from(device);
        self.touch();
    }

    /// Updates the shape and recalculates derived fields
    pub fn update_shape(&mut self, new_shape: Vec<usize>) {
        self.shape = new_shape;
        self.element_count = self.shape.iter().product();
        self.size_bytes = self.dtype.bytes_for_elements(self.element_count);
        self.touch();
    }

    /// Creates a copy of the metadata with a new ID (for cloning tensors)
    pub fn clone_with_new_id(&self, new_id: u64) -> Self {
        let mut metadata = self.clone();
        metadata.id = new_id;
        metadata.ref_count = 1;
        metadata.created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        metadata.last_accessed = metadata.created_at;
        metadata
    }
}

impl std::fmt::Display for TensorMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::get_cpu_device;

    #[test]
    fn test_metadata_creation() {
        let device = get_cpu_device();
        let metadata = TensorMetadata::new(
            1,
            vec![2, 3],
            BitNetDType::F32,
            &device,
            Some("test_tensor".to_string()),
        );

        assert_eq!(metadata.id, 1);
        assert_eq!(metadata.shape, vec![2, 3]);
        assert_eq!(metadata.dtype, BitNetDType::F32);
        assert_eq!(metadata.element_count, 6);
        assert_eq!(metadata.size_bytes, 24); // 6 * 4 bytes
        assert_eq!(metadata.ref_count, 1);
        assert_eq!(metadata.name, Some("test_tensor".to_string()));
        assert!(!metadata.is_migrating);
    }

    #[test]
    fn test_reference_counting() {
        let device = get_cpu_device();
        let mut metadata = TensorMetadata::new(1, vec![2], BitNetDType::F32, &device, None);

        assert_eq!(metadata.ref_count(), 1);

        metadata.add_ref();
        assert_eq!(metadata.ref_count(), 2);

        assert!(!metadata.remove_ref());
        assert_eq!(metadata.ref_count(), 1);

        assert!(metadata.remove_ref());
        assert_eq!(metadata.ref_count(), 0);
    }

    #[test]
    fn test_tags() {
        let device = get_cpu_device();
        let mut metadata = TensorMetadata::new(1, vec![2], BitNetDType::F32, &device, None);

        assert!(!metadata.has_tag("test"));

        metadata.add_tag("test".to_string());
        assert!(metadata.has_tag("test"));

        metadata.add_tag("test".to_string()); // Should not duplicate
        assert_eq!(metadata.tags.len(), 1);

        metadata.add_tag("another".to_string());
        assert_eq!(metadata.tags.len(), 2);

        metadata.remove_tag("test");
        assert!(!metadata.has_tag("test"));
        assert!(metadata.has_tag("another"));
    }

    #[test]
    fn test_tensor_classification() {
        let device = get_cpu_device();

        let scalar = TensorMetadata::new(1, vec![], BitNetDType::F32, &device, None);
        assert!(scalar.is_scalar());
        assert!(!scalar.is_vector());
        assert!(!scalar.is_matrix());
        assert_eq!(scalar.rank(), 0);

        let vector = TensorMetadata::new(2, vec![5], BitNetDType::F32, &device, None);
        assert!(!vector.is_scalar());
        assert!(vector.is_vector());
        assert!(!vector.is_matrix());
        assert_eq!(vector.rank(), 1);

        let matrix = TensorMetadata::new(3, vec![3, 4], BitNetDType::F32, &device, None);
        assert!(!matrix.is_scalar());
        assert!(!matrix.is_vector());
        assert!(matrix.is_matrix());
        assert_eq!(matrix.rank(), 2);
    }

    #[test]
    fn test_device_info_conversion() {
        let cpu_device = get_cpu_device();
        let device_info = DeviceInfo::from(&cpu_device);
        assert!(matches!(device_info, DeviceInfo::Cpu));
    }

    #[test]
    fn test_clone_with_new_id() {
        let device = get_cpu_device();
        let original = TensorMetadata::new(1, vec![2, 3], BitNetDType::F32, &device, Some("original".to_string()));
        
        // Add a small delay to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(1));
        
        let cloned = original.clone_with_new_id(2);
        
        assert_eq!(cloned.id, 2);
        assert_eq!(cloned.ref_count, 1);
        assert_eq!(cloned.shape, original.shape);
        assert_eq!(cloned.dtype, original.dtype);
        assert!(cloned.created_at >= original.created_at);
    }

    #[test]
    fn test_memory_efficiency() {
        let device = get_cpu_device();
        
        let f32_meta = TensorMetadata::new(1, vec![100], BitNetDType::F32, &device, None);
        assert_eq!(f32_meta.memory_efficiency(), 1.0);
        
        let i4_meta = TensorMetadata::new(2, vec![100], BitNetDType::I4, &device, None);
        assert_eq!(i4_meta.memory_efficiency(), 8.0);
    }

    #[test]
    fn test_serialization() {
        let device = get_cpu_device();
        let metadata = TensorMetadata::new(1, vec![2, 3], BitNetDType::BitNet158, &device, Some("test".to_string()));
        
        let serialized = serde_json::to_string(&metadata).unwrap();
        let deserialized: TensorMetadata = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(metadata.id, deserialized.id);
        assert_eq!(metadata.shape, deserialized.shape);
        assert_eq!(metadata.dtype, deserialized.dtype);
        assert_eq!(metadata.name, deserialized.name);
    }
}