//! Device Comparison Utilities
//!
//! This module provides custom device comparison logic to address the limitation
//! that `candle_core::Device` doesn't implement `PartialEq`. It offers detailed
//! equality checking and comparison utilities for testing and validation.

use candle_core::Device;
use std::fmt;

/// Detailed device comparison result
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceComparisonResult {
    /// Whether the devices are considered equal
    pub equal: bool,
    /// Detailed comparison information
    pub details: DeviceComparisonDetails,
}

/// Detailed information about device comparison
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceComparisonDetails {
    /// Type of the first device
    pub device1_type: DeviceType,
    /// Type of the second device
    pub device2_type: DeviceType,
    /// Additional comparison metadata
    pub metadata: DeviceComparisonMetadata,
}

/// Device type classification for comparison
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    Cpu,
    Metal { device_id: Option<usize> },
    Cuda { device_id: Option<usize> },
}

/// Additional metadata for device comparison
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceComparisonMetadata {
    /// Whether devices are on the same physical hardware
    pub same_hardware: bool,
    /// Whether devices support unified memory
    pub unified_memory_compatible: bool,
    /// Device capability comparison
    pub capability_match: CapabilityMatch,
}

/// Device capability matching information
#[derive(Debug, Clone, PartialEq)]
pub enum CapabilityMatch {
    /// Devices have identical capabilities
    Identical,
    /// Devices are compatible but with different capabilities
    Compatible,
    /// Devices have incompatible capabilities
    Incompatible,
    /// Capability comparison not available
    Unknown,
}

impl DeviceComparisonResult {
    /// Creates a new comparison result indicating equality
    pub fn equal(details: DeviceComparisonDetails) -> Self {
        Self {
            equal: true,
            details,
        }
    }

    /// Creates a new comparison result indicating inequality
    pub fn not_equal(details: DeviceComparisonDetails) -> Self {
        Self {
            equal: false,
            details,
        }
    }

    /// Returns true if devices are equal
    pub fn is_equal(&self) -> bool {
        self.equal
    }

    /// Returns true if devices are not equal
    pub fn is_not_equal(&self) -> bool {
        !self.equal
    }

    /// Gets the comparison details
    pub fn details(&self) -> &DeviceComparisonDetails {
        &self.details
    }
}

impl fmt::Display for DeviceComparisonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.equal {
            write!(
                f,
                "Devices are equal: {:?} == {:?}",
                self.details.device1_type, self.details.device2_type
            )
        } else {
            write!(
                f,
                "Devices are not equal: {:?} != {:?} ({})",
                self.details.device1_type,
                self.details.device2_type,
                self.details.metadata.capability_match.description()
            )
        }
    }
}

impl CapabilityMatch {
    /// Returns a human-readable description of the capability match
    pub fn description(&self) -> &'static str {
        match self {
            CapabilityMatch::Identical => "identical capabilities",
            CapabilityMatch::Compatible => "compatible capabilities",
            CapabilityMatch::Incompatible => "incompatible capabilities",
            CapabilityMatch::Unknown => "unknown capability compatibility",
        }
    }
}

/// Compares two devices for equality with detailed analysis
pub fn compare_devices(device1: &Device, device2: &Device) -> DeviceComparisonResult {
    let device1_type = classify_device(device1);
    let device2_type = classify_device(device2);

    let equal = device_types_equal(&device1_type, &device2_type);
    let metadata = analyze_device_compatibility(&device1_type, &device2_type);

    let details = DeviceComparisonDetails {
        device1_type,
        device2_type,
        metadata,
    };

    if equal {
        DeviceComparisonResult::equal(details)
    } else {
        DeviceComparisonResult::not_equal(details)
    }
}

/// Simple device equality check
pub fn devices_equal(device1: &Device, device2: &Device) -> bool {
    let type1 = classify_device(device1);
    let type2 = classify_device(device2);
    device_types_equal(&type1, &type2)
}

/// Checks if devices are compatible for operations
pub fn devices_compatible(device1: &Device, device2: &Device) -> bool {
    let comparison = compare_devices(device1, device2);
    comparison.equal
        || matches!(
            comparison.details.metadata.capability_match,
            CapabilityMatch::Compatible
        )
}

/// Classifies a device into our comparison type system
fn classify_device(device: &Device) -> DeviceType {
    match device {
        Device::Cpu => DeviceType::Cpu,
        Device::Metal(metal_device) => {
            // Extract device ID if available
            let device_id = extract_metal_device_id(metal_device);
            DeviceType::Metal { device_id }
        }
        Device::Cuda(cuda_device) => {
            // Extract device ID if available
            let device_id = extract_cuda_device_id(cuda_device);
            DeviceType::Cuda { device_id }
        }
    }
}

/// Checks equality between device types
fn device_types_equal(type1: &DeviceType, type2: &DeviceType) -> bool {
    match (type1, type2) {
        (DeviceType::Cpu, DeviceType::Cpu) => true,
        (DeviceType::Metal { device_id: id1 }, DeviceType::Metal { device_id: id2 }) => {
            // Metal devices are equal if they have the same ID or both have no ID
            match (id1, id2) {
                (Some(id1), Some(id2)) => id1 == id2,
                (None, None) => true,
                _ => false,
            }
        }
        (DeviceType::Cuda { device_id: id1 }, DeviceType::Cuda { device_id: id2 }) => {
            // CUDA devices are equal if they have the same ID or both have no ID
            match (id1, id2) {
                (Some(id1), Some(id2)) => id1 == id2,
                (None, None) => true,
                _ => false,
            }
        }
        _ => false,
    }
}

/// Analyzes compatibility between device types
fn analyze_device_compatibility(
    type1: &DeviceType,
    type2: &DeviceType,
) -> DeviceComparisonMetadata {
    let same_hardware = match (type1, type2) {
        (DeviceType::Cpu, DeviceType::Cpu) => true,
        (DeviceType::Metal { device_id: id1 }, DeviceType::Metal { device_id: id2 }) => {
            match (id1, id2) {
                (Some(id1), Some(id2)) => id1 == id2,
                _ => true, // Assume same hardware if IDs not available
            }
        }
        (DeviceType::Cuda { device_id: id1 }, DeviceType::Cuda { device_id: id2 }) => {
            match (id1, id2) {
                (Some(id1), Some(id2)) => id1 == id2,
                _ => false, // Different CUDA devices are typically different hardware
            }
        }
        _ => false,
    };

    let unified_memory_compatible = match (type1, type2) {
        (DeviceType::Cpu, DeviceType::Cpu) => true,
        (DeviceType::Metal { .. }, DeviceType::Metal { .. }) => true, // Apple Silicon unified memory
        (DeviceType::Cpu, DeviceType::Metal { .. })
        | (DeviceType::Metal { .. }, DeviceType::Cpu) => true,
        _ => false,
    };

    let capability_match = if device_types_equal(type1, type2) {
        CapabilityMatch::Identical
    } else {
        match (type1, type2) {
            (DeviceType::Cpu, DeviceType::Metal { .. })
            | (DeviceType::Metal { .. }, DeviceType::Cpu) => CapabilityMatch::Compatible,
            (DeviceType::Metal { .. }, DeviceType::Metal { .. }) => CapabilityMatch::Compatible,
            (DeviceType::Cuda { .. }, DeviceType::Cuda { .. }) => CapabilityMatch::Compatible,
            _ => CapabilityMatch::Incompatible,
        }
    };

    DeviceComparisonMetadata {
        same_hardware,
        unified_memory_compatible,
        capability_match,
    }
}

/// Extracts device ID from Metal device (if available)
fn extract_metal_device_id(metal_device: &candle_core::MetalDevice) -> Option<usize> {
    // This is a placeholder - the actual implementation would depend on
    // the internal structure of MetalDevice
    // For now, we'll return None to indicate ID extraction is not available
    let _ = metal_device;
    None
}

/// Extracts device ID from CUDA device (if available)
fn extract_cuda_device_id(cuda_device: &candle_core::CudaDevice) -> Option<usize> {
    // This is a placeholder - the actual implementation would depend on
    // the internal structure of CudaDevice
    // For now, we'll return None to indicate ID extraction is not available
    let _ = cuda_device;
    None
}

/// Assertion helper for tests - panics if devices are not equal
pub fn assert_devices_equal(device1: &Device, device2: &Device) {
    let comparison = compare_devices(device1, device2);
    if !comparison.equal {
        panic!("Device assertion failed: {}", comparison);
    }
}

/// Assertion helper for tests - panics if devices are equal
pub fn assert_devices_not_equal(device1: &Device, device2: &Device) {
    let comparison = compare_devices(device1, device2);
    if comparison.equal {
        panic!(
            "Device assertion failed: expected devices to be different, but they are equal: {}",
            comparison
        );
    }
}

/// Assertion helper for tests - panics if devices are not compatible
pub fn assert_devices_compatible(device1: &Device, device2: &Device) {
    if !devices_compatible(device1, device2) {
        let comparison = compare_devices(device1, device2);
        panic!("Device compatibility assertion failed: {}", comparison);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{auto_select_device, get_cpu_device};

    #[test]
    fn test_cpu_device_equality() {
        let cpu1 = get_cpu_device();
        let cpu2 = get_cpu_device();

        let comparison = compare_devices(&cpu1, &cpu2);
        assert!(comparison.equal);
        assert!(matches!(comparison.details.device1_type, DeviceType::Cpu));
        assert!(matches!(comparison.details.device2_type, DeviceType::Cpu));
        assert!(comparison.details.metadata.same_hardware);
        assert!(comparison.details.metadata.unified_memory_compatible);
        assert!(matches!(
            comparison.details.metadata.capability_match,
            CapabilityMatch::Identical
        ));
    }

    #[test]
    fn test_device_equality_helper() {
        let cpu1 = get_cpu_device();
        let cpu2 = get_cpu_device();

        assert!(devices_equal(&cpu1, &cpu2));
    }

    #[test]
    fn test_device_compatibility_helper() {
        let cpu = get_cpu_device();
        let auto_device = auto_select_device();

        // CPU should be compatible with any device
        assert!(devices_compatible(&cpu, &auto_device));
    }

    #[test]
    fn test_assertion_helpers() {
        let cpu1 = get_cpu_device();
        let cpu2 = get_cpu_device();

        // This should not panic
        assert_devices_equal(&cpu1, &cpu2);
        assert_devices_compatible(&cpu1, &cpu2);
    }

    #[test]
    fn test_device_type_classification() {
        let cpu = get_cpu_device();
        let cpu_type = classify_device(&cpu);

        assert!(matches!(cpu_type, DeviceType::Cpu));
    }

    #[test]
    fn test_comparison_result_display() {
        let cpu1 = get_cpu_device();
        let cpu2 = get_cpu_device();

        let comparison = compare_devices(&cpu1, &cpu2);
        let display_str = format!("{}", comparison);

        assert!(display_str.contains("Devices are equal"));
        assert!(display_str.contains("Cpu"));
    }

    #[test]
    fn test_capability_match_description() {
        assert_eq!(
            CapabilityMatch::Identical.description(),
            "identical capabilities"
        );
        assert_eq!(
            CapabilityMatch::Compatible.description(),
            "compatible capabilities"
        );
        assert_eq!(
            CapabilityMatch::Incompatible.description(),
            "incompatible capabilities"
        );
        assert_eq!(
            CapabilityMatch::Unknown.description(),
            "unknown capability compatibility"
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_device_comparison() {
        use crate::device::get_metal_device;

        if let Ok(metal1) = get_metal_device() {
            if let Ok(metal2) = get_metal_device() {
                let comparison = compare_devices(&metal1, &metal2);
                assert!(comparison.equal);
                assert!(matches!(
                    comparison.details.device1_type,
                    DeviceType::Metal { .. }
                ));
                assert!(matches!(
                    comparison.details.device2_type,
                    DeviceType::Metal { .. }
                ));
            }
        }
    }

    #[test]
    fn test_cross_device_type_comparison() {
        let cpu = get_cpu_device();
        let auto_device = auto_select_device();

        let comparison = compare_devices(&cpu, &auto_device);

        // The result depends on what auto_select_device returns
        match auto_device {
            Device::Cpu => {
                assert!(comparison.equal);
                assert!(matches!(
                    comparison.details.metadata.capability_match,
                    CapabilityMatch::Identical
                ));
            }
            Device::Metal(_) => {
                assert!(!comparison.equal);
                assert!(matches!(
                    comparison.details.metadata.capability_match,
                    CapabilityMatch::Compatible
                ));
                assert!(comparison.details.metadata.unified_memory_compatible);
            }
            Device::Cuda(_) => {
                assert!(!comparison.equal);
                assert!(matches!(
                    comparison.details.metadata.capability_match,
                    CapabilityMatch::Incompatible
                ));
                assert!(!comparison.details.metadata.unified_memory_compatible);
            }
        }
    }
}
