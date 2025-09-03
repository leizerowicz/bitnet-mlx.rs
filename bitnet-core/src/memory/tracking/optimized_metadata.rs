//! Optimized Memory Tracking Metadata
//!
//! This module provides compact, cache-optimized metadata structures for memory tracking
//! with minimal overhead while maintaining essential tracking functionality.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use candle_core::Device;

use super::AllocationId;

/// Optimized metadata structure designed for minimal memory overhead
///
/// This structure is carefully laid out to fit in exactly 16 bytes while 
/// providing essential tracking information with atomic safety.
///
/// Memory layout (16 bytes total):
/// - packed_data: 8 bytes (size + thread_id + flags)
/// - timestamp: 8 bytes (milliseconds since epoch)
#[derive(Debug)]
#[repr(C)]
pub struct OptimizedAllocationMetadata {
    /// Packed allocation data using atomic operations
    /// Layout: size(48 bits) | thread_id(8 bits) | flags(8 bits)
    packed_data: AtomicU64,
    
    /// Timestamp in milliseconds since epoch for age tracking
    timestamp: AtomicU64,
}

/// Allocation flags packed into 16 bits
#[derive(Debug, Clone, Copy)]
pub struct AllocationFlags {
    /// Allocation is still active
    pub is_active: bool,
    /// Allocation is under memory pressure monitoring
    pub is_monitored: bool,
    /// Allocation is large (>1MB)
    pub is_large: bool,
    /// Allocation failed cleanup attempt
    pub cleanup_failed: bool,
    /// Reserved flags for future use
    pub reserved: u8,
}

/// Size class for efficient storage and lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeClass {
    Tiny,        // 0-1KB
    Small,       // 1KB-64KB  
    Medium,      // 64KB-1MB
    Large,       // 1MB-16MB
    Huge,        // >16MB
}

/// Compact device identifier for efficient storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompactDeviceId {
    Cpu = 0,
    Cuda = 1,  
    Metal = 2,
    Mlx = 3,
    Other = 4,
}

impl OptimizedAllocationMetadata {
    // Bit field constants for packed data
    const ID_SHIFT: u64 = 32;
    const SIZE_CLASS_SHIFT: u64 = 24;
    const DEVICE_SHIFT: u64 = 16;
    const FLAGS_MASK: u64 = 0xFFFF;

    /// Creates new optimized metadata
    pub fn new(
        id: AllocationId,
        size: usize,
        device: &Device,
        tracking_start: Instant,
    ) -> Self {
        let size_class = SizeClass::from_size(size);
        let device_id = CompactDeviceId::from_device(device);
        let flags = AllocationFlags::new();
        let timestamp = tracking_start.elapsed().as_millis() as u32;
        
        let packed_data = Self::pack_data(
            id.0 as u32,
            size_class as u8,
            device_id as u8,
            flags.to_u16(),
        );
        
        Self {
            packed_data: AtomicU64::new(packed_data),
            timestamp: AtomicU64::new(timestamp.into()),
        }
    }

    /// Packs data into single u64 for atomic operations
    #[inline]
    fn pack_data(id: u32, size_class: u8, device_id: u8, flags: u16) -> u64 {
        ((id as u64) << Self::ID_SHIFT)
            | ((size_class as u64) << Self::SIZE_CLASS_SHIFT)
            | ((device_id as u64) << Self::DEVICE_SHIFT)
            | (flags as u64)
    }

    /// Extracts allocation ID from packed data
    #[inline]
    pub fn allocation_id(&self) -> AllocationId {
        let packed = self.packed_data.load(Ordering::Acquire);
        AllocationId((packed >> Self::ID_SHIFT) as u64)
    }

    /// Extracts size class from packed data
    #[inline]
    pub fn size_class(&self) -> SizeClass {
        let packed = self.packed_data.load(Ordering::Acquire);
        let size_class = ((packed >> Self::SIZE_CLASS_SHIFT) & 0xFF) as u8;
        SizeClass::from_u8(size_class)
    }

    /// Extracts device ID from packed data
    #[inline]
    pub fn device_id(&self) -> CompactDeviceId {
        let packed = self.packed_data.load(Ordering::Acquire);
        let device_id = ((packed >> Self::DEVICE_SHIFT) & 0xFF) as u8;
        CompactDeviceId::from_u8(device_id)
    }

    /// Extracts flags from packed data
    #[inline]
    pub fn flags(&self) -> AllocationFlags {
        let packed = self.packed_data.load(Ordering::Acquire);
        let flags = (packed & Self::FLAGS_MASK) as u16;
        AllocationFlags::from_u16(flags)
    }

    /// Gets relative timestamp
    #[inline]
    pub fn timestamp(&self) -> Duration {
        let millis = self.timestamp.load(Ordering::Acquire);
        Duration::from_millis(millis as u64)
    }

    /// Updates flags atomically
    pub fn update_flags(&self, new_flags: AllocationFlags) {
        let current = self.packed_data.load(Ordering::Acquire);
        let updated = (current & !Self::FLAGS_MASK) | (new_flags.to_u16() as u64);
        self.packed_data.store(updated, Ordering::Release);
    }

    /// Marks allocation as deallocated
    #[inline]
    pub fn mark_deallocated(&self) {
        let mut flags = self.flags();
        flags.is_active = false;
        self.update_flags(flags);
    }

    /// Checks if allocation is active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.flags().is_active
    }

    /// Total memory footprint (16 bytes vs ~80 bytes for original)
    pub const MEMORY_FOOTPRINT: usize = std::mem::size_of::<Self>();
}

impl SizeClass {
    pub fn from_size(size: usize) -> Self {
        match size {
            0..=1024 => Self::Tiny,
            1025..=65536 => Self::Small,
            65537..=1048576 => Self::Medium,
            1048577..=16777216 => Self::Large,
            _ => Self::Huge,
        }
    }

    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Tiny,
            1 => Self::Small,
            2 => Self::Medium,
            3 => Self::Large,
            _ => Self::Huge,
        }
    }

    /// Get approximate size for this class (for statistics)
    pub fn approximate_size(&self) -> usize {
        match self {
            Self::Tiny => 512,
            Self::Small => 32768,
            Self::Medium => 524288,
            Self::Large => 8388608,
            Self::Huge => 33554432,
        }
    }
}

impl CompactDeviceId {
    pub fn from_device(device: &Device) -> Self {
        // Simplified device identification based on debug string
        let device_str = format!("{:?}", device);
        if device_str.contains("Metal") {
            Self::Metal
        } else if device_str.contains("Mlx") {
            Self::Mlx
        } else if device_str.contains("Cpu") {
            Self::Cpu
        } else {
            Self::Other
        }
    }

    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Cpu,
            1 => Self::Cuda,
            2 => Self::Metal,
            3 => Self::Mlx,
            _ => Self::Other,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::Metal => "Metal",
            Self::Mlx => "MLX",
            Self::Other => "Other",
        }
    }
}

impl AllocationFlags {
    pub fn new() -> Self {
        Self {
            is_active: true,
            is_monitored: false,
            is_large: false,
            cleanup_failed: false,
            reserved: 0,
        }
    }

    pub fn to_u16(&self) -> u16 {
        let mut flags = 0u16;
        if self.is_active { flags |= 0x01; }
        if self.is_monitored { flags |= 0x02; }
        if self.is_large { flags |= 0x04; }
        if self.cleanup_failed { flags |= 0x08; }
        flags |= (self.reserved as u16) << 8;
        flags
    }

    pub fn from_u16(value: u16) -> Self {
        Self {
            is_active: (value & 0x01) != 0,
            is_monitored: (value & 0x02) != 0,
            is_large: (value & 0x04) != 0,
            cleanup_failed: (value & 0x08) != 0,
            reserved: ((value >> 8) & 0xFF) as u8,
        }
    }
}

/// Adaptive sampling controller to balance overhead vs accuracy
#[derive(Debug)]
pub struct AdaptiveSamplingController {
    /// Base sample rate (0.0 to 1.0)
    base_sample_rate: AtomicU32, // f32 as u32 bits
    /// Large allocation threshold (full tracking above this size)
    large_allocation_threshold: usize,
    /// Performance window for overhead calculation
    performance_window: std::sync::RwLock<Vec<f32>>,
    /// Target overhead percentage
    target_overhead: f32,
}

impl AdaptiveSamplingController {
    pub fn new(base_sample_rate: f32, large_threshold: usize, target_overhead: f32) -> Self {
        Self {
            base_sample_rate: AtomicU32::new(base_sample_rate.to_bits()),
            large_allocation_threshold: large_threshold,
            performance_window: std::sync::RwLock::new(Vec::with_capacity(1000)),
            target_overhead,
        }
    }

    /// Determines if an allocation should be tracked
    pub fn should_track(&self, size: usize) -> bool {
        // Always track large allocations
        if size >= self.large_allocation_threshold {
            return true;
        }

        // Use sampling for smaller allocations
        let sample_rate = f32::from_bits(self.base_sample_rate.load(Ordering::Acquire));
        fastrand::f32() < sample_rate
    }

    /// Updates sample rate based on current overhead
    pub fn update_sample_rate(&self, current_overhead: f32) {
        let current_rate = f32::from_bits(self.base_sample_rate.load(Ordering::Acquire));
        
        let adjustment = if current_overhead < self.target_overhead * 0.5 {
            1.2 // Increase sampling by 20%
        } else if current_overhead > self.target_overhead * 0.8 {
            0.8 // Decrease sampling by 20%
        } else {
            1.0 // No adjustment
        };
        
        let new_rate = (current_rate * adjustment).clamp(0.01, 0.5);
        self.base_sample_rate.store(new_rate.to_bits(), Ordering::Release);
        
        // Update performance window
        if let Ok(mut window) = self.performance_window.write() {
            window.push(current_overhead);
            if window.len() > 1000 {
                window.remove(0);
            }
        }
    }

    /// Gets current sample rate
    pub fn current_sample_rate(&self) -> f32 {
        f32::from_bits(self.base_sample_rate.load(Ordering::Acquire))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_optimized_metadata_size() {
        // Verify the metadata size is significantly smaller
        assert!(OptimizedAllocationMetadata::MEMORY_FOOTPRINT <= 16);
        // Original AllocationInfo is approximately 80+ bytes
        println!("Optimized metadata size: {} bytes", OptimizedAllocationMetadata::MEMORY_FOOTPRINT);
    }

    #[test]
    fn test_metadata_packing_unpacking() {
        let device = Device::Cpu;
        let tracking_start = Instant::now();
        let id = AllocationId(12345);
        
        let metadata = OptimizedAllocationMetadata::new(id, 2048, &device, tracking_start);
        
        // Verify data integrity
        assert_eq!(metadata.allocation_id(), id);
        assert_eq!(metadata.size_class(), SizeClass::Small);
        assert_eq!(metadata.device_id(), CompactDeviceId::Cpu);
        assert!(metadata.is_active());
    }

    #[test]
    fn test_flags_operations() {
        let device = Device::Cpu;
        let tracking_start = Instant::now();
        let id = AllocationId(1);
        
        let metadata = OptimizedAllocationMetadata::new(id, 1024, &device, tracking_start);
        assert!(metadata.is_active());
        
        metadata.mark_deallocated();
        assert!(!metadata.is_active());
    }

    #[test]
    fn test_adaptive_sampling() {
        let controller = AdaptiveSamplingController::new(0.1, 1_048_576, 0.1);
        
        // Small allocation should use sampling
        let should_track_small = (0..100)
            .map(|_| controller.should_track(1024))
            .filter(|&x| x)
            .count();
        
        // Should track roughly 10% (with some variance)
        assert!(should_track_small >= 5 && should_track_small <= 20);
        
        // Large allocation should always be tracked
        assert!(controller.should_track(2_000_000));
    }

    #[test]
    fn test_sample_rate_adaptation() {
        let controller = AdaptiveSamplingController::new(0.1, 1_048_576, 0.1);
        let initial_rate = controller.current_sample_rate();
        
        // High overhead should decrease sample rate
        controller.update_sample_rate(0.15);
        let new_rate = controller.current_sample_rate();
        assert!(new_rate < initial_rate);
        
        // Low overhead should increase sample rate  
        controller.update_sample_rate(0.03);
        let final_rate = controller.current_sample_rate();
        assert!(final_rate > new_rate);
    }
}

// Manual Clone implementation for OptimizedAllocationMetadata since AtomicU64 doesn't implement Clone
impl Clone for OptimizedAllocationMetadata {
    fn clone(&self) -> Self {
        Self {
            packed_data: AtomicU64::new(self.packed_data.load(Ordering::Relaxed)),
            timestamp: AtomicU64::new(self.timestamp.load(Ordering::Relaxed)),
        }
    }
}
