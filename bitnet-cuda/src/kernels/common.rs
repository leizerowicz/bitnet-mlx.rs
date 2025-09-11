//! Common CUDA utilities and definitions

pub mod constants {
    /// CUDA warp size
    pub const WARP_SIZE: usize = 32;
    
    /// Maximum threads per block
    pub const MAX_THREADS_PER_BLOCK: usize = 1024;
    
    /// BitNet quantization constants
    pub const BITNET_2BIT_MIN: i8 = -1;
    pub const BITNET_2BIT_MAX: i8 = 1;
}

pub mod utilities {
    /// Calculate optimal grid size for CUDA kernel launch
    pub fn calculate_grid_size(total_elements: usize, block_size: usize) -> usize {
        (total_elements + block_size - 1) / block_size
    }
    
    /// Align value to the next multiple of alignment
    pub fn align_up(value: usize, alignment: usize) -> usize {
        (value + alignment - 1) / alignment * alignment
    }
}

// Re-export commonly used items
pub use constants::*;
pub use utilities::*;
