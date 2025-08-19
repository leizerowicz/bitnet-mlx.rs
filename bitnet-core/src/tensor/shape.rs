//! Tensor Shape Management and Broadcasting
//!
//! This module provides comprehensive shape management for BitNet tensors,
//! including advanced broadcasting compatible with NumPy/PyTorch semantics,
//! shape validation, and dimension manipulation utilities.

use std::fmt;
use std::ops::Range;
use serde::{Deserialize, Serialize};

#[cfg(feature = "tracing")]
use tracing::{debug, warn, error};

/// Slice indexing operations for tensor views
///
/// This enum represents different ways to index into a tensor dimension
/// for creating views and performing slicing operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceIndex {
    /// Select all elements in the dimension
    Full,
    /// Select a single element (removes dimension)
    Index(usize),
    /// Select a range of elements
    Range(Range<usize>),
    /// Select elements with a step (start..end, step)
    Step(Range<usize>, usize),
}

/// Memory requirements for tensor storage
///
/// This struct provides detailed information about the memory requirements
/// for storing a tensor with a particular shape and element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRequirements {
    /// Total bytes required for tensor data
    pub total_bytes: usize,
    /// Bytes required including alignment padding
    pub aligned_bytes: usize,
    /// Required memory alignment in bytes
    pub alignment: usize,
    /// Number of elements in the tensor
    pub element_count: usize,
    /// Size of each element in bytes
    pub element_size: usize,
    /// Whether the tensor memory layout is contiguous
    pub is_contiguous: bool,
}

/// Shape operation types for operation chaining
///
/// This enum represents different shape manipulation operations
/// that can be applied in sequence to a tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeOperation {
    /// Reshape to new dimensions
    Reshape(Vec<usize>),
    /// Transpose with given axes
    Transpose(Vec<usize>),
    /// Squeeze dimensions (None for all size-1 dims)
    Squeeze(Option<usize>),
    /// Add dimension at specified axis
    Unsqueeze(usize),
    /// Pad to target rank with size-1 dimensions
    PadToRank(usize, bool),
}

/// Memory layout recommendations
///
/// This struct provides optimization recommendations for tensor
/// memory layout based on shape characteristics and usage patterns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutRecommendation {
    /// Whether contiguous layout is recommended
    pub is_contiguous_recommended: bool,
    /// Whether the tensor size is cache-friendly
    pub cache_friendly: bool,
    /// Whether the tensor dimensions are SIMD-friendly
    pub simd_friendly: bool,
    /// Recommended memory alignment in bytes
    pub recommended_alignment: usize,
    /// Expected memory access pattern
    pub memory_access_pattern: MemoryAccessPattern,
}

/// Memory access patterns for optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryAccessPattern {
    /// Sequential access (cache-friendly)
    Sequential,
    /// Strided access (potentially cache-unfriendly)
    Strided,
    /// Random access (cache-unfriendly)
    Random,
}

/// Tensor shape representation with advanced broadcasting support
///
/// TensorShape provides a comprehensive shape system that supports
/// multi-dimensional tensors with broadcasting operations compatible
/// with NumPy and PyTorch semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorShape {
    /// Dimensions of the tensor
    dims: Vec<usize>,
    /// Stride information for memory layout
    strides: Option<Vec<isize>>,
    /// Whether this shape uses C-contiguous (row-major) layout
    c_contiguous: bool,
}

impl TensorShape {
    /// Creates a new tensor shape from dimensions
    ///
    /// # Arguments
    ///
    /// * `dims` - Slice of dimension sizes
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[2, 3, 4]);
    /// assert_eq!(shape.dims(), &[2, 3, 4]);
    /// assert_eq!(shape.rank(), 3);
    /// ```
    pub fn new(dims: &[usize]) -> Self {
        let strides = Self::compute_c_contiguous_strides(dims);
        
        Self {
            dims: dims.to_vec(),
            strides: Some(strides),
            c_contiguous: true,
        }
    }

    /// Creates a new tensor shape with custom strides
    ///
    /// # Arguments
    ///
    /// * `dims` - Slice of dimension sizes
    /// * `strides` - Custom stride information
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::with_strides(&[2, 3], &[3, 1]);
    /// assert_eq!(shape.strides().unwrap(), &[3, 1]);
    /// ```
    pub fn with_strides(dims: &[usize], strides: &[isize]) -> Self {
        assert_eq!(dims.len(), strides.len(), "Dimensions and strides must have same length");
        
        let c_contiguous = Self::is_c_contiguous_strides(dims, strides);
        
        Self {
            dims: dims.to_vec(),
            strides: Some(strides.to_vec()),
            c_contiguous,
        }
    }

    /// Creates a scalar tensor shape (0 dimensions)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::scalar();
    /// assert_eq!(shape.rank(), 0);
    /// assert!(shape.is_scalar());
    /// ```
    pub fn scalar() -> Self {
        Self {
            dims: Vec::new(),
            strides: Some(Vec::new()),
            c_contiguous: true,
        }
    }

    /// Returns the dimensions of the tensor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[2, 3, 4]);
    /// assert_eq!(shape.dims(), &[2, 3, 4]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the number of dimensions (rank)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[2, 3, 4]);
    /// assert_eq!(shape.rank(), 3);
    /// ```
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[2, 3, 4]);
    /// assert_eq!(shape.num_elements(), 24);
    /// ```
    pub fn num_elements(&self) -> usize {
        if self.dims.is_empty() {
            1 // Scalar has 1 element
        } else {
            self.dims.iter().product()
        }
    }

    /// Returns the stride information
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[2, 3]);
    /// let strides = shape.strides().unwrap();
    /// assert_eq!(strides, &[3, 1]);
    /// ```
    pub fn strides(&self) -> Option<&[isize]> {
        self.strides.as_deref()
    }

    /// Returns true if the tensor is scalar (0-dimensional)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let scalar = TensorShape::scalar();
    /// assert!(scalar.is_scalar());
    ///
    /// let vector = TensorShape::new(&[5]);
    /// assert!(!vector.is_scalar());
    /// ```
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Returns true if the tensor is a vector (1-dimensional)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let vector = TensorShape::new(&[5]);
    /// assert!(vector.is_vector());
    ///
    /// let matrix = TensorShape::new(&[2, 3]);
    /// assert!(!matrix.is_vector());
    /// ```
    pub fn is_vector(&self) -> bool {
        self.dims.len() == 1
    }

    /// Returns true if the tensor is a matrix (2-dimensional)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let matrix = TensorShape::new(&[2, 3]);
    /// assert!(matrix.is_matrix());
    ///
    /// let tensor = TensorShape::new(&[2, 3, 4]);
    /// assert!(!tensor.is_matrix());
    /// ```
    pub fn is_matrix(&self) -> bool {
        self.dims.len() == 2
    }

    /// Returns true if this shape uses C-contiguous (row-major) layout
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[2, 3]);
    /// assert!(shape.is_c_contiguous());
    /// ```
    pub fn is_c_contiguous(&self) -> bool {
        self.c_contiguous
    }

    /// Reshapes the tensor to a new shape
    ///
    /// # Arguments
    ///
    /// * `new_dims` - New dimensions for the tensor
    ///
    /// # Returns
    ///
    /// Result containing new TensorShape or error if invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[2, 6]);
    /// let reshaped = shape.reshape(&[3, 4]).unwrap();
    /// assert_eq!(reshaped.dims(), &[3, 4]);
    /// ```
    pub fn reshape(&self, new_dims: &[usize]) -> ShapeResult<TensorShape> {
        let current_elements = self.num_elements();
        let new_elements: usize = new_dims.iter().product();
        
        if current_elements != new_elements {
            return Err(ShapeError::IncompatibleReshape {
                from: self.clone(),
                to: TensorShape::new(new_dims),
                reason: format!("Element count mismatch: {} != {}", current_elements, new_elements),
            });
        }
        
        Ok(TensorShape::new(new_dims))
    }

    /// Transposes the tensor by permuting dimensions
    ///
    /// # Arguments
    ///
    /// * `axes` - Permutation of axes indices
    ///
    /// # Returns
    ///
    /// Result containing transposed TensorShape or error if invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[2, 3, 4]);
    /// let transposed = shape.transpose(&[2, 0, 1]).unwrap();
    /// assert_eq!(transposed.dims(), &[4, 2, 3]);
    /// ```
    pub fn transpose(&self, axes: &[usize]) -> ShapeResult<TensorShape> {
        if axes.len() != self.rank() {
            return Err(ShapeError::InvalidTranspose {
                shape: self.clone(),
                axes: axes.to_vec(),
                reason: format!("Axes length {} != tensor rank {}", axes.len(), self.rank()),
            });
        }

        // Validate axes contains each dimension exactly once
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable();
        let expected: Vec<usize> = (0..self.rank()).collect();
        
        if sorted_axes != expected {
            return Err(ShapeError::InvalidTranspose {
                shape: self.clone(),
                axes: axes.to_vec(),
                reason: "Axes must be a permutation of dimension indices".to_string(),
            });
        }

        // Apply transpose to dimensions
        let new_dims: Vec<usize> = axes.iter().map(|&i| self.dims[i]).collect();
        
        // Apply transpose to strides if available
        let new_strides = if let Some(ref strides) = self.strides {
            Some(axes.iter().map(|&i| strides[i]).collect())
        } else {
            None
        };

        let mut result = TensorShape::new(&new_dims);
        if let Some(strides) = new_strides {
            result.strides = Some(strides);
            result.c_contiguous = Self::is_c_contiguous_strides(&new_dims, &result.strides.as_ref().unwrap());
        }

        Ok(result)
    }

    /// Squeezes dimensions of size 1
    ///
    /// # Arguments
    ///
    /// * `axis` - Optional specific axis to squeeze, or None for all size-1 dimensions
    ///
    /// # Returns
    ///
    /// Result containing squeezed TensorShape
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[2, 1, 3, 1]);
    /// let squeezed = shape.squeeze(None).unwrap();
    /// assert_eq!(squeezed.dims(), &[2, 3]);
    /// ```
    pub fn squeeze(&self, axis: Option<usize>) -> ShapeResult<TensorShape> {
        let new_dims = if let Some(axis) = axis {
            if axis >= self.rank() {
                return Err(ShapeError::InvalidAxis {
                    axis,
                    rank: self.rank(),
                });
            }
            if self.dims[axis] != 1 {
                return Err(ShapeError::CannotSqueeze {
                    shape: self.clone(),
                    axis,
                    size: self.dims[axis],
                });
            }
            let mut dims = self.dims.clone();
            dims.remove(axis);
            dims
        } else {
            self.dims.iter().filter(|&&d| d != 1).cloned().collect()
        };

        Ok(TensorShape::new(&new_dims))
    }

    /// Unsqueezes (adds) dimensions of size 1
    ///
    /// # Arguments
    ///
    /// * `axis` - Axis position to add dimension
    ///
    /// # Returns
    ///
    /// Result containing unsqueezed TensorShape
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[2, 3]);
    /// let unsqueezed = shape.unsqueeze(1).unwrap();
    /// assert_eq!(unsqueezed.dims(), &[2, 1, 3]);
    /// ```
    pub fn unsqueeze(&self, axis: usize) -> ShapeResult<TensorShape> {
        if axis > self.rank() {
            return Err(ShapeError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }

        let mut new_dims = self.dims.clone();
        new_dims.insert(axis, 1);

        Ok(TensorShape::new(&new_dims))
    }

    /// Computes C-contiguous strides for given dimensions
    fn compute_c_contiguous_strides(dims: &[usize]) -> Vec<isize> {
        if dims.is_empty() {
            return Vec::new();
        }

        let mut strides = Vec::with_capacity(dims.len());
        let mut stride = 1isize;
        
        for &dim in dims.iter().rev() {
            strides.push(stride);
            stride *= dim as isize;
        }
        
        strides.reverse();
        strides
    }

    /// Checks if given strides represent C-contiguous layout
    fn is_c_contiguous_strides(dims: &[usize], strides: &[isize]) -> bool {
        if dims.is_empty() {
            return true;
        }

        let expected = Self::compute_c_contiguous_strides(dims);
        strides == expected
    }

    // === ADVANCED INDEXING AND SLICING OPERATIONS ===

    /// Creates a view of the tensor with specified slicing operations
    ///
    /// A view shares the same underlying data but potentially with different
    /// shape and strides. This is a zero-copy operation when possible.
    ///
    /// # Arguments
    ///
    /// * `slices` - Slice operations for each dimension
    ///
    /// # Returns
    ///
    /// Result containing the view shape with updated strides
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::{TensorShape, SliceIndex};
    ///
    /// let shape = TensorShape::new(&[4, 6, 8]);
    /// let slices = vec![
    ///     SliceIndex::Range(0..2),  // First 2 rows
    ///     SliceIndex::Full,         // All columns  
    ///     SliceIndex::Range(2..6),  // Columns 2-5
    /// ];
    /// let view = shape.view(&slices).unwrap();
    /// assert_eq!(view.dims(), &[2, 6, 4]);
    /// ```
    pub fn view(&self, slices: &[SliceIndex]) -> ShapeResult<TensorShape> {
        if slices.len() > self.rank() {
            return Err(ShapeError::InvalidSlice {
                shape: self.clone(),
                reason: format!("Too many slice indices: {} for tensor of rank {}", slices.len(), self.rank()),
            });
        }

        let mut new_dims = Vec::new();
        let mut new_strides = Vec::new();
        let default_strides = Self::compute_c_contiguous_strides(&self.dims);
        let current_strides = self.strides().unwrap_or(&default_strides);

        for (i, slice) in slices.iter().enumerate() {
            let dim_size = self.dims[i];
            let stride = current_strides[i];

            match slice {
                SliceIndex::Full => {
                    new_dims.push(dim_size);
                    new_strides.push(stride);
                }
                SliceIndex::Index(idx) => {
                    // Single index removes dimension
                    if *idx >= dim_size {
                        return Err(ShapeError::IndexOutOfBounds {
                            shape: self.clone(),
                            axis: i,
                            index: *idx,
                            size: dim_size,
                        });
                    }
                    // No dimension added for single index
                }
                SliceIndex::Range(range) => {
                    let start = range.start.min(dim_size);
                    let end = range.end.min(dim_size);
                    
                    if start >= end {
                        return Err(ShapeError::InvalidSlice {
                            shape: self.clone(),
                            reason: format!("Invalid range {}..{} for dimension {} of size {}", 
                                           start, end, i, dim_size),
                        });
                    }
                    
                    new_dims.push(end - start);
                    new_strides.push(stride);
                }
                SliceIndex::Step(range, step) => {
                    if *step == 0 {
                        return Err(ShapeError::InvalidSlice {
                            shape: self.clone(),
                            reason: "Step cannot be zero".to_string(),
                        });
                    }

                    let start = range.start.min(dim_size);
                    let end = range.end.min(dim_size);
                    
                    if start >= end {
                        return Err(ShapeError::InvalidSlice {
                            shape: self.clone(),
                            reason: format!("Invalid range {}..{} for dimension {} of size {}", 
                                           start, end, i, dim_size),
                        });
                    }

                    let new_size = ((end - start) + step - 1) / step; // Ceiling division
                    new_dims.push(new_size);
                    new_strides.push(stride * (*step as isize));
                }
            }
        }

        // Add remaining dimensions that weren't sliced
        for i in slices.len()..self.rank() {
            new_dims.push(self.dims[i]);
            new_strides.push(current_strides[i]);
        }

        Ok(TensorShape::with_strides(&new_dims, &new_strides))
    }

    /// Validates multi-dimensional indices for tensor access
    ///
    /// # Arguments
    ///
    /// * `indices` - Multi-dimensional indices to validate
    ///
    /// # Returns
    ///
    /// Result indicating whether indices are valid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[3, 4, 5]);
    /// assert!(shape.validate_indices(&[1, 2, 3]).is_ok());
    /// assert!(shape.validate_indices(&[3, 2, 1]).is_err()); // First index out of bounds
    /// ```
    pub fn validate_indices(&self, indices: &[usize]) -> ShapeResult<()> {
        if indices.len() != self.rank() {
            return Err(ShapeError::InvalidIndices {
                shape: self.clone(),
                indices: indices.to_vec(),
                reason: format!("Index count {} doesn't match tensor rank {}", indices.len(), self.rank()),
            });
        }

        for (axis, (&index, &dim_size)) in indices.iter().zip(self.dims.iter()).enumerate() {
            if index >= dim_size {
                return Err(ShapeError::IndexOutOfBounds {
                    shape: self.clone(),
                    axis,
                    index,
                    size: dim_size,
                });
            }
        }

        Ok(())
    }

    /// Computes the linear memory offset for multi-dimensional indices
    ///
    /// # Arguments
    ///
    /// * `indices` - Multi-dimensional indices
    ///
    /// # Returns
    ///
    /// Linear offset in memory or error if indices are invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[3, 4]);
    /// let offset = shape.linear_offset(&[1, 2]).unwrap();
    /// assert_eq!(offset, 6); // 1*4 + 2
    /// ```
    pub fn linear_offset(&self, indices: &[usize]) -> ShapeResult<usize> {
        self.validate_indices(indices)?;
        
        let default_strides = Self::compute_c_contiguous_strides(&self.dims);
        let strides = self.strides().unwrap_or(&default_strides);
        let mut offset = 0isize;
        
        for (&index, &stride) in indices.iter().zip(strides.iter()) {
            offset += (index as isize) * stride;
        }
        
        Ok(offset as usize)
    }

    /// Computes multi-dimensional indices from linear offset
    ///
    /// # Arguments
    ///
    /// * `linear_offset` - Linear offset in memory
    ///
    /// # Returns
    ///
    /// Multi-dimensional indices or error if offset is invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[3, 4]);
    /// let indices = shape.indices_from_offset(6).unwrap();
    /// assert_eq!(indices, vec![1, 2]);
    /// ```
    pub fn indices_from_offset(&self, linear_offset: usize) -> ShapeResult<Vec<usize>> {
        if linear_offset >= self.num_elements() {
            return Err(ShapeError::IndexOutOfBounds {
                shape: self.clone(),
                axis: 0, // Not applicable for linear offset
                index: linear_offset,
                size: self.num_elements(),
            });
        }

        let mut indices = Vec::with_capacity(self.rank());
        let mut remaining_offset = linear_offset;

        // Work backwards through dimensions
        for &dim_size in self.dims.iter().rev() {
            let index = remaining_offset % dim_size;
            indices.push(index);
            remaining_offset /= dim_size;
        }

        indices.reverse();
        Ok(indices)
    }

    /// Checks if the tensor memory layout is contiguous
    ///
    /// A tensor is contiguous if elements are stored in a single block
    /// of memory without gaps, following C-contiguous (row-major) order.
    ///
    /// # Returns
    ///
    /// True if tensor is contiguous
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[3, 4]);
    /// assert!(shape.is_contiguous());
    ///
    /// let custom_strides = TensorShape::with_strides(&[3, 4], &[8, 1]); // Non-contiguous
    /// assert!(!custom_strides.is_contiguous());
    /// ```
    pub fn is_contiguous(&self) -> bool {
        self.c_contiguous
    }

    /// Creates a contiguous version of the shape
    ///
    /// If the shape is already contiguous, returns a clone.
    /// Otherwise, returns a new shape with C-contiguous strides.
    ///
    /// # Returns
    ///
    /// Contiguous version of the shape
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::with_strides(&[3, 4], &[8, 1]);
    /// let contiguous = shape.contiguous();
    /// assert!(contiguous.is_contiguous());
    /// ```
    pub fn contiguous(&self) -> TensorShape {
        if self.is_contiguous() {
            self.clone()
        } else {
            TensorShape::new(&self.dims)
        }
    }

    /// Calculates memory requirements for the tensor shape
    ///
    /// # Arguments
    ///
    /// * `element_size` - Size of each element in bytes
    ///
    /// # Returns
    ///
    /// Memory requirements information
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[100, 200]);
    /// let memory_req = shape.memory_requirements(4); // 4 bytes per f32
    /// assert_eq!(memory_req.total_bytes, 80000);
    /// ```
    pub fn memory_requirements(&self, element_size: usize) -> MemoryRequirements {
        let num_elements = self.num_elements();
        let total_bytes = num_elements * element_size;
        
        // Calculate alignment requirements
        let alignment = if element_size <= 4 {
            element_size
        } else if element_size <= 8 {
            8
        } else {
            16 // For larger types, use 16-byte alignment for SIMD
        };

        let aligned_size = (total_bytes + alignment - 1) & !(alignment - 1);

        MemoryRequirements {
            total_bytes,
            aligned_bytes: aligned_size,
            alignment,
            element_count: num_elements,
            element_size,
            is_contiguous: self.is_contiguous(),
        }
    }

    /// Pads the shape with dimensions of size 1 to match a target rank
    ///
    /// This is useful for aligning tensors for broadcasting operations.
    ///
    /// # Arguments
    ///
    /// * `target_rank` - Target rank to pad to
    /// * `prepend` - Whether to add dimensions at the beginning (true) or end (false)
    ///
    /// # Returns
    ///
    /// New shape with padded dimensions
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[3, 4]);
    /// let padded = shape.pad_to_rank(4, true);
    /// assert_eq!(padded.dims(), &[1, 1, 3, 4]);
    /// ```
    pub fn pad_to_rank(&self, target_rank: usize, prepend: bool) -> TensorShape {
        if target_rank <= self.rank() {
            return self.clone();
        }

        let padding_needed = target_rank - self.rank();
        let mut new_dims = Vec::with_capacity(target_rank);

        if prepend {
            // Add 1s at the beginning
            for _ in 0..padding_needed {
                new_dims.push(1);
            }
            new_dims.extend_from_slice(&self.dims);
        } else {
            // Add 1s at the end
            new_dims.extend_from_slice(&self.dims);
            for _ in 0..padding_needed {
                new_dims.push(1);
            }
        }

        TensorShape::new(&new_dims)
    }

    /// Computes the shape after applying multiple operations in sequence
    ///
    /// This is useful for operation chaining and optimization planning.
    ///
    /// # Arguments
    ///
    /// * `operations` - Sequence of shape operations to apply
    ///
    /// # Returns
    ///
    /// Result containing the final shape after all operations
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::{TensorShape, ShapeOperation};
    ///
    /// let shape = TensorShape::new(&[2, 3, 4]);
    /// let ops = vec![
    ///     ShapeOperation::Transpose(vec![2, 0, 1]),
    ///     ShapeOperation::Squeeze(None),
    /// ];
    /// let final_shape = shape.apply_operations(&ops).unwrap();
    /// ```
    pub fn apply_operations(&self, operations: &[ShapeOperation]) -> ShapeResult<TensorShape> {
        let mut current_shape = self.clone();
        
        for operation in operations {
            current_shape = match operation {
                ShapeOperation::Reshape(new_dims) => current_shape.reshape(new_dims)?,
                ShapeOperation::Transpose(axes) => current_shape.transpose(axes)?,
                ShapeOperation::Squeeze(axis) => current_shape.squeeze(*axis)?,
                ShapeOperation::Unsqueeze(axis) => current_shape.unsqueeze(*axis)?,
                ShapeOperation::PadToRank(rank, prepend) => current_shape.pad_to_rank(*rank, *prepend),
            };
        }
        
        Ok(current_shape)
    }

    /// Checks if this shape can be reshaped to the target shape
    ///
    /// # Arguments
    ///
    /// * `target` - Target shape to reshape to
    ///
    /// # Returns
    ///
    /// True if reshape is possible (same number of elements)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[2, 6]);
    /// let target = TensorShape::new(&[3, 4]);
    /// assert!(shape.can_reshape_to(&target));
    ///
    /// let invalid_target = TensorShape::new(&[3, 5]);
    /// assert!(!shape.can_reshape_to(&invalid_target));
    /// ```
    pub fn can_reshape_to(&self, target: &TensorShape) -> bool {
        self.num_elements() == target.num_elements()
    }

    /// Finds the optimal memory layout for this shape given usage patterns
    ///
    /// This analyzes the shape and provides recommendations for optimal
    /// memory layout based on common access patterns.
    ///
    /// # Returns
    ///
    /// Recommended memory layout configuration
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::tensor::TensorShape;
    ///
    /// let shape = TensorShape::new(&[100, 200, 300]);
    /// let layout = shape.optimal_layout();
    /// assert!(layout.is_contiguous_recommended);
    /// ```
    pub fn optimal_layout(&self) -> LayoutRecommendation {
        let num_elements = self.num_elements();
        let rank = self.rank();

        // Heuristics for optimal layout
        let is_contiguous_recommended = match rank {
            0..=1 => true, // Scalars and vectors are always contiguous
            2 => num_elements > 1000, // Large matrices benefit from contiguous layout
            3..=4 => num_elements > 10000, // 3D/4D tensors need larger sizes
            _ => num_elements > 100000, // High-dimensional tensors need very large sizes
        };

        let cache_friendly = num_elements < 1_000_000; // Fits in typical L3 cache
        let simd_friendly = self.dims.last().map_or(false, |&d| d % 4 == 0 || d % 8 == 0);
        
        LayoutRecommendation {
            is_contiguous_recommended,
            cache_friendly,
            simd_friendly,
            recommended_alignment: if simd_friendly { 32 } else { 16 },
            memory_access_pattern: if rank <= 2 { 
                MemoryAccessPattern::Sequential 
            } else { 
                MemoryAccessPattern::Strided 
            },
        }
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.dims)
    }
}

/// Broadcasting compatibility trait
///
/// This trait provides methods for checking and performing broadcasting
/// operations between tensor shapes, following NumPy/PyTorch semantics.
pub trait BroadcastCompatible {
    /// Checks if two shapes are broadcast compatible
    ///
    /// # Arguments
    ///
    /// * `other` - Other shape to check compatibility with
    ///
    /// # Returns
    ///
    /// True if shapes are broadcast compatible
    fn is_broadcast_compatible(&self, other: &Self) -> bool;

    /// Computes the broadcast shape for two shapes
    ///
    /// # Arguments
    ///
    /// * `other` - Other shape to broadcast with
    ///
    /// # Returns
    ///
    /// Result containing broadcast shape or error if incompatible
    fn broadcast_shape(&self, other: &Self) -> ShapeResult<TensorShape>;

    /// Checks if this shape can be broadcast to a target shape
    ///
    /// # Arguments
    ///
    /// * `target` - Target shape to broadcast to
    ///
    /// # Returns
    ///
    /// True if broadcasting is possible
    fn can_broadcast_to(&self, target: &Self) -> bool;
}

impl BroadcastCompatible for TensorShape {
    fn is_broadcast_compatible(&self, other: &Self) -> bool {
        let dims1 = &self.dims;
        let dims2 = &other.dims;
        
        // Iterate from the trailing dimensions
        let max_rank = dims1.len().max(dims2.len());
        
        for i in 0..max_rank {
            let dim1 = if i < dims1.len() {
                Some(dims1[dims1.len() - 1 - i])
            } else {
                None
            };
            
            let dim2 = if i < dims2.len() {
                Some(dims2[dims2.len() - 1 - i])
            } else {
                None
            };
            
            match (dim1, dim2) {
                (Some(d1), Some(d2)) => {
                    // Dimensions must be equal or one must be 1
                    if d1 != d2 && d1 != 1 && d2 != 1 {
                        return false;
                    }
                }
                (None, Some(_)) | (Some(_), None) => {
                    // Missing dimensions are implicitly 1, always compatible
                    continue;
                }
                (None, None) => break,
            }
        }
        
        true
    }

    fn broadcast_shape(&self, other: &Self) -> ShapeResult<TensorShape> {
        if !self.is_broadcast_compatible(other) {
            return Err(ShapeError::IncompatibleBroadcast {
                lhs: self.clone(),
                rhs: other.clone(),
            });
        }

        let dims1 = &self.dims;
        let dims2 = &other.dims;
        let max_rank = dims1.len().max(dims2.len());
        let mut result_dims = Vec::with_capacity(max_rank);
        
        for i in 0..max_rank {
            let dim1 = if i < dims1.len() {
                dims1[dims1.len() - 1 - i]
            } else {
                1
            };
            
            let dim2 = if i < dims2.len() {
                dims2[dims2.len() - 1 - i]
            } else {
                1
            };
            
            result_dims.push(dim1.max(dim2));
        }
        
        result_dims.reverse();
        Ok(TensorShape::new(&result_dims))
    }

    fn can_broadcast_to(&self, target: &Self) -> bool {
        let src_dims = &self.dims;
        let target_dims = &target.dims;
        
        // Source must have rank <= target rank
        if src_dims.len() > target_dims.len() {
            return false;
        }
        
        // Check from trailing dimensions
        for i in 0..src_dims.len() {
            let src_dim = src_dims[src_dims.len() - 1 - i];
            let target_dim = target_dims[target_dims.len() - 1 - i];
            
            // Source dimension must be 1 or equal to target dimension
            if src_dim != 1 && src_dim != target_dim {
                return false;
            }
        }
        
        true
    }
}

/// Shape operation errors
#[derive(Debug, thiserror::Error)]
pub enum ShapeError {
    /// Incompatible shapes for broadcasting
    #[error("Cannot broadcast shapes {lhs} and {rhs}")]
    IncompatibleBroadcast { lhs: TensorShape, rhs: TensorShape },
    
    /// Invalid reshape operation
    #[error("Cannot reshape {from} to {to}: {reason}")]
    IncompatibleReshape { from: TensorShape, to: TensorShape, reason: String },
    
    /// Invalid transpose operation
    #[error("Cannot transpose {shape:?} with axes {axes:?}: {reason}")]
    InvalidTranspose { shape: TensorShape, axes: Vec<usize>, reason: String },
    
    /// Invalid axis for operation
    #[error("Axis {axis} is invalid for tensor of rank {rank}")]
    InvalidAxis { axis: usize, rank: usize },
    
    /// Cannot squeeze dimension that is not size 1
    #[error("Cannot squeeze dimension {axis} of size {size} in shape {shape:?}")]
    CannotSqueeze { shape: TensorShape, axis: usize, size: usize },
    
    /// Invalid shape dimensions
    #[error("Invalid shape dimensions: {reason}")]
    InvalidDimensions { reason: String },

    /// Invalid slicing operation
    #[error("Invalid slice operation on {shape:?}: {reason}")]
    InvalidSlice { shape: TensorShape, reason: String },

    /// Index out of bounds
    #[error("Index {index} is out of bounds for axis {axis} of size {size} in shape {shape:?}")]
    IndexOutOfBounds { shape: TensorShape, axis: usize, index: usize, size: usize },

    /// Invalid multi-dimensional indices
    #[error("Invalid indices {indices:?} for shape {shape:?}: {reason}")]
    InvalidIndices { shape: TensorShape, indices: Vec<usize>, reason: String },
}

/// Result type for shape operations
pub type ShapeResult<T> = std::result::Result<T, ShapeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape_creation() {
        let shape = TensorShape::new(&[2, 3, 4]);
        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.num_elements(), 24);
        assert!(!shape.is_scalar());
        assert!(!shape.is_vector());
        assert!(!shape.is_matrix());
    }

    #[test]
    fn test_scalar_shape() {
        let scalar = TensorShape::scalar();
        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.num_elements(), 1);
        assert!(scalar.is_scalar());
    }

    #[test]
    fn test_vector_shape() {
        let vector = TensorShape::new(&[5]);
        assert_eq!(vector.rank(), 1);
        assert!(vector.is_vector());
        assert!(!vector.is_matrix());
    }

    #[test]
    fn test_matrix_shape() {
        let matrix = TensorShape::new(&[2, 3]);
        assert_eq!(matrix.rank(), 2);
        assert!(matrix.is_matrix());
        assert!(!matrix.is_scalar());
    }

    #[test]
    fn test_strides() {
        let shape = TensorShape::new(&[2, 3, 4]);
        let strides = shape.strides().unwrap();
        assert_eq!(strides, &[12, 4, 1]);
        assert!(shape.is_c_contiguous());
    }

    #[test]
    fn test_custom_strides() {
        let shape = TensorShape::with_strides(&[2, 3], &[1, 2]);
        assert_eq!(shape.strides().unwrap(), &[1, 2]);
        assert!(!shape.is_c_contiguous());
    }

    #[test]
    fn test_reshape() {
        let shape = TensorShape::new(&[2, 6]);
        let reshaped = shape.reshape(&[3, 4]).unwrap();
        assert_eq!(reshaped.dims(), &[3, 4]);
        
        // Invalid reshape
        let result = shape.reshape(&[3, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose() {
        let shape = TensorShape::new(&[2, 3, 4]);
        let transposed = shape.transpose(&[2, 0, 1]).unwrap();
        assert_eq!(transposed.dims(), &[4, 2, 3]);
        
        // Invalid transpose
        let result = shape.transpose(&[0, 1]); // Wrong number of axes
        assert!(result.is_err());
        
        let result = shape.transpose(&[0, 1, 3]); // Invalid axis
        assert!(result.is_err());
    }

    #[test]
    fn test_squeeze() {
        let shape = TensorShape::new(&[2, 1, 3, 1]);
        
        // Squeeze all
        let squeezed = shape.squeeze(None).unwrap();
        assert_eq!(squeezed.dims(), &[2, 3]);
        
        // Squeeze specific axis
        let squeezed = shape.squeeze(Some(1)).unwrap();
        assert_eq!(squeezed.dims(), &[2, 3, 1]);
        
        // Invalid squeeze
        let result = shape.squeeze(Some(0)); // Dimension is not 1
        assert!(result.is_err());
    }

    #[test]
    fn test_unsqueeze() {
        let shape = TensorShape::new(&[2, 3]);
        let unsqueezed = shape.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.dims(), &[2, 1, 3]);
        
        let unsqueezed = shape.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.dims(), &[1, 2, 3]);
        
        // Invalid axis
        let result = shape.unsqueeze(3);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_compatibility() {
        let shape1 = TensorShape::new(&[2, 3, 4]);
        let shape2 = TensorShape::new(&[3, 1]);
        let shape3 = TensorShape::new(&[2, 5]); // This should be incompatible (3 != 5, neither is 1)
        
        assert!(shape1.is_broadcast_compatible(&shape2));
        assert!(!shape1.is_broadcast_compatible(&shape3));
        
        // Additional compatibility tests
        let shape4 = TensorShape::new(&[1, 3, 1]);
        let shape5 = TensorShape::new(&[2, 1, 4]);
        assert!(shape4.is_broadcast_compatible(&shape5)); // Should result in [2, 3, 4]
        
        let shape6 = TensorShape::new(&[3, 4]);
        let shape7 = TensorShape::new(&[5, 4]);
        assert!(!shape6.is_broadcast_compatible(&shape7)); // Incompatible: 3 != 5, neither is 1
    }

    #[test]
    fn test_broadcast_shape() {
        let shape1 = TensorShape::new(&[2, 3, 1]);
        let shape2 = TensorShape::new(&[4]);
        
        let broadcast = shape1.broadcast_shape(&shape2).unwrap();
        assert_eq!(broadcast.dims(), &[2, 3, 4]);
        
        // This should actually be compatible: [2,3,1] + [5] -> [2,3,5]
        let shape3 = TensorShape::new(&[5]);
        let broadcast2 = shape1.broadcast_shape(&shape3).unwrap();
        assert_eq!(broadcast2.dims(), &[2, 3, 5]);
        
        // Test truly incompatible broadcast: [2,3,4] + [5] -> incompatible (4 != 5, neither is 1)
        let incompatible1 = TensorShape::new(&[2, 3, 4]);
        let incompatible2 = TensorShape::new(&[5]);
        let result = incompatible1.broadcast_shape(&incompatible2);
        assert!(result.is_err());
    }

    #[test]
    fn test_can_broadcast_to() {
        let source = TensorShape::new(&[1, 3]);
        let target = TensorShape::new(&[2, 3]);
        
        assert!(source.can_broadcast_to(&target));
        
        let invalid_target = TensorShape::new(&[2, 4]);
        assert!(!source.can_broadcast_to(&invalid_target));
    }

    #[test]
    fn test_display() {
        let shape = TensorShape::new(&[2, 3, 4]);
        assert_eq!(format!("{}", shape), "Shape([2, 3, 4])");
    }

    // === ADVANCED INDEXING AND SLICING TESTS ===

    #[test]
    fn test_view_slicing() {
        let shape = TensorShape::new(&[4, 6, 8]);
        
        // Test full slice
        let slices = vec![SliceIndex::Full, SliceIndex::Full, SliceIndex::Full];
        let view = shape.view(&slices).unwrap();
        assert_eq!(view.dims(), &[4, 6, 8]);
        
        // Test range slice
        let slices = vec![
            SliceIndex::Range(0..2),  // First 2 rows
            SliceIndex::Full,         // All columns  
            SliceIndex::Range(2..6),  // Columns 2-5
        ];
        let view = shape.view(&slices).unwrap();
        assert_eq!(view.dims(), &[2, 6, 4]);
        
        // Test index slice (removes dimension)
        let slices = vec![
            SliceIndex::Index(1),     // Select second row
            SliceIndex::Full,         // All columns
        ];
        let view = shape.view(&slices).unwrap();
        assert_eq!(view.dims(), &[6, 8]);
    }

    #[test]
    fn test_view_step_slicing() {
        let shape = TensorShape::new(&[10, 20]);
        
        // Test step slicing
        let slices = vec![
            SliceIndex::Step(0..8, 2),  // Every other element from 0 to 8
            SliceIndex::Full,
        ];
        let view = shape.view(&slices).unwrap();
        assert_eq!(view.dims(), &[4, 20]); // (8-0)/2 = 4 elements
    }

    #[test]
    fn test_view_error_cases() {
        let shape = TensorShape::new(&[4, 6]);
        
        // Too many slice indices
        let slices = vec![SliceIndex::Full, SliceIndex::Full, SliceIndex::Full];
        let result = shape.view(&slices);
        assert!(result.is_err());
        
        // Index out of bounds
        let slices = vec![SliceIndex::Index(5), SliceIndex::Full]; // Index 5 >= dimension 4
        let result = shape.view(&slices);
        assert!(result.is_err());
        
        // Invalid range
        let slices = vec![SliceIndex::Range(3..2), SliceIndex::Full]; // start >= end
        let result = shape.view(&slices);
        assert!(result.is_err());
        
        // Zero step
        let slices = vec![SliceIndex::Step(0..4, 0), SliceIndex::Full];
        let result = shape.view(&slices);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_indices() {
        let shape = TensorShape::new(&[3, 4, 5]);
        
        // Valid indices
        assert!(shape.validate_indices(&[1, 2, 3]).is_ok());
        assert!(shape.validate_indices(&[0, 0, 0]).is_ok());
        assert!(shape.validate_indices(&[2, 3, 4]).is_ok());
        
        // Invalid indices - wrong count
        assert!(shape.validate_indices(&[1, 2]).is_err());
        assert!(shape.validate_indices(&[1, 2, 3, 4]).is_err());
        
        // Invalid indices - out of bounds
        assert!(shape.validate_indices(&[3, 2, 1]).is_err()); // First index out of bounds
        assert!(shape.validate_indices(&[1, 4, 1]).is_err()); // Second index out of bounds
        assert!(shape.validate_indices(&[1, 2, 5]).is_err()); // Third index out of bounds
    }

    #[test]
    fn test_linear_offset() {
        let shape = TensorShape::new(&[3, 4]);
        
        // Test various positions
        assert_eq!(shape.linear_offset(&[0, 0]).unwrap(), 0);
        assert_eq!(shape.linear_offset(&[0, 1]).unwrap(), 1);
        assert_eq!(shape.linear_offset(&[1, 0]).unwrap(), 4);
        assert_eq!(shape.linear_offset(&[1, 2]).unwrap(), 6); // 1*4 + 2
        assert_eq!(shape.linear_offset(&[2, 3]).unwrap(), 11); // 2*4 + 3
        
        // Test 3D tensor
        let shape_3d = TensorShape::new(&[2, 3, 4]);
        assert_eq!(shape_3d.linear_offset(&[1, 1, 1]).unwrap(), 17); // 1*12 + 1*4 + 1
        
        // Invalid indices
        assert!(shape.linear_offset(&[3, 0]).is_err());
        assert!(shape.linear_offset(&[0, 4]).is_err());
        assert!(shape.linear_offset(&[0]).is_err()); // Wrong number of indices
    }

    #[test]
    fn test_indices_from_offset() {
        let shape = TensorShape::new(&[3, 4]);
        
        // Test various offsets
        assert_eq!(shape.indices_from_offset(0).unwrap(), vec![0, 0]);
        assert_eq!(shape.indices_from_offset(1).unwrap(), vec![0, 1]);
        assert_eq!(shape.indices_from_offset(4).unwrap(), vec![1, 0]);
        assert_eq!(shape.indices_from_offset(6).unwrap(), vec![1, 2]);
        assert_eq!(shape.indices_from_offset(11).unwrap(), vec![2, 3]);
        
        // Test 3D tensor
        let shape_3d = TensorShape::new(&[2, 3, 4]);
        assert_eq!(shape_3d.indices_from_offset(17).unwrap(), vec![1, 1, 1]);
        
        // Invalid offset
        assert!(shape.indices_from_offset(12).is_err()); // >= num_elements (12)
    }

    #[test]
    fn test_offset_indices_roundtrip() {
        let shape = TensorShape::new(&[5, 6, 7]);
        
        // Test roundtrip: indices -> offset -> indices
        for i in 0..5 {
            for j in 0..6 {
                for k in 0..7 {
                    let indices = vec![i, j, k];
                    let offset = shape.linear_offset(&indices).unwrap();
                    let recovered_indices = shape.indices_from_offset(offset).unwrap();
                    assert_eq!(indices, recovered_indices);
                }
            }
        }
    }

    #[test]
    fn test_is_contiguous() {
        let shape = TensorShape::new(&[3, 4]);
        assert!(shape.is_contiguous());
        
        let custom_strides = TensorShape::with_strides(&[3, 4], &[8, 1]); // Non-contiguous
        assert!(!custom_strides.is_contiguous());
    }

    #[test]
    fn test_contiguous() {
        let shape = TensorShape::new(&[3, 4]);
        let contiguous = shape.contiguous();
        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.dims(), shape.dims());
        
        let custom_strides = TensorShape::with_strides(&[3, 4], &[8, 1]); // Non-contiguous
        let made_contiguous = custom_strides.contiguous();
        assert!(made_contiguous.is_contiguous());
        assert_eq!(made_contiguous.dims(), custom_strides.dims());
    }

    #[test]
    fn test_memory_requirements() {
        let shape = TensorShape::new(&[100, 200]);
        let memory_req = shape.memory_requirements(4); // 4 bytes per f32
        
        assert_eq!(memory_req.total_bytes, 80000);
        assert_eq!(memory_req.element_count, 20000);
        assert_eq!(memory_req.element_size, 4);
        assert_eq!(memory_req.alignment, 4);
        assert!(memory_req.aligned_bytes >= memory_req.total_bytes);
        assert!(memory_req.is_contiguous);
        
        // Test larger element size alignment
        let memory_req_f64 = shape.memory_requirements(8); // 8 bytes per f64
        assert_eq!(memory_req_f64.alignment, 8);
        
        let memory_req_large = shape.memory_requirements(32); // Large element
        assert_eq!(memory_req_large.alignment, 16); // Max 16-byte alignment for SIMD
    }

    #[test]
    fn test_slice_index_equality() {
        assert_eq!(SliceIndex::Full, SliceIndex::Full);
        assert_eq!(SliceIndex::Index(5), SliceIndex::Index(5));
        assert_eq!(SliceIndex::Range(0..5), SliceIndex::Range(0..5));
        assert_eq!(SliceIndex::Step(0..10, 2), SliceIndex::Step(0..10, 2));
        
        assert_ne!(SliceIndex::Index(5), SliceIndex::Index(6));
        assert_ne!(SliceIndex::Range(0..5), SliceIndex::Range(0..6));
        assert_ne!(SliceIndex::Step(0..10, 2), SliceIndex::Step(0..10, 3));
    }

    #[test]
    fn test_memory_requirements_equality() {
        let req1 = MemoryRequirements {
            total_bytes: 100,
            aligned_bytes: 104,
            alignment: 4,
            element_count: 25,
            element_size: 4,
            is_contiguous: true,
        };
        
        let req2 = MemoryRequirements {
            total_bytes: 100,
            aligned_bytes: 104,
            alignment: 4,
            element_count: 25,
            element_size: 4,
            is_contiguous: true,
        };
        
        assert_eq!(req1, req2);
        
        let req3 = MemoryRequirements {
            total_bytes: 200,
            aligned_bytes: 104,
            alignment: 4,
            element_count: 25,
            element_size: 4,
            is_contiguous: true,
        };
        
        assert_ne!(req1, req3);
    }
}
