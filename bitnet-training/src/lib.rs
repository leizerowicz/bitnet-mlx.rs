//! BitNet Training Library - Phase 3: Quantization-Aware Training
//! 
//! This crate provides comprehensive training utilities for BitNet models with
//! quantization-aware training (QAT) support, straight-through estimators, 
//! custom autograd functions, and specialized optimizers.
//!
//! ## Key Features
//! 
//! - **Straight-Through Estimator (STE)**: Multiple variants for quantization-aware training
//! - **Custom Autograd**: Integration with candle-core's autograd system
//! - **QAT Loss Functions**: Specialized loss functions for quantized training  
//! - **QAT Optimizers**: Quantization-aware Adam/AdamW optimizers
//! - **Progressive Quantization**: Layer-wise and phase-based quantization scheduling
//! - **Knowledge Distillation**: Teacher-student training for quantized models
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use bitnet_training::qat::{
//!     straight_through::{STEConfig, STEVariant, StraightThroughEstimator},
//!     loss::QATLossFactory,
//!     state_tracking::QATTrainingState,
//! };
//! use candle_core::{Tensor, Device, DType};
//!
//! let device = Device::Cpu;
//! 
//! // Configure STE for binary quantization
//! let config = STEConfig {
//!     variant: STEVariant::Standard,
//!     bits: 1,
//!     range: 1.0,
//!     ..Default::default()
//! };
//!
//! // Create Straight-Through Estimator
//! let mut ste = StraightThroughEstimator::new(config, device.clone())?;
//!
//! // Quantize tensor with straight-through gradients  
//! let input = Tensor::from_slice(&[0.8, -0.3, 0.1, -0.9], (4,), &device)?;
//! let quantized = ste.forward(&input)?;
//!
//! // Create QAT loss factory
//! let loss_factory = QATLossFactory;
//!
//! // Initialize training state
//! let training_state = QATTrainingState::new();
//! # Ok::<(), candle_core::Error>(())
//! ```

pub mod qat;