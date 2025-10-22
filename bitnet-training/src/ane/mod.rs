// bitnet-training/src/ane/mod.rs
//! Apple Neural Engine Integration Module
//!
//! This module provides ANE acceleration capabilities for BitNet model training and inference.

pub mod training_integration;
pub mod ane_model;
pub mod coreml_converter;
pub mod performance_monitor;
pub mod device;
pub mod mlx_mock;

pub use training_integration::{
    ANETrainingConfig, ANETrainingStrategy, ANEConversionSettings,
    ANETrainingResult, ANEError,
};
pub use ane_model::{ANEBitNetModel, ANEBitNetAccelerator};
pub use coreml_converter::ANECoreMLConverter;
pub use performance_monitor::{ANEPerformanceMonitor, ANEPerformanceReport, ANEOperation};
pub use device::ExtendedDevice;
pub use mlx_mock::{MLXArray, MLXDevice, mlx_array_to_tensor, tensor_to_mlx_array};