//! BitNet Training Library - Phase 2: Training Infrastructure
//!
//! This crate provides training infrastructure for BitNet models with
//! support for custom model training, tokenization, and dataset management.
//!
//! ## Key Features
//!
//! - **Custom BitNet Models**: Training infrastructure for 1.58-bit quantized models
//! - **Tokenization**: Simple tokenization for training data
//! - **Dataset Management**: Training dataset loading and batching
//! - **Training Loop**: Basic training infrastructure with checkpointing
//! - **Advanced Fine-tuning**: DoRA, MoLoRA, and continual learning capabilities
//! - **ANE Integration**: Apple Neural Engine acceleration for training and inference

// Allow dead code for work-in-progress training implementations
#![allow(dead_code, unused_variables, unused_imports)]

// Core modules
pub mod models;
pub mod tokenization;
pub mod dataset;
pub mod training;
pub mod qat;
pub mod export;
pub mod lora;
pub mod finetuning;

// ANE integration module
#[cfg(target_os = "macos")]
pub mod ane;

// Re-exports for convenience
pub use models::{CustomBitNetModel, CustomBitNetConfig};
pub use tokenization::{TrainingTokenizer, TokenizerConfig};
pub use dataset::{TrainingDataset, DatasetConfig, TrainingExample};
pub use training::{BitNetTrainer, TrainingConfig, TrainingStats};
pub use export::{GgufExporter, GgufMetadata, ExportStats};
pub use lora::{BitNetLoRA, BitNetQLoRA, LoRAConfig, LoRAQuantConfig, QLoRALayer};
pub use finetuning::{
    AdvancedFinetuner, AdvancedFinetuningConfig,
    BitNetDoRA, DoRAConfig,
    MixtureOfLoRAs, MoLoRAConfig, 
    ContinualLearner, ContinualLearningConfig
};

// ANE exports for macOS platforms
#[cfg(target_os = "macos")]
pub use ane::{
    ANETrainingConfig, ANETrainingStrategy, ANEConversionSettings,
    ANETrainingResult, ANEError, ANEBitNetModel, ANEBitNetAccelerator,
    ANECoreMLConverter, ANEPerformanceMonitor, ANEPerformanceReport, ANEOperation
};