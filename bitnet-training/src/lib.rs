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

// Allow dead code for work-in-progress training implementations
#![allow(dead_code, unused_variables, unused_imports)]

// Core modules
pub mod models;
pub mod tokenization;
pub mod dataset;
pub mod training;
pub mod qat;
pub mod export;

// Re-exports for convenience
pub use models::{CustomBitNetModel, CustomBitNetConfig};
pub use tokenization::{TrainingTokenizer, TokenizerConfig};
pub use dataset::{TrainingDataset, DatasetConfig, TrainingExample};
pub use training::{BitNetTrainer, TrainingConfig, TrainingStats};
pub use export::{GgufExporter, GgufMetadata, ExportStats};