//! Customer Tools Module
//!
//! Essential customer onboarding tools for BitNet-Rust CLI, enabling new customers
//! to quickly validate, setup, and deploy BitNet-Rust in under 30 minutes.
//!
//! This module implements Story 2.1 from Epic 2: Essential Customer Tools
//! 
//! ## Core Components
//! 
//! - **conversion**: Model format conversion (SafeTensors, ONNX, PyTorch → BitNet)
//! - **setup**: Interactive setup wizard with environment validation  
//! - **validation**: System health validation and performance benchmarking
//! - **quickstart**: Quick start automation with example models

pub mod conversion;
pub mod setup;
pub mod validation;
pub mod quickstart;

pub use conversion::ModelConversionEngine;
pub use setup::SetupWizard;
pub use validation::SystemValidator;
pub use quickstart::QuickStartEngine;

/// Customer tools error types
#[derive(Debug, thiserror::Error)]
pub enum CustomerToolsError {
    #[error("Model conversion failed: {0}")]
    ConversionError(String),
    
    #[error("Setup wizard failed: {0}")]
    SetupError(String),
    
    #[error("System validation failed: {0}")]
    ValidationError(String),
    
    #[error("Quick start failed: {0}")]
    QuickStartError(String),
    
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub type Result<T> = std::result::Result<T, CustomerToolsError>;

/// Customer onboarding progress tracking
#[derive(Debug, Clone)]
pub struct OnboardingProgress {
    pub current_step: String,
    pub completed_steps: Vec<String>,
    pub total_steps: usize,
    pub estimated_remaining_minutes: u32,
}

impl OnboardingProgress {
    pub fn new(total_steps: usize) -> Self {
        Self {
            current_step: "Starting".to_string(),
            completed_steps: Vec::new(),
            total_steps,
            estimated_remaining_minutes: 25, // Target <30 minutes
        }
    }
    
    pub fn progress_percentage(&self) -> f32 {
        if self.total_steps == 0 {
            return 100.0;
        }
        (self.completed_steps.len() as f32 / self.total_steps as f32) * 100.0
    }
    
    pub fn complete_step(&mut self, step: String) {
        self.completed_steps.push(self.current_step.clone());
        self.current_step = step;
        
        // Update time estimate based on progress
        let progress = self.progress_percentage() / 100.0;
        self.estimated_remaining_minutes = ((1.0 - progress) * 25.0) as u32;
    }
}
