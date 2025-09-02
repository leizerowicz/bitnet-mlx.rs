//! Customer Tools Commands
//!
//! CLI command implementations for essential customer onboarding tools.
//! Implements Story 2.1 from Epic 2: Essential Customer Tools

pub mod convert;
pub mod setup;
pub mod validate;
pub mod quickstart;

pub use convert::ConvertCommand;
pub use setup::SetupCommand;
pub use validate::ValidateCommand;
pub use quickstart::QuickStartCommand;

use clap::Subcommand;

/// Customer tools command interface
#[derive(Subcommand)]
pub enum CustomerCommand {
    /// Convert models between formats (SafeTensors, ONNX, PyTorch → BitNet)
    #[command(name = "convert")]
    Convert(ConvertCommand),
    
    /// Interactive setup wizard for environment validation
    #[command(name = "setup")]
    Setup(SetupCommand),
    
    /// System health validation and performance benchmarking
    #[command(name = "validate")]
    Validate(ValidateCommand),
    
    /// Quick start automation with example models
    #[command(name = "quickstart")]
    QuickStart(QuickStartCommand),
}
