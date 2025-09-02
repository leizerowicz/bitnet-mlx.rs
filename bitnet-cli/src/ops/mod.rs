//! Production operations commands for BitNet CLI

use clap::Subcommand;
use crate::{Cli, error::CliError};

pub mod error;
pub mod validation;
pub mod profiling;
pub mod monitoring;

#[derive(Subcommand)]
pub enum OpsCommand {
    /// Validate deployment configuration and system readiness
    #[command(name = "validate")]
    Validate(validation::ValidateCommand),
    
    /// Profile system and application performance
    #[command(name = "profile")]
    Profile(profiling::ProfileCommand),
    
    /// Setup and manage monitoring integration
    #[command(name = "monitor")]
    Monitor(monitoring::MonitorCommand),
    
    /// Show operations system status
    Status,
}

impl OpsCommand {
    pub async fn execute(&self, cli: &Cli) -> Result<(), CliError> {
        match self {
            Self::Validate(cmd) => cmd.execute(cli).await,
            Self::Profile(cmd) => cmd.execute(cli).await,
            Self::Monitor(cmd) => cmd.execute(cli).await,
            Self::Status => {
                println!("BitNet Operations Status:");
                println!("  Version: 1.0.0");
                println!("  Build: {} ({})", env!("CARGO_PKG_VERSION"), "development");
                println!("  Features: validation, profiling, monitoring");
                println!("  Status: Ready for production operations");
                Ok(())
            }
        }
    }
}
