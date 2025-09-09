//! BitNet CLI Application
//!
//! Command-line interface for BitNet operations including model management,
//! inference, training, benchmarking, and production operations support.

// Allow dead code for work-in-progress CLI implementations
#![allow(dead_code, unused_variables, unused_imports)]

use clap::{Parser, Subcommand};
use std::process;

mod ops;
mod error;
mod config;
mod customer_tools;
mod commands;

use ops::OpsCommand;

#[derive(Parser)]
#[command(name = "bitnet")]
#[command(version = "1.0.0")]
#[command(about = "BitNet CLI - High-performance neural network quantization tools")]
#[command(long_about = "
BitNet CLI provides comprehensive tools for working with BitNet neural networks,
including model conversion, inference, training, benchmarking, and production 
operations support for DevOps teams.

For commercial customers, this CLI enables rapid onboarding, production deployment
validation, performance optimization, and operational monitoring integration.
")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Configuration file path
    #[arg(short, long, global = true)]
    pub config: Option<std::path::PathBuf>,

    /// Output format (json, yaml, table)
    #[arg(long, global = true, default_value = "table")]
    pub output: String,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Production operations commands for DevOps teams
    #[command(subcommand)]
    Ops(OpsCommand),
    
    /// Convert models between formats (SafeTensors, ONNX, PyTorch → BitNet)
    Convert(commands::ConvertCommand),
    
    /// Interactive setup wizard for environment validation
    Setup(commands::SetupCommand),
    
    /// System health validation and performance benchmarking
    Validate(commands::ValidateCommand),
    
    /// Quick start automation with example models
    Quickstart(commands::QuickStartCommand),
    
    /// Model operations (conversion, quantization, analysis) [Coming Soon]
    #[command(subcommand)]
    Model(ModelCommand),
    
    /// Inference operations (chat, completion, batch) [Coming Soon]
    #[command(subcommand)]
    Infer(InferCommand),
    
    /// Training operations (fine-tuning, QAT) [Coming Soon]
    #[command(subcommand)]
    Train(TrainCommand),
    
    /// Benchmarking and profiling [Coming Soon]
    #[command(subcommand)]
    Benchmark(BenchmarkCommand),
}

#[derive(Subcommand)]
pub enum ModelCommand {
    /// Convert model formats [Coming Soon]
    Convert {
        /// Input model path
        #[arg(short, long)]
        input: std::path::PathBuf,
        
        /// Output model path  
        #[arg(short, long)]
        output: std::path::PathBuf,
    },
}

#[derive(Subcommand)]
pub enum InferCommand {
    /// Interactive chat mode [Coming Soon]
    Chat {
        /// Model path
        #[arg(short, long)]
        model: std::path::PathBuf,
    },
}

#[derive(Subcommand)]
pub enum TrainCommand {
    /// Start training [Coming Soon]
    Start {
        /// Model path
        #[arg(short, long)]
        model: std::path::PathBuf,
    },
}

#[derive(Subcommand)]
pub enum BenchmarkCommand {
    /// Benchmark inference performance [Coming Soon]
    Inference {
        /// Model path
        #[arg(short, long)]
        model: std::path::PathBuf,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    let log_level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .init();

    let result = match &cli.command {
        Commands::Ops(ops_cmd) => ops_cmd.execute(&cli).await,
        Commands::Convert(convert_cmd) => convert_cmd.execute().await.map_err(|e| e.into()),
        Commands::Setup(setup_cmd) => setup_cmd.execute().await.map_err(|e| e.into()),
        Commands::Validate(validate_cmd) => validate_cmd.execute().await.map_err(|e| e.into()),
        Commands::Quickstart(quickstart_cmd) => quickstart_cmd.execute().await.map_err(|e| e.into()),
        Commands::Model(_) => {
            eprintln!("Model operations are coming soon in the next release!");
            eprintln!("Use 'bitnet convert' for model format conversion.");
            Ok(())
        }
        Commands::Infer(_) => {
            eprintln!("Inference operations are coming soon in the next release!");
            eprintln!("Current focus: Essential customer tools (Story 2.1)");
            Ok(())
        }
        Commands::Train(_) => {
            eprintln!("Training operations are coming soon in the next release!");
            eprintln!("Current focus: Essential customer tools (Story 2.1)");
            Ok(())
        }
        Commands::Benchmark(_) => {
            eprintln!("Benchmarking operations are coming soon in the next release!");
            eprintln!("Use 'bitnet validate' for system performance validation.");
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        
        // Display error chain if available
        let mut current_error = &e as &dyn std::error::Error;
        while let Some(source) = current_error.source() {
            eprintln!("  Caused by: {}", source);
            current_error = source;
        }
        
        process::exit(1);
    }
}
