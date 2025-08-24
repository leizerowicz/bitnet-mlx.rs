//! BitNet Benchmarks CLI
//!
//! Command-line interface for running MLX vs Candle performance benchmarks.

use bitnet_benchmarks::run_cli;

fn main() -> anyhow::Result<()> {
    run_cli()
}
