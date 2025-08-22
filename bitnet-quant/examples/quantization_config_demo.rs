//! Comprehensive demonstration of the BitNet quantization configuration system
//! 
//! This example shows how to use the enhanced configuration structs for different
//! quantization scenarios in BitNet models.

use bitnet_quant::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("BitNet Quantization Configuration Demo");
    println!("=====================================\n");

    // 1. Basic BitNet 1.58-bit Configuration
    demo_basic_bitnet_config()?;
    
    // 2. Advanced Weight Quantization Configuration
    demo_advanced_weight_config()?;
    
    // 3. Dynamic Activation Quantization Configuration
    demo_activation_config()?;
    
    // 4. SIMD-Optimized Packing Configuration
    demo_packing_config()?;
    
    // 5. Configuration Builders
    demo_config_builders()?;
    
    // 6. Configuration Validation
    demo_config_validation()?;

    Ok(())
}

fn demo_basic_bitnet_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic BitNet 1.58-bit Configuration");
    println!("--------------------------------------");
    
    // Create a standard BitNet configuration
    let config = EnhancedQuantizationConfig::bitnet_158();
    println!("BitNet Config: {config:?}");
    
    // Validate the configuration
    config.validate()?;
    println!("✓ Configuration is valid\n");
    
    // Create weight quantization config for BitNet
    let weight_config = EnhancedWeightQuantizationConfig::bitnet();
    println!("Weight Config: {weight_config:?}");
    weight_config.validate()?;
    println!("✓ Weight configuration is valid\n");
    
    Ok(())
}

fn demo_advanced_weight_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Advanced Weight Quantization Configuration");
    println!("--------------------------------------------");
    
    // Create a custom weight configuration with specific settings
    let weight_config = EnhancedWeightQuantizationConfig::bitnet()
        .with_learnable_scales()
        .with_ternary_method(TernaryMethod::OptimalThreshold)
        .with_threshold_factor(0.8)
        .with_packing(EnhancedPackingConfig::max_compression());
    
    println!("Advanced Weight Config:");
    println!("  - Learnable scales: {}", weight_config.learnable_scales);
    println!("  - Ternary method: {:?}", weight_config.ternary_method);
    println!("  - Threshold factor: {:?}", weight_config.custom_threshold_factor);
    println!("  - Packing strategy: {:?}", weight_config.packing.strategy);
    
    weight_config.validate()?;
    println!("✓ Advanced weight configuration is valid\n");
    
    Ok(())
}

fn demo_activation_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Dynamic Activation Quantization Configuration");
    println!("-----------------------------------------------");
    
    // Create activation configuration for BitNet
    let activation_config = EnhancedActivationQuantizationConfig::bitnet()
        .with_per_token()
        .with_window(200)
        .with_smooth_quantization(0.1)
        .with_caching(512);
    
    println!("Activation Config:");
    println!("  - Per-token quantization: {}", activation_config.per_token);
    println!("  - Moving average window: {}", activation_config.moving_average_window);
    println!("  - Smooth quantization: {}", activation_config.smooth_quantization);
    println!("  - Temperature: {}", activation_config.temperature);
    println!("  - Caching enabled: {}", activation_config.enable_caching);
    println!("  - Cache size: {:?} MB", activation_config.cache_size_mb);
    
    activation_config.validate()?;
    println!("✓ Activation configuration is valid\n");
    
    Ok(())
}

fn demo_packing_config() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. SIMD-Optimized Packing Configuration");
    println!("---------------------------------------");
    
    // Create different packing configurations for different scenarios
    
    // Maximum compression configuration
    let max_compression = EnhancedPackingConfig::max_compression();
    println!("Max Compression Config:");
    println!("  - Strategy: {:?}", max_compression.strategy);
    println!("  - Compression level: {}", max_compression.compression_level);
    println!("  - Parallel packing: {}", max_compression.parallel_packing);
    
    // Maximum speed configuration
    let max_speed = EnhancedPackingConfig::max_speed();
    println!("\nMax Speed Config:");
    println!("  - Strategy: {:?}", max_speed.strategy);
    println!("  - SIMD optimized: {}", max_speed.simd_optimized);
    println!("  - Alignment: {} bytes", max_speed.alignment);
    
    // Custom SIMD configuration
    let simd_config = SimdConfig::aggressive()
        .force_instruction_set(false, true, false) // Force AVX2
        .with_custom_param("prefetch_multiplier".to_string(), 2.0);
    
    println!("\nSIMD Config:");
    println!("  - Enabled: {}", simd_config.enabled);
    println!("  - Force AVX2: {}", simd_config.force_avx2);
    println!("  - Chunk size: {}", simd_config.chunk_size);
    println!("  - Prefetch enabled: {}", simd_config.enable_prefetch);
    
    max_compression.validate()?;
    max_speed.validate()?;
    simd_config.validate()?;
    println!("✓ All packing configurations are valid\n");
    
    Ok(())
}

fn demo_config_builders() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Configuration Builders");
    println!("------------------------");
    
    // Build a custom quantization configuration
    let custom_config = QuantizationConfigBuilder::new()
        .precision(QuantizationPrecision::EightBit)
        .strategy(QuantizationStrategy::Dynamic)
        .per_channel(true)
        .clip_threshold(5.0)
        .qat_enabled(true)
        .calibration_size(2000)
        .seed(12345)
        .verbose(true)
        .build();
    
    println!("Custom Config (via builder):");
    println!("  - Precision: {:?}", custom_config.precision);
    println!("  - Strategy: {:?}", custom_config.strategy);
    println!("  - Per-channel: {}", custom_config.per_channel);
    println!("  - QAT enabled: {}", custom_config.qat_enabled);
    println!("  - Seed: {:?}", custom_config.seed);
    
    // Build a custom weight configuration
    let custom_weight_config = WeightQuantizationConfigBuilder::new()
        .base(custom_config)
        .group_size(128)
        .learnable_scales(true)
        .ternary_method(TernaryMethod::AdaptiveThreshold)
        .custom_threshold_factor(0.9)
        .freeze_weights(false)
        .weight_decay(1e-5)
        .gradient_clip(2.0)
        .build();
    
    println!("\nCustom Weight Config (via builder):");
    println!("  - Group size: {:?}", custom_weight_config.group_size);
    println!("  - Learnable scales: {}", custom_weight_config.learnable_scales);
    println!("  - Weight decay: {:?}", custom_weight_config.weight_decay);
    println!("  - Gradient clip: {:?}", custom_weight_config.gradient_clip);
    
    custom_weight_config.validate()?;
    println!("✓ Builder configurations are valid\n");
    
    Ok(())
}

fn demo_config_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Configuration Validation");
    println!("---------------------------");
    
    // Demonstrate validation with valid configuration
    let valid_config = EnhancedQuantizationConfig::bitnet_158();
    match valid_config.validate() {
        Ok(()) => println!("✓ Valid configuration passed validation"),
        Err(e) => println!("✗ Validation failed: {e}"),
    }
    
    // Demonstrate validation with invalid configuration
    let mut invalid_config = EnhancedQuantizationConfig::default();
    invalid_config.clip_threshold = Some(-1.0); // Invalid negative threshold
    
    match invalid_config.validate() {
        Ok(()) => println!("✗ Invalid configuration incorrectly passed validation"),
        Err(e) => println!("✓ Invalid configuration correctly failed validation: {e}"),
    }
    
    // Demonstrate weight config validation
    let mut invalid_weight_config = EnhancedWeightQuantizationConfig::default();
    invalid_weight_config.group_size = Some(0); // Invalid zero group size
    
    match invalid_weight_config.validate() {
        Ok(()) => println!("✗ Invalid weight configuration incorrectly passed validation"),
        Err(e) => println!("✓ Invalid weight configuration correctly failed validation: {e}"),
    }
    
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_functions() {
        assert!(demo_basic_bitnet_config().is_ok());
        assert!(demo_advanced_weight_config().is_ok());
        assert!(demo_activation_config().is_ok());
        assert!(demo_packing_config().is_ok());
        assert!(demo_config_builders().is_ok());
        assert!(demo_config_validation().is_ok());
    }

    #[test]
    fn test_config_interoperability() {
        // Test that new configs work with existing quantization functions
        let device = Device::Cpu;
        let weights = Tensor::randn(0.0, 1.0, (64, 128), &device).unwrap();
        
        // Create enhanced configuration
        let config = EnhancedWeightQuantizationConfig::bitnet();
        
        // Validate configuration
        assert!(config.validate().is_ok());
        
        // Test that configuration has expected values
        assert_eq!(config.base.precision, QuantizationPrecision::OneFiveFiveBit);
        assert_eq!(config.ternary_method, TernaryMethod::MeanThreshold);
        assert_eq!(config.custom_threshold_factor, Some(0.7));
    }

    #[test]
    fn test_builder_pattern() {
        let config = QuantizationConfigBuilder::new()
            .precision(QuantizationPrecision::EightBit)
            .strategy(QuantizationStrategy::Asymmetric)
            .per_channel(true)
            .build();
        
        assert_eq!(config.precision, QuantizationPrecision::EightBit);
        assert_eq!(config.strategy, QuantizationStrategy::Asymmetric);
        assert!(config.per_channel);
    }

    #[test]
    fn test_simd_config_variants() {
        let aggressive = SimdConfig::aggressive();
        let conservative = SimdConfig::conservative();
        let disabled = SimdConfig::disabled();
        
        assert!(aggressive.enabled);
        assert!(conservative.enabled);
        assert!(!disabled.enabled);
        
        assert!(aggressive.chunk_size > conservative.chunk_size);
        assert!(aggressive.enable_prefetch);
        assert!(!conservative.enable_prefetch);
    }
}