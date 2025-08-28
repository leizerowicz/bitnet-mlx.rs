//! Mixed Precision Demo
//!
//! This example demonstrates how to use the mixed precision system in BitNet,
//! showing layer-specific precision configuration, automatic conversion,
//! and performance optimization.

use bitnet_core::{
    device::get_cpu_device,
    memory::{
        tensor::{BitNetDType, BitNetTensor},
        HybridMemoryPool,
    },
    mixed_precision::{
        conversion::{ConversionConfig, ConversionStrategy, PrecisionConverter},
        layer_precision::{LayerPrecisionManager, LayerPrecisionSpec},
        policy::{
            ConditionOperator, ConditionType, ConditionValue, PolicyAction, PolicyCondition,
            PolicyEngine, PolicyRule, PrecisionPolicy,
        },
        precision_manager::{OptimizationObjective, PrecisionManager},
        validation::PrecisionValidator,
        ComponentPrecisionConfig, ComponentType, LayerPrecisionConfig, LayerType,
        MixedPrecisionConfig, MixedPrecisionStrategy,
    },
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 BitNet Mixed Precision Demo");
    println!("===============================\n");

    // Initialize components
    let device = get_cpu_device();
    let memory_pool = HybridMemoryPool::new()?;

    // Demo 1: Basic Mixed Precision Configuration
    demo_basic_configuration()?;

    // Demo 2: Layer-Specific Precision Management
    demo_layer_precision_management(&device, &memory_pool)?;

    // Demo 3: Precision Conversion
    demo_precision_conversion(&device, &memory_pool)?;

    // Demo 4: Policy-Based Precision Selection
    demo_policy_based_precision()?;

    // Demo 5: Validation and Optimization
    demo_validation_and_optimization()?;

    println!("✅ Mixed precision demo completed successfully!");
    Ok(())
}

fn demo_basic_configuration() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Demo 1: Basic Mixed Precision Configuration");
    println!("----------------------------------------------");

    // Create different strategy configurations
    let conservative_config = MixedPrecisionConfig::conservative();
    let balanced_config = MixedPrecisionConfig::balanced();
    let aggressive_config = MixedPrecisionConfig::aggressive();

    println!("Conservative strategy:");
    println!(
        "  - Default layer precision: {:?}",
        conservative_config.default_layer_precision
    );
    println!("  - Strategy: {:?}", conservative_config.strategy);

    println!("\nBalanced strategy:");
    println!(
        "  - Default layer precision: {:?}",
        balanced_config.default_layer_precision
    );
    println!("  - Strategy: {:?}", balanced_config.strategy);

    println!("\nAggressive strategy:");
    println!(
        "  - Default layer precision: {:?}",
        aggressive_config.default_layer_precision
    );
    println!("  - Strategy: {:?}", aggressive_config.strategy);

    // Create custom configuration
    let custom_config = MixedPrecisionConfig::new(MixedPrecisionStrategy::Custom)
        .with_layer_config(
            "transformer_layer_0".to_string(),
            LayerPrecisionConfig::new(LayerType::Attention, BitNetDType::F16)
                .with_component_override(ComponentType::Weights, BitNetDType::I8)
                .with_component_override(ComponentType::Bias, BitNetDType::F16),
        )
        .with_component_config(
            ComponentType::Activations,
            ComponentPrecisionConfig::new(ComponentType::Activations, BitNetDType::I8),
        );

    println!("\nCustom configuration created with layer-specific settings");
    println!("  - Attention layer: F16 with I8 weights");
    println!("  - Global activations: I8");

    // Validate configuration
    custom_config.validate()?;
    println!("  ✅ Configuration validated successfully");

    // Estimate memory savings
    let memory_savings = custom_config.estimate_memory_savings();
    println!("  📊 Estimated memory efficiency: {memory_savings:.2}x");

    println!();
    Ok(())
}

fn demo_layer_precision_management(
    _device: &candle_core::Device,
    _memory_pool: &HybridMemoryPool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️  Demo 2: Layer-Specific Precision Management");
    println!("----------------------------------------------");

    let layer_manager = LayerPrecisionManager::new();

    // Register different layer types with specific precision requirements
    let layers = vec![
        LayerPrecisionSpec::new(
            "embedding".to_string(),
            LayerType::Embedding,
            BitNetDType::F16,
            BitNetDType::F16,
            BitNetDType::I8,
        )
        .with_component_precision(ComponentType::Weights, BitNetDType::I8),
        LayerPrecisionSpec::new(
            "attention_0".to_string(),
            LayerType::Attention,
            BitNetDType::I8,
            BitNetDType::I8,
            BitNetDType::I8,
        )
        .with_component_precision(ComponentType::AttentionScores, BitNetDType::F16),
        LayerPrecisionSpec::new(
            "linear_0".to_string(),
            LayerType::Linear,
            BitNetDType::I8,
            BitNetDType::I8,
            BitNetDType::BitNet158,
        )
        .with_dynamic_adjustment(),
        LayerPrecisionSpec::new(
            "output".to_string(),
            LayerType::Output,
            BitNetDType::I8,
            BitNetDType::F16,
            BitNetDType::F16,
        ),
    ];

    for layer_spec in layers {
        println!("Registering layer: {}", layer_spec.layer_id);
        println!("  - Type: {:?}", layer_spec.layer_type);
        println!("  - Weight precision: {:?}", layer_spec.weight_precision);
        println!(
            "  - Input/Output: {:?} -> {:?}",
            layer_spec.input_precision, layer_spec.output_precision
        );

        layer_manager.register_layer(layer_spec)?;
    }

    // Query layers by type
    let attention_layers = layer_manager.get_layers_by_type(LayerType::Attention);
    println!("\nFound {} attention layers", attention_layers.len());

    // Query layers by precision
    let bitnet_layers = layer_manager.get_layers_by_precision(BitNetDType::BitNet158);
    println!(
        "Found {} layers using BitNet 1.58 precision",
        bitnet_layers.len()
    );

    // Optimize for memory efficiency
    println!("\nOptimizing for 50% memory reduction...");
    let optimizations = layer_manager.optimize_for_memory(0.5)?;
    println!("Suggested optimizations:");
    for (layer_id, new_precision) in &optimizations {
        println!("  - {layer_id}: {new_precision:?}");
    }

    // Analyze precision impact
    let analysis = layer_manager.analyze_precision_impact();
    println!("\nPrecision Impact Analysis:");
    println!(
        "  - Average memory savings: {:.2}%",
        analysis.average_memory_savings * 100.0
    );
    println!("  - Total layers: {}", analysis.total_layers);
    println!("  - Precision distribution:");
    for (precision, count) in &analysis.precision_distribution {
        println!("    {precision:?}: {count} layers");
    }

    println!();
    Ok(())
}

fn demo_precision_conversion(
    device: &candle_core::Device,
    memory_pool: &HybridMemoryPool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Demo 3: Precision Conversion");
    println!("-------------------------------");

    // Create test tensors with different precisions
    let f32_tensor = BitNetTensor::from_data(
        vec![1.5, -2.3, 0.8, -1.1, 2.7, -0.5],
        &[2, 3],
        device,
        memory_pool,
    )?;

    println!("Original tensor (F32):");
    println!("  - Shape: {:?}", f32_tensor.shape());
    println!("  - Dtype: {:?}", f32_tensor.dtype());
    println!("  - Size: {} bytes", f32_tensor.size_bytes());

    // Create precision converter with different strategies
    let strategies = vec![
        ("Direct", ConversionStrategy::Direct),
        ("Scaled", ConversionStrategy::Scaled),
        ("Quantization-Aware", ConversionStrategy::QuantizationAware),
        ("Stochastic", ConversionStrategy::StochasticRounding),
    ];

    for (name, strategy) in strategies {
        println!("\n{name} conversion strategy:");

        let config = ConversionConfig {
            strategy,
            preserve_metadata: true,
            validate_results: true,
            validation_tolerance: 1e-3,
            use_simd: true,
            custom_params: HashMap::new(),
        };

        let mut converter = PrecisionConverter::new(config)?;

        // Convert to different precisions
        let target_precisions = vec![BitNetDType::I8, BitNetDType::I4, BitNetDType::BitNet158];

        for target_precision in target_precisions {
            match converter.convert_tensor(&f32_tensor, target_precision) {
                Ok(converted_tensor) => {
                    let compression_ratio =
                        f32_tensor.size_bytes() as f32 / converted_tensor.size_bytes() as f32;
                    println!(
                        "  - {} -> {:?}: {:.2}x compression",
                        f32_tensor.dtype(),
                        target_precision,
                        compression_ratio
                    );
                }
                Err(e) => {
                    println!(
                        "  - {} -> {:?}: Failed ({})",
                        f32_tensor.dtype(),
                        target_precision,
                        e
                    );
                }
            }
        }

        // Show conversion statistics
        let stats = converter.get_stats();
        println!("  - Total conversions: {}", stats.total_conversions);
        println!(
            "  - Average time: {:.2}ms",
            stats.average_conversion_time_ms()
        );
        println!(
            "  - Memory efficiency: {:.2}MB saved",
            stats.memory_efficiency()
        );
    }

    println!();
    Ok(())
}

fn demo_policy_based_precision() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Demo 4: Policy-Based Precision Selection");
    println!("-------------------------------------------");

    let mut policy_engine = PolicyEngine::new();

    // Create a custom policy for memory-critical scenarios
    let memory_critical_policy = PrecisionPolicy::new(
        "memory_critical".to_string(),
        "Memory Critical Policy".to_string(),
        "Aggressive quantization when memory usage is high".to_string(),
    )
    .with_priority(100)
    .add_rule(
        PolicyRule::new(
            "high_memory_usage".to_string(),
            PolicyAction::SetPrecision(BitNetDType::I4),
        )
        .add_condition(PolicyCondition::new(
            ConditionType::MemoryUsage,
            ConditionOperator::GreaterThan,
            ConditionValue::Float(80.0), // 80MB threshold
        ))
        .with_weight(1.0),
    );

    // Create a policy for accuracy-critical layers
    let accuracy_critical_policy = PrecisionPolicy::new(
        "accuracy_critical".to_string(),
        "Accuracy Critical Policy".to_string(),
        "High precision for attention and output layers".to_string(),
    )
    .with_priority(90)
    .add_rule(
        PolicyRule::new(
            "attention_high_precision".to_string(),
            PolicyAction::SetPrecision(BitNetDType::F16),
        )
        .add_condition(PolicyCondition::new(
            ConditionType::LayerType,
            ConditionOperator::Equal,
            ConditionValue::LayerType(LayerType::Attention),
        )),
    )
    .add_rule(
        PolicyRule::new(
            "output_high_precision".to_string(),
            PolicyAction::SetPrecision(BitNetDType::F16),
        )
        .add_condition(PolicyCondition::new(
            ConditionType::LayerType,
            ConditionOperator::Equal,
            ConditionValue::LayerType(LayerType::Output),
        )),
    );

    policy_engine.add_policy(memory_critical_policy);
    policy_engine.add_policy(accuracy_critical_policy);

    println!("Created custom policies:");
    for policy in policy_engine.list_policies() {
        println!(
            "  - {}: {} (priority: {})",
            policy.id, policy.name, policy.priority
        );
        println!("    Rules: {}", policy.rules.len());
    }

    // Simulate policy application
    println!("\nSimulating policy applications:");

    // Test scenarios
    let scenarios = vec![
        ("Low memory, linear layer", LayerType::Linear, 30.0),
        ("High memory, linear layer", LayerType::Linear, 90.0),
        ("Low memory, attention layer", LayerType::Attention, 30.0),
        ("High memory, attention layer", LayerType::Attention, 90.0),
        ("Low memory, output layer", LayerType::Output, 30.0),
    ];

    for (scenario_name, layer_type, memory_usage) in scenarios {
        println!("\nScenario: {scenario_name}");
        println!("  - Layer type: {layer_type:?}");
        println!("  - Memory usage: {memory_usage:.1}MB");

        // This would normally be called by the precision manager
        // For demo purposes, we'll show what policies would apply
        println!("  - Applicable policies: [simulated]");
    }

    // Show policy statistics
    let stats = policy_engine.get_stats();
    println!("\nPolicy Statistics:");
    println!("  - Total applications: {}", stats.total_applications);
    if let Some(most_used) = stats.most_used_policy() {
        println!("  - Most used policy: {most_used}");
    }

    println!();
    Ok(())
}

fn demo_validation_and_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("✅ Demo 5: Validation and Optimization");
    println!("--------------------------------------");

    let validator = PrecisionValidator::new();

    // Create test layer specifications
    let test_specs = vec![
        // Valid specification
        LayerPrecisionSpec::new(
            "valid_layer".to_string(),
            LayerType::Linear,
            BitNetDType::I8,
            BitNetDType::I8,
            BitNetDType::BitNet158,
        ),
        // Potentially problematic specification
        LayerPrecisionSpec::new(
            "aggressive_layer".to_string(),
            LayerType::Attention,
            BitNetDType::I4,
            BitNetDType::I4,
            BitNetDType::I1,
        ),
        // Invalid specification (normalization with integer precision)
        LayerPrecisionSpec::new(
            "invalid_layer".to_string(),
            LayerType::Normalization,
            BitNetDType::I8,
            BitNetDType::I8,
            BitNetDType::I4,
        ),
    ];

    println!("Validating layer specifications:");

    for spec in &test_specs {
        println!("\nLayer: {}", spec.layer_id);
        println!("  - Type: {:?}", spec.layer_type);
        println!("  - Precision: {:?}", spec.weight_precision);

        match validator.validate_layer_spec(spec) {
            Ok(result) => {
                if result.passed {
                    println!("  ✅ Validation passed");
                } else {
                    println!("  ❌ Validation failed");
                    for issue in &result.issues {
                        println!("    - {}: {}", issue.severity, issue.description);
                        if let Some(fix) = &issue.suggested_fix {
                            println!("      Suggested fix: {fix}");
                        }
                    }
                }

                if !result.warnings.is_empty() {
                    println!("  ⚠️  Warnings:");
                    for warning in &result.warnings {
                        println!("    - {}: {}", warning.component, warning.message);
                    }
                }

                if !result.suggestions.is_empty() {
                    println!("  💡 Suggestions:");
                    for suggestion in &result.suggestions {
                        println!(
                            "    - {} ({})",
                            suggestion.description, suggestion.expected_benefit
                        );
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Validation error: {e}");
            }
        }
    }

    // Validate multiple layers for cross-layer compatibility
    println!("\nCross-layer validation:");
    match validator.validate_multiple_layers(&test_specs) {
        Ok(result) => {
            println!(
                "  - Overall validation: {}",
                if result.passed {
                    "✅ Passed"
                } else {
                    "❌ Failed"
                }
            );
            println!("  - Total issues: {}", result.issues.len());
            println!("  - Total warnings: {}", result.warnings.len());
            println!("  - Total suggestions: {}", result.suggestions.len());
        }
        Err(e) => {
            println!("  ❌ Cross-layer validation error: {e}");
        }
    }

    // Demonstrate precision manager optimization
    println!("\nPrecision Manager Optimization:");
    let config = MixedPrecisionConfig::balanced();
    let precision_manager = PrecisionManager::new(config)?;

    // Register test layers
    for spec in test_specs {
        if let Err(e) = precision_manager.register_layer(spec) {
            println!("  - Failed to register layer: {e}");
        }
    }

    // Demonstrate different optimization objectives
    let objectives = vec![
        (
            "Memory efficiency",
            OptimizationObjective::Memory {
                target_reduction: 0.5,
            },
        ),
        (
            "Speed optimization",
            OptimizationObjective::Speed {
                target_speedup: 2.0,
            },
        ),
        (
            "Accuracy preservation",
            OptimizationObjective::Accuracy { min_accuracy: 0.95 },
        ),
        (
            "Balanced optimization",
            OptimizationObjective::Balanced {
                memory_weight: 0.4,
                speed_weight: 0.3,
                accuracy_weight: 0.3,
            },
        ),
    ];

    for (name, objective) in objectives {
        println!("  - {name}: ");
        match precision_manager.optimize_precision(objective) {
            Ok(()) => println!("✅ Optimization completed"),
            Err(e) => println!("❌ Optimization failed: {e}"),
        }
    }

    // Show final analysis
    match precision_manager.analyze_configuration() {
        Ok(analysis) => {
            println!("\nFinal Configuration Analysis:");
            println!(
                "  - Memory savings: {:.1}%",
                analysis.memory_savings * 100.0
            );
            println!(
                "  - Accuracy impact: {:.1}%",
                analysis.accuracy_impact * 100.0
            );
            println!(
                "  - Conversion overhead: {:.2}ms",
                analysis.conversion_overhead
            );
            println!("  - Total layers: {}", analysis.total_layers);
            println!("  - Recommendations: {}", analysis.recommendations.len());

            for rec in &analysis.recommendations {
                println!("    - {}: {}", rec.recommendation_type, rec.description);
            }
        }
        Err(e) => {
            println!("❌ Analysis failed: {e}");
        }
    }

    println!();
    Ok(())
}
