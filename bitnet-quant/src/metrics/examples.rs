// bitnet-quant/src/metrics/examples.rs
//! Example implementations and usage demonstrations for quantization metrics
//! 
//! This module provides comprehensive examples showing how to use the BitNet
//! quantization metrics system in various real-world scenarios.

use candle_core::{Tensor, Device, DType, Result};
use std::collections::HashMap;

use crate::metrics::{
    QuantizationMetrics, MetricsCalculator, ErrorThresholds,
    error_analysis::{ErrorAnalyzer, ErrorAnalysisConfig},
    mse::MSECalculator,
    sqnr::SQNRCalculator, 
    cosine_similarity::CosineSimilarityCalculator,
    layer_wise::{LayerWiseAnalyzer, LayerWiseAnalysisConfig},
    visualization::VisualizationEngine,
    mitigation::{ErrorMitigationEngine, MitigationConfig},
    reporting::{ReportingEngine, ReportFormat},
};

/// Comprehensive example demonstrating the full quantization metrics workflow
pub struct MetricsWorkflowDemo {
    device: Device,
    error_analyzer: ErrorAnalyzer,
    layer_wise_analyzer: LayerWiseAnalyzer,
    visualization_engine: VisualizationEngine,
    mitigation_engine: ErrorMitigationEngine,
    reporting_engine: ReportingEngine,
}

impl MetricsWorkflowDemo {
    pub fn new(device: Device) -> Result<Self> {
        let error_config = ErrorAnalysisConfig {
            enable_streaming: true,
            histogram_bins: 50,
            pattern_analysis_enabled: true,
            statistical_analysis_enabled: true,
        };

        let layer_config = LayerWiseAnalysisConfig {
            correlation_threshold: 0.7,
            sensitivity_analysis_enabled: true,
            optimization_planning_enabled: true,
            temporal_analysis_enabled: true,
        };

        let mitigation_config = MitigationConfig {
            enable_adaptive_mitigation: true,
            quality_improvement_threshold: 0.1,
            implementation_complexity_preference: 
                crate::metrics::mitigation::ComplexityPreference::Balanced,
            risk_tolerance: crate::metrics::mitigation::RiskTolerance::Medium,
        };

        Ok(Self {
            device: device.clone(),
            error_analyzer: ErrorAnalyzer::new(error_config),
            layer_wise_analyzer: LayerWiseAnalyzer::new(layer_config),
            visualization_engine: VisualizationEngine::new("./visualizations".to_string()),
            mitigation_engine: ErrorMitigationEngine::new(mitigation_config),
            reporting_engine: ReportingEngine::new("./reports".to_string()),
        })
    }

    /// Complete workflow example: from tensors to final report
    pub fn run_complete_analysis_workflow(&mut self, 
        layer_data: &[(String, Tensor, Tensor)]) -> Result<()> {
        
        println!("🔍 Starting comprehensive quantization analysis workflow...\n");

        // Step 1: Calculate basic metrics for each layer
        println!("📊 Step 1: Calculating basic quantization metrics...");
        let mut layer_metrics = HashMap::new();
        
        for (layer_name, original, quantized) in layer_data {
            let metrics = self.calculate_layer_metrics(original, quantized, layer_name)?;
            layer_metrics.insert(layer_name.clone(), metrics);
            println!("  ✓ Analyzed layer: {} (MSE: {:.6}, SQNR: {:.2} dB)", 
                layer_name, metrics.mse, metrics.sqnr);
        }

        // Step 2: Perform comprehensive error analysis
        println!("\n🔬 Step 2: Performing error analysis...");
        let mut error_results = HashMap::new();
        
        for (layer_name, original, quantized) in layer_data {
            let result = self.error_analyzer.analyze_comprehensive_error(
                original, quantized, layer_name
            )?;
            error_results.insert(layer_name.clone(), result);
            println!("  ✓ Error analysis complete for: {}", layer_name);
        }

        // Step 3: Layer-wise sensitivity analysis
        println!("\n🎯 Step 3: Conducting layer-wise analysis...");
        let layer_analysis = self.layer_wise_analyzer.analyze_layers(&layer_metrics)?;
        
        println!("  📈 Global statistics:");
        println!("    Mean MSE: {:.6}", layer_analysis.global_statistics.mean_mse);
        println!("    Mean SQNR: {:.2} dB", layer_analysis.global_statistics.mean_sqnr);
        println!("    Mean Cosine Similarity: {:.4}", layer_analysis.global_statistics.mean_cosine_similarity);
        
        println!("  🚨 Problematic layers found: {}", layer_analysis.problematic_layers.len());
        println!("  ⭐ High priority layers: {}", layer_analysis.optimization_plan.high_priority_layers.len());

        // Step 4: Generate visualizations
        println!("\n📈 Step 4: Creating visualizations...");
        let dashboard = self.visualization_engine.create_error_dashboard(&layer_analysis)?;
        println!("  ✓ Dashboard created with {} visualizations", dashboard.visualizations.len());

        // Step 5: Execute error mitigation
        println!("\n🛠️ Step 5: Planning error mitigation...");
        let mitigation_result = self.mitigation_engine.execute_mitigation_plan(&layer_analysis)?;
        
        println!("  📋 Mitigation actions: {}", mitigation_result.mitigation_actions.len());
        println!("  💡 Expected improvement: {:.1}%", mitigation_result.overall_improvement * 100.0);
        println!("  ⏱️ Estimated implementation: {} hours", 
            mitigation_result.implementation_plan.total_estimated_duration_hours);

        // Step 6: Generate comprehensive report
        println!("\n📄 Step 6: Generating comprehensive report...");
        let report = self.reporting_engine.generate_comprehensive_report(&layer_analysis)?;
        let exported_files = self.reporting_engine.export_report(&report)?;
        
        println!("  ✓ Report generated successfully");
        println!("  📁 Exported files: {}", exported_files.len());
        for file in &exported_files {
            println!("    - {}", file);
        }

        // Step 7: Summary and recommendations
        println!("\n📋 Analysis Summary:");
        println!("==================");
        println!("Overall Quality: {:?}", report.executive_summary.overall_quality_grade);
        println!("Key Findings: {}", report.executive_summary.key_findings.len());
        println!("High Priority Actions: {}", report.executive_summary.high_priority_actions_count);
        
        println!("\n🎯 Next Steps:");
        for (i, step) in report.executive_summary.next_steps.iter().enumerate() {
            println!("  {}. {}", i + 1, step);
        }

        println!("\n✅ Workflow completed successfully!");

        Ok(())
    }

    fn calculate_layer_metrics(&self, original: &Tensor, quantized: &Tensor, 
        layer_name: &str) -> Result<QuantizationMetrics> {
        
        let mse_calc = MSECalculator::new();
        let sqnr_calc = SQNRCalculator::new();
        let cosine_calc = CosineSimilarityCalculator::new();

        let mse = mse_calc.calculate_mse(original, quantized)?;
        let sqnr = sqnr_calc.calculate_sqnr(original, quantized)?;
        let cosine_sim = cosine_calc.calculate_cosine_similarity(original, quantized)?;

        // Calculate additional metrics
        let error_tensor = (original - quantized)?;
        let abs_error = error_tensor.abs()?;
        let mean_abs_error = abs_error.mean_all()?.to_scalar::<f32>()?;
        let max_error = abs_error.max_keepdim(0)?
            .max_keepdim(1)?
            .to_scalar::<f32>()?;
        
        let original_mean = original.abs()?.mean_all()?.to_scalar::<f32>()?;
        let relative_error = if original_mean > 1e-8 {
            mean_abs_error / original_mean
        } else {
            mean_abs_error
        };

        Ok(QuantizationMetrics {
            layer_name: layer_name.to_string(),
            mse,
            sqnr,
            cosine_similarity: cosine_sim,
            mean_absolute_error: mean_abs_error,
            max_error,
            relative_error,
            ..Default::default()
        })
    }
}

/// Simple example for basic metrics calculation
pub struct BasicMetricsExample;

impl BasicMetricsExample {
    /// Demonstrate basic MSE calculation
    pub fn calculate_mse_example() -> Result<()> {
        println!("🔢 Basic MSE Calculation Example");
        println!("================================\n");

        let device = Device::Cpu;
        
        // Create sample tensors
        let original = Tensor::randn(0f32, 1f32, (100, 50), &device)?;
        let noise = Tensor::randn(0f32, 0.1f32, (100, 50), &device)?;
        let quantized = (&original + &noise)?;
        
        let mse_calc = MSECalculator::new();
        let mse = mse_calc.calculate_mse(&original, &quantized)?;
        
        println!("Original tensor shape: {:?}", original.shape());
        println!("Quantized tensor shape: {:?}", quantized.shape());
        println!("Mean Squared Error: {:.6}", mse);
        
        // Expected MSE should be close to noise variance (0.01)
        println!("Expected MSE (noise variance): 0.01");
        println!("Actual vs Expected ratio: {:.2}", mse / 0.01);
        
        if mse > 0.005 && mse < 0.015 {
            println!("✅ MSE calculation appears correct!");
        } else {
            println!("⚠️ MSE calculation may need review");
        }
        
        Ok(())
    }

    /// Demonstrate SQNR calculation with different noise levels
    pub fn calculate_sqnr_example() -> Result<()> {
        println!("\n📡 SQNR Calculation Example");
        println!("===========================\n");

        let device = Device::Cpu;
        let sqnr_calc = SQNRCalculator::new();
        
        // Test with different noise levels
        let noise_levels = vec![0.01, 0.05, 0.1, 0.2];
        let original = Tensor::randn(0f32, 1f32, (200, 100), &device)?;
        
        println!("Signal power (estimated): 1.0");
        println!("Testing different noise levels:\n");
        
        for noise_level in noise_levels {
            let noise = Tensor::randn(0f32, noise_level, (200, 100), &device)?;
            let quantized = (&original + &noise)?;
            
            let sqnr = sqnr_calc.calculate_sqnr(&original, &quantized)?;
            let theoretical_sqnr = -10.0 * (noise_level * noise_level).log10();
            
            println!("Noise Level: {:.3}", noise_level);
            println!("  Calculated SQNR: {:.2} dB", sqnr);
            println!("  Theoretical SQNR: {:.2} dB", theoretical_sqnr);
            println!("  Difference: {:.2} dB", (sqnr - theoretical_sqnr).abs());
            println!();
        }
        
        Ok(())
    }

    /// Demonstrate cosine similarity for different alignment levels
    pub fn calculate_cosine_similarity_example() -> Result<()> {
        println!("📐 Cosine Similarity Example");
        println!("============================\n");

        let device = Device::Cpu;
        let cosine_calc = CosineSimilarityCalculator::new();
        
        // Create base vector
        let original = Tensor::randn(0f32, 1f32, (1000,), &device)?;
        
        // Test different similarity levels
        let test_cases = vec![
            ("Identical", 1.0, 0.0),
            ("Very Similar", 0.95, 0.05),
            ("Moderately Similar", 0.8, 0.2),
            ("Somewhat Similar", 0.5, 0.5),
            ("Different", 0.0, 1.0),
        ];
        
        println!("Testing different vector alignments:\n");
        
        for (description, scale_factor, noise_factor) in test_cases {
            let scaled = (&original * scale_factor)?;
            let noise = Tensor::randn(0f32, noise_factor, (1000,), &device)?;
            let test_vector = (&scaled + &noise)?;
            
            let similarity = cosine_calc.calculate_cosine_similarity(&original, &test_vector)?;
            
            println!("{}: {:.4}", description, similarity);
        }
        
        Ok(())
    }
}

/// Advanced example showing streaming processing for large models
pub struct StreamingProcessingExample {
    mse_calc: MSECalculator,
    sqnr_calc: SQNRCalculator,
    cosine_calc: CosineSimilarityCalculator,
}

impl StreamingProcessingExample {
    pub fn new() -> Self {
        Self {
            mse_calc: MSECalculator::new(),
            sqnr_calc: SQNRCalculator::new(),
            cosine_calc: CosineSimilarityCalculator::new(),
        }
    }

    /// Demonstrate streaming processing for large tensor analysis
    pub fn streaming_analysis_example(&mut self) -> Result<()> {
        println!("🌊 Streaming Processing Example");
        println!("===============================\n");

        let device = Device::Cpu;
        let chunk_size = 1000;
        let total_elements = 10000;
        let chunks = total_elements / chunk_size;
        
        println!("Processing {} elements in {} chunks of {}", 
            total_elements, chunks, chunk_size);
        println!("Simulating large model layer analysis...\n");

        // Initialize streaming calculators
        let mut streaming_mse = 0.0;
        let mut streaming_sqnr_components = (0.0, 0.0); // (signal_power, noise_power)
        let mut streaming_cosine_components = (0.0, 0.0, 0.0); // (dot_product, norm1, norm2)
        let mut total_processed = 0;

        for chunk_idx in 0..chunks {
            // Simulate loading chunk data
            let original_chunk = Tensor::randn(0f32, 1f32, (chunk_size,), &device)?;
            let noise = Tensor::randn(0f32, 0.1f32, (chunk_size,), &device)?;
            let quantized_chunk = (&original_chunk + &noise)?;

            // Update streaming MSE
            let chunk_mse = self.mse_calc.calculate_mse(&original_chunk, &quantized_chunk)?;
            streaming_mse = (streaming_mse * total_processed as f32 + 
                chunk_mse * chunk_size as f32) / (total_processed + chunk_size) as f32;

            // Update streaming SQNR components
            let signal_power = original_chunk.sqr()?.sum_all()?.to_scalar::<f32>()?;
            let error = (&original_chunk - &quantized_chunk)?;
            let noise_power = error.sqr()?.sum_all()?.to_scalar::<f32>()?;
            
            streaming_sqnr_components.0 += signal_power;
            streaming_sqnr_components.1 += noise_power;

            // Update streaming cosine similarity components
            let dot_product = (&original_chunk * &quantized_chunk)?.sum_all()?.to_scalar::<f32>()?;
            let norm1_sq = original_chunk.sqr()?.sum_all()?.to_scalar::<f32>()?;
            let norm2_sq = quantized_chunk.sqr()?.sum_all()?.to_scalar::<f32>()?;
            
            streaming_cosine_components.0 += dot_product;
            streaming_cosine_components.1 += norm1_sq;
            streaming_cosine_components.2 += norm2_sq;

            total_processed += chunk_size;
            
            println!("Chunk {}/{} processed - Running MSE: {:.6}", 
                chunk_idx + 1, chunks, streaming_mse);
        }

        // Calculate final streaming results
        let final_sqnr = 10.0 * (streaming_sqnr_components.0 / streaming_sqnr_components.1).log10();
        let final_cosine = streaming_cosine_components.0 / 
            (streaming_cosine_components.1.sqrt() * streaming_cosine_components.2.sqrt());

        println!("\n📊 Streaming Processing Results:");
        println!("================================");
        println!("Final MSE: {:.6}", streaming_mse);
        println!("Final SQNR: {:.2} dB", final_sqnr);
        println!("Final Cosine Similarity: {:.4}", final_cosine);
        println!("Total elements processed: {}", total_processed);

        // Compare with batch processing
        println!("\n🔄 Verification with batch processing...");
        let full_original = Tensor::randn(0f32, 1f32, (total_elements,), &device)?;
        let full_noise = Tensor::randn(0f32, 0.1f32, (total_elements,), &device)?;
        let full_quantized = (&full_original + &full_noise)?;

        let batch_mse = self.mse_calc.calculate_mse(&full_original, &full_quantized)?;
        let batch_sqnr = self.sqnr_calc.calculate_sqnr(&full_original, &full_quantized)?;
        let batch_cosine = self.cosine_calc.calculate_cosine_similarity(&full_original, &full_quantized)?;

        println!("Batch MSE: {:.6} (diff: {:.6})", batch_mse, (streaming_mse - batch_mse).abs());
        println!("Batch SQNR: {:.2} dB (diff: {:.2} dB)", batch_sqnr, (final_sqnr - batch_sqnr).abs());
        println!("Batch Cosine: {:.4} (diff: {:.4})", batch_cosine, (final_cosine - batch_cosine).abs());

        Ok(())
    }
}

/// Real-world scenario examples
pub struct RealWorldScenarios;

impl RealWorldScenarios {
    /// Example: Model comparison scenario
    pub fn model_comparison_example() -> Result<()> {
        println!("🔍 Model Comparison Scenario");
        println!("============================\n");

        let device = Device::Cpu;
        
        // Simulate different quantization schemes
        let scenarios = vec![
            ("8-bit Uniform", 0.01),
            ("8-bit Per-Channel", 0.008),
            ("Mixed Precision", 0.005),
            ("16-bit Uniform", 0.001),
        ];

        let original_activations = Tensor::randn(0f32, 1f32, (1000, 512), &device)?;
        
        println!("Comparing quantization schemes on sample activations:");
        println!("Original shape: {:?}\n", original_activations.shape());

        let mse_calc = MSECalculator::new();
        let sqnr_calc = SQNRCalculator::new();
        let cosine_calc = CosineSimilarityCalculator::new();

        for (scheme_name, noise_level) in scenarios {
            let noise = Tensor::randn(0f32, noise_level, (1000, 512), &device)?;
            let quantized = (&original_activations + &noise)?;

            let mse = mse_calc.calculate_mse(&original_activations, &quantized)?;
            let sqnr = sqnr_calc.calculate_sqnr(&original_activations, &quantized)?;
            let cosine_sim = cosine_calc.calculate_cosine_similarity(&original_activations, &quantized)?;

            println!("{}: ", scheme_name);
            println!("  MSE: {:.6}", mse);
            println!("  SQNR: {:.2} dB", sqnr);
            println!("  Cosine Similarity: {:.4}", cosine_sim);
            
            // Quality assessment
            let quality = if mse < 0.001 && sqnr > 30.0 && cosine_sim > 0.99 {
                "Excellent"
            } else if mse < 0.01 && sqnr > 20.0 && cosine_sim > 0.95 {
                "Good"
            } else if mse < 0.1 && sqnr > 10.0 && cosine_sim > 0.9 {
                "Fair"
            } else {
                "Poor"
            };
            
            println!("  Quality Assessment: {}\n", quality);
        }

        Ok(())
    }

    /// Example: Layer sensitivity analysis scenario
    pub fn layer_sensitivity_example() -> Result<()> {
        println!("🎯 Layer Sensitivity Analysis Scenario");
        println!("======================================\n");

        let device = Device::Cpu;
        
        // Simulate different layer types with varying sensitivities
        let layer_configs = vec![
            ("embedding", (10000, 512), 0.001), // Low sensitivity
            ("attention_qkv", (512, 1536), 0.01), // Medium sensitivity  
            ("attention_out", (512, 512), 0.005), // Medium sensitivity
            ("ffn_up", (512, 2048), 0.02), // High sensitivity
            ("ffn_down", (2048, 512), 0.015), // High sensitivity
            ("layer_norm", (512,), 0.001), // Low sensitivity
            ("output_proj", (512, 50000), 0.03), // Very high sensitivity
        ];

        println!("Analyzing quantization sensitivity across different layer types:\n");

        let mut layer_metrics = HashMap::new();
        let mse_calc = MSECalculator::new();
        let sqnr_calc = SQNRCalculator::new();
        let cosine_calc = CosineSimilarityCalculator::new();

        for (layer_name, shape, noise_level) in &layer_configs {
            let original = Tensor::randn(0f32, 1f32, *shape, &device)?;
            let noise = Tensor::randn(0f32, *noise_level, *shape, &device)?;
            let quantized = (&original + &noise)?;

            let mse = mse_calc.calculate_mse(&original, &quantized)?;
            let sqnr = sqnr_calc.calculate_sqnr(&original, &quantized)?;
            let cosine_sim = cosine_calc.calculate_cosine_similarity(&original, &quantized)?;

            let metrics = QuantizationMetrics {
                layer_name: layer_name.to_string(),
                mse,
                sqnr,
                cosine_similarity: cosine_sim,
                ..Default::default()
            };

            layer_metrics.insert(layer_name.to_string(), metrics);

            println!("{} ({}): ", layer_name, format!("{:?}", shape));
            println!("  MSE: {:.6}", mse);
            println!("  SQNR: {:.2} dB", sqnr);  
            println!("  Cosine Similarity: {:.4}", cosine_sim);
        }

        // Perform layer-wise analysis
        println!("\n📊 Layer-wise Analysis Results:");
        println!("================================");
        
        let config = crate::metrics::layer_wise::LayerWiseAnalysisConfig {
            correlation_threshold: 0.5,
            sensitivity_analysis_enabled: true,
            optimization_planning_enabled: true,
            temporal_analysis_enabled: false,
        };
        
        let analyzer = LayerWiseAnalyzer::new(config);
        let analysis = analyzer.analyze_layers(&layer_metrics)?;

        println!("Global Statistics:");
        println!("  Mean MSE: {:.6}", analysis.global_statistics.mean_mse);
        println!("  Mean SQNR: {:.2} dB", analysis.global_statistics.mean_sqnr);
        println!("  Mean Cosine Similarity: {:.4}", analysis.global_statistics.mean_cosine_similarity);

        println!("\nSensitivity Ranking (most sensitive first):");
        for (i, (layer_name, sensitivity)) in analysis.sensitivity_ranking.iter().enumerate() {
            println!("  {}. {}: {:.2}", i + 1, layer_name, sensitivity);
        }

        println!("\nOptimization Plan:");
        println!("  High Priority: {:?}", analysis.optimization_plan.high_priority_layers);
        println!("  Medium Priority: {:?}", analysis.optimization_plan.medium_priority_layers);
        println!("  Low Priority: {:?}", analysis.optimization_plan.low_priority_layers);

        Ok(())
    }

    /// Example: Production monitoring scenario
    pub fn production_monitoring_example() -> Result<()> {
        println!("📈 Production Monitoring Scenario");
        println!("=================================\n");

        let device = Device::Cpu;
        let thresholds = ErrorThresholds {
            max_acceptable_mse: 0.01,
            min_acceptable_sqnr: 20.0,
            min_acceptable_cosine_similarity: 0.95,
            max_acceptable_relative_error: 0.05,
        };

        println!("Simulating production inference monitoring...");
        println!("Quality Thresholds:");
        println!("  Max MSE: {}", thresholds.max_acceptable_mse);
        println!("  Min SQNR: {} dB", thresholds.min_acceptable_sqnr);
        println!("  Min Cosine Similarity: {}", thresholds.min_acceptable_cosine_similarity);
        println!("  Max Relative Error: {}\n", thresholds.max_acceptable_relative_error);

        let mse_calc = MSECalculator::new();
        let sqnr_calc = SQNRCalculator::new(); 
        let cosine_calc = CosineSimilarityCalculator::new();

        let mut alerts = Vec::new();
        let batch_count = 10;

        for batch_idx in 0..batch_count {
            // Simulate varying quality over time
            let quality_degradation = (batch_idx as f32) * 0.002; // Gradual degradation
            let noise_level = 0.005 + quality_degradation;
            
            let original = Tensor::randn(0f32, 1f32, (512, 768), &device)?;
            let noise = Tensor::randn(0f32, noise_level, (512, 768), &device)?;
            let quantized = (&original + &noise)?;

            let mse = mse_calc.calculate_mse(&original, &quantized)?;
            let sqnr = sqnr_calc.calculate_sqnr(&original, &quantized)?;
            let cosine_sim = cosine_calc.calculate_cosine_similarity(&original, &quantized)?;

            println!("Batch {}: MSE={:.6}, SQNR={:.2}dB, Cosine={:.4}", 
                batch_idx + 1, mse, sqnr, cosine_sim);

            // Check thresholds and generate alerts
            if mse > thresholds.max_acceptable_mse {
                alerts.push(format!("Batch {}: MSE threshold exceeded ({:.6} > {})", 
                    batch_idx + 1, mse, thresholds.max_acceptable_mse));
            }

            if sqnr < thresholds.min_acceptable_sqnr {
                alerts.push(format!("Batch {}: SQNR threshold exceeded ({:.2} < {})", 
                    batch_idx + 1, sqnr, thresholds.min_acceptable_sqnr));
            }

            if cosine_sim < thresholds.min_acceptable_cosine_similarity {
                alerts.push(format!("Batch {}: Cosine similarity threshold exceeded ({:.4} < {})", 
                    batch_idx + 1, cosine_sim, thresholds.min_acceptable_cosine_similarity));
            }
        }

        println!("\n🚨 Quality Alerts Generated:");
        if alerts.is_empty() {
            println!("  No alerts - All batches within acceptable thresholds ✅");
        } else {
            for alert in &alerts {
                println!("  ⚠️ {}", alert);
            }
        }

        println!("\n📊 Monitoring Summary:");
        println!("  Total batches processed: {}", batch_count);
        println!("  Alerts generated: {}", alerts.len());
        println!("  Alert rate: {:.1}%", (alerts.len() as f32 / batch_count as f32) * 100.0);

        Ok(())
    }
}

/// Run all examples
pub fn run_all_examples() -> Result<()> {
    println!("🚀 BitNet Quantization Metrics Examples");
    println!("========================================\n");

    // Basic examples
    BasicMetricsExample::calculate_mse_example()?;
    BasicMetricsExample::calculate_sqnr_example()?;
    BasicMetricsExample::calculate_cosine_similarity_example()?;

    // Streaming processing
    let mut streaming_example = StreamingProcessingExample::new();
    streaming_example.streaming_analysis_example()?;

    // Real-world scenarios
    RealWorldScenarios::model_comparison_example()?;
    RealWorldScenarios::layer_sensitivity_example()?;
    RealWorldScenarios::production_monitoring_example()?;

    // Complete workflow (requires tensor data)
    println!("\n💡 For complete workflow example, use MetricsWorkflowDemo::run_complete_analysis_workflow()");
    println!("   with your actual model layer tensors.\n");

    println!("✅ All examples completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_mse_example() -> Result<()> {
        BasicMetricsExample::calculate_mse_example()
    }

    #[test]
    fn test_basic_sqnr_example() -> Result<()> {
        BasicMetricsExample::calculate_sqnr_example()
    }

    #[test]
    fn test_basic_cosine_example() -> Result<()> {
        BasicMetricsExample::calculate_cosine_similarity_example()
    }

    #[test]
    fn test_streaming_processing() -> Result<()> {
        let mut example = StreamingProcessingExample::new();
        example.streaming_analysis_example()
    }

    #[test]
    fn test_model_comparison() -> Result<()> {
        RealWorldScenarios::model_comparison_example()
    }

    #[test]
    fn test_layer_sensitivity() -> Result<()> {
        RealWorldScenarios::layer_sensitivity_example()
    }

    #[test]
    fn test_production_monitoring() -> Result<()> {
        RealWorldScenarios::production_monitoring_example()
    }

    #[test]
    fn test_metrics_workflow_demo_creation() -> Result<()> {
        let device = Device::Cpu;
        let _demo = MetricsWorkflowDemo::new(device)?;
        Ok(())
    }
}
