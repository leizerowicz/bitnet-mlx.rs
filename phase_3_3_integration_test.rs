// Phase 3.3 Integration Test - Error Analysis and Metrics Validation
//
// This test validates all Phase 3.3 components work together correctly

use bitnet_quant::metrics::{
    QuantizationMetrics, ErrorThresholds,
    mse::MSECalculator,
    sqnr::SQNRCalculator,
    cosine_similarity::CosineSimilarityCalculator,
    error_analysis::ErrorAnalyzer,
    layer_wise::LayerWiseAnalyzer,
    visualization::VisualizationEngine,
    mitigation::ErrorMitigationEngine,
    reporting::ReportingEngine,
    examples::MetricsWorkflowDemo
};
use candle_core::{Device, Tensor, DType};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Phase 3.3 Integration Testing - Error Analysis and Metrics");
    println!("=" .repeat(70));

    let device = Device::Cpu;

    // 1. MSE Calculator Integration Test
    println!("\n📊 1. Testing MSE Calculator Integration");
    println!("-".repeat(50));
    test_mse_integration(&device)?;

    // 2. SQNR Calculator Integration Test
    println!("\n📊 2. Testing SQNR Calculator Integration");
    println!("-".repeat(50));
    test_sqnr_integration(&device)?;

    // 3. Cosine Similarity Integration Test
    println!("\n📊 3. Testing Cosine Similarity Integration");
    println!("-".repeat(50));
    test_cosine_similarity_integration(&device)?;

    // 4. Error Analysis Integration Test
    println!("\n📊 4. Testing Error Analysis Integration");
    println!("-".repeat(50));
    test_error_analysis_integration(&device)?;

    // 5. Layer-wise Analysis Integration Test
    println!("\n📊 5. Testing Layer-wise Analysis Integration");
    println!("-".repeat(50));
    test_layer_wise_integration(&device)?;

    // 6. Visualization Integration Test
    println!("\n📊 6. Testing Visualization Integration");
    println!("-".repeat(50));
    test_visualization_integration(&device)?;

    // 7. Error Mitigation Integration Test
    println!("\n📊 7. Testing Error Mitigation Integration");
    println!("-".repeat(50));
    test_error_mitigation_integration(&device)?;

    // 8. Reporting Integration Test
    println!("\n📊 8. Testing Reporting Integration");
    println!("-".repeat(50));
    test_reporting_integration(&device)?;

    // 9. Complete Workflow Integration Test
    println!("\n📊 9. Testing Complete Workflow Integration");
    println!("-".repeat(50));
    test_complete_workflow_integration(&device)?;

    println!("\n🎉 Phase 3.3 Integration Testing Complete!");
    println!("✅ All metrics and error analysis components validated successfully");
    println!("=" .repeat(70));

    Ok(())
}

fn test_mse_integration(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    // Create test tensors
    let original = Tensor::randn(0f32, 1f32, (1000,), device)?;
    let quantized = &original + &Tensor::randn(0f32, 0.1f32, (1000,), device)?;

    let calculator = MSECalculator::new();
    let mse = calculator.calculate_mse(&original, &quantized)?;

    println!("  MSE calculated: {:.6}", mse);
    println!("  ✅ MSE Calculator integration successful");

    Ok(())
}

fn test_sqnr_integration(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    // Create test tensors
    let original = Tensor::randn(0f32, 1f32, (1000,), device)?;
    let quantized = &original + &Tensor::randn(0f32, 0.1f32, (1000,), device)?;

    let calculator = SQNRCalculator::new();
    let sqnr = calculator.calculate_sqnr(&original, &quantized)?;

    println!("  SQNR calculated: {:.2} dB", sqnr);
    println!("  ✅ SQNR Calculator integration successful");

    Ok(())
}

fn test_cosine_similarity_integration(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    // Create test tensors
    let original = Tensor::randn(0f32, 1f32, (1000,), device)?;
    let quantized = &original * 0.9f32 + &Tensor::randn(0f32, 0.1f32, (1000,), device)?;

    let calculator = CosineSimilarityCalculator::new();
    let similarity = calculator.calculate_similarity(&original, &quantized)?;

    println!("  Cosine similarity: {:.4}", similarity);
    println!("  ✅ Cosine Similarity Calculator integration successful");

    Ok(())
}

fn test_error_analysis_integration(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    // Create test tensors
    let original = Tensor::randn(0f32, 1f32, (1000,), device)?;
    let quantized = &original + &Tensor::randn(0f32, 0.1f32, (1000,), device)?;

    let analyzer = ErrorAnalyzer::new(Default::default());
    let analysis = analyzer.analyze_quantization_errors(&original, &quantized)?;

    println!("  Error patterns detected: {}", analysis.error_patterns.len());
    println!("  Bit flip ratio: {:.4}", analysis.bit_flip_ratio);
    println!("  ✅ Error Analysis integration successful");

    Ok(())
}

fn test_layer_wise_integration(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    // Create test layer metrics
    let mut layer_metrics = HashMap::new();

    for i in 0..3 {
        let layer_name = format!("layer_{}", i);
        let original = Tensor::randn(0f32, 1f32, (100,), device)?;
        let quantized = &original + &Tensor::randn(0f32, 0.1f32 * (i + 1) as f32, (100,), device)?;

        let mse_calc = MSECalculator::new();
        let mse = mse_calc.calculate_mse(&original, &quantized)?;

        let metrics = QuantizationMetrics {
            mse,
            sqnr: 20.0 - (i as f32 * 3.0),
            cosine_similarity: 0.98 - (i as f32 * 0.02),
            max_error: 0.1 * (i + 1) as f32,
            mean_absolute_error: 0.05 * (i + 1) as f32,
            relative_error: 0.02 * (i + 1) as f32,
            bit_flip_ratio: 0.001 * (i + 1) as f32,
            layer_name: layer_name.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        layer_metrics.insert(layer_name, metrics);
    }

    let analyzer = LayerWiseAnalyzer::new(Default::default());
    let analysis = analyzer.analyze_layers(&layer_metrics)?;

    println!("  Layers analyzed: {}", analysis.layer_rankings.len());
    println!("  Problematic layers: {}", analysis.problematic_layers.len());
    println!("  ✅ Layer-wise Analysis integration successful");

    Ok(())
}

fn test_visualization_integration(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    // Create test data
    let original = Tensor::randn(0f32, 1f32, (100,), device)?;
    let quantized = &original + &Tensor::randn(0f32, 0.1f32, (100,), device)?;

    let analyzer = ErrorAnalyzer::new(Default::default());
    let analysis = analyzer.analyze_quantization_errors(&original, &quantized)?;

    let viz_engine = VisualizationEngine::new();
    let dashboard = viz_engine.create_error_analysis_dashboard(&analysis)?;

    println!("  Dashboard charts created: {}", dashboard.charts.len());
    println!("  ✅ Visualization integration successful");

    Ok(())
}

fn test_error_mitigation_integration(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    // Create test error analysis
    let original = Tensor::randn(0f32, 1f32, (100,), device)?;
    let quantized = &original + &Tensor::randn(0f32, 0.2f32, (100,), device)?;

    let analyzer = ErrorAnalyzer::new(Default::default());
    let analysis = analyzer.analyze_quantization_errors(&original, &quantized)?;

    let mitigation_engine = ErrorMitigationEngine::new();
    let strategy = mitigation_engine.select_mitigation_strategy(&analysis)?;

    println!("  Mitigation strategy selected: {:?}", strategy.strategy_type);
    println!("  Implementation complexity: {:?}", strategy.implementation_complexity);
    println!("  ✅ Error Mitigation integration successful");

    Ok(())
}

fn test_reporting_integration(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    // Create comprehensive test analysis
    let mut layer_metrics = HashMap::new();

    let original = Tensor::randn(0f32, 1f32, (100,), device)?;
    let quantized = &original + &Tensor::randn(0f32, 0.1f32, (100,), device)?;

    let mse_calc = MSECalculator::new();
    let mse = mse_calc.calculate_mse(&original, &quantized)?;

    let metrics = QuantizationMetrics {
        mse,
        sqnr: 18.5,
        cosine_similarity: 0.95,
        max_error: 0.15,
        mean_absolute_error: 0.08,
        relative_error: 0.03,
        bit_flip_ratio: 0.002,
        layer_name: "test_layer".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    layer_metrics.insert("test_layer".to_string(), metrics);

    let analyzer = LayerWiseAnalyzer::new(Default::default());
    let analysis = analyzer.analyze_layers(&layer_metrics)?;

    let reporting = ReportingEngine::new("./test_reports".to_string());
    let report = reporting.generate_comprehensive_report(&analysis)?;

    println!("  Report sections: {}", report.sections.len());
    println!("  Executive summary generated: {}", !report.executive_summary.is_empty());
    println!("  ✅ Reporting integration successful");

    Ok(())
}

fn test_complete_workflow_integration(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    // Test the complete workflow demo
    let mut demo = MetricsWorkflowDemo::new(device.clone())?;

    // Create sample layer data
    let mut layer_data = HashMap::new();
    for i in 0..2 {
        let layer_name = format!("workflow_layer_{}", i);
        let original = Tensor::randn(0f32, 1f32, (50,), device)?;
        let quantized = &original + &Tensor::randn(0f32, 0.1f32, (50,), device)?;

        layer_data.insert(layer_name, (original, quantized));
    }

    demo.run_complete_analysis_workflow(&layer_data)?;

    println!("  Complete workflow executed successfully");
    println!("  ✅ Complete Workflow integration successful");

    Ok(())
}
