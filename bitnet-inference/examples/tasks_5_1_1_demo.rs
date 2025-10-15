//! Example demonstrating Tasks 5.1.1c and 5.1.1d implementation
//!
//! This example shows how to use the new reference output collection
//! and benchmark dataset integration systems.

use bitnet_inference::{
    ReferenceOutputCollector, ReferenceCollectionConfig, CollectionMethod,
    BenchmarkIntegrator, BenchmarkConfig,
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎯 BitNet Reference Outputs and Benchmark Datasets Demo");
    println!("Tasks 5.1.1c and 5.1.1d Implementation Example\n");

    // Task 5.1.1c: Reference Output Collection Demo
    println!("📊 Task 5.1.1c: Reference Output Collection");
    println!("===========================================");
    
    let config = ReferenceCollectionConfig {
        model_id: "microsoft/bitnet-b1.58-2B-4T-gguf".to_string(),
        temperature: 0.7,
        max_tokens: 50,
        seed: Some(42),
        collection_method: CollectionMethod::ManualCollection,
    };
    
    let mut collector = ReferenceOutputCollector::new(config);
    collector.create_standard_references()?;
    
    let references = collector.get_reference_outputs();
    println!("✅ Created {} reference outputs", references.len());
    
    // Show some examples
    for (i, reference) in references.iter().take(3).enumerate() {
        println!("  {}. Prompt: \"{}\"", i + 1, reference.prompt);
        println!("     Expected tokens: {:?}", reference.expected_tokens);
        println!("     Expected text: \"{}\"", reference.expected_text);
        println!();
    }
    
    // Demonstrate validation
    println!("🔍 Reference Validation Example:");
    let test_prompt = "Hello, my name is";
    let mock_tokens = vec![1000, 1001, 1002];
    let mock_logits = vec![0.1, -0.2, 0.3];
    
    if let Ok(result) = collector.validate_against_reference(test_prompt, &mock_tokens, &mock_logits, 0.1) {
        println!("  ✅ Validation passed: {}", result.passed);
        println!("  📈 Token accuracy: {:.1}%", result.token_accuracy * 100.0);
        println!("  📊 Logit MAE: {:.4}", result.logit_mae);
    }
    
    println!();
    
    // Task 5.1.1d: Benchmark Dataset Integration Demo
    println!("📚 Task 5.1.1d: Benchmark Dataset Integration");
    println!("=============================================");
    
    let benchmark_config = BenchmarkConfig {
        max_examples: Some(10),
        batch_size: 4,
        temperature: 0.7,
        max_tokens: 50,
        seed: Some(42),
        save_detailed_results: false,
    };
    
    let mut integrator = BenchmarkIntegrator::new(benchmark_config);
    integrator.create_sample_glue_datasets()?;
    integrator.create_sample_superglue_datasets()?;
    
    let datasets = integrator.get_available_datasets();
    println!("✅ Created {} benchmark datasets", datasets.len());
    println!("📋 Available datasets:");
    for dataset in &datasets {
        println!("  - {}", dataset);
        
        if let Some(examples) = integrator.get_dataset(dataset) {
            println!("    {} examples", examples.len());
            if let Some(first_example) = examples.first() {
                let preview = if first_example.input_text.len() > 80 {
                    format!("{}...", &first_example.input_text[..77])
                } else {
                    first_example.input_text.clone()
                };
                println!("    Example: \"{}\"", preview);
            }
        }
    }
    
    println!();
    
    // Demonstrate benchmark evaluation
    println!("🔬 Benchmark Evaluation Example:");
    let evaluation_result = integrator.evaluate_dataset("glue_sst2", "test_model").await?;
    
    println!("  📊 Dataset: {:?}", evaluation_result.dataset);
    println!("  📈 Examples evaluated: {}", evaluation_result.num_examples);
    println!("  ⏱️  Evaluation time: {:.2}s", evaluation_result.evaluation_time_secs);
    
    if let Some(accuracy) = evaluation_result.metrics.accuracy {
        println!("  🎯 Accuracy: {:.1}%", accuracy * 100.0);
    }
    
    if let Some(f1) = evaluation_result.metrics.f1_score {
        println!("  📏 F1 Score: {:.3}", f1);
    }
    
    // Show some example results
    println!("  🔍 Sample predictions:");
    for (i, example) in evaluation_result.example_results.iter().take(3).enumerate() {
        let status = if example.correct { "✅" } else { "❌" };
        println!("    {}. {} ID: {} | Correct: {} | Time: {:.1}ms", 
                i + 1, status, example.id, example.correct, example.inference_time_ms);
    }
    
    println!();
    println!("🎉 Tasks 5.1.1c and 5.1.1d Successfully Completed!");
    println!("✅ Reference output collection system operational");
    println!("✅ Benchmark dataset integration system operational");
    println!("✅ Both systems ready for BitNet model validation");
    
    Ok(())
}