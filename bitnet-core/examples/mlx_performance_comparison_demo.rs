//! MLX Performance Comparison Demo
//!
//! This example demonstrates how to use the comprehensive MLX performance comparison tools
//! to analyze, benchmark, and optimize MLX operations on Apple Silicon devices.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "mlx")]
    {
        run_mlx_demo()
    }

    #[cfg(not(feature = "mlx"))]
    {
        run_stub_demo();
        Ok(())
    }
}

#[cfg(feature = "mlx")]
fn run_mlx_demo() -> Result<(), Box<dyn std::error::Error>> {
    use anyhow::Result;
    use bitnet_core::mlx::{
        track_allocation, track_deallocation, BenchmarkConfig, BitNetMlxDevice, ComparisonResult,
        DeviceComparisonConfig, ExportFormat, ImplementationEffort, MemoryEvent, MemoryEventType,
        MemoryMetrics, MemoryOptimization, MemoryUsage, MetricsConfig, MlxAdvancedProfiler,
        MlxDeviceComparison, MlxMemoryTracker, MlxMetrics, MlxMetricsCollector,
        MlxPerformanceBenchmarker, MlxRegressionTester, OperationContext, OptimizationPriority,
        OptimizationType, PerformanceMetrics, PerformanceReportGenerator, ProfileOutputFormat,
        ProfilerConfig, RegressionTestConfig, SystemMetrics,
    };
    use std::time::{Duration, SystemTime};

    println!("🚀 MLX Performance Comparison Demo");
    println!("===================================\n");

    // 1. Basic Performance Benchmarking
    demo_basic_benchmarking()?;

    // 2. Memory Tracking and Analysis
    demo_memory_tracking()?;

    // 3. Comprehensive Device Comparison
    demo_device_comparison()?;

    // 4. Advanced Profiling
    demo_advanced_profiling()?;

    // 5. Metrics Collection
    demo_metrics_collection()?;

    // 6. Regression Testing
    demo_regression_testing()?;

    // 7. Report Generation
    demo_report_generation()?;

    println!("\n✅ Demo completed successfully!");
    Ok(())
}

#[cfg(not(feature = "mlx"))]
fn run_stub_demo() {
    println!("🚀 MLX Performance Comparison Demo");
    println!("===================================");
    println!();
    println!("MLX feature not enabled. Please run with --features mlx");
    println!();
    println!("This demo showcases:");
    println!("• Basic performance benchmarking");
    println!("• Memory tracking and analysis");
    println!("• Comprehensive device comparison");
    println!("• Advanced profiling capabilities");
    println!("• Metrics collection and export");
    println!("• Regression testing");
    println!("• Performance report generation");
    println!();
    println!("To see these features in action, rebuild with:");
    println!("cargo run --example mlx_performance_comparison_demo --features mlx");
}

/// Demonstrate basic performance benchmarking
#[cfg(feature = "mlx")]
fn demo_basic_benchmarking() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 1. Basic Performance Benchmarking");
    println!("------------------------------------");

    // Create benchmark configuration
    let config = BenchmarkConfig {
        warmup_iterations: 3,
        measurement_iterations: 5,
        tensor_sizes: vec![vec![512, 512], vec![1024, 1024]],
        data_types: vec!["f32".to_string()],
        devices: vec!["cpu".to_string(), "gpu".to_string()],
        timeout: Duration::from_secs(30),
    };

    let mut benchmarker = MlxPerformanceBenchmarker::new(config);

    // Test both CPU and GPU
    let cpu_device = BitNetMlxDevice::cpu();
    let gpu_device = BitNetMlxDevice::gpu();

    println!("Benchmarking matrix multiplication...");

    // Benchmark CPU
    let cpu_metrics = benchmarker.benchmark_matmul(&cpu_device)?;
    println!("CPU Results:");
    println!("  Execution time: {:?}", cpu_metrics.execution_time);
    println!("  Throughput: {:.2} ops/sec", cpu_metrics.throughput);
    println!(
        "  Memory usage: {:.2} MB",
        cpu_metrics.memory_usage.allocated_memory_mb
    );

    // Benchmark GPU
    let gpu_metrics = benchmarker.benchmark_matmul(&gpu_device)?;
    println!("GPU Results:");
    println!("  Execution time: {:?}", gpu_metrics.execution_time);
    println!("  Throughput: {:.2} ops/sec", gpu_metrics.throughput);
    println!(
        "  Memory usage: {:.2} MB",
        gpu_metrics.memory_usage.allocated_memory_mb
    );

    // Calculate speedup
    let speedup =
        cpu_metrics.execution_time.as_secs_f64() / gpu_metrics.execution_time.as_secs_f64();
    println!("GPU Speedup: {:.2}x", speedup);

    // Test other operations
    println!("\nBenchmarking quantization...");
    let cpu_quant = benchmarker.benchmark_quantization(&cpu_device)?;
    let gpu_quant = benchmarker.benchmark_quantization(&gpu_device)?;

    let quant_speedup =
        cpu_quant.execution_time.as_secs_f64() / gpu_quant.execution_time.as_secs_f64();
    println!("Quantization GPU speedup: {:.2}x", quant_speedup);

    println!("✅ Basic benchmarking completed\n");
    Ok(())
}

/// Demonstrate memory tracking and analysis
#[cfg(feature = "mlx")]
fn demo_memory_tracking() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 2. Memory Tracking and Analysis");
    println!("----------------------------------");

    let mut tracker = MlxMemoryTracker::new();
    tracker.set_tracking_enabled(true);

    let device = BitNetMlxDevice::gpu();

    println!("Simulating memory operations...");

    // Simulate tensor allocations
    for i in 0..5 {
        let tensor_id = format!("tensor_{}", i);
        let size = (i + 1) * 1024 * 1024; // 1MB, 2MB, 3MB, etc.

        track_allocation(tensor_id.clone(), size, &device, "matmul".to_string())?;

        println!("  Allocated {}: {} bytes", tensor_id, size);
    }

    // Check memory pressure
    let pressure = tracker.get_memory_pressure(&device);
    println!("Memory pressure level: {:?}", pressure);

    // Get optimization recommendations
    let optimizations = tracker.generate_optimizations(&device);
    println!("Memory optimization recommendations:");
    for opt in &optimizations {
        println!("  - {}", opt.description);
        println!("    Potential savings: {} bytes", opt.potential_savings);
        println!("    Priority: {:?}", opt.priority);
    }

    // Simulate some deallocations
    for i in 0..3 {
        let tensor_id = format!("tensor_{}", i);
        track_deallocation(tensor_id, &device, "cleanup".to_string())?;
    }

    // Generate memory report
    let report = tracker.generate_report();
    println!("\nMemory Usage Report:");
    println!("{}", report);

    println!("✅ Memory tracking completed\n");
    Ok(())
}

/// Demonstrate comprehensive device comparison
#[cfg(feature = "mlx")]
fn demo_device_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚖️  3. Comprehensive Device Comparison");
    println!("-------------------------------------");

    let config = DeviceComparisonConfig {
        devices_to_compare: vec!["cpu".to_string(), "gpu".to_string()],
        operations_to_test: vec!["matmul".to_string(), "quantization".to_string()],
        tensor_sizes: vec![vec![256, 256], vec![1024, 1024]],
        data_types: vec!["f32".to_string()],
        iterations_per_test: 3,
        warmup_iterations: 2,
        enable_memory_analysis: true,
        enable_profiling: false,
        enable_power_analysis: true,
        comparison_timeout: Duration::from_secs(60),
    };

    let mut comparison = MlxDeviceComparison::new(config);

    println!("Running comprehensive device comparison...");
    let results = comparison.run_comparison()?;

    // Display summary
    println!("Comparison Results:");
    println!(
        "  Best overall device: {}",
        results.summary.best_overall_device
    );
    println!(
        "  Best performance device: {}",
        results.summary.best_performance_device
    );
    println!(
        "  Best efficiency device: {}",
        results.summary.best_efficiency_device
    );
    println!(
        "  Best memory device: {}",
        results.summary.best_memory_device
    );

    // Display key insights
    println!("\nKey Insights:");
    for insight in &results.summary.key_insights {
        println!("  • {}", insight);
    }

    // Display device recommendations
    println!("\nDevice Recommendations:");
    for (use_case, device) in &results.device_selection_guide.use_case_recommendations {
        println!("  {} → {}", use_case, device);
    }

    // Display optimization recommendations
    println!("\nOptimization Recommendations:");
    for rec in &results.optimization_recommendations {
        println!("  {} on {}: {}", rec.device, rec.operation, rec.description);
        println!("    Expected improvement: {:.1}x", rec.expected_improvement);
    }

    // Export detailed results
    let summary = comparison.export_results(&results, "summary")?;
    println!("\nDetailed Summary:");
    println!("{}", summary);

    println!("✅ Device comparison completed\n");
    Ok(())
}

/// Demonstrate advanced profiling capabilities
#[cfg(feature = "mlx")]
fn demo_advanced_profiling() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 4. Advanced Profiling");
    println!("------------------------");

    let config = ProfilerConfig {
        enable_call_stack_tracking: true,
        enable_memory_profiling: true,
        enable_gpu_profiling: true,
        enable_hotspot_detection: true,
        sampling_interval: Duration::from_millis(10),
        max_call_stack_depth: 16,
        profile_duration: Some(Duration::from_secs(5)),
        output_format: ProfileOutputFormat::Json,
    };

    let mut profiler = MlxAdvancedProfiler::new(config);

    println!("Starting profiling session...");
    profiler.start_session("demo_profiling".to_string())?;

    let device = BitNetMlxDevice::gpu();

    // Profile matrix multiplication
    let (_, metrics) = profiler.profile_operation("matmul", &device, || {
        // Simulate MLX operation
        std::thread::sleep(Duration::from_millis(50));
        Ok(())
    })?;

    println!(
        "Profiled operation execution time: {:?}",
        metrics.execution_time
    );

    // Profile quantization
    let (_, _) = profiler.profile_operation("quantization", &device, || {
        // Simulate quantization operation
        std::thread::sleep(Duration::from_millis(30));
        Ok(())
    })?;

    // Stop profiling and get results
    let session = profiler.stop_session()?;

    println!("Profiling session completed:");
    println!("  Session ID: {}", session.session_id);
    println!("  Call stack samples: {}", session.call_stacks.len());
    println!("  Hotspots detected: {}", session.hotspots.len());
    println!("  Bottlenecks found: {}", session.bottlenecks.len());

    // Display hotspots
    if !session.hotspots.is_empty() {
        println!("\nTop Hotspots:");
        for (i, hotspot) in session.hotspots.iter().take(3).enumerate() {
            println!(
                "  {}. {} - {:.2}% of total time",
                i + 1,
                hotspot.function_name,
                hotspot.percentage_of_total
            );
            println!(
                "     Calls: {}, Avg time: {:?}",
                hotspot.call_count, hotspot.average_time
            );
        }
    }

    // Generate flame graph
    let flame_graph = profiler.generate_flame_graph(&session)?;
    println!(
        "\nFlame graph data generated ({} lines)",
        flame_graph.lines().count()
    );

    // Generate call tree
    let call_tree = profiler.generate_call_tree(&session)?;
    println!("Call tree analysis:");
    println!(
        "{}",
        call_tree.lines().take(10).collect::<Vec<_>>().join("\n")
    );

    println!("✅ Advanced profiling completed\n");
    Ok(())
}

/// Demonstrate metrics collection
#[cfg(feature = "mlx")]
fn demo_metrics_collection() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 5. Metrics Collection");
    println!("------------------------");

    let config = MetricsConfig {
        collect_performance: true,
        collect_memory: true,
        collect_system: true,
        collection_interval: Duration::from_millis(100),
        max_history_size: 1000,
        enable_detailed_profiling: false,
        export_format: ExportFormat::Json,
    };

    let collector = MlxMetricsCollector::new(config);
    collector.start_collection();

    let device = BitNetMlxDevice::gpu();

    println!("Collecting metrics for operations...");

    // Simulate multiple operations with metrics collection
    for i in 0..3 {
        let context = OperationContext {
            operation_name: "matmul".to_string(),
            batch_size: 32,
            sequence_length: Some(512),
            model_parameters: Some(1000000),
            precision: "f32".to_string(),
            optimization_level: "O2".to_string(),
            parallel_execution: true,
        };

        let execution_time = Duration::from_millis(50 + i * 10);
        let tensor_shapes = vec![vec![512, 512], vec![512, 512]];

        let metrics = collector.collect_operation_metrics(
            "matmul",
            &device,
            execution_time,
            tensor_shapes,
            context,
        )?;

        println!(
            "  Operation {}: {:?} execution time",
            i + 1,
            metrics.performance.execution_time
        );
    }

    // Get aggregated statistics
    let stats = collector.get_aggregated_stats();
    println!("\nAggregated Statistics:");
    println!("  Total operations: {}", stats.total_operations());
    println!(
        "  Average throughput: {:.2} ops/sec",
        stats.average_throughput()
    );
    println!("  Peak memory usage: {} bytes", stats.peak_memory_usage());

    // Export metrics in different formats
    let json_export = collector.export_metrics(Some(ExportFormat::Json))?;
    println!("JSON export generated ({} characters)", json_export.len());

    let csv_export = collector.export_metrics(Some(ExportFormat::Csv))?;
    println!(
        "CSV export generated ({} lines)",
        csv_export.lines().count()
    );

    collector.stop_collection();
    println!("✅ Metrics collection completed\n");
    Ok(())
}

/// Demonstrate regression testing
#[cfg(feature = "mlx")]
fn demo_regression_testing() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 6. Regression Testing");
    println!("------------------------");

    let config = RegressionTestConfig {
        baseline_directory: std::path::PathBuf::from("./demo_baselines"),
        test_operations: vec!["matmul".to_string(), "quantization".to_string()],
        test_devices: vec!["cpu".to_string(), "gpu".to_string()],
        tensor_sizes: vec![vec![512, 512], vec![1024, 1024]],
        data_types: vec!["f32".to_string()],
        regression_threshold: 15.0, // 15% threshold for demo
        improvement_threshold: 5.0,
        iterations_per_test: 3,
        warmup_iterations: 1,
        enable_memory_regression_testing: true,
        enable_automated_bisection: false, // Disabled for demo
        max_bisection_iterations: 5,
        test_timeout: Duration::from_secs(30),
    };

    let mut tester = MlxRegressionTester::new(config);

    println!("Creating performance baseline...");
    let baseline = tester.create_baseline(
        "demo_v1.0".to_string(),
        "Demo baseline for MLX performance comparison".to_string(),
    )?;

    println!("Baseline created: {}", baseline.baseline_id);
    println!(
        "  Performance metrics: {}",
        baseline.performance_metrics.len()
    );
    println!("  Memory baselines: {}", baseline.memory_baselines.len());

    // Simulate some time passing and potential changes
    std::thread::sleep(Duration::from_millis(100));

    println!("\nRunning regression tests...");
    let test_results = tester.run_regression_tests("demo_v1.0")?;

    println!("Regression test results:");
    println!("  Test ID: {}", test_results.test_id);
    println!("  Overall status: {:?}", test_results.overall_status);
    println!("  Total tests: {}", test_results.summary.total_tests);
    println!("  Passed tests: {}", test_results.summary.passed_tests);
    println!("  Regressions: {}", test_results.summary.regression_count);
    println!("  Improvements: {}", test_results.summary.improvement_count);

    // Display any regressions
    if !test_results.regressions_detected.is_empty() {
        println!("\nRegressions detected:");
        for regression in &test_results.regressions_detected {
            println!(
                "  {} on {}: {:.1}% degradation ({:?})",
                regression.operation,
                regression.device,
                regression.performance_degradation,
                regression.severity
            );
        }
    }

    // Display any improvements
    if !test_results.improvements_detected.is_empty() {
        println!("\nImprovements detected:");
        for improvement in &test_results.improvements_detected {
            println!(
                "  {} on {}: {:.1}% improvement",
                improvement.operation, improvement.device, improvement.performance_improvement
            );
        }
    }

    // Generate regression report
    let report = tester.generate_report(&test_results)?;
    println!("\nRegression Test Report:");
    println!("{}", report.lines().take(20).collect::<Vec<_>>().join("\n"));
    println!("... (truncated)");

    println!("✅ Regression testing completed\n");
    Ok(())
}

/// Demonstrate report generation
#[cfg(feature = "mlx")]
fn demo_report_generation() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 7. Report Generation");
    println!("-----------------------");

    // Create sample data for report generation
    let generator = PerformanceReportGenerator::new();

    // Generate sample metrics (in a real scenario, these would come from actual benchmarks)
    let sample_metrics = create_sample_metrics();
    let sample_comparisons = create_sample_comparisons();
    let sample_memory_events = create_sample_memory_events();
    let sample_optimizations = create_sample_optimizations();

    println!("Generating comprehensive performance report...");

    let report = generator.generate_comprehensive_report(
        &sample_metrics,
        &sample_comparisons,
        &sample_memory_events,
        &sample_optimizations,
    )?;

    println!("Report generated:");
    println!(
        "  Operations analyzed: {}",
        report.performance_analysis.operation_performance.len()
    );
    println!("  Device comparisons: {}", report.device_comparisons.len());
    println!(
        "  Optimization recommendations: {}",
        report.optimization_recommendations.len()
    );
    println!(
        "  Overall performance score: {:.1}/100",
        report.executive_summary.overall_score
    );

    // Display key findings
    println!("\nKey Findings:");
    for finding in &report.executive_summary.key_findings {
        println!("  • {}", finding);
    }

    // Display top recommendations
    println!("\nTop Recommendations:");
    for rec in report.optimization_recommendations.iter().take(3) {
        println!("  • {}: {}", rec.title, rec.description);
        if let Some(perf_gain) = rec.expected_improvement.performance_gain {
            println!("    Expected improvement: {:.1}x", perf_gain);
        }
    }

    // Generate HTML report
    println!("\nGenerating HTML report...");
    let html_report = generator.generate_html_report(&report)?;

    // Save HTML report to file
    std::fs::write("demo_performance_report.html", &html_report)?;
    println!(
        "HTML report saved to: demo_performance_report.html ({} characters)",
        html_report.len()
    );

    // Generate JSON report for programmatic access
    let json_report = generator.generate_json_report(&sample_metrics, &sample_comparisons)?;
    std::fs::write("demo_performance_data.json", &json_report)?;
    println!("JSON data saved to: demo_performance_data.json");

    println!("✅ Report generation completed\n");
    Ok(())
}

// Helper functions to create sample data for demonstration

#[cfg(feature = "mlx")]
fn create_sample_metrics() -> Vec<MlxMetrics> {
    vec![MlxMetrics {
        performance: PerformanceMetrics {
            operation_name: "matmul".to_string(),
            device_type: "gpu".to_string(),
            execution_time: Duration::from_millis(25),
            memory_usage: MemoryUsage {
                peak_memory_mb: 128.0,
                allocated_memory_mb: 64.0,
                freed_memory_mb: 0.0,
                memory_efficiency: 0.85,
            },
            throughput: 40.0,
            tensor_shapes: vec![vec![1024, 1024]],
            data_type: "f32".to_string(),
            timestamp: SystemTime::now(),
        },
        memory: MemoryMetrics {
            current_usage: MemoryUsage {
                peak_memory_mb: 128.0,
                allocated_memory_mb: 64.0,
                freed_memory_mb: 0.0,
                memory_efficiency: 0.85,
            },
            pressure_level: "Low".to_string(),
            allocation_events: 10,
            deallocation_events: 8,
            transfer_events: 2,
            fragmentation_ratio: 0.1,
            efficiency_score: 0.85,
        },
        system: SystemMetrics {
            cpu_usage: 25.0,
            gpu_usage: 75.0,
            system_memory_usage: 60.0,
            gpu_memory_usage: 40.0,
            temperature: Some(65.0),
            power_consumption: Some(25.0),
            thermal_state: "Normal".to_string(),
        },
        operation_context: OperationContext {
            operation_name: "matmul".to_string(),
            batch_size: 32,
            sequence_length: Some(512),
            model_parameters: Some(1000000),
            precision: "f32".to_string(),
            optimization_level: "O2".to_string(),
            parallel_execution: true,
        },
    }]
}

#[cfg(feature = "mlx")]
fn create_sample_comparisons() -> Vec<ComparisonResult> {
    vec![ComparisonResult {
        baseline_metrics: PerformanceMetrics {
            operation_name: "matmul".to_string(),
            device_type: "cpu".to_string(),
            execution_time: Duration::from_millis(100),
            memory_usage: MemoryUsage {
                peak_memory_mb: 64.0,
                allocated_memory_mb: 32.0,
                freed_memory_mb: 0.0,
                memory_efficiency: 0.8,
            },
            throughput: 10.0,
            tensor_shapes: vec![vec![1024, 1024]],
            data_type: "f32".to_string(),
            timestamp: SystemTime::now(),
        },
        comparison_metrics: PerformanceMetrics {
            operation_name: "matmul".to_string(),
            device_type: "gpu".to_string(),
            execution_time: Duration::from_millis(25),
            memory_usage: MemoryUsage {
                peak_memory_mb: 128.0,
                allocated_memory_mb: 64.0,
                freed_memory_mb: 0.0,
                memory_efficiency: 0.85,
            },
            throughput: 40.0,
            tensor_shapes: vec![vec![1024, 1024]],
            data_type: "f32".to_string(),
            timestamp: SystemTime::now(),
        },
        speedup: 4.0,
        memory_improvement: 0.05,
        throughput_improvement: 4.0,
        recommendation: "Use GPU for large matrix operations".to_string(),
    }]
}

#[cfg(feature = "mlx")]
fn create_sample_memory_events() -> Vec<MemoryEvent> {
    vec![MemoryEvent {
        event_type: MemoryEventType::Allocation,
        size_bytes: 1024 * 1024,
        device_type: "gpu".to_string(),
        operation: "matmul".to_string(),
        timestamp: SystemTime::now(),
        tensor_id: "tensor_1".to_string(),
        stack_trace: Some("mlx_matmul -> allocate_tensor".to_string()),
    }]
}

#[cfg(feature = "mlx")]
fn create_sample_optimizations() -> Vec<MemoryOptimization> {
    vec![MemoryOptimization {
        suggestion_type: OptimizationType::TensorReuse,
        description: "Implement tensor reuse for frequently allocated sizes".to_string(),
        potential_savings: 1024 * 1024,
        priority: OptimizationPriority::Medium,
        implementation_effort: ImplementationEffort::Low,
    }]
}
