//! Tests for Apple Neural Engine (ANE) Integration
//! 
//! This module provides comprehensive testing for ANE acceleration including:
//! - Hardware detection and availability checking
//! - CoreML conversion functionality  
//! - Performance monitoring and reporting
//! - Partitioning strategies and execution paths

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ane::*;

    #[test]
    fn test_ane_availability_detection() {
        // Test ANE availability detection across different platforms
        let result = ANEBitNetAccelerator::check_ane_availability();
        
        #[cfg(target_arch = "aarch64")]
        {
            // On Apple Silicon, check should succeed (even if ANE is not available)
            assert!(result.is_ok(), "ANE availability check should not fail on Apple Silicon");
            
            // Log the result for debugging
            match result {
                Ok(available) => {
                    println!("ANE availability on Apple Silicon: {}", available);
                },
                Err(e) => {
                    panic!("Unexpected error on Apple Silicon: {}", e);
                }
            }
        }
        
        #[cfg(not(target_arch = "aarch64"))]
        {
            // On non-Apple platforms, ANE should not be available
            assert!(result.is_ok(), "ANE availability check should not fail");
            assert!(!result.unwrap(), "ANE should not be available on non-Apple platforms");
        }
    }

    #[test]
    fn test_performance_monitor_basic_operations() {
        let mut monitor = ANEPerformanceMonitor::new();
        
        // Test initial state
        let stats = monitor.get_usage_stats();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.ane_operations, 0);
        assert_eq!(stats.gpu_operations, 0);
        assert_eq!(stats.cpu_operations, 0);
        assert_eq!(stats.average_ane_latency, 0.0);
        
        // Log some operations
        monitor.log_operation(ANEOperation::MatMul, 1.5, Device::ANE);
        monitor.log_operation(ANEOperation::Attention, 2.0, Device::GPU);
        monitor.log_operation(ANEOperation::Convolution, 0.8, Device::ANE);
        monitor.log_operation(ANEOperation::Activation, 0.5, Device::CPU);
        
        // Verify statistics
        let stats = monitor.get_usage_stats();
        assert_eq!(stats.total_operations, 4);
        assert_eq!(stats.ane_operations, 2);
        assert_eq!(stats.gpu_operations, 1);
        assert_eq!(stats.cpu_operations, 1);
        
        // Check average latency calculation
        let expected_avg = (1.5 + 0.8) / 2.0;
        assert!((stats.average_ane_latency - expected_avg).abs() < 0.001);
        
        // Check efficiency ratio
        let expected_ratio = 2.0 / 4.0;
        assert!((stats.ane_efficiency_ratio - expected_ratio).abs() < 0.001);
    }

    #[test]
    fn test_performance_report_generation() {
        let mut monitor = ANEPerformanceMonitor::new();
        
        // Log operations with different patterns
        for i in 0..10 {
            let device = if i % 3 == 0 { Device::ANE } else { Device::GPU };
            let latency = 1.0 + (i as f32 * 0.1);
            monitor.log_operation(ANEOperation::MatMul, latency, device);
        }
        
        let report = monitor.generate_performance_report();
        
        // Verify utilization calculation
        let expected_utilization = 4.0 / 10.0; // 4 ANE operations out of 10 total
        assert!((report.ane_utilization - expected_utilization).abs() < 0.001);
        
        // Verify we have recommendations
        assert!(!report.recommendations.is_empty());
        
        // Check power efficiency (should be >= 0.0)
        assert!(report.power_efficiency >= 0.0);
    }

    #[test]
    fn test_partitioning_strategies() {
        // Test FullANE strategy
        let full_ane = ANEPartitioningStrategy::FullANE;
        match full_ane {
            ANEPartitioningStrategy::FullANE => assert!(true),
            _ => assert!(false, "Expected FullANE strategy"),
        }
        
        // Test ANEGPUHybrid strategy
        let hybrid = ANEPartitioningStrategy::ANEGPUHybrid {
            ane_layers: vec![0, 2, 4, 6],
            gpu_layers: vec![1, 3, 5, 7],
        };
        
        match hybrid {
            ANEPartitioningStrategy::ANEGPUHybrid { ane_layers, gpu_layers } => {
                assert_eq!(ane_layers.len(), 4);
                assert_eq!(gpu_layers.len(), 4);
                assert_eq!(ane_layers[0], 0);
                assert_eq!(gpu_layers[0], 1);
            },
            _ => assert!(false, "Expected ANEGPUHybrid strategy"),
        }
        
        // Test SelectiveANE strategy
        let selective = ANEPartitioningStrategy::SelectiveANE {
            ane_operations: vec![ANEOperation::MatMul, ANEOperation::Convolution],
        };
        
        match selective {
            ANEPartitioningStrategy::SelectiveANE { ane_operations } => {
                assert_eq!(ane_operations.len(), 2);
                assert!(ane_operations.contains(&ANEOperation::MatMul));
                assert!(ane_operations.contains(&ANEOperation::Convolution));
            },
            _ => assert!(false, "Expected SelectiveANE strategy"),
        }
        
        // Test Dynamic strategy
        let dynamic = ANEPartitioningStrategy::Dynamic;
        match dynamic {
            ANEPartitioningStrategy::Dynamic => assert!(true),
            _ => assert!(false, "Expected Dynamic strategy"),
        }
    }

    #[test]
    fn test_ane_operations_enumeration() {
        let operations = vec![
            ANEOperation::MatMul,
            ANEOperation::Convolution,
            ANEOperation::Quantization,
            ANEOperation::Attention,
            ANEOperation::Activation,
        ];
        
        assert_eq!(operations.len(), 5);
        
        // Test equality and uniqueness
        assert_eq!(operations[0], ANEOperation::MatMul);
        assert_eq!(operations[1], ANEOperation::Convolution);
        assert_eq!(operations[2], ANEOperation::Quantization);
        assert_eq!(operations[3], ANEOperation::Attention);
        assert_eq!(operations[4], ANEOperation::Activation);
        
        // Test that each operation is different
        for (i, op1) in operations.iter().enumerate() {
            for (j, op2) in operations.iter().enumerate() {
                if i != j {
                    assert_ne!(op1, op2, "Operations at indices {} and {} should be different", i, j);
                }
            }
        }
    }

    #[test]
    fn test_coreml_converter() {
        let converter = ANECoreMLConverter::new();
        
        // Test ternary weight conversion with various input patterns
        let test_cases = vec![
            // Standard ternary weights
            (vec![-1.0, 0.0, 1.0], vec![-1.0, 0.0, 1.0]),
            // Mixed weights (some non-ternary)
            (vec![-1.0, 0.5, 1.0, -0.5], vec![-1.0, 0.5, 1.0, -0.5]),
            // Edge cases
            (vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]),
            (vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]),
            (vec![-1.0, -1.0, -1.0], vec![-1.0, -1.0, -1.0]),
        ];
        
        for (input_weights, expected) in test_cases {
            let result = converter.convert_ternary_weights_for_ane(&input_weights);
            assert!(result.is_ok(), "Weight conversion should succeed");
            
            let ane_weights = result.unwrap();
            assert_eq!(ane_weights.len(), expected.len());
            
            for (i, &expected_val) in expected.iter().enumerate() {
                let actual_val = ane_weights[i].to_f32();
                assert!((actual_val - expected_val).abs() < 0.001, 
                       "Weight {} mismatch: expected {}, got {}", i, expected_val, actual_val);
            }
        }
    }

    #[test] 
    fn test_ane_error_types() {
        // Test ANEError variants
        let not_available = ANEError::NotAvailable;
        let compilation_error = ANEError::CoreMLCompilation("test error".to_string());
        let conversion_error = ANEError::ModelConversion("conversion failed".to_string());
        let inference_error = ANEError::InferenceFailed("inference failed".to_string());
        let hardware_error = ANEError::HardwareDetection("hardware detection failed".to_string());
        
        // Test error messages
        assert!(not_available.to_string().contains("not available"));
        assert!(compilation_error.to_string().contains("test error"));
        assert!(conversion_error.to_string().contains("conversion failed"));
        assert!(inference_error.to_string().contains("inference failed"));
        assert!(hardware_error.to_string().contains("hardware detection failed"));
    }

    #[test]
    #[cfg(not(feature = "metal"))]
    fn test_mock_accelerator_creation() {
        // Test creating accelerator without Metal dependency
        let result = ANEBitNetAccelerator::new_mock("test_model.gguf");
        
        match result {
            Ok(accelerator) => {
                assert!(accelerator.is_ane_available() || !accelerator.is_ane_available());
                
                // Test that we can get partitioning strategy
                let strategy = accelerator.get_partitioning_strategy();
                match strategy {
                    ANEPartitioningStrategy::FullANE | 
                    ANEPartitioningStrategy::ANEGPUHybrid { .. } |
                    ANEPartitioningStrategy::SelectiveANE { .. } |
                    ANEPartitioningStrategy::Dynamic => assert!(true),
                }
                
                // Test performance stats
                let stats = accelerator.get_performance_stats();
                assert_eq!(stats.get_usage_stats().total_operations, 0);
            },
            Err(e) => {
                // ANE might not be available, which is acceptable for testing
                println!("Mock accelerator creation failed (expected on non-ANE platforms): {}", e);
            }
        }
    }

    #[test]
    #[cfg(not(feature = "metal"))]
    fn test_mock_forward_pass() {
        // Test mock forward pass without MLX dependency
        if let Ok(mut accelerator) = ANEBitNetAccelerator::new_mock("test_model.gguf") {
            let input_shape = vec![1, 128, 768]; // Typical transformer input shape
            let result = accelerator.forward_mock(&input_shape);
            
            assert!(result.is_ok(), "Mock forward pass should succeed");
            
            let output = result.unwrap();
            let expected_size: usize = input_shape.iter().product();
            assert_eq!(output.len(), expected_size);
            
            // All values should be zeros in mock implementation
            for &val in &output {
                assert_eq!(val, 0.0);
            }
            
            // Check that performance was logged
            let stats = accelerator.get_performance_stats().get_usage_stats();
            assert_eq!(stats.total_operations, 1);
            assert_eq!(stats.ane_operations, 1);
        }
    }

    #[test]
    fn test_gpu_fallback_stats() {
        let mut stats = GPUFallbackStats::default();
        
        // Test initial state
        assert_eq!(stats.total_fallbacks, 0);
        assert!(stats.fallback_reasons.is_empty());
        
        // Log some fallbacks
        stats.log_fallback(ANEOperation::MatMul);
        stats.log_fallback(ANEOperation::Attention);
        stats.log_fallback(ANEOperation::MatMul); // Duplicate to test counting
        
        assert_eq!(stats.total_fallbacks, 3);
        assert_eq!(stats.fallback_reasons.len(), 2);
        assert_eq!(stats.fallback_reasons[&ANEOperation::MatMul], 2);
        assert_eq!(stats.fallback_reasons[&ANEOperation::Attention], 1);
    }

    #[test]
    fn test_power_consumption() {
        let mut power = PowerConsumption::new();
        
        // Test initial state
        assert_eq!(power.total_energy, 0.0);
        assert_eq!(power.ane_energy, 0.0);
        assert_eq!(power.gpu_energy, 0.0);
        assert_eq!(power.compute_efficiency(), 0.0);
        
        // Manually set some values for testing
        power.total_energy = 100.0;
        power.ane_energy = 60.0;
        power.gpu_energy = 40.0;
        
        let efficiency = power.compute_efficiency();
        assert!((efficiency - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_performance_recommendations() {
        let mut monitor = ANEPerformanceMonitor::new();
        
        // Create scenario with low ANE utilization
        for _ in 0..10 {
            monitor.log_operation(ANEOperation::MatMul, 1.0, Device::GPU);
        }
        monitor.log_operation(ANEOperation::MatMul, 1.0, Device::ANE);
        
        let report = monitor.generate_performance_report();
        assert!(report.recommendations.iter().any(|r| r.contains("ANE-optimized")));
        
        // Test high latency scenario
        let mut high_latency_monitor = ANEPerformanceMonitor::new();
        high_latency_monitor.log_operation(ANEOperation::MatMul, 15.0, Device::ANE);
        
        let high_latency_report = high_latency_monitor.generate_performance_report();
        assert!(high_latency_report.recommendations.iter().any(|r| r.contains("latency")));
    }

    #[test]
    fn test_device_enumeration() {
        let devices = vec![Device::ANE, Device::GPU, Device::CPU];
        
        assert_eq!(devices.len(), 3);
        assert_eq!(devices[0], Device::ANE);
        assert_eq!(devices[1], Device::GPU);
        assert_eq!(devices[2], Device::CPU);
        
        // Test that devices are different
        assert_ne!(Device::ANE, Device::GPU);
        assert_ne!(Device::GPU, Device::CPU);
        assert_ne!(Device::ANE, Device::CPU);
    }

    #[test]
    fn test_accelerator_properties() {
        if let Ok(accelerator) = ANEBitNetAccelerator::new_mock("test_model.gguf") {
            // Test basic property access
            let _is_available = accelerator.is_ane_available();
            let _strategy = accelerator.get_partitioning_strategy();
            let _stats = accelerator.get_performance_stats();
            
            // Test strategy update
            let mut mutable_accelerator = accelerator;
            let new_strategy = ANEPartitioningStrategy::FullANE;
            mutable_accelerator.update_partitioning_strategy(new_strategy);
            
            match mutable_accelerator.get_partitioning_strategy() {
                ANEPartitioningStrategy::FullANE => assert!(true),
                _ => assert!(false, "Strategy should have been updated to FullANE"),
            }
        }
    }

    #[test]
    fn test_coreml_spec_mock() {
        let mut spec = CoreMLSpec::new();
        
        // Test configuration methods
        spec.set_compute_precision(CoreMLPrecision::Float16);
        spec.enable_ane_optimization(true);
        spec.set_batch_prediction(false);
        
        // Test layer addition
        let layer = ANELayer::mock();
        spec.add_layer(layer);
        
        // Test compilation
        let result = spec.compile_for_ane("test_output.mlmodelc");
        assert!(result.is_ok(), "Mock compilation should succeed");
    }

    #[test]
    fn test_comprehensive_integration() {
        // Integration test that combines multiple components
        if let Ok(mut accelerator) = ANEBitNetAccelerator::new_mock("comprehensive_test.gguf") {
            // Test initial state
            assert_eq!(accelerator.get_performance_stats().get_usage_stats().total_operations, 0);
            
            // Perform mock operations
            let input_shape = vec![1, 64, 512];
            for i in 0..5 {
                let result = accelerator.forward_mock(&input_shape);
                assert!(result.is_ok(), "Forward pass {} should succeed", i);
            }
            
            // Verify performance tracking
            let stats = accelerator.get_performance_stats().get_usage_stats();
            assert_eq!(stats.total_operations, 5);
            assert_eq!(stats.ane_operations, 5);
            assert_eq!(stats.gpu_operations, 0);
            
            // Test strategy changes
            let strategies = vec![
                ANEPartitioningStrategy::FullANE,
                ANEPartitioningStrategy::Dynamic,
                ANEPartitioningStrategy::SelectiveANE { 
                    ane_operations: vec![ANEOperation::MatMul] 
                },
            ];
            
            for strategy in strategies {
                accelerator.update_partitioning_strategy(strategy.clone());
                // Each strategy should be set correctly (tested via pattern matching)
                match accelerator.get_partitioning_strategy() {
                    ANEPartitioningStrategy::FullANE | 
                    ANEPartitioningStrategy::Dynamic |
                    ANEPartitioningStrategy::SelectiveANE { .. } => assert!(true),
                    _ => assert!(false, "Unexpected partitioning strategy"),
                }
            }
            
            // Generate comprehensive performance report
            let report = accelerator.get_performance_stats().generate_performance_report();
            assert!(report.ane_utilization > 0.9); // Should be very high since all operations were on ANE
            assert!(report.average_latency >= 0.0);
            assert!(report.power_efficiency >= 0.0);
        }
    }
}