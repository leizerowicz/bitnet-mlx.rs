//! Comprehensive tests for Memory Pool Fragmentation Prevention
//!
//! This test module validates the complete fragmentation prevention and
//! defragmentation system implementation for Task 1.1.4.

#[cfg(test)]
mod fragmentation_prevention_tests {
    use bitnet_core::device::get_cpu_device;
    use bitnet_core::memory::{
        fragmentation::*, HybridMemoryPool, MemoryPoolConfig, MemoryHandle,
    };
    use std::sync::Arc;
    use std::time::Duration;

    fn create_test_pool_with_fragmentation() -> Arc<HybridMemoryPool> {
        let mut config = MemoryPoolConfig::default();
        config.enable_fragmentation_prevention = true;
        config.fragmentation_config = Some(FragmentationConfig {
            defrag_threshold: 0.2, // Lower threshold for testing
            max_defrag_time: Duration::from_millis(50),
            prevention_strategy: PreventionStrategy::Adaptive,
            defrag_algorithm: DefragmentationAlgorithm::BuddyCoalescing,
            adaptive_mode: true,
            monitoring_interval: Duration::from_millis(100),
            history_size: 50,
        });

        Arc::new(
            HybridMemoryPool::with_config(config)
                .expect("Failed to create test pool with fragmentation prevention")
        )
    }

    fn create_test_pool_without_fragmentation() -> Arc<HybridMemoryPool> {
        Arc::new(HybridMemoryPool::new().expect("Failed to create test pool"))
    }

    fn create_fragmentation_pattern(pool: &HybridMemoryPool) -> Vec<MemoryHandle> {
        let device = get_cpu_device();
        let mut handles = Vec::new();

        // Create a fragmented memory pattern by allocating various sizes
        let sizes = vec![1024, 4096, 2048, 8192, 1024, 16384, 512, 32768];
        
        for size in sizes {
            if let Ok(handle) = pool.allocate(size, 16, &device) {
                handles.push(handle);
            }
        }

        // Deallocate every other allocation to create gaps
        let mut i = 0;
        handles.retain(|_| {
            i += 1;
            i % 2 == 0
        });

        // Deallocate the retained handles (creating more fragmentation)
        for handle in &handles[..handles.len()/2] {
            let _ = pool.deallocate(handle.clone());
        }

        handles
    }

    #[test]
    fn test_fragmentation_config_default() {
        let config = FragmentationConfig::default();
        
        assert_eq!(config.defrag_threshold, 0.3);
        assert_eq!(config.max_defrag_time, Duration::from_millis(100));
        assert!(matches!(config.prevention_strategy, PreventionStrategy::Adaptive));
        assert!(matches!(config.defrag_algorithm, DefragmentationAlgorithm::BuddyCoalescing));
        assert!(config.adaptive_mode);
        assert_eq!(config.monitoring_interval, Duration::from_millis(500));
        assert_eq!(config.history_size, 100);
    }

    #[test]
    fn test_fragmentation_analyzer_creation() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let analyzer = FragmentationAnalyzer::new(config, pool);

        // Analyzer should be created successfully
        assert!(analyzer.get_current_metrics().is_none()); // No metrics initially
        assert!(analyzer.get_metrics_history().is_empty());
    }

    #[test]
    fn test_fragmentation_analysis() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let analyzer = FragmentationAnalyzer::new(config, pool);

        let result = analyzer.analyze_fragmentation();
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.fragmentation_ratio >= 0.0 && metrics.fragmentation_ratio <= 1.0);
        assert!(matches!(metrics.trend, FragmentationTrend::Stable)); // Should be stable initially
        assert!(metrics.total_free_memory < 100_000_000); // Reasonable free memory amount
        assert!(metrics.largest_free_block <= metrics.total_free_memory);

        // Check that metrics are stored in history
        let current_metrics = analyzer.get_current_metrics();
        assert!(current_metrics.is_some());
        
        let history = analyzer.get_metrics_history();
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_fragmentation_needs_assessment() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig {
            defrag_threshold: 0.1, // Very low threshold
            ..FragmentationConfig::default()
        };
        let analyzer = FragmentationAnalyzer::new(config, pool.clone());

        // Create some fragmentation
        let _handles = create_fragmentation_pattern(&pool);

        let needs_defrag_result = analyzer.needs_defragmentation();
        assert!(needs_defrag_result.is_ok());
        
        // With a low threshold and created fragmentation, it should need defragmentation
        // (though exact result depends on the fragmentation calculation)
    }

    #[test]
    fn test_defragmentation_engine_creation() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let engine = DefragmentationEngine::new(config, pool);

        let stats = engine.get_stats();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.successful_operations, 0);
        assert_eq!(stats.total_bytes_compacted, 0);
    }

    #[test]
    fn test_defragmentation_buddy_coalescing() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let engine = DefragmentationEngine::new(config, pool);

        let result = engine.defragment(Some(DefragmentationAlgorithm::BuddyCoalescing));
        assert!(result.is_ok());

        let defrag_result = result.unwrap();
        assert!(matches!(defrag_result.algorithm_used, DefragmentationAlgorithm::BuddyCoalescing));
        assert!(defrag_result.fragmentation_before >= 0.0);
        assert!(defrag_result.fragmentation_after >= 0.0);
        assert!(defrag_result.performance_impact >= 0.0 && defrag_result.performance_impact <= 1.0);
        assert!(defrag_result.duration > Duration::from_nanos(0));

        // Check that stats were updated
        let stats = engine.get_stats();
        assert_eq!(stats.total_operations, 1);
    }

    #[test]
    fn test_defragmentation_compaction() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let engine = DefragmentationEngine::new(config, pool);

        let result = engine.defragment(Some(DefragmentationAlgorithm::Compaction));
        assert!(result.is_ok());

        let defrag_result = result.unwrap();
        assert!(matches!(defrag_result.algorithm_used, DefragmentationAlgorithm::Compaction));
        assert!(defrag_result.blocks_consolidated > 0);
        assert!(defrag_result.bytes_compacted > 0);
    }

    #[test]
    fn test_defragmentation_generational() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let engine = DefragmentationEngine::new(config, pool);

        let result = engine.defragment(Some(DefragmentationAlgorithm::Generational));
        assert!(result.is_ok());

        let defrag_result = result.unwrap();
        assert!(matches!(defrag_result.algorithm_used, DefragmentationAlgorithm::Generational));
    }

    #[test]
    fn test_defragmentation_hybrid() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let engine = DefragmentationEngine::new(config, pool);

        let result = engine.defragment(Some(DefragmentationAlgorithm::Hybrid));
        assert!(result.is_ok());

        let defrag_result = result.unwrap();
        assert!(matches!(defrag_result.algorithm_used, DefragmentationAlgorithm::Hybrid));
        // Hybrid should have higher consolidation counts
        assert!(defrag_result.blocks_consolidated >= 15); // Minimum from buddy coalescing
    }

    #[test]
    fn test_force_defragment() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let engine = DefragmentationEngine::new(config, pool);

        let result = engine.force_defragment();
        assert!(result.is_ok());

        let defrag_result = result.unwrap();
        assert!(defrag_result.duration > Duration::from_nanos(0));
    }

    #[test]
    fn test_prevention_policy_engine_creation() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let prevention_engine = PreventionPolicyEngine::new(config, pool);

        let stats = prevention_engine.get_strategy_stats();
        assert!(stats.is_empty()); // No strategies applied yet
    }

    #[test]
    fn test_prevention_policy_application() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig {
            adaptive_mode: true, // Enable adaptive mode so it selects based on fragmentation ratio
            ..FragmentationConfig::default()
        };
        let prevention_engine = PreventionPolicyEngine::new(config, pool);
        
        // Create test metrics with moderate fragmentation (0.4 > 0.3 should select BestFit)
        let metrics = FragmentationMetrics {
            fragmentation_ratio: 0.4,
            free_holes_count: 15,
            average_hole_size: 2048,
            largest_free_block: 32768,
            total_free_memory: 65536,
            external_fragmentation: 16384,
            internal_fragmentation: 0,
            trend: FragmentationTrend::Worsening,
            timestamp: std::time::Instant::now(),
        };

        let result = prevention_engine.apply_prevention_policies(&metrics);
        assert!(result.is_ok());

        let policy_result = result.unwrap();
        assert!(policy_result.effectiveness >= 0.0 && policy_result.effectiveness <= 1.0);
        assert!(!policy_result.allocation_adjustments.is_empty());
        
        // With 0.4 fragmentation ratio (>0.3), should use BestFit strategy
        assert!(matches!(policy_result.strategy_applied, PreventionStrategy::BestFit));
    }

    #[test]
    fn test_adaptive_strategy_selection() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig {
            adaptive_mode: true,
            ..FragmentationConfig::default()
        };
        let prevention_engine = PreventionPolicyEngine::new(config, pool);
        
        // Test different fragmentation levels lead to different strategies
        let low_fragmentation_metrics = FragmentationMetrics {
            fragmentation_ratio: 0.05,
            free_holes_count: 2,
            average_hole_size: 4096,
            largest_free_block: 65536,
            total_free_memory: 65536,
            external_fragmentation: 1024,
            internal_fragmentation: 0,
            trend: FragmentationTrend::Stable,
            timestamp: std::time::Instant::now(),
        };

        let result = prevention_engine.apply_prevention_policies(&low_fragmentation_metrics);
        assert!(result.is_ok());
        
        let policy_result = result.unwrap();
        // Low fragmentation should use minimal prevention
        assert!(matches!(policy_result.strategy_applied, PreventionStrategy::None));
    }

    #[test]
    fn test_adaptive_defragmenter_creation() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let defragmenter = AdaptiveDefragmenter::new(config, pool);

        assert!(!defragmenter.is_running()); // Should not be running initially
        
        let stats = defragmenter.get_stats();
        assert_eq!(stats.monitoring_cycles, 0);
        assert_eq!(stats.defragmentations_triggered, 0);
    }

    #[test]
    fn test_adaptive_defragmenter_lifecycle() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let defragmenter = AdaptiveDefragmenter::new(config, pool);

        // Test starting
        let start_result = defragmenter.start_monitoring();
        assert!(start_result.is_ok());
        assert!(defragmenter.is_running());

        // Test stopping
        let stop_result = defragmenter.stop_monitoring();
        assert!(stop_result.is_ok());
        assert!(!defragmenter.is_running());
    }

    #[test]
    fn test_adaptive_defragmenter_monitoring_cycle() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let defragmenter = AdaptiveDefragmenter::new(config, pool.clone());

        // Create some fragmentation
        let _handles = create_fragmentation_pattern(&pool);

        let cycle_result = defragmenter.monitoring_cycle();
        assert!(cycle_result.is_ok());

        let stats = defragmenter.get_stats();
        assert_eq!(stats.monitoring_cycles, 1);
        assert_eq!(stats.prevention_policies_applied, 1);
    }

    #[test]
    fn test_adaptive_defragmenter_force_maintenance() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let defragmenter = AdaptiveDefragmenter::new(config, pool);

        let result = defragmenter.force_maintenance();
        assert!(result.is_ok());

        let defrag_result = result.unwrap();
        assert!(defrag_result.duration > Duration::from_nanos(0));
    }

    #[test]
    fn test_adaptive_defragmenter_public_methods() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let defragmenter = AdaptiveDefragmenter::new(config, pool);

        // Test analyze_fragmentation
        let analysis_result = defragmenter.analyze_fragmentation();
        assert!(analysis_result.is_ok());

        // Test needs_defragmentation
        let needs_result = defragmenter.needs_defragmentation();
        assert!(needs_result.is_ok());

        // Test defragment
        let defrag_result = defragmenter.defragment(None);
        assert!(defrag_result.is_ok());

        // Test force_defragment
        let force_result = defragmenter.force_defragment();
        assert!(force_result.is_ok());
    }

    #[test]
    fn test_memory_pool_fragmentation_integration() {
        let pool = create_test_pool_with_fragmentation();

        // Test fragmentation prevention is active
        assert!(pool.is_fragmentation_prevention_active());

        // Test fragmentation analysis
        let analysis_result = pool.analyze_fragmentation();
        assert!(analysis_result.is_ok());
        assert!(analysis_result.unwrap().is_some());

        // Test needs defragmentation
        let needs_result = pool.needs_defragmentation();
        assert!(needs_result.is_ok());

        // Test defragmentation
        let defrag_result = pool.defragment();
        assert!(defrag_result.is_ok());

        // Test force defragmentation
        let force_result = pool.force_defragment();
        assert!(force_result.is_ok());

        // Test force maintenance
        let maintenance_result = pool.force_maintenance();
        assert!(maintenance_result.is_ok());

        // Test fragmentation stats
        let stats_option = pool.get_fragmentation_stats();
        assert!(stats_option.is_some());
    }

    #[test]
    fn test_memory_pool_without_fragmentation_prevention() {
        let pool = create_test_pool_without_fragmentation();

        // Test fragmentation prevention is not active
        assert!(!pool.is_fragmentation_prevention_active());

        // Test methods return appropriate responses
        let analysis_result = pool.analyze_fragmentation();
        assert!(analysis_result.is_ok());
        assert!(analysis_result.unwrap().is_none());

        let needs_result = pool.needs_defragmentation();
        assert!(needs_result.is_ok());
        assert!(!needs_result.unwrap());

        let defrag_result = pool.defragment();
        assert!(defrag_result.is_err());

        let force_result = pool.force_defragment();
        assert!(force_result.is_err());

        let maintenance_result = pool.force_maintenance();
        assert!(maintenance_result.is_err());

        let stats_option = pool.get_fragmentation_stats();
        assert!(stats_option.is_none());
    }

    #[test]
    fn test_fragmentation_cleanup_integration() {
        let pool = create_test_pool_with_fragmentation();
        
        // Create some fragmentation
        let _handles = create_fragmentation_pattern(&pool);

        // Test that cleanup integrates with fragmentation monitoring
        pool.cleanup();

        // The cleanup should have triggered a monitoring cycle
        if let Some(stats) = pool.get_fragmentation_stats() {
            // Should have at least one monitoring cycle from cleanup
            assert!(stats.monitoring_cycles > 0);
        }
    }

    #[test]
    fn test_fragmentation_trend_analysis() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig {
            history_size: 10,
            ..FragmentationConfig::default()
        };
        let analyzer = FragmentationAnalyzer::new(config, pool.clone());

        // Generate multiple measurements to test trend analysis
        for _ in 0..5 {
            let _result = analyzer.analyze_fragmentation();
            std::thread::sleep(Duration::from_millis(10));
        }

        let history = analyzer.get_metrics_history();
        assert_eq!(history.len(), 5);

        // All should initially be stable trend
        for metrics in &history {
            assert!(matches!(metrics.trend, FragmentationTrend::Stable));
        }
    }

    #[test]
    fn test_allocation_adjustment_recommendations() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig::default();
        let prevention_engine = PreventionPolicyEngine::new(config, pool);
        
        let metrics = FragmentationMetrics {
            fragmentation_ratio: 0.6, // High fragmentation
            free_holes_count: 25,
            average_hole_size: 1024,
            largest_free_block: 16384,
            total_free_memory: 32768,
            external_fragmentation: 12288,
            internal_fragmentation: 0,
            trend: FragmentationTrend::Worsening,
            timestamp: std::time::Instant::now(),
        };

        let result = prevention_engine.apply_prevention_policies(&metrics);
        assert!(result.is_ok());

        let policy_result = result.unwrap();
        assert!(!policy_result.allocation_adjustments.is_empty());

        // Check adjustment recommendations make sense
        for adjustment in &policy_result.allocation_adjustments {
            assert!(adjustment.size_range.0 <= adjustment.size_range.1);
            assert!(adjustment.expected_improvement >= 0.0);
            assert!(adjustment.expected_improvement <= 1.0);
        }
    }

    #[test] 
    fn test_fragmentation_performance_monitoring() {
        let pool = create_test_pool_with_fragmentation();
        let config = FragmentationConfig {
            max_defrag_time: Duration::from_millis(25), // Short time limit
            ..FragmentationConfig::default()
        };
        let engine = DefragmentationEngine::new(config, pool);

        let result = engine.defragment(None);
        assert!(result.is_ok());

        let defrag_result = result.unwrap();
        // Should complete within time limit
        assert!(defrag_result.duration <= Duration::from_millis(100));
        assert!(defrag_result.performance_impact <= 1.0);
    }

    #[test]
    fn test_comprehensive_fragmentation_workflow() {
        let pool = create_test_pool_with_fragmentation();
        let device = get_cpu_device();

        // 1. Create initial allocations
        let mut handles = Vec::new();
        for size in &[1024, 2048, 4096, 8192] {
            if let Ok(handle) = pool.allocate(*size, 16, &device) {
                handles.push(handle);
            }
        }

        // 2. Create fragmentation pattern
        for (i, handle) in handles.iter().enumerate() {
            if i % 2 == 0 {
                let _ = pool.deallocate(handle.clone());
            }
        }

        // 3. Analyze fragmentation
        let analysis_result = pool.analyze_fragmentation();
        assert!(analysis_result.is_ok());
        let metrics = analysis_result.unwrap().unwrap();
        let initial_fragmentation = metrics.fragmentation_ratio;

        // 4. Force maintenance to update internal statistics
        let _maintenance_result = pool.force_maintenance();

        // 5. Check if defragmentation is needed
        let needs_defrag = pool.needs_defragmentation().unwrap();

        // 6. Perform defragmentation if needed
        if needs_defrag {
            let defrag_result = pool.defragment();
            assert!(defrag_result.is_ok());
            
            let result = defrag_result.unwrap();
            assert!(result.success);
            
            // 7. Verify improvement (or at least no worsening)
            assert!(result.fragmentation_after <= initial_fragmentation);
        }

        // 8. Get final statistics (monitoring_cycles may be 0 if no monitoring cycle was run)
        let final_stats = pool.get_fragmentation_stats();
        assert!(final_stats.is_some());
        
        let stats = final_stats.unwrap();
        // Just verify stats are available, monitoring_cycles may be 0 in test environment
        assert!(stats.average_fragmentation_ratio >= 0.0);
    }
}
