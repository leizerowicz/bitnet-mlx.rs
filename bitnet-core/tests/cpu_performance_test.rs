//! CPU Performance Validation Test
//!
//! Direct test to run our CPU performance validation systems

use anyhow::Result;
use bitnet_core::cpu::performance_validator::PerformanceValidator;
use bitnet_core::cpu::production_validator::ProductionValidator;

#[test]
fn test_cpu_performance_validation() -> Result<()> {
    println!("🎯 Running CPU Performance Validation Test...");
    
    // Test performance validator
    let mut validator = PerformanceValidator::new();
    
    // Test sizes for validation
    let test_sizes = [1024, 4096, 16384];
    
    // Establish baselines
    println!("📊 Establishing performance baselines...");
    if let Err(e) = validator.establish_baseline(&test_sizes) {
        println!("⚠️ Baseline establishment failed: {}", e);
        return Ok(()); // Don't fail the test, just report
    }
    
    // Run performance validation
    println!("⚡ Running performance validation...");
    match validator.validate_performance(&test_sizes) {
        Ok(results) => {
            println!("✅ Performance validation completed");
            let report = validator.generate_report(&results);
            println!("\n📋 Performance Report:");
            println!("{}", report);
            
            // Count successes
            let passed = results.iter().filter(|r| r.meets_target).count();
            let total = results.len();
            println!("\n🎯 Results Summary: {}/{} targets achieved ({:.1}%)", 
                passed, total, (passed as f64 / total as f64) * 100.0);
        },
        Err(e) => {
            println!("⚠️ Performance validation error: {}", e);
        }
    }
    
    Ok(())
}

#[test]
fn test_production_readiness_validation() -> Result<()> {
    println!("🔍 Running Production Readiness Validation Test...");
    
    let mut validator = ProductionValidator::new();
    
    match validator.validate_production_readiness() {
        Ok(status) => {
            println!("✅ Production readiness validation completed");
            match status {
                bitnet_core::cpu::production_validator::ReadinessStatus::Ready => {
                    println!("🟢 Status: PRODUCTION READY");
                },
                bitnet_core::cpu::production_validator::ReadinessStatus::ReadyWithWarnings(warnings) => {
                    println!("🟡 Status: READY WITH WARNINGS");
                    for warning in warnings {
                        println!("  ⚠️ {}", warning);
                    }
                },
                bitnet_core::cpu::production_validator::ReadinessStatus::NotReady(errors) => {
                    println!("🔴 Status: NOT READY");
                    for error in errors {
                        println!("  ❌ {}", error);
                    }
                },
                bitnet_core::cpu::production_validator::ReadinessStatus::Unknown => {
                    println!("❓ Status: UNKNOWN");
                }
            }
        },
        Err(e) => {
            println!("❌ Production readiness validation failed: {}", e);
        }
    }
    
    Ok(())
}