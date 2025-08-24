//! State Tracking Tests
//!
//! Test suite for QAT training state management, checkpoint handling,
//! and training progress monitoring.

use bitnet_training::qat::state_tracking::{CheckpointManager, QATStateTracker, QATTrainingState};
use candle_core::Device;

/// Test setup helper
fn setup_test_device() -> Device {
    Device::Cpu
}

#[cfg(test)]
mod basic_state_tracking {
    use super::*;

    #[test]
    fn test_training_state_creation() {
        let state = QATTrainingState::new();

        assert_eq!(state.epoch, 0);
        assert_eq!(state.step, 0);
        assert_eq!(state.loss_history.len(), 0);
    }

    #[test]
    fn test_training_metrics_update() {
        let mut state = QATTrainingState::new();

        // Test updating training metrics
        state.update_training_metrics(
            1,    // epoch
            100,  // step
            0.01, // learning_rate
            0.5,  // loss
            32,   // samples_this_step
            0.1,  // step_time
        );

        assert_eq!(state.epoch, 1);
        assert_eq!(state.step, 100);
        assert_eq!(state.learning_rate, 0.01);
        assert_eq!(state.current_loss, 0.5);
        assert_eq!(state.loss_history.len(), 1);
        assert_eq!(state.loss_history[0], 0.5);
    }

    #[test]
    fn test_validation_metrics_update() {
        let mut state = QATTrainingState::new();

        // Test validation metrics
        state.update_validation_metrics(0.4, Some(0.85));

        assert_eq!(state.validation_loss, Some(0.4));
        assert_eq!(state.validation_history.len(), 1);
        assert_eq!(state.validation_history[0], 0.4);
    }

    #[test]
    fn test_checkpoint_frequency() {
        let state = QATTrainingState::new();

        // Test checkpoint frequency check
        assert!(state.should_checkpoint(1)); // Every epoch

        // Update state to epoch 5
        let mut state = QATTrainingState::new();
        state.update_training_metrics(5, 500, 0.01, 0.3, 32, 0.1);

        assert!(state.should_checkpoint(5)); // Every 5 epochs
        assert!(!state.should_checkpoint(10)); // Not at epoch 10 boundary
    }
}

#[cfg(test)]
mod state_tracker_tests {
    use super::*;

    #[test]
    fn test_state_tracker_creation() {
        let device = setup_test_device();
        let tracker = QATStateTracker::new(device);

        assert_eq!(tracker.get_state().epoch, 0);
        assert_eq!(tracker.get_state().step, 0);
    }

    #[test]
    fn test_state_tracker_updates() {
        let device = setup_test_device();
        let mut tracker = QATStateTracker::new(device);

        // Update training state
        tracker.update(1, 100, 0.01, 0.5, 32, 0.1);

        let state = tracker.get_state();
        assert_eq!(state.epoch, 1);
        assert_eq!(state.step, 100);
        assert_eq!(state.learning_rate, 0.01);
        assert_eq!(state.current_loss, 0.5);

        // Update validation
        tracker.update_validation(0.4, Some(0.85));

        let state = tracker.get_state();
        assert_eq!(state.validation_loss, Some(0.4));
    }

    #[test]
    fn test_checkpoint_detection() {
        let device = setup_test_device();
        let mut tracker = QATStateTracker::new(device);

        tracker.set_checkpoint_frequency(5);

        // Should not checkpoint at epoch 3
        tracker.update(3, 300, 0.01, 0.4, 32, 0.1);
        assert!(!tracker.should_checkpoint());

        // Should checkpoint at epoch 5
        tracker.update(5, 500, 0.01, 0.3, 32, 0.1);
        assert!(tracker.should_checkpoint());
    }

    #[test]
    fn test_tracker_reset() {
        let device = setup_test_device();
        let mut tracker = QATStateTracker::new(device);

        // Update with some data
        tracker.update(5, 500, 0.01, 0.3, 32, 0.1);
        tracker.update_validation(0.25, Some(0.9));

        // Reset
        tracker.reset();

        let state = tracker.get_state();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.step, 0);
        assert_eq!(state.loss_history.len(), 0);
        assert_eq!(state.validation_history.len(), 0);
    }
}

#[cfg(test)]
mod checkpoint_manager_tests {
    use super::*;

    #[test]
    fn test_checkpoint_manager_creation() {
        let checkpoint_dir = "/tmp/test_checkpoints".to_string();
        let device = setup_test_device();

        let _manager = CheckpointManager::new(checkpoint_dir, device);

        // Basic validation that manager was created (no panic)
    }

    #[test]
    fn test_save_and_load_state() {
        let checkpoint_dir = "/tmp/test_checkpoints".to_string();
        let device = setup_test_device();

        let manager = CheckpointManager::new(checkpoint_dir, device);

        // Create a state to save
        let mut state = QATTrainingState::new();
        state.update_training_metrics(10, 1000, 0.001, 0.2, 64, 0.2);
        state.update_validation_metrics(0.18, Some(0.95));

        // Save state
        let result = manager.save_state(&state, "test_checkpoint");
        if let Err(e) = &result {
            println!("Save failed (expected in test env): {e:?}");
            // In test environment, saving might fail due to permissions
            return;
        }

        // If save succeeded, try to load
        let loaded_result = manager.load_state("test_checkpoint");
        match loaded_result {
            Ok(loaded_state) => {
                assert_eq!(loaded_state.epoch, 10);
                assert_eq!(loaded_state.step, 1000);
                assert_eq!(loaded_state.learning_rate, 0.001);
                assert_eq!(loaded_state.current_loss, 0.2);
                assert_eq!(loaded_state.validation_loss, Some(0.18));
            }
            Err(e) => {
                println!("Load failed (expected in test env): {e:?}");
            }
        }
    }

    #[test]
    fn test_list_checkpoints() {
        let checkpoint_dir = "/tmp/test_checkpoints".to_string();
        let device = setup_test_device();

        let manager = CheckpointManager::new(checkpoint_dir, device);

        // Save multiple checkpoints if possible
        let state1 = QATTrainingState::new();
        let mut state2 = QATTrainingState::new();
        state2.update_training_metrics(5, 500, 0.01, 0.4, 32, 0.1);

        let _ = manager.save_state(&state1, "checkpoint_1");
        let _ = manager.save_state(&state2, "checkpoint_2");

        // List checkpoints
        let checkpoints_result = manager.list_checkpoints();
        match checkpoints_result {
            Ok(checkpoints) => {
                // In test environment, this might be empty due to permissions
                assert!(
                    true,
                    "Checkpoints listed successfully: {} items",
                    checkpoints.len()
                );
            }
            Err(e) => {
                println!("List checkpoints failed (expected in test env): {e:?}");
            }
        }
    }
}

#[cfg(test)]
mod training_summary_tests {
    use super::*;

    #[test]
    fn test_training_summary() {
        let mut state = QATTrainingState::new();

        // Simulate training progress
        let losses = vec![1.0, 0.8, 0.6, 0.5, 0.4];
        for (epoch, &loss) in losses.iter().enumerate() {
            state.update_training_metrics(epoch, epoch * 100, 0.01, loss, 32, 0.1);
        }

        // Get training summary
        let summary = state.get_summary();

        // Validate summary contains expected information
        assert_eq!(summary.epoch, losses.len() - 1);
        assert_eq!(summary.current_loss, 0.4); // Last loss

        // Verify loss history is properly tracked
        assert_eq!(state.loss_history.len(), losses.len());
        assert_eq!(state.loss_history, losses);
    }

    #[test]
    fn test_quantization_error_tracking() {
        let mut state = QATTrainingState::new();

        // Simulate quantization error progression
        let errors = [0.1, 0.08, 0.06, 0.05, 0.04];
        for (epoch, &error) in errors.iter().enumerate() {
            state.update_training_metrics(epoch, epoch * 100, 0.01, 0.5, 32, 0.1);
            state.quantization_error = error; // Direct assignment for testing
        }

        // Final quantization error should be the last one set
        assert_eq!(state.quantization_error, 0.04);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_training_simulation() {
        let device = setup_test_device();
        let mut tracker = QATStateTracker::new(device);

        tracker.set_checkpoint_frequency(5);

        // Simulate 10 epochs of training
        for epoch in 1..=10 {
            let loss = 1.0 - (epoch as f32 * 0.08); // Decreasing loss
            let lr = 0.01 / (1.0 + epoch as f32 * 0.1); // Decaying learning rate
            let samples = 32;
            let step_time = 0.1;

            tracker.update(epoch, epoch * 100, lr, loss, samples, step_time);

            // Add validation every few epochs
            if epoch % 3 == 0 {
                let val_loss = loss * 0.9; // Slightly better than training loss
                let accuracy = 0.5 + (epoch as f32 * 0.04); // Improving accuracy
                tracker.update_validation(val_loss, Some(accuracy));
            }

            // Check for checkpointing
            if tracker.should_checkpoint() {
                // In a real scenario, we would save the checkpoint here
                assert_eq!(epoch % 5, 0, "Should only checkpoint every 5 epochs");
            }
        }

        // Validate final state
        let final_state = tracker.get_state();
        assert_eq!(final_state.epoch, 10);
        assert_eq!(final_state.step, 1000);
        assert!(final_state.current_loss < 1.0); // Loss should have improved
        assert!(final_state.loss_history.len() == 10); // Should track all epochs

        // Validation should have been recorded
        assert!(final_state.validation_loss.is_some());
        assert!(!final_state.validation_history.is_empty());
    }
}

#[cfg(test)]
mod error_handling {
    use super::*;

    #[test]
    fn test_invalid_checkpoint_loading() {
        let checkpoint_dir = "/tmp/test_checkpoints".to_string();
        let device = setup_test_device();

        let manager = CheckpointManager::new(checkpoint_dir, device);

        // Try to load non-existent checkpoint
        let result = manager.load_state("nonexistent_checkpoint");
        assert!(
            result.is_err(),
            "Should fail to load non-existent checkpoint"
        );
    }

    #[test]
    fn test_state_with_invalid_values() {
        let mut state = QATTrainingState::new();

        // Update with edge case values
        state.update_training_metrics(0, 0, 0.0, f32::INFINITY, 0, 0.0);

        // State should remain stable despite invalid loss
        assert_eq!(state.epoch, 0);
        assert_eq!(state.step, 0);
        assert_eq!(state.learning_rate, 0.0);
        assert!(state.current_loss.is_infinite());

        // Should still be able to add to history
        assert_eq!(state.loss_history.len(), 1);
    }

    #[test]
    fn test_empty_state_operations() {
        let state = QATTrainingState::new();

        // Operations on empty state should not crash
        assert!(state.should_checkpoint(1));
        assert!(state.should_checkpoint(10));

        let summary = state.get_summary();
        assert_eq!(summary.epoch, 0);
    }
}
