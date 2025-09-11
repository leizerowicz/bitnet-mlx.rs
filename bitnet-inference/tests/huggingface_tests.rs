//! Tests for HuggingFace integration functionality

use bitnet_inference::{HuggingFaceLoader, ModelRepo, HuggingFaceConfig};
use tempfile::TempDir;

#[tokio::test]
async fn test_huggingface_loader_creation() {
    let temp_dir = TempDir::new().unwrap();
    let config = HuggingFaceConfig {
        cache_dir: temp_dir.path().to_path_buf(),
        offline: true, // Offline mode for testing
        ..Default::default()
    };

    let _loader = HuggingFaceLoader::with_config(config).unwrap();
    // Successfully created - test passes
}

#[tokio::test]
async fn test_model_repo_creation() {
    let repo = ModelRepo::new("microsoft", "bitnet-b1.58-large");
    assert_eq!(repo.repo_id(), "microsoft/bitnet-b1.58-large");
    assert_eq!(repo.revision, None);

    let repo_with_revision = repo.with_revision("v1.0");
    assert_eq!(repo_with_revision.revision, Some("v1.0".to_string()));
}

#[tokio::test]
async fn test_cache_stats() {
    let temp_dir = TempDir::new().unwrap();
    let config = HuggingFaceConfig {
        cache_dir: temp_dir.path().to_path_buf(),
        offline: true,
        ..Default::default()
    };

    let loader = HuggingFaceLoader::with_config(config).unwrap();
    let stats = loader.cache_stats().await.unwrap();
    
    assert_eq!(stats.model_count, 0);
    assert_eq!(stats.total_size, 0);
}

#[tokio::test]
async fn test_offline_model_loading() {
    let temp_dir = TempDir::new().unwrap();
    let config = HuggingFaceConfig {
        cache_dir: temp_dir.path().to_path_buf(),
        offline: true,
        ..Default::default()
    };

    let loader = HuggingFaceLoader::with_config(config).unwrap();
    let repo = ModelRepo::new("test", "model");
    
    // Should fail in offline mode when model doesn't exist
    let result = loader.load_model(&repo).await;
    assert!(result.is_err());
}

#[test]
fn test_parse_repo_id() {
    // Test valid repo ID
    let repo_id = "microsoft/bitnet-b1.58-large";
    let parts: Vec<&str> = repo_id.split('/').collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], "microsoft");
    assert_eq!(parts[1], "bitnet-b1.58-large");
}

#[tokio::test]
async fn test_clear_cache() {
    let temp_dir = TempDir::new().unwrap();
    let config = HuggingFaceConfig {
        cache_dir: temp_dir.path().to_path_buf(),
        offline: true,
        ..Default::default()
    };

    let loader = HuggingFaceLoader::with_config(config).unwrap();
    
    // Clear cache should not fail even if cache is empty
    let result = loader.clear_cache().await;
    assert!(result.is_ok());
}
