// Re-export orchestration module functionality
#[path = "../automation/container-orchestration.rs"]
mod container_orchestration;

pub use container_orchestration::*;