//! Error Context Management
//!
//! This module provides utilities for adding rich context information to errors,
//! making debugging and error analysis more effective.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Rich context information for errors
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ErrorContext {
    /// Key-value pairs of context information
    pub data: HashMap<String, String>,
    /// Stack trace information (if available)
    pub stack_trace: Option<String>,
    /// Source location information
    pub source_location: Option<SourceLocation>,
    /// Related operation or component
    pub operation: Option<String>,
    /// User-friendly description
    pub description: Option<String>,
}

/// Source code location information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct SourceLocation {
    /// File name
    pub file: String,
    /// Line number
    pub line: u32,
    /// Column number (if available)
    pub column: Option<u32>,
    /// Function or method name (if available)
    pub function: Option<String>,
}

impl ErrorContext {
    /// Creates a new empty error context
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            stack_trace: None,
            source_location: None,
            operation: None,
            description: None,
        }
    }

    /// Adds a key-value pair to the context
    pub fn add(&mut self, key: &str, value: &str) {
        self.data.insert(key.to_string(), value.to_string());
    }

    /// Adds multiple key-value pairs to the context
    pub fn add_multiple(&mut self, pairs: &[(&str, &str)]) {
        for (key, value) in pairs {
            self.add(key, value);
        }
    }

    /// Sets the operation name
    pub fn set_operation(&mut self, operation: &str) {
        self.operation = Some(operation.to_string());
    }

    /// Sets the user-friendly description
    pub fn set_description(&mut self, description: &str) {
        self.description = Some(description.to_string());
    }

    /// Sets the source location
    pub fn set_source_location(&mut self, location: SourceLocation) {
        self.source_location = Some(location);
    }

    /// Sets the stack trace
    pub fn set_stack_trace(&mut self, stack_trace: String) {
        self.stack_trace = Some(stack_trace);
    }

    /// Gets a value from the context
    pub fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }

    /// Checks if the context is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
            && self.stack_trace.is_none()
            && self.source_location.is_none()
            && self.operation.is_none()
            && self.description.is_none()
    }

    /// Merges another context into this one
    pub fn merge(&mut self, other: &ErrorContext) {
        for (key, value) in &other.data {
            self.data.insert(key.clone(), value.clone());
        }

        if other.stack_trace.is_some() {
            self.stack_trace = other.stack_trace.clone();
        }

        if other.source_location.is_some() {
            self.source_location = other.source_location.clone();
        }

        if other.operation.is_some() {
            self.operation = other.operation.clone();
        }

        if other.description.is_some() {
            self.description = other.description.clone();
        }
    }

    /// Returns all context keys
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }

    /// Returns the number of context entries
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();

        if let Some(ref operation) = self.operation {
            parts.push(format!("operation={}", operation));
        }

        if let Some(ref description) = self.description {
            parts.push(format!("description=\"{}\"", description));
        }

        for (key, value) in &self.data {
            parts.push(format!("{}={}", key, value));
        }

        if let Some(ref location) = self.source_location {
            parts.push(format!("location={}:{}", location.file, location.line));
            if let Some(ref function) = location.function {
                parts.push(format!("function={}", function));
            }
        }

        write!(f, "{}", parts.join(", "))
    }
}

/// Builder for creating error contexts
#[allow(dead_code)]
pub struct ErrorContextBuilder {
    context: ErrorContext,
}

impl ErrorContextBuilder {
    /// Creates a new context builder
    pub fn new() -> Self {
        Self {
            context: ErrorContext::new(),
        }
    }

    /// Adds a key-value pair
    pub fn add(mut self, key: &str, value: &str) -> Self {
        self.context.add(key, value);
        self
    }

    /// Sets the operation
    pub fn operation(mut self, operation: &str) -> Self {
        self.context.set_operation(operation);
        self
    }

    /// Sets the description
    pub fn description(mut self, description: &str) -> Self {
        self.context.set_description(description);
        self
    }

    /// Sets the source location
    pub fn source_location(mut self, file: &str, line: u32) -> Self {
        self.context.set_source_location(SourceLocation {
            file: file.to_string(),
            line,
            column: None,
            function: None,
        });
        self
    }

    /// Sets the source location with function
    pub fn source_location_with_function(mut self, file: &str, line: u32, function: &str) -> Self {
        self.context.set_source_location(SourceLocation {
            file: file.to_string(),
            line,
            column: None,
            function: Some(function.to_string()),
        });
        self
    }

    /// Builds the context
    pub fn build(self) -> ErrorContext {
        self.context
    }
}

impl Default for ErrorContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can provide error context
pub trait ContextualError {
    /// Adds context to the error
    fn with_context(self, context: ErrorContext) -> Self;

    /// Adds a simple context message
    fn with_context_msg(self, message: &str) -> Self;

    /// Adds operation context
    fn with_operation(self, operation: &str) -> Self;
}

/// Macro for creating error context with source location
#[macro_export]
macro_rules! error_context {
    () => {
        $crate::error::context::ErrorContextBuilder::new()
            .source_location(file!(), line!())
            .build()
    };
    ($operation:expr) => {
        $crate::error::context::ErrorContextBuilder::new()
            .operation($operation)
            .source_location(file!(), line!())
            .build()
    };
    ($operation:expr, $description:expr) => {
        $crate::error::context::ErrorContextBuilder::new()
            .operation($operation)
            .description($description)
            .source_location(file!(), line!())
            .build()
    };
}

/// Macro for adding context to results
#[macro_export]
macro_rules! with_context {
    ($result:expr, $context:expr) => {
        $result.map_err(|e| e.with_context($context))
    };
    ($result:expr, $operation:expr, $description:expr) => {
        $result.map_err(|e| {
            let context = $crate::error::context::ErrorContextBuilder::new()
                .operation($operation)
                .description($description)
                .source_location(file!(), line!())
                .build();
            e.with_context(context)
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let mut context = ErrorContext::new();
        assert!(context.is_empty());

        context.add("key1", "value1");
        context.add("key2", "value2");

        assert!(!context.is_empty());
        assert_eq!(context.len(), 2);
        assert_eq!(context.get("key1"), Some(&"value1".to_string()));
    }

    #[test]
    fn test_context_builder() {
        let context = ErrorContextBuilder::new()
            .add("device", "Metal")
            .add("operation", "buffer_allocation")
            .operation("memory_management")
            .description("Failed to allocate GPU buffer")
            .source_location("memory.rs", 42)
            .build();

        assert_eq!(context.get("device"), Some(&"Metal".to_string()));
        assert_eq!(context.operation, Some("memory_management".to_string()));
        assert_eq!(
            context.description,
            Some("Failed to allocate GPU buffer".to_string())
        );
        assert!(context.source_location.is_some());
    }

    #[test]
    fn test_context_merge() {
        let mut context1 = ErrorContext::new();
        context1.add("key1", "value1");
        context1.set_operation("op1");

        let mut context2 = ErrorContext::new();
        context2.add("key2", "value2");
        context2.set_description("desc2");

        context1.merge(&context2);

        assert_eq!(context1.get("key1"), Some(&"value1".to_string()));
        assert_eq!(context1.get("key2"), Some(&"value2".to_string()));
        assert_eq!(context1.operation, Some("op1".to_string()));
        assert_eq!(context1.description, Some("desc2".to_string()));
    }

    #[test]
    fn test_context_display() {
        let context = ErrorContextBuilder::new()
            .operation("test_operation")
            .add("device", "CPU")
            .add("size", "1024")
            .build();

        let display_str = format!("{}", context);
        assert!(display_str.contains("operation=test_operation"));
        assert!(display_str.contains("device=CPU"));
        assert!(display_str.contains("size=1024"));
    }

    #[test]
    fn test_source_location() {
        let location = SourceLocation {
            file: "test.rs".to_string(),
            line: 42,
            column: Some(10),
            function: Some("test_function".to_string()),
        };

        let mut context = ErrorContext::new();
        context.set_source_location(location);

        assert!(context.source_location.is_some());
        let loc = context.source_location.as_ref().unwrap();
        assert_eq!(loc.file, "test.rs");
        assert_eq!(loc.line, 42);
        assert_eq!(loc.function, Some("test_function".to_string()));
    }
}
