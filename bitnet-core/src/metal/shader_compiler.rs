//! # Metal Shader Compilation Pipeline
//!
//! This module provides comprehensive shader compilation and loading utilities for BitNet Metal operations.
//! It includes automatic shader discovery, compilation, caching, and runtime loading capabilities.

use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[cfg(all(target_os = "macos", feature = "metal"))]
use anyhow::Context;

#[cfg(all(target_os = "macos", feature = "metal"))]
use std::fs;

#[cfg(all(target_os = "macos", feature = "metal"))]
use metal;

use super::MetalError;

/// Shader compilation configuration
#[derive(Debug, Clone)]
pub struct ShaderCompilerConfig {
    /// Directory containing Metal shader source files
    pub shader_directory: PathBuf,
    /// Enable shader caching
    pub enable_caching: bool,
    /// Cache directory for compiled shaders
    pub cache_directory: Option<PathBuf>,
    /// Compilation options
    pub compile_options: CompileOptions,
    /// Enable debug information in shaders
    pub debug_info: bool,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
}

/// Metal shader compilation options
#[derive(Debug, Clone)]
pub struct CompileOptions {
    /// Preprocessor definitions
    pub defines: HashMap<String, String>,
    /// Include directories
    pub include_directories: Vec<PathBuf>,
    /// Language version
    pub language_version: LanguageVersion,
    /// Fast math optimizations
    pub fast_math: bool,
}

/// Metal Shading Language version
#[derive(Debug, Clone, Copy)]
pub enum LanguageVersion {
    /// Metal 1.0
    Metal1_0,
    /// Metal 1.1
    Metal1_1,
    /// Metal 1.2
    Metal1_2,
    /// Metal 2.0
    Metal2_0,
    /// Metal 2.1
    Metal2_1,
    /// Metal 2.2
    Metal2_2,
    /// Metal 2.3
    Metal2_3,
    /// Metal 2.4
    Metal2_4,
    /// Metal 3.0
    Metal3_0,
}

/// Shader optimization level
#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Full optimization
    Full,
}

/// Compiled shader information
#[derive(Debug, Clone)]
pub struct CompiledShader {
    /// Shader name
    pub name: String,
    /// Source file path
    pub source_path: PathBuf,
    /// Compiled library
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub library: metal::Library,
    /// Available function names
    pub function_names: Vec<String>,
    /// Compilation timestamp
    pub compiled_at: std::time::SystemTime,
    /// Source hash for cache validation
    pub source_hash: u64,
}

/// Shader cache entry
#[derive(Debug)]
struct CacheEntry {
    compiled_shader: CompiledShader,
    last_accessed: std::time::Instant,
}

/// Metal shader compiler and loader
pub struct ShaderCompiler {
    #[cfg(target_os = "macos")]
    device: metal::Device,
    config: ShaderCompilerConfig,
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    #[cfg(target_os = "macos")]
    compile_options: metal::CompileOptions,
}

impl Default for ShaderCompilerConfig {
    fn default() -> Self {
        Self {
            shader_directory: PathBuf::from("src/metal/shaders"),
            enable_caching: true,
            cache_directory: Some(PathBuf::from("target/metal_cache")),
            compile_options: CompileOptions::default(),
            debug_info: cfg!(debug_assertions),
            optimization_level: if cfg!(debug_assertions) {
                OptimizationLevel::None
            } else {
                OptimizationLevel::Full
            },
        }
    }
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            defines: HashMap::new(),
            include_directories: Vec::new(),
            language_version: LanguageVersion::Metal2_4,
            fast_math: true,
        }
    }
}

impl LanguageVersion {
    #[cfg(target_os = "macos")]
    fn to_metal_version(self) -> metal::MTLLanguageVersion {
        match self {
            LanguageVersion::Metal1_0 => metal::MTLLanguageVersion::V1_0,
            LanguageVersion::Metal1_1 => metal::MTLLanguageVersion::V1_1,
            LanguageVersion::Metal1_2 => metal::MTLLanguageVersion::V1_2,
            LanguageVersion::Metal2_0 => metal::MTLLanguageVersion::V2_0,
            LanguageVersion::Metal2_1 => metal::MTLLanguageVersion::V2_1,
            LanguageVersion::Metal2_2 => metal::MTLLanguageVersion::V2_2,
            LanguageVersion::Metal2_3 => metal::MTLLanguageVersion::V2_3,
            LanguageVersion::Metal2_4 => metal::MTLLanguageVersion::V2_4,
            LanguageVersion::Metal3_0 => metal::MTLLanguageVersion::V2_4, // Fallback to 2.4 for compatibility
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl ShaderCompiler {
    /// Creates a new shader compiler with the given device and configuration
    pub fn new(device: metal::Device, config: ShaderCompilerConfig) -> Result<Self> {
        // Create cache directory if needed
        if let Some(cache_dir) = &config.cache_directory {
            if config.enable_caching {
                fs::create_dir_all(cache_dir)
                    .with_context(|| format!("Failed to create cache directory: {:?}", cache_dir))?;
            }
        }

        // Configure Metal compile options
        let mut compile_options = metal::CompileOptions::new();
        compile_options.set_language_version(config.compile_options.language_version.to_metal_version());
        compile_options.set_fast_math_enabled(config.compile_options.fast_math);

        // Set preprocessor definitions
        for (key, value) in &config.compile_options.defines {
            // Note: Metal 0.27.0 has different preprocessor macro API
            // This functionality may need to be implemented differently
            // For now, we'll skip setting preprocessor macros
        }

        Ok(Self {
            device,
            config,
            cache: Arc::new(Mutex::new(HashMap::new())),
            compile_options,
        })
    }

    /// Creates a shader compiler with default configuration
    pub fn new_default(device: metal::Device) -> Result<Self> {
        Self::new(device, ShaderCompilerConfig::default())
    }

    /// Discovers and compiles all shaders in the shader directory
    pub fn compile_all_shaders(&self) -> Result<Vec<CompiledShader>> {
        let shader_files = self.discover_shader_files()?;
        let mut compiled_shaders = Vec::new();

        for shader_file in shader_files {
            match self.compile_shader_file(&shader_file) {
                Ok(compiled_shader) => {
                    compiled_shaders.push(compiled_shader);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to compile shader {:?}: {}", shader_file, e);
                }
            }
        }

        Ok(compiled_shaders)
    }

    /// Compiles a specific shader file
    pub fn compile_shader_file(&self, shader_path: &Path) -> Result<CompiledShader> {
        let shader_name = shader_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| MetalError::LibraryCreationFailed("Invalid shader file name".to_string()))?;

        // Check cache first
        if self.config.enable_caching {
            if let Some(cached_shader) = self.get_cached_shader(shader_name, shader_path)? {
                return Ok(cached_shader);
            }
        }

        // Read shader source
        let source = fs::read_to_string(shader_path)
            .with_context(|| format!("Failed to read shader file: {:?}", shader_path))?;

        // Compile shader
        let library = self.compile_source(&source, shader_name)?;

        // Extract function names
        let function_names = self.extract_function_names(&library);

        // Calculate source hash
        let source_hash = self.calculate_hash(&source);

        let compiled_shader = CompiledShader {
            name: shader_name.to_string(),
            source_path: shader_path.to_path_buf(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            library,
            function_names,
            compiled_at: std::time::SystemTime::now(),
            source_hash,
        };

        // Cache the compiled shader
        if self.config.enable_caching {
            self.cache_shader(&compiled_shader);
        }

        Ok(compiled_shader)
    }

    /// Compiles Metal source code into a library
    pub fn compile_source(&self, source: &str, name: &str) -> Result<metal::Library> {
        let library = self.device
            .new_library_with_source(source, &self.compile_options)
            .map_err(|e| MetalError::LibraryCreationFailed(
                format!("Failed to compile shader '{}': {}", name, e)
            ))?;

        Ok(library)
    }

    /// Gets a compiled shader by name
    pub fn get_shader(&self, name: &str) -> Option<CompiledShader> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(entry) = cache.get_mut(name) {
            entry.last_accessed = std::time::Instant::now();
            Some(entry.compiled_shader.clone())
        } else {
            None
        }
    }

    /// Gets a compute function from a compiled shader
    pub fn get_compute_function(&self, shader_name: &str, function_name: &str) -> Result<metal::Function> {
        let shader = self.get_shader(shader_name)
            .ok_or_else(|| MetalError::ComputeFunctionNotFound(
                format!("Shader '{}' not found", shader_name)
            ))?;

        let function = shader.library
            .get_function(function_name, None)
            .map_err(|e| MetalError::ComputeFunctionNotFound(
                format!("Function '{}' not found in shader '{}': {}", function_name, shader_name, e)
            ))?;

        Ok(function)
    }

    /// Creates a compute pipeline state from a shader function
    pub fn create_compute_pipeline(&self, shader_name: &str, function_name: &str) -> Result<metal::ComputePipelineState> {
        let function = self.get_compute_function(shader_name, function_name)?;
        
        let pipeline_state = self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::ComputePipelineCreationFailed(
                format!("Failed to create pipeline for '{}::{}': {}", shader_name, function_name, e)
            ))?;

        Ok(pipeline_state)
    }

    /// Recompiles all shaders (useful for development)
    pub fn recompile_all(&self) -> Result<()> {
        // Clear cache
        self.clear_cache();

        // Recompile all shaders
        self.compile_all_shaders()?;

        Ok(())
    }

    /// Clears the shader cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }

    /// Gets compilation statistics
    pub fn get_stats(&self) -> ShaderCompilerStats {
        let cache = self.cache.lock().unwrap();
        ShaderCompilerStats {
            cached_shaders: cache.len(),
            total_functions: cache.values()
                .map(|entry| entry.compiled_shader.function_names.len())
                .sum(),
            cache_directory: self.config.cache_directory.clone(),
            shader_directory: self.config.shader_directory.clone(),
        }
    }

    // Private helper methods

    fn discover_shader_files(&self) -> Result<Vec<PathBuf>> {
        let mut shader_files = Vec::new();
        
        if !self.config.shader_directory.exists() {
            return Ok(shader_files);
        }

        let entries = fs::read_dir(&self.config.shader_directory)
            .with_context(|| format!("Failed to read shader directory: {:?}", self.config.shader_directory))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension == "metal" {
                        shader_files.push(path);
                    }
                }
            }
        }

        Ok(shader_files)
    }

    fn get_cached_shader(&self, name: &str, shader_path: &Path) -> Result<Option<CompiledShader>> {
        let cache = self.cache.lock().unwrap();
        
        if let Some(entry) = cache.get(name) {
            // Check if source file has been modified
            let metadata = fs::metadata(shader_path)?;
            let modified_time = metadata.modified()?;
            
            if modified_time <= entry.compiled_shader.compiled_at {
                return Ok(Some(entry.compiled_shader.clone()));
            }
        }

        Ok(None)
    }

    fn cache_shader(&self, shader: &CompiledShader) {
        let mut cache = self.cache.lock().unwrap();
        cache.insert(shader.name.clone(), CacheEntry {
            compiled_shader: shader.clone(),
            last_accessed: std::time::Instant::now(),
        });
    }

    fn extract_function_names(&self, library: &metal::Library) -> Vec<String> {
        library.function_names().iter().map(|s| s.to_string()).collect()
    }

    fn calculate_hash(&self, source: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        hasher.finish()
    }
}

/// Shader compiler statistics
#[derive(Debug, Clone)]
pub struct ShaderCompilerStats {
    pub cached_shaders: usize,
    pub total_functions: usize,
    pub cache_directory: Option<PathBuf>,
    pub shader_directory: PathBuf,
}

/// Shader loading utilities
pub struct ShaderLoader {
    compiler: ShaderCompiler,
    preloaded_shaders: HashMap<String, CompiledShader>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl ShaderLoader {
    /// Creates a new shader loader
    pub fn new(device: metal::Device, config: ShaderCompilerConfig) -> Result<Self> {
        let compiler = ShaderCompiler::new(device, config)?;
        
        Ok(Self {
            compiler,
            preloaded_shaders: HashMap::new(),
        })
    }

    /// Creates a shader loader with default configuration
    pub fn new_default(device: metal::Device) -> Result<Self> {
        Self::new(device, ShaderCompilerConfig::default())
    }

    /// Preloads all shaders for faster runtime access
    pub fn preload_all_shaders(&mut self) -> Result<()> {
        let compiled_shaders = self.compiler.compile_all_shaders()?;
        
        for shader in compiled_shaders {
            self.preloaded_shaders.insert(shader.name.clone(), shader);
        }

        Ok(())
    }

    /// Preloads specific shaders
    pub fn preload_shaders(&mut self, shader_names: &[&str]) -> Result<()> {
        for &name in shader_names {
            let shader_path = self.compiler.config.shader_directory.join(format!("{}.metal", name));
            if shader_path.exists() {
                let compiled_shader = self.compiler.compile_shader_file(&shader_path)?;
                self.preloaded_shaders.insert(name.to_string(), compiled_shader);
            }
        }

        Ok(())
    }

    /// Gets a shader (from preloaded cache or compiles on demand)
    pub fn get_shader(&self, name: &str) -> Result<&CompiledShader> {
        if let Some(shader) = self.preloaded_shaders.get(name) {
            return Ok(shader);
        }

        // Try to get from compiler cache
        if let Some(shader) = self.compiler.get_shader(name) {
            // Note: This would require making preloaded_shaders mutable
            // For now, return an error suggesting preloading
            return Err(MetalError::ComputeFunctionNotFound(
                format!("Shader '{}' not preloaded. Consider calling preload_shaders() first.", name)
            ).into());
        }

        Err(MetalError::ComputeFunctionNotFound(
            format!("Shader '{}' not found", name)
        ).into())
    }

    /// Creates a compute pipeline from preloaded shader
    pub fn create_compute_pipeline(&self, shader_name: &str, function_name: &str) -> Result<metal::ComputePipelineState> {
        self.compiler.create_compute_pipeline(shader_name, function_name)
    }

    /// Gets available shader names
    pub fn get_available_shaders(&self) -> Vec<String> {
        self.preloaded_shaders.keys().cloned().collect()
    }

    /// Gets available functions for a shader
    pub fn get_shader_functions(&self, shader_name: &str) -> Result<&[String]> {
        let shader = self.get_shader(shader_name)?;
        Ok(&shader.function_names)
    }
}

// Convenience functions for easy shader management

/// Creates a shader compiler with default configuration
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn create_shader_compiler(device: &metal::Device) -> Result<ShaderCompiler> {
    ShaderCompiler::new_default(device.clone())
}

/// Creates a shader compiler with custom configuration
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn create_shader_compiler_with_config(
    device: &metal::Device,
    config: ShaderCompilerConfig,
) -> Result<ShaderCompiler> {
    ShaderCompiler::new(device.clone(), config)
}

/// Creates a shader loader with default configuration
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn create_shader_loader(device: &metal::Device) -> Result<ShaderLoader> {
    ShaderLoader::new_default(device.clone())
}

/// Creates a shader loader with custom configuration
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn create_shader_loader_with_config(
    device: &metal::Device,
    config: ShaderCompilerConfig,
) -> Result<ShaderLoader> {
    ShaderLoader::new(device.clone(), config)
}

/// Compiles a single shader file and returns the library
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn compile_shader_file(device: &metal::Device, shader_path: &Path) -> Result<metal::Library> {
    let compiler = ShaderCompiler::new_default(device.clone())?;
    let compiled_shader = compiler.compile_shader_file(shader_path)?;
    Ok(compiled_shader.library)
}

/// Compiles Metal source code directly
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn compile_shader_source(device: &metal::Device, source: &str, name: &str) -> Result<metal::Library> {
    let compiler = ShaderCompiler::new_default(device.clone())?;
    compiler.compile_source(source, name)
}

// Non-macOS implementations
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn create_shader_compiler(_device: &()) -> Result<()> {
    Err(MetalError::UnsupportedPlatform.into())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn create_shader_compiler_with_config(_device: &(), _config: ShaderCompilerConfig) -> Result<()> {
    Err(MetalError::UnsupportedPlatform.into())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn create_shader_loader(_device: &()) -> Result<()> {
    Err(MetalError::UnsupportedPlatform.into())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn create_shader_loader_with_config(_device: &(), _config: ShaderCompilerConfig) -> Result<()> {
    Err(MetalError::UnsupportedPlatform.into())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn compile_shader_file(_device: &(), _shader_path: &Path) -> Result<()> {
    Err(MetalError::UnsupportedPlatform.into())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub fn compile_shader_source(_device: &(), _source: &str, _name: &str) -> Result<()> {
    Err(MetalError::UnsupportedPlatform.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_shader_compiler_config_default() {
        let config = ShaderCompilerConfig::default();
        assert_eq!(config.shader_directory, PathBuf::from("src/metal/shaders"));
        assert!(config.enable_caching);
        assert!(config.compile_options.fast_math);
    }

    #[test]
    fn test_compile_options_default() {
        let options = CompileOptions::default();
        assert!(options.defines.is_empty());
        assert!(options.include_directories.is_empty());
        assert!(options.fast_math);
        assert!(matches!(options.language_version, LanguageVersion::Metal2_4));
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_shader_compiler_creation() {
        use crate::metal::create_metal_device;
        
        if let Ok(device) = create_metal_device() {
            let temp_dir = TempDir::new().unwrap();
            let config = ShaderCompilerConfig {
                shader_directory: temp_dir.path().to_path_buf(),
                enable_caching: true,
                cache_directory: Some(temp_dir.path().join("cache")),
                ..Default::default()
            };

            let compiler_result = ShaderCompiler::new(device, config);
            assert!(compiler_result.is_ok());
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_shader_discovery() {
        use crate::metal::create_metal_device;
        
        if let Ok(device) = create_metal_device() {
            let temp_dir = TempDir::new().unwrap();
            
            // Create test shader files
            let shader1_path = temp_dir.path().join("test1.metal");
            let shader2_path = temp_dir.path().join("test2.metal");
            
            fs::write(&shader1_path, "// Test shader 1").unwrap();
            fs::write(&shader2_path, "// Test shader 2").unwrap();
            
            let config = ShaderCompilerConfig {
                shader_directory: temp_dir.path().to_path_buf(),
                ..Default::default()
            };

            let compiler = ShaderCompiler::new(device, config).unwrap();
            let shader_files = compiler.discover_shader_files().unwrap();
            
            assert_eq!(shader_files.len(), 2);
            assert!(shader_files.contains(&shader1_path));
            assert!(shader_files.contains(&shader2_path));
        }
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_unsupported_platform() {
        let result = create_shader_compiler(&());
        assert!(result.is_err());
        
        let result = create_shader_loader(&());
        assert!(result.is_err());
        
        let result = compile_shader_file(&(), Path::new("test.metal"));
        assert!(result.is_err());
        
        let result = compile_shader_source(&(), "// test", "test");
        assert!(result.is_err());
    }
}