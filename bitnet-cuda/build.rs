// CUDA build configuration for BitNet kernels
use std::env;
use std::path::PathBuf;

fn main() {
    // Only build CUDA kernels if the cuda feature is enabled
    if cfg!(feature = "cuda") {
        build_cuda_kernels();
    }
    
    println!("cargo:rerun-if-changed=src/kernels/");
    println!("cargo:rerun-if-changed=build.rs");
}

fn build_cuda_kernels() {
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    let cuda_include = PathBuf::from(&cuda_path).join("include");
    let cuda_lib = PathBuf::from(&cuda_path).join("lib64");
    
    // Check if CUDA is available
    if !cuda_include.exists() || !cuda_lib.exists() {
        println!("cargo:warning=CUDA not found at {}, skipping CUDA kernel compilation", cuda_path);
        return;
    }

    let mut nvcc = cc::Build::new();
    
    // Configure NVCC compiler
    nvcc.compiler("nvcc")
        .include(&cuda_include)
        .include("src/kernels")
        .flag("--std=c++17")
        .flag("--compiler-options")
        .flag("-fPIC")
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("--expt-relaxed-constexpr");
    
    // Add compute capabilities for modern GPUs
    let compute_caps = ["75", "80", "86", "89", "90"]; // RTX 20xx, A100, RTX 30xx, RTX 40xx, H100
    for cap in &compute_caps {
        nvcc.flag(&format!("--generate-code"))
            .flag(&format!("arch=compute_{},code=sm_{}", cap, cap));
    }
    
    // Add source files
    let kernel_files = [
        "src/kernels/w2a8_gemv.cu",
        "src/kernels/common.cu",
    ];
    
    for file in &kernel_files {
        let path = PathBuf::from(file);
        if path.exists() {
            nvcc.file(file);
        }
    }
    
    // Compile CUDA kernels
    nvcc.compile("bitnet_cuda_kernels");
    
    // Link CUDA runtime
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=curand");
}
