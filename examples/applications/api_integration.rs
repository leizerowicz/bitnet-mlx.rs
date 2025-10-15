use anyhow::Result;
use bitnet_inference::{InferenceEngine, EngineConfig};
use bitnet_inference::api::{TextGenerator, GenerationConfig, GenerationResult};
use bitnet_inference::bitnet_config::TokenizerConfig;
use bitnet_core::Device;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use warp::Filter;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use colored::*;

/// BitNet API Integration Example
/// 
/// Demonstrates how to integrate BitNet-Rust into applications with:
/// - REST API server
/// - WebSocket streaming
/// - Client libraries
/// - Production patterns
#[derive(Parser, Debug)]
#[command(name = "bitnet-api")]
#[command(version = "1.0.0")]
#[command(about = "API integration examples for BitNet neural networks")]
pub struct ApiArgs {
    /// Model name or path
    #[arg(short, long, default_value = "microsoft/bitnet-b1.58-2B-4T-gguf")]
    pub model: String,

    /// Device to use (auto, cpu, metal, cuda)
    #[arg(short, long, default_value = "auto")]
    pub device: String,

    /// API server port
    #[arg(short, long, default_value = "8080")]
    pub port: u16,

    /// API server host
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Enable CORS for web clients
    #[arg(long)]
    pub cors: bool,

    /// API key for authentication (optional)
    #[arg(long)]
    pub api_key: Option<String>,

    /// Maximum concurrent requests
    #[arg(long, default_value = "10")]
    pub max_concurrent: usize,

    /// Request timeout in seconds
    #[arg(long, default_value = "30")]
    pub timeout_seconds: u64,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,
}

// API Request/Response Types

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub id: String,
    pub text: String,
    pub token_count: usize,
    pub generation_time_ms: u64,
    pub finish_reason: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub chunk: String,
    pub token_count: usize,
    pub is_final: bool,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub device: String,
    pub loaded_at: DateTime<Utc>,
    pub total_requests: u64,
    pub average_tokens_per_second: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
    pub device: String,
    pub memory_usage_mb: f64,
    pub uptime_seconds: u64,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub error_code: String,
    pub timestamp: DateTime<Utc>,
}

// Application State

#[derive(Clone)]
pub struct AppState {
    pub generator: Arc<TextGenerator>,
    pub model_info: Arc<RwLock<ModelInfo>>,
    pub config: ApiArgs,
    pub request_count: Arc<RwLock<u64>>,
    pub start_time: DateTime<Utc>,
}

// API Server Implementation

pub struct BitNetApiServer {
    state: AppState,
}

impl BitNetApiServer {
    /// Initialize the API server
    pub async fn new(config: ApiArgs) -> Result<Self> {
        println!("{}", "🚀 BitNet API Server".bright_blue().bold());
        println!("Initializing inference engine...");

        // Parse device selection
        let device = match config.device.to_lowercase().as_str() {
            "auto" => Device::best_available(),
            "cpu" => Device::Cpu,
            "metal" => Device::Metal,
            "cuda" => Device::Cuda,
            _ => {
                println!("{} Unknown device '{}', using auto-detection", "Warning:".yellow(), config.device);
                Device::best_available()
            }
        };

        println!("Selected device: {}", format!("{:?}", device).green());

        // Configure for API server
        let engine_config = EngineConfig {
            device: device.clone(),
            memory_limit_mb: Some(4096),
            thread_count: Some(num_cpus::get()),
            enable_profiling: config.verbose,
            ..Default::default()
        };

        let engine = InferenceEngine::with_config(engine_config).await?;

        // Load model
        println!("Loading model: {}", config.model.cyan());
        let start_time = tokio::time::Instant::now();
        
        let model_handle = if config.model.contains('/') && !config.model.starts_with('.') && !config.model.starts_with('/') {
            engine.load_model_from_hub(&config.model).await?
        } else {
            engine.load_model(&config.model).await?
        };

        let load_time = start_time.elapsed();
        println!("Model loaded in {:.2}s", load_time.as_secs_f64());

        // Configure text generation with defaults
        let generation_config = GenerationConfig {
            temperature: 0.7,
            top_k: Some(50),
            top_p: Some(0.9),
            typical_p: Some(0.95),
            max_length: 512,
            max_context_length: Some(4096),
            do_sample: true,
            stop_tokens: vec!["<|endoftext|>".to_string(), "</s>".to_string()],
            seed: None,
            early_stopping: true,
            repetition_penalty: Some(1.1),
            length_penalty: Some(1.0),
            use_lut_acceleration: true,
            target_latency_ms: Some(50),
        };

        // Default tokenizer config
        let tokenizer_config = TokenizerConfig {
            vocab_size: 128256,
            tokenizer_type: "llama3".to_string(),
            bos_token_id: Some(128000),
            eos_token_id: Some(128001),
            pad_token_id: Some(128002),
        };

        let generator = Arc::new(TextGenerator::new(
            engine,
            model_handle,
            generation_config,
            tokenizer_config,
        ));

        // Create application state
        let model_info = ModelInfo {
            name: config.model.clone(),
            device: format!("{:?}", device),
            loaded_at: Utc::now(),
            total_requests: 0,
            average_tokens_per_second: 0.0,
        };

        let state = AppState {
            generator,
            model_info: Arc::new(RwLock::new(model_info)),
            config,
            request_count: Arc::new(RwLock::new(0)),
            start_time: Utc::now(),
        };

        Ok(Self { state })
    }

    /// Start the API server
    pub async fn run(&self) -> Result<()> {
        let state = self.state.clone();

        // Health check endpoint
        let health = warp::path("health")
            .and(warp::get())
            .and(with_state(state.clone()))
            .and_then(health_handler);

        // Model info endpoint
        let model_info = warp::path("model")
            .and(warp::get())
            .and(with_state(state.clone()))
            .and_then(model_info_handler);

        // Generate text endpoint
        let generate = warp::path("generate")
            .and(warp::post())
            .and(warp::body::json())
            .and(with_state(state.clone()))
            .and_then(generate_handler);

        // Stream generation endpoint (WebSocket)
        let stream = warp::path("stream")
            .and(warp::ws())
            .and(with_state(state.clone()))
            .map(|ws: warp::ws::Ws, state| {
                ws.on_upgrade(move |socket| stream_handler(socket, state))
            });

        // Combine routes
        let mut routes = health
            .or(model_info)
            .or(generate)
            .or(stream)
            .recover(handle_rejection);

        // Add CORS if enabled
        if self.state.config.cors {
            routes = routes
                .with(warp::cors()
                    .allow_any_origin()
                    .allow_headers(vec!["content-type", "authorization"])
                    .allow_methods(vec!["GET", "POST", "OPTIONS"]));
        }

        // Add authentication if API key is provided
        if self.state.config.api_key.is_some() {
            routes = routes.with(warp::log("bitnet_api"));
        }

        println!("🌐 Starting API server at {}:{}", self.state.config.host, self.state.config.port);
        println!("📖 API Documentation:");
        println!("  GET  /health           - Server health check");
        println!("  GET  /model            - Model information");
        println!("  POST /generate         - Generate text");
        println!("  WS   /stream           - Stream generation");
        println!();

        warp::serve(routes)
            .run((
                self.state.config.host.parse::<std::net::IpAddr>()?,
                self.state.config.port,
            ))
            .await;

        Ok(())
    }
}

// Helper function to pass state to handlers
fn with_state(state: AppState) -> impl Filter<Extract = (AppState,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || state.clone())
}

// Handler Functions

async fn health_handler(state: AppState) -> Result<impl warp::Reply, warp::Rejection> {
    let uptime = (Utc::now() - state.start_time).num_seconds();
    
    let response = HealthResponse {
        status: "healthy".to_string(),
        model_loaded: true,
        device: state.model_info.read().await.device.clone(),
        memory_usage_mb: 0.0, // TODO: Get actual memory usage
        uptime_seconds: uptime as u64,
        version: "1.0.0".to_string(),
    };

    Ok(warp::reply::json(&response))
}

async fn model_info_handler(state: AppState) -> Result<impl warp::Reply, warp::Rejection> {
    let model_info = state.model_info.read().await.clone();
    Ok(warp::reply::json(&model_info))
}

async fn generate_handler(
    request: GenerateRequest,
    state: AppState,
) -> Result<impl warp::Reply, warp::Rejection> {
    let request_id = Uuid::new_v4().to_string();
    let start_time = tokio::time::Instant::now();

    // Increment request count
    {
        let mut count = state.request_count.write().await;
        *count += 1;
    }

    // Create custom generation config if parameters provided
    let mut generation_config = GenerationConfig {
        temperature: request.temperature.unwrap_or(0.7),
        top_k: request.top_k.map(|k| k as usize),
        top_p: request.top_p,
        typical_p: Some(0.95),
        max_length: request.max_tokens.unwrap_or(512) as usize,
        max_context_length: Some(4096),
        do_sample: true,
        stop_tokens: vec!["<|endoftext|>".to_string(), "</s>".to_string()],
        seed: None,
        early_stopping: true,
        repetition_penalty: Some(1.1),
        length_penalty: Some(1.0),
        use_lut_acceleration: true,
        target_latency_ms: Some(50),
    };

    // Generate text
    match state.generator.generate(&request.prompt).await {
        Ok(result) => {
            let generation_time = start_time.elapsed();
            
            // Update model statistics
            {
                let mut model_info = state.model_info.write().await;
                model_info.total_requests += 1;
                // Update average tokens per second (simplified calculation)
                let total_time_ms = generation_time.as_millis() as f64;
                if total_time_ms > 0.0 {
                    model_info.average_tokens_per_second = 
                        (result.token_count as f64) / (total_time_ms / 1000.0);
                }
            }

            let response = GenerateResponse {
                id: request_id,
                text: result.text,
                token_count: result.token_count,
                generation_time_ms: generation_time.as_millis() as u64,
                finish_reason: format!("{:?}", result.finish_reason),
                timestamp: Utc::now(),
            };

            Ok(warp::reply::json(&response))
        }
        Err(e) => {
            let error_response = ErrorResponse {
                error: e.to_string(),
                error_code: "generation_failed".to_string(),
                timestamp: Utc::now(),
            };
            Ok(warp::reply::json(&error_response))
        }
    }
}

async fn stream_handler(ws: warp::ws::WebSocket, state: AppState) {
    // WebSocket streaming implementation would go here
    // This is a simplified version
    println!("WebSocket connection established");
    // TODO: Implement streaming generation
}

async fn handle_rejection(err: warp::Rejection) -> Result<impl warp::Reply, std::convert::Infallible> {
    let error_response = ErrorResponse {
        error: format!("Request failed: {:?}", err),
        error_code: "request_failed".to_string(),
        timestamp: Utc::now(),
    };

    Ok(warp::reply::with_status(
        warp::reply::json(&error_response),
        warp::http::StatusCode::BAD_REQUEST,
    ))
}

// Client Library Example

pub struct BitNetClient {
    base_url: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl BitNetClient {
    /// Create a new BitNet API client
    pub fn new(base_url: &str, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.to_string(),
            api_key,
            client: reqwest::Client::new(),
        }
    }

    /// Generate text using the API
    pub async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let url = format!("{}/generate", self.base_url);
        
        let mut req = self.client.post(&url).json(&request);
        
        if let Some(api_key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = req.send().await?;
        
        if response.status().is_success() {
            let result: GenerateResponse = response.json().await?;
            Ok(result)
        } else {
            let error: ErrorResponse = response.json().await?;
            Err(anyhow::anyhow!("API Error: {}", error.error))
        }
    }

    /// Get model information
    pub async fn model_info(&self) -> Result<ModelInfo> {
        let url = format!("{}/model", self.base_url);
        let response = self.client.get(&url).send().await?;
        let model_info: ModelInfo = response.json().await?;
        Ok(model_info)
    }

    /// Check server health
    pub async fn health(&self) -> Result<HealthResponse> {
        let url = format!("{}/health", self.base_url);
        let response = self.client.get(&url).send().await?;
        let health: HealthResponse = response.json().await?;
        Ok(health)
    }
}

// Example Integration Patterns

/// Example: Integrate BitNet into a web application
pub async fn web_app_integration() -> Result<()> {
    println!("{}", "🌐 Web Application Integration Example".bright_green().bold());
    
    let client = BitNetClient::new("http://localhost:8080", None);
    
    // Check if server is healthy
    match client.health().await {
        Ok(health) => println!("✅ Server status: {}", health.status),
        Err(e) => {
            println!("❌ Server health check failed: {}", e);
            return Ok(());
        }
    }
    
    // Get model information
    let model_info = client.model_info().await?;
    println!("📋 Model: {} on {}", model_info.name, model_info.device);
    
    // Generate text
    let request = GenerateRequest {
        prompt: "Explain the benefits of neural network quantization".to_string(),
        temperature: Some(0.7),
        top_k: Some(50),
        top_p: Some(0.9),
        max_tokens: Some(256),
        stream: Some(false),
    };
    
    println!("🔄 Generating response...");
    let response = client.generate(request).await?;
    
    println!("✅ Generated {} tokens in {}ms", 
        response.token_count, response.generation_time_ms);
    println!("📝 Response: {}", response.text);
    
    Ok(())
}

/// Example: Batch processing with API
pub async fn batch_api_processing() -> Result<()> {
    println!("{}", "📦 Batch API Processing Example".bright_green().bold());
    
    let client = BitNetClient::new("http://localhost:8080", None);
    
    let prompts = vec![
        "What is artificial intelligence?",
        "Explain machine learning",
        "Describe neural networks",
        "What is deep learning?",
    ];
    
    println!("Processing {} prompts...", prompts.len());
    
    for (i, prompt) in prompts.iter().enumerate() {
        let request = GenerateRequest {
            prompt: prompt.to_string(),
            temperature: Some(0.7),
            max_tokens: Some(100),
            ..Default::default()
        };
        
        match client.generate(request).await {
            Ok(response) => {
                println!("✅ Prompt {}: {} tokens", i + 1, response.token_count);
            }
            Err(e) => {
                println!("❌ Prompt {} failed: {}", i + 1, e);
            }
        }
    }
    
    Ok(())
}

/// Example: Microservice integration
pub async fn microservice_integration() -> Result<()> {
    println!("{}", "🔧 Microservice Integration Example".bright_green().bold());
    
    // This would typically be in a separate microservice
    struct TextProcessingService {
        bitnet_client: BitNetClient,
    }
    
    impl TextProcessingService {
        fn new() -> Self {
            Self {
                bitnet_client: BitNetClient::new("http://bitnet-service:8080", None),
            }
        }
        
        async fn process_text(&self, input: &str) -> Result<String> {
            let request = GenerateRequest {
                prompt: format!("Summarize this text: {}", input),
                max_tokens: Some(150),
                temperature: Some(0.3), // Lower temperature for summarization
                ..Default::default()
            };
            
            let response = self.bitnet_client.generate(request).await?;
            Ok(response.text)
        }
    }
    
    let service = TextProcessingService::new();
    let input_text = "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.";
    
    match service.process_text(input_text).await {
        Ok(summary) => println!("📝 Summary: {}", summary),
        Err(e) => println!("❌ Processing failed: {}", e),
    }
    
    Ok(())
}

// Default implementation for GenerateRequest
impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            temperature: None,
            top_k: None,
            top_p: None,
            max_tokens: None,
            stream: None,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = ApiArgs::parse();
    
    match std::env::args().nth(1).as_deref() {
        Some("server") => {
            // Run API server
            let server = BitNetApiServer::new(config).await?;
            server.run().await?;
        }
        Some("client") => {
            // Run client examples
            println!("{}", "🚀 BitNet API Client Examples".bright_blue().bold());
            
            // Wait a moment for server to be ready
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            
            web_app_integration().await?;
            println!();
            batch_api_processing().await?;
            println!();
            microservice_integration().await?;
        }
        _ => {
            println!("Usage:");
            println!("  cargo run --bin api_integration server  # Start API server");
            println!("  cargo run --bin api_integration client  # Run client examples");
        }
    }
    
    Ok(())
}

// Example usage:
// Terminal 1: cargo run --bin api_integration server
// Terminal 2: cargo run --bin api_integration client