use bitnet_inference::{
    LlamaTokenizer, ChatFormat, InputProcessor, InputProcessingConfig,
    Message, Role, Dialog, Result,
};
use std::io::Write;
use tempfile::NamedTempFile;

/// Example demonstrating the LLaMA 3 tokenizer and input processing implementation
/// 
/// This example shows:
/// 1. Creating a LLaMA 3 tokenizer
/// 2. Processing single prompts and dialogs
/// 3. Batch processing
/// 4. Input validation and memory management
fn main() -> Result<()> {
    println!("🚀 BitNet-Rust LLaMA 3 Tokenizer and Input Processing Example");
    println!("================================================================\n");

    // Create a test tokenizer (in production, this would use real model files)
    let tokenizer = create_demo_tokenizer()?;
    println!("✅ Created LLaMA 3 tokenizer with vocab size: {}", tokenizer.vocab_size());
    
    // Create input processor with default configuration
    let config = InputProcessingConfig::default();
    let processor = InputProcessor::new(tokenizer.clone(), config);
    println!("✅ Created input processor with max context length: {}\n", processor.config().max_context_length);

    // Example 1: Single prompt processing
    println!("📝 Example 1: Single Prompt Processing");
    println!("=====================================");
    
    let prompt = "What is the capital of France?";
    let result = processor.process_prompt(prompt)?;
    
    println!("Input: \"{}\"", prompt);
    println!("Tokens: {:?}", result.tokens);
    println!("Token count: {}", result.metadata.final_token_count);
    println!("Memory usage: {} bytes", result.metadata.memory_usage);
    println!("Was truncated: {}\n", result.metadata.was_truncated);

    // Example 2: Dialog processing
    println!("💬 Example 2: Dialog Processing");
    println!("==============================");
    
    let dialog: Dialog = vec![
        Message {
            role: Role::System,
            content: "You are a helpful AI assistant.".to_string(),
        },
        Message {
            role: Role::User,
            content: "Hello! How are you today?".to_string(),
        },
    ];
    
    let dialog_result = processor.process_dialog(&dialog)?;
    println!("Dialog with {} messages", dialog.len());
    println!("Total tokens: {}", dialog_result.metadata.final_token_count);
    println!("Token sequence: {:?}", dialog_result.tokens);
    println!("Memory usage: {} bytes\n", dialog_result.metadata.memory_usage);

    // Example 3: Batch processing
    println!("📦 Example 3: Batch Processing");
    println!("=============================");
    
    let batch_inputs = vec![
        "Hello world!",
        "How does BitNet work?",
        "What is 1.58-bit quantization?",
    ];
    
    let batch = processor.process_batch(batch_inputs)?;
    println!("Batch ID: {}", batch.metadata.batch_id);
    println!("Batch size: {}", batch.inputs.len());
    println!("Total tokens: {}", batch.metadata.total_tokens);
    println!("Max token length: {}", batch.metadata.max_token_length);
    println!("Estimated processing time: {:?}\n", batch.metadata.estimated_processing_time);

    // Example 4: Chat format demonstration
    println!("🗣️  Example 4: Chat Format Demonstration");
    println!("=======================================");
    
    let chat_format = ChatFormat::new(tokenizer.clone());
    
    let user_message = Message {
        role: Role::User,
        content: "Explain BitNet in simple terms.".to_string(),
    };
    
    let message_tokens = chat_format.encode_message(&user_message)?;
    println!("User message: \"{}\"", user_message.content);
    println!("Encoded tokens: {:?}", message_tokens);
    
    let conversation = vec![
        Message {
            role: Role::System,
            content: "You are an expert in neural networks.".to_string(),
        },
        user_message,
    ];
    
    let prompt_tokens = chat_format.encode_dialog_prompt(&conversation)?;
    println!("Dialog prompt tokens: {:?}", prompt_tokens);
    println!("Dialog ready for completion (ends with assistant header)\n");

    // Example 5: Token buffer management
    println!("🔧 Example 5: Token Buffer Management");
    println!("====================================");
    
    let mut buffer = processor.get_token_buffer();
    println!("Got token buffer with capacity: {}", buffer.remaining_capacity());
    
    buffer.push_tokens(&[1, 2, 3, 4, 5])?;
    println!("Added 5 tokens, buffer length: {}", buffer.len());
    
    let tokens = buffer.get_all_tokens();
    println!("Buffer contents: {:?}", tokens);
    
    // Return buffer to pool for reuse
    processor.return_token_buffer(buffer);
    println!("Returned buffer to pool for reuse\n");

    // Example 6: Processing statistics
    println!("📊 Example 6: Processing Statistics");
    println!("==================================");
    
    let stats = processor.get_stats();
    println!("Total inputs processed: {}", stats.total_processed);
    println!("Total tokens processed: {}", stats.total_tokens);
    println!("Total truncations: {}", stats.total_truncations);
    println!("Average processing time: {:?}", stats.average_processing_time);
    println!("Current memory usage: {} bytes", stats.current_memory_usage);
    println!("Peak memory usage: {} bytes\n", stats.peak_memory_usage);

    // Example 7: Special token handling
    println!("🎯 Example 7: Special Token Handling");
    println!("===================================");
    
    println!("BOS token ID: {}", tokenizer.bos_id());
    println!("EOS token ID: {}", tokenizer.eos_id());
    println!("PAD token ID: {}", tokenizer.pad_id());
    
    if let Some(eot_id) = tokenizer.special_token_id("<|eot_id|>") {
        println!("End of turn token ID: {}", eot_id);
        println!("Is stop token: {}", tokenizer.is_stop_token(eot_id));
    }
    
    if let Some(start_header_id) = tokenizer.special_token_id("<|start_header_id|>") {
        println!("Start header token ID: {}", start_header_id);
    }

    println!("\n🎉 All examples completed successfully!");
    println!("Ready for integration with inference engine and text generation.");
    
    Ok(())
}

/// Create a demo tokenizer for testing (in production, use real model files)
fn create_demo_tokenizer() -> Result<LlamaTokenizer> {
    // Create a temporary file to simulate a tokenizer model file
    let mut temp_file = NamedTempFile::new()
        .map_err(|e| bitnet_inference::InferenceError::TokenizerError(format!("Failed to create temp file: {}", e)))?;
    
    writeln!(temp_file, "# Demo tokenizer model file")
        .map_err(|e| bitnet_inference::InferenceError::TokenizerError(format!("Failed to write temp file: {}", e)))?;
    
    LlamaTokenizer::new(temp_file.path())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_example() {
        let result = main();
        assert!(result.is_ok());
    }
}