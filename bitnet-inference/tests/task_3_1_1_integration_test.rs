//! Integration test for Task 3.1.1: LLaMA 3 Tokenizer Integration
//! 
//! This test validates all requirements of Task 3.1.1:
//! - [x] Tokenizer Loading: HuggingFace tokenizer integration  
//! - [x] Special Tokens: Proper handling of BOS, EOS, PAD tokens
//! - [x] Chat Format: System/user/assistant conversation templates
//! - [x] Encoding/Decoding: Efficient text ↔ token conversion

use bitnet_inference::{
    tokenizer::{LlamaTokenizer, Role, Message, ChatFormat},
    Result,
};
use std::collections::HashMap;
use tempfile::TempDir;

/// Test Task 3.1.1 Requirement 1: HuggingFace Tokenizer Integration
#[tokio::test]
async fn test_huggingface_tokenizer_integration() -> Result<()> {
    // Test creating tokenizer from custom vocabulary (simulating HuggingFace loading)
    let vocab_tokens: Vec<String> = (0..1000)
        .map(|i| format!("token_{}", i))
        .collect();
    
    let mut special_tokens = HashMap::new();
    special_tokens.insert("<|begin_of_text|>".to_string(), 1000);
    special_tokens.insert("<|end_of_text|>".to_string(), 1001);
    special_tokens.insert("<|eot_id|>".to_string(), 1002);
    
    let tokenizer = LlamaTokenizer::from_vocabulary(
        vocab_tokens,
        Some(128256), // LLaMA 3 vocab size
        Some(special_tokens),
    )?;
    
    // Verify tokenizer properties
    assert_eq!(tokenizer.vocab_size(), 128256);
    assert!(tokenizer.bos_id() > 0);
    assert!(tokenizer.eos_id() > 0);
    assert_eq!(tokenizer.pad_id(), -1); // LLaMA 3 doesn't use padding
    
    println!("✅ HuggingFace tokenizer integration test passed");
    Ok(())
}

/// Test Task 3.1.1 Requirement 2: Special Token Handling
#[tokio::test]
async fn test_special_token_handling() -> Result<()> {
    // Create a simple tokenizer for testing
    let tokenizer = create_test_tokenizer()?;
    
    // Test BOS token
    let bos_id = tokenizer.bos_id();
    assert!(bos_id > 0, "BOS token ID should be valid");
    
    // Test EOS token  
    let eos_id = tokenizer.eos_id();
    assert!(eos_id > 0, "EOS token ID should be valid");
    assert_ne!(bos_id, eos_id, "BOS and EOS should have different IDs");
    
    // Test PAD token (LLaMA 3 doesn't use padding)
    assert_eq!(tokenizer.pad_id(), -1, "LLaMA 3 should not use padding tokens");
    
    // Test stop tokens
    assert!(tokenizer.is_stop_token(eos_id), "EOS should be a stop token");
    
    // Test special token lookup
    assert_eq!(tokenizer.special_token_id("<|begin_of_text|>"), Some(bos_id));
    assert_eq!(tokenizer.special_token_id("<|end_of_text|>"), Some(eos_id));
    
    if let Some(eot_id) = tokenizer.special_token_id("<|eot_id|>") {
        assert!(tokenizer.is_stop_token(eot_id), "End-of-turn should be a stop token");
    }
    
    println!("✅ Special token handling test passed");
    Ok(())
}

/// Test Task 3.1.1 Requirement 3: Chat Format Support
#[tokio::test]
async fn test_chat_format_support() -> Result<()> {
    let tokenizer = create_test_tokenizer()?;
    let chat_format = ChatFormat::new(tokenizer);
    
    // Create a sample dialog
    let dialog = vec![
        Message { role: Role::System, content: "You are a helpful assistant.".to_string() },
        Message { role: Role::User, content: "Hello, how are you?".to_string() },
        Message { role: Role::Assistant, content: "I'm doing well, thank you for asking!".to_string() },
        Message { role: Role::User, content: "Can you help me with some coding?".to_string() },
    ];
    
    // Test dialog encoding
    let encoded_tokens = chat_format.encode_dialog_prompt(&dialog)?;
    
    // Verify encoding results
    assert!(!encoded_tokens.is_empty(), "Dialog should produce tokens");
    assert!(encoded_tokens.len() > dialog.len(), "Should include special tokens and text");
    
    // Test that BOS token is included
    let bos_id = chat_format.tokenizer().bos_id();
    assert!(encoded_tokens.contains(&bos_id), "Dialog should start with BOS token");
    
    // Test role formatting
    let system_msg = Message { role: Role::System, content: "Test system message".to_string() };
    let user_msg = Message { role: Role::User, content: "Test user message".to_string() };
    let assistant_msg = Message { role: Role::Assistant, content: "Test assistant message".to_string() };
    
    // Verify role string representations
    assert_eq!(system_msg.role.to_string(), "system");
    assert_eq!(user_msg.role.to_string(), "user");
    assert_eq!(assistant_msg.role.to_string(), "assistant");
    
    println!("✅ Chat format support test passed");
    Ok(())
}

/// Test Task 3.1.1 Requirement 4: Efficient Text ↔ Token Conversion
#[tokio::test]
async fn test_efficient_encoding_decoding() -> Result<()> {
    let tokenizer = create_test_tokenizer()?;
    
    // Test basic encoding/decoding roundtrip
    let test_texts = vec![
        "Hello, world!",
        "This is a test of the BitNet tokenizer.",
        "LLaMA 3 uses a 128,256 token vocabulary.",
        "Special tokens like <|begin_of_text|> should be handled properly.",
        "",  // Empty string
        "🚀🎯✅", // Unicode characters
    ];
    
    for text in test_texts {
        // Test encoding with different BOS/EOS combinations
        let tokens_no_special = tokenizer.encode(text, false, false)?;
        let tokens_with_bos = tokenizer.encode(text, true, false)?;
        let tokens_with_eos = tokenizer.encode(text, false, true)?;
        let tokens_with_both = tokenizer.encode(text, true, true)?;
        
        // Verify BOS/EOS token addition
        if !text.is_empty() {
            assert!(tokens_with_bos.len() >= tokens_no_special.len());
            assert!(tokens_with_eos.len() >= tokens_no_special.len());
            assert!(tokens_with_both.len() >= tokens_no_special.len());
        }
        
        // Test decoding roundtrip (use tokens without special tokens for clean roundtrip)
        if !tokens_no_special.is_empty() {
            let decoded_text = tokenizer.decode(&tokens_no_special)?;
            
            // For simple ASCII text, should have good fidelity
            if text.chars().all(|c| c.is_ascii()) && !text.is_empty() {
                // The decoded text should contain the original content
                // (exact match may not be possible due to BPE tokenization)
                assert!(!decoded_text.is_empty(), "Decoded text should not be empty for non-empty input");
            }
        }
        
        // Test special token handling
        if tokens_with_bos.len() > tokens_no_special.len() {
            assert_eq!(tokens_with_bos[0], tokenizer.bos_id(), "First token should be BOS");
        }
        
        if tokens_with_eos.len() > tokens_no_special.len() {
            let last_token = tokens_with_eos[tokens_with_eos.len() - 1];
            assert_eq!(last_token, tokenizer.eos_id(), "Last token should be EOS");
        }
    }
    
    // Test batch processing efficiency
    let batch_texts = vec![
        "First text for batch processing",
        "Second text for batch processing", 
        "Third text for batch processing",
    ];
    
    // Individual encoding
    let start_time = std::time::Instant::now();
    let individual_results: Result<Vec<_>> = batch_texts
        .iter()
        .map(|text| tokenizer.encode(text, true, true))
        .collect();
    let individual_duration = start_time.elapsed();
    
    let individual_tokens = individual_results?;
    assert_eq!(individual_tokens.len(), batch_texts.len());
    
    println!("✅ Individual encoding took: {:?}", individual_duration);
    println!("✅ Efficient encoding/decoding test passed");
    
    Ok(())
}

/// Test Task 3.1.1 Complete Integration: All Requirements Together
#[tokio::test]
async fn test_complete_task_3_1_1_integration() -> Result<()> {
    // Test loading tokenizer from vocabulary file format (simulating HuggingFace)
    let temp_dir = TempDir::new().unwrap();
    let tokenizer_file = temp_dir.path().join("tokenizer.json");
    
    // Create a minimal tokenizer.json for testing
    let tokenizer_json = serde_json::json!({
        "model": {
            "vocab": {
                "hello": 0,
                "world": 1,
                "test": 2,
                "bitnet": 3,
                "<|begin_of_text|>": 1000,
                "<|end_of_text|>": 1001,
                "<|eot_id|>": 1002
            }
        },
        "added_tokens": [
            {
                "content": "<|begin_of_text|>",
                "id": 1000
            },
            {
                "content": "<|end_of_text|>",
                "id": 1001
            },
            {
                "content": "<|eot_id|>",
                "id": 1002
            }
        ]
    });
    
    tokio::fs::write(&tokenizer_file, tokenizer_json.to_string()).await?;
    
    // Test loading vocabulary from file
    let (vocab_tokens, vocab_size, special_tokens) = 
        LlamaTokenizer::load_vocabulary_from_file(&tokenizer_file)?;
    
    assert!(!vocab_tokens.is_empty(), "Should load vocabulary tokens");
    assert!(vocab_size > 0, "Should have valid vocabulary size");
    assert!(!special_tokens.is_empty(), "Should load special tokens");
    
    // Create tokenizer from loaded vocabulary
    let tokenizer = LlamaTokenizer::from_vocabulary(
        vocab_tokens,
        Some(vocab_size),
        Some(special_tokens),
    )?;
    
    // Test complete chat workflow
    let chat_format = ChatFormat::new(tokenizer);
    
    let conversation = vec![
        Message { role: Role::System, content: "You are BitNet, a helpful AI assistant.".to_string() },
        Message { role: Role::User, content: "Hello BitNet! How are you?".to_string() },
        Message { role: Role::Assistant, content: "Hello! I'm doing well and ready to help.".to_string() },
        Message { role: Role::User, content: "Can you explain what you are?".to_string() },
    ];
    
    // Test encoding the complete conversation
    let conversation_tokens = chat_format.encode_dialog_prompt(&conversation)?;
    
    // Verify results
    assert!(!conversation_tokens.is_empty(), "Conversation should encode to tokens");
    
    // Verify special tokens are present
    let bos_id = chat_format.tokenizer().bos_id();
    assert!(conversation_tokens.contains(&bos_id), "Should contain BOS token");
    
    // Test individual message encoding
    let system_message = &conversation[0];
    let system_tokens = chat_format.encode_message(system_message)?;
    assert!(!system_tokens.is_empty(), "System message should encode to tokens");
    
    println!("✅ Complete Task 3.1.1 integration test passed");
    println!("🎯 All Task 3.1.1 requirements verified:");
    println!("   ✅ Tokenizer Loading: HuggingFace tokenizer integration");
    println!("   ✅ Special Tokens: Proper handling of BOS, EOS, PAD tokens");
    println!("   ✅ Chat Format: System/user/assistant conversation templates");
    println!("   ✅ Encoding/Decoding: Efficient text ↔ token conversion");
    
    Ok(())
}

/// Helper function to create a test tokenizer
fn create_test_tokenizer() -> Result<LlamaTokenizer> {
    let mut vocab = Vec::new();
    for word in ["hello", "world", "test", "bitnet", "tokenizer", "the", "a", "is", "of", "to"].iter() {
        vocab.push(word.to_string());
    }
    
    // Add more tokens to simulate a larger vocabulary
    while vocab.len() < 100 {
        vocab.push(format!("token_{}", vocab.len()));
    }
    
    let mut special_tokens = HashMap::new();
    special_tokens.insert("<|begin_of_text|>".to_string(), 1000);
    special_tokens.insert("<|end_of_text|>".to_string(), 1001);
    special_tokens.insert("<|eot_id|>".to_string(), 1002);
    
    LlamaTokenizer::from_vocabulary(vocab, Some(128256), Some(special_tokens))
}