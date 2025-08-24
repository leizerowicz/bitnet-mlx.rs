//! Tokenizer Demo
//!
//! This example demonstrates the usage of the text encoding/decoding functions
//! provided by the BitNet tokenizer module.

use anyhow::Result;
use bitnet_core::tokenizer::{create_simple_tokenizer, decode_tokens, encode_batch, encode_text};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("BitNet Tokenizer Demo");
    println!("====================");

    // Create a simple tokenizer for demonstration
    let tokenizer = create_demo_tokenizer();

    // Demo 1: Basic text encoding
    println!("\n1. Basic Text Encoding:");
    let text = "hello world from bitnet";
    let tokens = encode_text(&tokenizer, text)?;
    println!("Text: '{text}'");
    println!("Tokens: {tokens:?}");

    // Demo 2: Token decoding
    println!("\n2. Token Decoding:");
    let decoded_text = decode_tokens(&tokenizer, &tokens)?;
    println!("Tokens: {tokens:?}");
    println!("Decoded: '{decoded_text}'");

    // Demo 3: Batch encoding
    println!("\n3. Batch Encoding:");
    let texts = vec![
        "hello world",
        "bitnet is awesome",
        "machine learning rocks",
        "hello bitnet",
    ];

    let batch_tokens = encode_batch(&tokenizer, &texts)?;
    println!("Input texts:");
    for (i, text) in texts.iter().enumerate() {
        println!("  {i}: '{text}'");
    }

    println!("Batch tokens:");
    for (i, tokens) in batch_tokens.iter().enumerate() {
        println!("  {i}: {tokens:?}");
    }

    // Demo 4: Round-trip encoding/decoding for batch
    println!("\n4. Round-trip Batch Processing:");
    for (i, tokens) in batch_tokens.iter().enumerate() {
        let decoded = decode_tokens(&tokenizer, tokens)?;
        println!("  Original: '{}' -> Decoded: '{}'", texts[i], decoded);
    }

    // Demo 5: Handling unknown tokens
    println!("\n5. Unknown Token Handling:");
    let text_with_unknown = "hello unknown_word world";
    let tokens_with_unk = encode_text(&tokenizer, text_with_unknown)?;
    let decoded_with_unk = decode_tokens(&tokenizer, &tokens_with_unk)?;
    println!("Text: '{text_with_unknown}'");
    println!("Tokens: {tokens_with_unk:?}");
    println!("Decoded: '{decoded_with_unk}'");

    println!("\nDemo completed successfully!");
    Ok(())
}

/// Create a demo tokenizer with a simple vocabulary
fn create_demo_tokenizer() -> bitnet_core::tokenizer::Tokenizer {
    // Create vocabulary map
    let mut vocab = HashMap::new();
    vocab.insert("hello".to_string(), 0);
    vocab.insert("world".to_string(), 1);
    vocab.insert("from".to_string(), 2);
    vocab.insert("bitnet".to_string(), 3);
    vocab.insert("is".to_string(), 4);
    vocab.insert("awesome".to_string(), 5);
    vocab.insert("machine".to_string(), 6);
    vocab.insert("learning".to_string(), 7);
    vocab.insert("rocks".to_string(), 8);
    vocab.insert("<unk>".to_string(), 9); // Unknown token

    // Create tokenizer using the new public constructor
    create_simple_tokenizer(vocab)
}
