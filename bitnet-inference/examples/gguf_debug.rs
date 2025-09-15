//! GGUF Debugging Tool
//!
//! This tool helps debug GGUF parsing issues by providing detailed parsing information

use bitnet_inference::GgufValueType;
use std::io::{Read, Seek, SeekFrom};
use std::fs::File;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let path = "/Users/wavegoodvybe/Library/Caches/bitnet-inference/huggingface/microsoft/bitnet-b1.58-2B-4T-gguf/main/ggml-model-i2_s.gguf";
    
    println!("Debugging GGUF file: {}", path);
    
    let mut file = File::open(path)?;
    
    // Get file size
    let file_size = file.metadata()?.len();
    println!("File size: {} bytes ({:.2} MB)", file_size, file_size as f64 / (1024.0 * 1024.0));
    
    // Read magic
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    println!("Magic: {:?} (expected: GGUF)", std::str::from_utf8(&magic));
    
    // Read version
    let version = read_u32(&mut file)?;
    println!("Version: {}", version);
    
    // Read tensor count
    let tensor_count = read_u64(&mut file)?;
    println!("Tensor count: {}", tensor_count);
    
    // Read metadata KV count
    let metadata_kv_count = read_u64(&mut file)?;
    println!("Metadata KV count: {}", metadata_kv_count);
    
    // Skip metadata for now to find tensor information
    for i in 0..metadata_kv_count {
        println!("Skipping metadata KV {} ...", i);
        
        let key = read_string(&mut file)?;
        println!("Key: {}", key);
        
        // Read value type
        let value_type_raw = read_u32(&mut file)?;
        
        match GgufValueType::try_from(value_type_raw) {
            Ok(value_type) => {
                // Skip the actual value
                match value_type {
                    GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool => {
                        file.seek(SeekFrom::Current(1))?;
                    }
                    GgufValueType::Uint16 | GgufValueType::Int16 => {
                        file.seek(SeekFrom::Current(2))?;
                    }
                    GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => {
                        file.seek(SeekFrom::Current(4))?;
                    }
                    GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64 => {
                        file.seek(SeekFrom::Current(8))?;
                    }
                    GgufValueType::String => {
                        let len = read_u64(&mut file)?;
                        file.seek(SeekFrom::Current(len as i64))?;
                    }
                    GgufValueType::Array => {
                        let array_type = read_u32(&mut file)?;
                        let array_len = read_u64(&mut file)?;
                        println!("Array type: {} (0x{:x}), length: {}", array_type, array_type, array_len);
                        
                        // Calculate array skip size based on type
                        let element_size = match array_type {
                            8 => { // String array - need to read each string length
                                for _ in 0..array_len {
                                    let str_len = read_u64(&mut file)?;
                                    file.seek(SeekFrom::Current(str_len as i64))?;
                                }
                                0 // Already skipped
                            }
                            0 | 1 | 7 => 1, // UINT8, INT8, BOOL
                            2 | 3 => 2,     // UINT16, INT16
                            4 | 5 | 6 => 4, // UINT32, INT32, FLOAT32
                            10 | 11 | 12 => 8, // UINT64, INT64, FLOAT64
                            _ => {
                                println!("Unknown array element type: {}", array_type);
                                4 // Default fallback
                            }
                        };
                        
                        if element_size > 0 {
                            file.seek(SeekFrom::Current((array_len * element_size) as i64))?;
                        }
                    }
                }
            }
            Err(e) => {
                println!("ERROR: {}", e);
                break;
            }
        }
    }
    
    println!("\n--- Tensor Information ---");
    let tensor_info_start = file.stream_position()?;
    println!("Tensor info starts at offset: {}", tensor_info_start);
    
    // Read first few tensors to understand the structure
    for i in 0..std::cmp::min(5, tensor_count) {
        println!("\n--- Tensor {} ---", i);
        
        let name = read_string(&mut file)?;
        println!("Name: {}", name);
        
        let n_dimensions = read_u32(&mut file)?;
        println!("Dimensions: {}", n_dimensions);
        
        let mut dimensions = Vec::new();
        for d in 0..n_dimensions {
            let dim = read_u64(&mut file)?;
            dimensions.push(dim);
            println!("  Dim {}: {}", d, dim);
        }
        
        let tensor_type = read_u32(&mut file)?;
        println!("Tensor type: {} (0x{:x})", tensor_type, tensor_type);
        
        let offset = read_u64(&mut file)?;
        println!("Offset: {} (0x{:x})", offset, offset);
        
        // Calculate expected tensor size
        let element_count: u64 = dimensions.iter().product();
        println!("Element count: {}", element_count);
        
        // Check if offset is reasonable
        if offset > file_size {
            println!("⚠️  Offset {} exceeds file size {}", offset, file_size);
        }
        
        // Show how much data is available from this offset
        let remaining = file_size.saturating_sub(offset);
        println!("Data available from offset: {} bytes", remaining);
    }
    
    // Calculate total header size
    let tensor_data_start = file.stream_position()?;
    println!("\nTensor data should start at offset: {}", tensor_data_start);
    println!("Actual file size: {}", file_size);
    println!("Header size: {} bytes", tensor_data_start);
    
    Ok(())
}

fn read_u32<R: Read>(reader: &mut R) -> std::result::Result<u32, Box<dyn std::error::Error>> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> std::result::Result<u64, Box<dyn std::error::Error>> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_string<R: Read>(reader: &mut R) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let len = read_u64(reader)?;
    let mut bytes = vec![0u8; len as usize];
    reader.read_exact(&mut bytes)?;
    Ok(String::from_utf8(bytes)?)
}