use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use sysreflect_model::*;

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn unique_temp_dir() -> PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    std::env::temp_dir().join(format!("lmsafe_test_{}_{}", std::process::id(), id))
}

#[test]
fn test_model_family_detection_unknown() {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("unknown.xyz");
    fs::write(&path, b"random data").unwrap();
    assert_eq!(detect_model_family(&path), ModelFamily::Unknown);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_model_family_detection_gguf() {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.gguf");
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    fs::write(&path, &data).unwrap();
    assert_eq!(fs::metadata(&path).unwrap().len(), 24);
    assert_eq!(detect_model_family(&path), ModelFamily::LlamaCpp);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_model_family_detection_litertlm() {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.litertlm");
    let mut data = vec![0u8; 32];
    data[0..8].copy_from_slice(b"LITERTLM");
    fs::write(&path, &data).unwrap();
    assert_eq!(detect_model_family(&path), ModelFamily::LiteRtLm);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_model_family_detection_safetensors() {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.safetensors");
    let json_header = r#"{"__metadata__":{"name":"Test"}}"#;
    let json_bytes = json_header.as_bytes();
    let header_len = json_bytes.len() as u64;
    let mut data = Vec::new();
    data.extend_from_slice(&header_len.to_le_bytes());
    data.extend_from_slice(json_bytes);
    fs::write(&path, &data).unwrap();
    assert_eq!(detect_model_family(&path), ModelFamily::SafeTensors);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_model_family_detection_onnx_file() {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.onnx");
    fs::write(&path, b"\x00\x00\x00\x00").unwrap();
    assert_eq!(detect_model_family(&path), ModelFamily::Onnx);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_model_family_detection_onnx_dir() {
    let dir = unique_temp_dir().join("onnx_model");
    let onnx_subdir = dir.join("onnx");
    fs::create_dir_all(&onnx_subdir).unwrap();
    fs::write(onnx_subdir.join("model.onnx"), b"\x00").unwrap();
    assert!(onnx_subdir.join("model.onnx").exists());
    assert_eq!(detect_model_family(&dir), ModelFamily::Onnx);
    let _ = fs::remove_dir_all(unique_temp_dir());
}

#[test]
fn test_get_recursive_size_file() {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.bin");
    let data = vec![0u8; 1024];
    fs::write(&path, &data).unwrap();
    assert!(path.exists());
    assert_eq!(fs::metadata(&path).unwrap().len(), 1024);
    assert_eq!(get_recursive_size(&path), 1024);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_get_recursive_size_directory() {
    let dir = unique_temp_dir().join("multi");
    fs::create_dir_all(dir.join("sub")).unwrap();
    fs::write(dir.join("a.bin"), vec![0u8; 500]).unwrap();
    fs::write(dir.join("sub").join("b.bin"), vec![0u8; 700]).unwrap();
    assert!(dir.join("a.bin").exists());
    assert!(dir.join("sub").join("b.bin").exists());
    assert_eq!(get_recursive_size(&dir), 1200);
    let _ = fs::remove_dir_all(dir.parent().unwrap());
}

#[test]
fn test_get_recursive_size_empty() {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("empty.bin");
    fs::write(&path, b"").unwrap();
    assert_eq!(get_recursive_size(&path), 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_model_family_as_str_and_from_str() {
    assert_eq!(ModelFamily::LlamaCpp.as_str(), "llama-cpp");
    assert_eq!(ModelFamily::WhisperCpp.as_str(), "whisper-cpp");
    assert_eq!(ModelFamily::LiteRtLm.as_str(), "litert-lm");
    assert_eq!(ModelFamily::Onnx.as_str(), "onnx");
    assert_eq!(ModelFamily::SafeTensors.as_str(), "safetensors");
    assert_eq!(
        ModelFamily::AudioSegmentation.as_str(),
        "audio-segmentation"
    );
    assert_eq!(ModelFamily::Unknown.as_str(), "unknown");

    assert_eq!(ModelFamily::from_str("llama-cpp"), ModelFamily::LlamaCpp);
    assert_eq!(
        ModelFamily::from_str("whisper-cpp"),
        ModelFamily::WhisperCpp
    );
    assert_eq!(ModelFamily::from_str("litert-lm"), ModelFamily::LiteRtLm);
    assert_eq!(ModelFamily::from_str("onnx"), ModelFamily::Onnx);
    assert_eq!(
        ModelFamily::from_str("safetensors"),
        ModelFamily::SafeTensors
    );
    assert_eq!(ModelFamily::from_str("unknown"), ModelFamily::Unknown);
    assert_eq!(ModelFamily::from_str("invalid"), ModelFamily::Unknown);
}

#[test]
fn test_inspect_media_file_image() {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.jpg");
    fs::write(&path, vec![0u8; 100]).unwrap();
    let result = inspect_media_file(&path);
    assert!(result.is_ok());
    let report = result.unwrap();
    assert_eq!(report.file_name, "test.jpg");
    assert_eq!(report.format, "image/jpg");
    assert!(report.is_supported);
    assert_eq!(report.size_bytes, 100);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_inspect_media_file_nonexistent() {
    let result = inspect_media_file("/nonexistent/file.jpg");
    assert!(result.is_err());
}

#[test]
fn test_inspect_media_file_unsupported() {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.xyz");
    fs::write(&path, vec![0u8; 100]).unwrap();
    let result = inspect_media_file(&path);
    assert!(result.is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_capability_inference_text_generation() {
    let mut report = ModelInspectionReport {
        family: ModelFamily::LlamaCpp,
        name: "test".to_string(),
        version: "1.0".to_string(),
        author: "test".to_string(),
        uuid: String::new(),
        capabilities: ModelCapabilities {
            is_instruct: true,
            has_chat_template: true,
            has_tool_support: false,
            has_json_support: false,
            is_moe: false,
            is_vision: false,
            context_length: 2048,
            quantization: Some("Q4_K_M".to_string()),
            parameter_count: Some("7B".to_string()),
        },
        inferred_capabilities: Vec::new(),
        raw_metadata: std::collections::HashMap::new(),
    };
    report.infer_capabilities();
    assert!(
        report
            .inferred_capabilities
            .contains(&Capability::TextGeneration)
    );
}

#[test]
fn test_capability_inference_vision() {
    let mut report = ModelInspectionReport {
        family: ModelFamily::LlamaCpp,
        name: "vision-model".to_string(),
        version: "1.0".to_string(),
        author: "test".to_string(),
        uuid: String::new(),
        capabilities: ModelCapabilities {
            is_instruct: false,
            has_chat_template: false,
            has_tool_support: false,
            has_json_support: false,
            is_moe: false,
            is_vision: true,
            context_length: 4096,
            quantization: None,
            parameter_count: None,
        },
        inferred_capabilities: Vec::new(),
        raw_metadata: std::collections::HashMap::new(),
    };
    report.infer_capabilities();
    assert!(report.inferred_capabilities.contains(&Capability::Vision));
    assert!(
        report
            .inferred_capabilities
            .contains(&Capability::Multimodal)
    );
}

#[test]
fn test_capability_inference_tools() {
    let mut report = ModelInspectionReport {
        family: ModelFamily::LlamaCpp,
        name: "tool-model".to_string(),
        version: "1.0".to_string(),
        author: "test".to_string(),
        uuid: String::new(),
        capabilities: ModelCapabilities {
            is_instruct: true,
            has_chat_template: true,
            has_tool_support: true,
            has_json_support: true,
            is_moe: false,
            is_vision: false,
            context_length: 8192,
            quantization: None,
            parameter_count: None,
        },
        inferred_capabilities: Vec::new(),
        raw_metadata: std::collections::HashMap::new(),
    };
    report.infer_capabilities();
    assert!(
        report
            .inferred_capabilities
            .contains(&Capability::TextGeneration)
    );
    assert!(report.inferred_capabilities.contains(&Capability::Tools));
}

#[test]
fn test_capability_inference_whisper() {
    let mut report = ModelInspectionReport {
        family: ModelFamily::WhisperCpp,
        name: "whisper-base".to_string(),
        version: "1.0".to_string(),
        author: "OpenAI".to_string(),
        uuid: String::new(),
        capabilities: ModelCapabilities {
            is_instruct: false,
            has_chat_template: false,
            has_tool_support: false,
            has_json_support: false,
            is_moe: false,
            is_vision: false,
            context_length: 0,
            quantization: None,
            parameter_count: None,
        },
        inferred_capabilities: Vec::new(),
        raw_metadata: std::collections::HashMap::new(),
    };
    report.infer_capabilities();
    assert!(
        report
            .inferred_capabilities
            .contains(&Capability::SpeechToText)
    );
    assert!(
        report
            .inferred_capabilities
            .contains(&Capability::AudioInput)
    );
}

#[test]
fn test_capability_inference_litertlm() {
    let mut report = ModelInspectionReport {
        family: ModelFamily::LiteRtLm,
        name: "gemma-3".to_string(),
        version: "1.0".to_string(),
        author: "Google".to_string(),
        uuid: String::new(),
        capabilities: ModelCapabilities {
            is_instruct: true,
            has_chat_template: true,
            has_tool_support: true,
            has_json_support: false,
            is_moe: false,
            is_vision: false,
            context_length: 8192,
            quantization: None,
            parameter_count: None,
        },
        inferred_capabilities: Vec::new(),
        raw_metadata: std::collections::HashMap::new(),
    };
    report.infer_capabilities();
    assert!(
        report
            .inferred_capabilities
            .contains(&Capability::TextGeneration)
    );
    assert!(report.inferred_capabilities.contains(&Capability::Tools));
}
