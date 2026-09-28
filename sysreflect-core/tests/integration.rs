use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use sysreflect_core::model::*;
use sysreflect_core::storage::*;
use sysreflect_core::*;

static COUNTER: AtomicU32 = AtomicU32::new(0);

// Storage tests share one process-global SafeZone base path (first
// init_storage wins), so they must not run concurrently, and tests that
// assert on directory contents clean their zone first.
static STORAGE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn unique_temp_dir() -> PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    std::env::temp_dir().join(format!("lmsafe_test_{}_{}", std::process::id(), id))
}

fn try_init_storage(dir: &Path) -> bool {
    let storage_path = dir.join("storage");
    match init_storage(&storage_path, false) {
        Ok(()) => true,
        Err(_) => false, // Already initialized
    }
}

#[test]
fn test_system_monitor_creation() {
    let monitor = SystemMonitor::new();
    // Should not panic
    let _ = monitor.global_cpu_usage();
    let _ = monitor.memory_info();
    let _ = monitor.core_count();
}

#[test]
fn test_system_monitor_host_info() {
    let monitor = SystemMonitor::new();
    let (host, os, kernel, uptime) = monitor.host_info();
    assert!(!host.is_empty());
    assert!(!os.is_empty());
    // kernel can be empty on some systems
    let _ = kernel;
    // uptime is a u64, always >= 0
    let _ = uptime;
}

#[test]
fn test_system_monitor_refresh() {
    let mut monitor = SystemMonitor::new();
    monitor.refresh();
    // After refresh, we should still be able to get values
    let _ = monitor.global_cpu_usage();
    let _ = monitor.memory_info();
}

#[test]
fn test_memory_info_returns_valid_values() {
    let mut monitor = SystemMonitor::new();
    monitor.refresh();
    if let MetricStatus::Available((total, used)) = monitor.memory_info() {
        assert!(total > 0, "Total memory should be > 0");
        assert!(used <= total, "Used memory should be <= total");
    }
}

#[test]
fn test_core_count_positive() {
    let monitor = SystemMonitor::new();
    let cores = monitor.core_count();
    assert!(cores > 0, "Core count should be > 0");
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
    fs::write(dir.join("a.bin"), &vec![0u8; 500]).unwrap();
    fs::write(dir.join("sub").join("b.bin"), &vec![0u8; 700]).unwrap();
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
fn test_model_is_safe_to_load_nonexistent() {
    assert!(!model_is_safe_to_load("/nonexistent/model.gguf", 1.15));
}

#[test]
fn test_model_is_safe_to_load_zero_size() {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("empty.bin");
    fs::write(&path, b"").unwrap();
    assert!(!model_is_safe_to_load(&path, 1.15));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_model_is_safe_to_load_small_file() {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.bin");
    fs::write(&path, &vec![0u8; 100]).unwrap();
    let result = model_is_safe_to_load(&path, 1.15);
    let _ = result;
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_storage_init_and_get_base_path() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    try_init_storage(&dir);
    let base = get_base_path();
    assert!(base.to_string_lossy().contains("lmsafe_test"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_storage_get_safe_zone_dir() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    try_init_storage(&dir);
    let safe_dir = get_safe_zone_dir(ModelFamily::LlamaCpp).unwrap();
    assert!(safe_dir.to_string_lossy().contains("llama-cpp"));
    let whisper_dir = get_safe_zone_dir(ModelFamily::WhisperCpp).unwrap();
    assert!(whisper_dir.to_string_lossy().contains("whisper-cpp"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_storage_get_media_safe_zone_dir() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    try_init_storage(&dir);
    let video_dir = get_media_safe_zone_dir(MediaFamily::Video).unwrap();
    assert!(video_dir.to_string_lossy().contains("video"));
    let image_dir = get_media_safe_zone_dir(MediaFamily::Image).unwrap();
    assert!(image_dir.to_string_lossy().contains("image"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_storage_clean_safe_zone() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    try_init_storage(&dir);
    let safe_dir = get_safe_zone_dir(ModelFamily::LlamaCpp).unwrap();
    fs::create_dir_all(&safe_dir).unwrap();
    fs::write(safe_dir.join("test.gguf"), b"test").unwrap();
    assert!(safe_dir.join("test.gguf").exists());
    clean_safe_zone(ModelFamily::LlamaCpp).unwrap();
    assert!(!safe_dir.join("test.gguf").exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_storage_copy_to_safe_zone() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    try_init_storage(&dir);
    let source_dir = dir.join("source");
    fs::create_dir_all(&source_dir).unwrap();
    let source_path = source_dir.join("model.gguf");
    let data = vec![0u8; 1024];
    fs::write(&source_path, &data).unwrap();
    assert!(source_path.exists());
    let result = copy_to_safe_zone(&source_path, ModelFamily::LlamaCpp, 1.2);
    assert!(result.is_ok(), "Copy should succeed: {:?}", result.err());
    let dest = result.unwrap();
    assert!(dest.exists());
    assert_eq!(fs::metadata(&dest).unwrap().len(), 1024);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_storage_copy_to_safe_zone_nonexistent_source() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    try_init_storage(&dir);
    let result = copy_to_safe_zone("/nonexistent/file.gguf", ModelFamily::LlamaCpp, 1.2);
    assert!(result.is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_storage_get_first_stored_model() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    try_init_storage(&dir);
    clean_safe_zone(ModelFamily::LlamaCpp).unwrap();
    let safe_dir = get_safe_zone_dir(ModelFamily::LlamaCpp).unwrap();
    fs::create_dir_all(&safe_dir).unwrap();
    fs::write(safe_dir.join("model.gguf"), b"test").unwrap();
    let first = get_first_stored_model(ModelFamily::LlamaCpp).unwrap();
    assert!(first.is_some());
    assert!(first.unwrap().to_string_lossy().contains("model.gguf"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_media_family_from_extension() {
    assert_eq!(MediaFamily::from_extension("mp4"), Some(MediaFamily::Video));
    assert_eq!(MediaFamily::from_extension("mov"), Some(MediaFamily::Video));
    assert_eq!(MediaFamily::from_extension("m4v"), Some(MediaFamily::Video));
    assert_eq!(MediaFamily::from_extension("avi"), Some(MediaFamily::Video));
    assert_eq!(MediaFamily::from_extension("mkv"), Some(MediaFamily::Video));
    assert_eq!(
        MediaFamily::from_extension("webm"),
        Some(MediaFamily::Video)
    );

    assert_eq!(MediaFamily::from_extension("jpg"), Some(MediaFamily::Image));
    assert_eq!(
        MediaFamily::from_extension("jpeg"),
        Some(MediaFamily::Image)
    );
    assert_eq!(MediaFamily::from_extension("png"), Some(MediaFamily::Image));
    assert_eq!(
        MediaFamily::from_extension("webp"),
        Some(MediaFamily::Image)
    );
    assert_eq!(MediaFamily::from_extension("gif"), Some(MediaFamily::Image));
    assert_eq!(
        MediaFamily::from_extension("heic"),
        Some(MediaFamily::Image)
    );

    assert_eq!(MediaFamily::from_extension("txt"), None);
    assert_eq!(MediaFamily::from_extension("xyz"), None);
}

#[test]
fn test_media_family_as_str() {
    assert_eq!(MediaFamily::Video.as_str(), "video");
    assert_eq!(MediaFamily::Image.as_str(), "image");
}

#[test]
fn test_list_media_safe_zone_empty() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    try_init_storage(&dir);
    clean_media_safe_zone(MediaFamily::Video).unwrap();
    let files = list_media_safe_zone(MediaFamily::Video).unwrap();
    assert!(files.is_empty());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_list_media_safe_zone_with_files() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    try_init_storage(&dir);
    clean_media_safe_zone(MediaFamily::Image).unwrap();
    let media_dir = get_media_safe_zone_dir(MediaFamily::Image).unwrap();
    fs::create_dir_all(&media_dir).unwrap();
    let f1 = media_dir.join("photo.jpg");
    let f2 = media_dir.join("icon.png");
    fs::write(&f1, &vec![0u8; 2048]).unwrap();
    fs::write(&f2, &vec![0u8; 512]).unwrap();

    let files = list_media_safe_zone(MediaFamily::Image).unwrap();
    assert!(files
        .iter()
        .any(|f| f.file_name == "photo.jpg" && f.size_bytes == 2048));
    assert!(files
        .iter()
        .any(|f| f.file_name == "icon.png" && f.size_bytes == 512));
    let _ = fs::remove_file(&f1);
    let _ = fs::remove_file(&f2);
}

#[test]
fn test_remove_media_from_safe_zone() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    try_init_storage(&dir);
    clean_media_safe_zone(MediaFamily::Video).unwrap();
    let media_dir = get_media_safe_zone_dir(MediaFamily::Video).unwrap();
    fs::create_dir_all(&media_dir).unwrap();
    let keep = media_dir.join("other.mp4");
    let remove = media_dir.join("clip.mp4");
    fs::write(&remove, &vec![0u8; 4096]).unwrap();
    fs::write(&keep, &vec![0u8; 1024]).unwrap();

    remove_media_from_safe_zone(MediaFamily::Video, "clip.mp4").unwrap();

    assert!(!remove.exists());
    assert!(keep.exists());
    let _ = fs::remove_file(&keep);
}

#[test]
fn test_remove_media_from_safe_zone_not_found() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    try_init_storage(&dir);
    let media_dir = get_media_safe_zone_dir(MediaFamily::Video).unwrap();
    fs::create_dir_all(&media_dir).unwrap();

    let result = remove_media_from_safe_zone(MediaFamily::Video, "nonexistent.mp4");
    assert!(result.is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_remove_media_from_safe_zone_path_traversal() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    try_init_storage(&dir);
    let media_dir = get_media_safe_zone_dir(MediaFamily::Video).unwrap();
    fs::create_dir_all(&media_dir).unwrap();

    let result = remove_media_from_safe_zone(MediaFamily::Video, "../../etc/passwd");
    assert!(result.is_err());
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
    fs::write(&path, &vec![0u8; 100]).unwrap();
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
    fs::write(&path, &vec![0u8; 100]).unwrap();
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
    assert!(report
        .inferred_capabilities
        .contains(&Capability::TextGeneration));
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
    assert!(report
        .inferred_capabilities
        .contains(&Capability::Multimodal));
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
    assert!(report
        .inferred_capabilities
        .contains(&Capability::TextGeneration));
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
    assert!(report
        .inferred_capabilities
        .contains(&Capability::SpeechToText));
    assert!(report
        .inferred_capabilities
        .contains(&Capability::AudioInput));
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
    assert!(report
        .inferred_capabilities
        .contains(&Capability::TextGeneration));
    assert!(report.inferred_capabilities.contains(&Capability::Tools));
}

#[test]
fn test_is_storage_safe_for_duplication() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.bin");
    fs::write(&path, &vec![0u8; 1024]).unwrap();
    let result = is_storage_safe_for_duplication(&path, 1.2);
    let _ = result;
    assert!(!is_storage_safe_for_duplication(
        "/nonexistent/file.bin",
        1.2
    ));
    let _ = fs::remove_dir_all(&dir);
}
