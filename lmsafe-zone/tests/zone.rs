use lmsafe_zone::storage::*;
use lmsafe_zone::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use sysreflect_model::ModelFamily;

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

// ─────────────────────────────────────────────────────────────────────────────
// Admission control: the single rule every consumer renders.
// These pin the semantics that used to differ between core, CLI and TUI.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_admission_zero_byte_model_is_rejected() {
    // The old CLI/TUI copies answered ACCEPTED here (required = 0).
    let report = assess_admission(Some(8_000_000_000), 0, 1.15);
    assert_eq!(report.verdict, Verdict::Rejected);
    assert!(!report.is_accepted());
}

#[test]
fn test_admission_unknown_when_telemetry_restricted() {
    // The old core copy answered "assume safe" here.
    let report = assess_admission(None, 100, 1.15);
    assert_eq!(report.verdict, Verdict::Unknown);
    assert!(!report.is_accepted());
    assert_eq!(report.available_memory, None);
}

#[test]
fn test_admission_accepted_when_free_memory_exceeds_requirement() {
    // 100 * 2.0 = 200 required, 1000 available. Binary-exact values on
    // purpose: f64 margins like 1.15 truncate (100 * 1.15 -> 114).
    let report = assess_admission(Some(1000), 100, 2.0);
    assert_eq!(report.verdict, Verdict::Accepted);
    assert!(report.is_accepted());
    assert_eq!(report.required_bytes, 200);
    assert_eq!(report.file_size, 100);
    assert_eq!(report.safety_margin, 2.0);
}

#[test]
fn test_admission_rejected_when_free_memory_is_short() {
    let report = assess_admission(Some(100), 100, 1.15);
    assert_eq!(report.verdict, Verdict::Rejected);
}

#[test]
fn test_admission_rejected_when_free_memory_equals_requirement() {
    // Strict inequality: exactly enough headroom is not enough headroom.
    let report = assess_admission(Some(200), 100, 2.0);
    assert_eq!(report.verdict, Verdict::Rejected);
}

#[test]
fn test_admission_zero_margin_disables_headroom_gate() {
    let report = assess_admission(Some(100), 100, 0.0);
    assert_eq!(report.verdict, Verdict::Accepted);
    assert_eq!(report.required_bytes, 0);
}

#[test]
fn test_admission_negative_margin_fails_safe_to_default() {
    // Invalid input must not silently disable the gate: 100 * 1.2 = 120.
    let report = assess_admission(Some(1000), 100, -1.0);
    assert_eq!(report.safety_margin, 1.2);
    assert_eq!(report.required_bytes, 120);
    assert_eq!(report.verdict, Verdict::Accepted);
}

#[test]
fn test_admission_rejects_when_zero_memory_available() {
    let report = assess_admission(Some(0), 100, 1.15);
    assert_eq!(report.verdict, Verdict::Rejected);
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
    fs::write(&path, vec![0u8; 100]).unwrap();
    let result = model_is_safe_to_load(&path, 1.15);
    let _ = result;
    let _ = fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe zone storage
// ─────────────────────────────────────────────────────────────────────────────

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
    fs::write(&f1, vec![0u8; 2048]).unwrap();
    fs::write(&f2, vec![0u8; 512]).unwrap();

    let files = list_media_safe_zone(MediaFamily::Image).unwrap();
    assert!(
        files
            .iter()
            .any(|f| f.file_name == "photo.jpg" && f.size_bytes == 2048)
    );
    assert!(
        files
            .iter()
            .any(|f| f.file_name == "icon.png" && f.size_bytes == 512)
    );
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
    fs::write(&remove, vec![0u8; 4096]).unwrap();
    fs::write(&keep, vec![0u8; 1024]).unwrap();

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
fn test_is_storage_safe_for_duplication() {
    let _guard = STORAGE_TEST_LOCK.lock().unwrap();
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.bin");
    fs::write(&path, vec![0u8; 1024]).unwrap();
    let result = is_storage_safe_for_duplication(&path, 1.2);
    let _ = result;
    assert!(!is_storage_safe_for_duplication(
        "/nonexistent/file.bin",
        1.2
    ));
    let _ = fs::remove_dir_all(&dir);
}
