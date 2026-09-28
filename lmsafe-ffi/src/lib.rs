#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use sysreflect_core::model::{inspect_model_file, model_is_safe_to_load, ModelFamily};
use sysreflect_core::storage::{
    clean_safe_zone, copy_to_safe_zone, get_first_stored_model, init_storage, list_media_safe_zone,
    remove_media_from_safe_zone,
};
use sysreflect_core::{MetricStatus, SystemMonitor};

// ──────────────────────────────────────────────────────────────────────────────
// Error Handling Infrastructure
// ──────────────────────────────────────────────────────────────────────────────

/// Error codes returned by lmsafe functions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmsafeError {
    Ok = 0,
    NullPointer = 1,
    InvalidUtf8 = 2,
    IoError = 3,
    ParseError = 4,
    InsufficientSpace = 5,
    ModelNotFound = 6,
    UnknownFamily = 7,
    StorageAlreadyInit = 8,
    TelemetryRestricted = 9,
}

thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(msg);
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

/// Get the last error message. Returns null if no error.
/// The caller MUST free the returned string with lmsafe_free_string.
#[no_mangle]
pub extern "C" fn lmsafe_last_error() -> *mut c_char {
    LAST_ERROR.with(|e| {
        let err = e.borrow();
        match err.as_ref() {
            Some(msg) => CString::new(msg.as_str())
                .unwrap_or_else(|_| CString::new("Invalid error message").unwrap())
                .into_raw(),
            None => std::ptr::null_mut(),
        }
    })
}

/// Get the last error code. Returns 0 (LMSAFE_OK) if no error.
#[no_mangle]
pub extern "C" fn lmsafe_last_error_code() -> std::os::raw::c_int {
    LAST_ERROR.with(|e| {
        let err = e.borrow();
        match err.as_ref() {
            None => LmsafeError::Ok as std::os::raw::c_int,
            Some(msg) => {
                let lower = msg.to_lowercase();
                if lower.contains("null") {
                    LmsafeError::NullPointer as std::os::raw::c_int
                } else if lower.contains("utf-8") || lower.contains("utf8") {
                    LmsafeError::InvalidUtf8 as std::os::raw::c_int
                } else if lower.contains("no such file") || lower.contains("not found") {
                    LmsafeError::ModelNotFound as std::os::raw::c_int
                } else if lower.contains("insufficient") || lower.contains("space") {
                    LmsafeError::InsufficientSpace as std::os::raw::c_int
                } else if lower.contains("already") {
                    LmsafeError::StorageAlreadyInit as std::os::raw::c_int
                } else if lower.contains("restricted") || lower.contains("denied") {
                    LmsafeError::TelemetryRestricted as std::os::raw::c_int
                } else if lower.contains("parse") || lower.contains("invalid") {
                    LmsafeError::ParseError as std::os::raw::c_int
                } else {
                    LmsafeError::IoError as std::os::raw::c_int
                }
            }
        }
    })
}

#[no_mangle]
pub extern "C" fn lmsafe_new() -> *mut SystemMonitor {
    Box::into_raw(Box::new(SystemMonitor::new()))
}

#[no_mangle]
pub extern "C" fn lmsafe_free(ptr: *mut SystemMonitor) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_refresh(ptr: *mut SystemMonitor) {
    if ptr.is_null() {
        return;
    }
    let monitor = unsafe { &mut *ptr };
    monitor.refresh();
}

#[no_mangle]
pub extern "C" fn lmsafe_get_cpu_usage(ptr: *mut SystemMonitor) -> f32 {
    if ptr.is_null() {
        return -1.0;
    }
    let monitor = unsafe { &*ptr };
    match monitor.global_cpu_usage() {
        MetricStatus::Available(v) => v,
        MetricStatus::Restricted => -1.0,
        MetricStatus::Unavailable => -2.0,
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_get_total_memory(ptr: *mut SystemMonitor) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    let monitor = unsafe { &*ptr };
    match monitor.memory_info() {
        MetricStatus::Available((total, _)) => total,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_get_used_memory(ptr: *mut SystemMonitor) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    let monitor = unsafe { &*ptr };
    match monitor.memory_info() {
        MetricStatus::Available((_, used)) => used,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_get_core_count(ptr: *mut SystemMonitor) -> usize {
    if ptr.is_null() {
        return 0;
    }
    let monitor = unsafe { &*ptr };
    monitor.core_count()
}

#[no_mangle]
pub extern "C" fn lmsafe_get_host_name(ptr: *mut SystemMonitor) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    let monitor = unsafe { &*ptr };
    let (host, _, _, _) = monitor.host_info();
    CString::new(host)
        .unwrap_or_else(|_| CString::new("Invalid").unwrap())
        .into_raw()
}

#[no_mangle]
pub extern "C" fn lmsafe_get_os_version(ptr: *mut SystemMonitor) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    let monitor = unsafe { &*ptr };
    let (_, os, _, _) = monitor.host_info();
    CString::new(os)
        .unwrap_or_else(|_| CString::new("Invalid").unwrap())
        .into_raw()
}

#[no_mangle]
pub extern "C" fn lmsafe_get_kernel_version(ptr: *mut SystemMonitor) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    let monitor = unsafe { &*ptr };
    let (_, _, kernel, _) = monitor.host_info();
    CString::new(kernel)
        .unwrap_or_else(|_| CString::new("Invalid").unwrap())
        .into_raw()
}

#[no_mangle]
pub extern "C" fn lmsafe_get_uptime(ptr: *mut SystemMonitor) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    let monitor = unsafe { &*ptr };
    let (_, _, _, uptime) = monitor.host_info();
    uptime
}

#[no_mangle]
pub extern "C" fn lmsafe_get_net_rx(ptr: *mut SystemMonitor) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    let monitor = unsafe { &*ptr };
    match monitor.network_throughput() {
        MetricStatus::Available((rx, _)) => rx,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_get_net_tx(ptr: *mut SystemMonitor) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    let monitor = unsafe { &*ptr };
    match monitor.network_throughput() {
        MetricStatus::Available((_, tx)) => tx,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_inspect_model(path: *const c_char) -> *mut c_char {
    clear_last_error();
    if path.is_null() {
        set_last_error("Null pointer passed for path".to_string());
        return std::ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in path: {}", e));
            return std::ptr::null_mut();
        }
    };

    match inspect_model_file(path_str) {
        Ok(report) => {
            let json = serde_json::to_string(&report).unwrap_or_else(|_| "{}".to_string());
            CString::new(json)
                .unwrap_or_else(|_| CString::new("{}").unwrap())
                .into_raw()
        }
        Err(e) => {
            set_last_error(format!("Failed to inspect model: {}", e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_model_is_safe_to_load(path: *const c_char, safety_margin: f64) -> bool {
    clear_last_error();
    if path.is_null() {
        set_last_error("Null pointer passed for path".to_string());
        return false;
    }
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in path: {}", e));
            return false;
        }
    };

    model_is_safe_to_load(path_str, safety_margin)
}

#[no_mangle]
pub extern "C" fn lmsafe_init_storage(base_path: *const c_char, sandboxed: bool) -> bool {
    clear_last_error();
    if base_path.is_null() {
        set_last_error("Null pointer passed for base_path".to_string());
        return false;
    }
    let c_str = unsafe { CStr::from_ptr(base_path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in base_path: {}", e));
            return false;
        }
    };

    match init_storage(path_str, sandboxed) {
        Ok(()) => true,
        Err(e) => {
            set_last_error(format!("Failed to initialize storage: {}", e));
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_copy_to_safe_zone(
    source_path: *const c_char,
    family_str: *const c_char,
) -> *mut c_char {
    clear_last_error();
    if source_path.is_null() || family_str.is_null() {
        set_last_error("Null pointer passed for source_path or family_str".to_string());
        return std::ptr::null_mut();
    }

    let src_cstr = unsafe { CStr::from_ptr(source_path) };
    let fam_cstr = unsafe { CStr::from_ptr(family_str) };

    let src = match src_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in source_path: {}", e));
            return std::ptr::null_mut();
        }
    };
    let fam_str = match fam_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in family_str: {}", e));
            return std::ptr::null_mut();
        }
    };

    let family = ModelFamily::from_str(fam_str);

    match copy_to_safe_zone(src, family, 1.2) {
        Ok(path) => {
            let path_str = path.to_string_lossy().into_owned();
            CString::new(path_str)
                .unwrap_or_else(|_| CString::new("").unwrap())
                .into_raw()
        }
        Err(e) => {
            set_last_error(format!("Failed to copy to safe zone: {}", e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_copy_media_to_safe_zone(
    source_path_str: *const c_char,
    media_family_str: *const c_char,
    safety_margin: f64,
) -> *mut c_char {
    clear_last_error();
    if source_path_str.is_null() || media_family_str.is_null() {
        set_last_error("Null pointer passed for source_path_str or media_family_str".to_string());
        return std::ptr::null_mut();
    }
    let src_cstr = unsafe { CStr::from_ptr(source_path_str) };
    let media_cstr = unsafe { CStr::from_ptr(media_family_str) };

    let src = match src_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in source_path_str: {}", e));
            return std::ptr::null_mut();
        }
    };
    let media_str = match media_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in media_family_str: {}", e));
            return std::ptr::null_mut();
        }
    };

    let family = match sysreflect_core::storage::MediaFamily::from_extension(media_str) {
        Some(f) => f,
        None => {
            set_last_error(format!("Unknown media family: {}", media_str));
            return std::ptr::null_mut();
        }
    };

    let margin = if safety_margin <= 0.0 {
        1.2
    } else {
        safety_margin
    };

    match sysreflect_core::storage::copy_media_to_safe_zone(src, family, margin) {
        Ok(path) => {
            let path_str = path.to_string_lossy().into_owned();
            CString::new(path_str)
                .unwrap_or_else(|_| CString::new("").unwrap())
                .into_raw()
        }
        Err(e) => {
            set_last_error(format!("Failed to copy media to safe zone: {}", e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_clean_safe_zone(family_str: *const c_char) -> bool {
    clear_last_error();
    if family_str.is_null() {
        set_last_error("Null pointer passed for family_str".to_string());
        return false;
    }
    let fam_cstr = unsafe { CStr::from_ptr(family_str) };
    let fam_str = match fam_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in family_str: {}", e));
            return false;
        }
    };

    let family = ModelFamily::from_str(fam_str);

    match clean_safe_zone(family) {
        Ok(()) => true,
        Err(e) => {
            set_last_error(format!("Failed to clean safe zone: {}", e));
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_clean_media_safe_zone(media_family_str: *const c_char) -> bool {
    clear_last_error();
    if media_family_str.is_null() {
        set_last_error("Null pointer passed for media_family_str".to_string());
        return false;
    }
    let media_cstr = unsafe { CStr::from_ptr(media_family_str) };
    let media_str = match media_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in media_family_str: {}", e));
            return false;
        }
    };

    let family = match sysreflect_core::storage::MediaFamily::from_extension(media_str) {
        Some(f) => f,
        None => {
            set_last_error(format!("Unknown media family: {}", media_str));
            return false;
        }
    };

    match sysreflect_core::storage::clean_media_safe_zone(family) {
        Ok(()) => true,
        Err(e) => {
            set_last_error(format!("Failed to clean media safe zone: {}", e));
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_list_media_safe_zone(media_family_str: *const c_char) -> *mut c_char {
    clear_last_error();
    if media_family_str.is_null() {
        set_last_error("Null pointer passed for media_family_str".to_string());
        return std::ptr::null_mut();
    }
    let media_cstr = unsafe { CStr::from_ptr(media_family_str) };
    let media_str = match media_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in media_family_str: {}", e));
            return std::ptr::null_mut();
        }
    };

    let family = match sysreflect_core::storage::MediaFamily::from_extension(media_str) {
        Some(f) => f,
        None => {
            set_last_error(format!("Unknown media family: {}", media_str));
            return std::ptr::null_mut();
        }
    };

    match list_media_safe_zone(family) {
        Ok(files) => {
            let json = serde_json::to_string(&files).unwrap_or_else(|_| "[]".to_string());
            CString::new(json)
                .unwrap_or_else(|_| CString::new("[]").unwrap())
                .into_raw()
        }
        Err(e) => {
            set_last_error(format!("Failed to list media safe zone: {}", e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_remove_media_from_safe_zone(
    media_family_str: *const c_char,
    file_name: *const c_char,
) -> bool {
    clear_last_error();
    if media_family_str.is_null() || file_name.is_null() {
        set_last_error("Null pointer passed for media_family_str or file_name".to_string());
        return false;
    }
    let media_cstr = unsafe { CStr::from_ptr(media_family_str) };
    let media_str = match media_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in media_family_str: {}", e));
            return false;
        }
    };
    let name_cstr = unsafe { CStr::from_ptr(file_name) };
    let name_str = match name_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in file_name: {}", e));
            return false;
        }
    };

    let family = match sysreflect_core::storage::MediaFamily::from_extension(media_str) {
        Some(f) => f,
        None => {
            set_last_error(format!("Unknown media family: {}", media_str));
            return false;
        }
    };

    match remove_media_from_safe_zone(family, name_str) {
        Ok(()) => true,
        Err(e) => {
            set_last_error(format!("Failed to remove media from safe zone: {}", e));
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_inspect_media_file(path_str: *const c_char) -> *mut c_char {
    clear_last_error();
    if path_str.is_null() {
        set_last_error("Null pointer passed for path_str".to_string());
        return std::ptr::null_mut();
    }
    let cstr = unsafe { CStr::from_ptr(path_str) };
    let path = match cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in path_str: {}", e));
            return std::ptr::null_mut();
        }
    };

    match sysreflect_core::model::inspect_media_file(path) {
        Ok(report) => {
            let json = serde_json::to_string(&report).unwrap_or_default();
            CString::new(json)
                .unwrap_or_else(|_| CString::new("").unwrap())
                .into_raw()
        }
        Err(e) => {
            set_last_error(format!("Failed to inspect media file: {}", e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_get_first_stored_model(family_str: *const c_char) -> *mut c_char {
    clear_last_error();
    if family_str.is_null() {
        set_last_error("Null pointer passed for family_str".to_string());
        return std::ptr::null_mut();
    }
    let fam_cstr = unsafe { CStr::from_ptr(family_str) };
    let fam_str = match fam_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in family_str: {}", e));
            return std::ptr::null_mut();
        }
    };

    let family = ModelFamily::from_str(fam_str);

    match get_first_stored_model(family) {
        Ok(Some(path)) => {
            let path_str = path.to_string_lossy().into_owned();
            CString::new(path_str)
                .unwrap_or_else(|_| CString::new("").unwrap())
                .into_raw()
        }
        Ok(None) => std::ptr::null_mut(),
        Err(e) => {
            set_last_error(format!("Failed to get first stored model: {}", e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn lmsafe_get_safe_zone_dir(family_str: *const c_char) -> *mut c_char {
    clear_last_error();
    if family_str.is_null() {
        set_last_error("Null pointer passed for family_str".to_string());
        return std::ptr::null_mut();
    }
    let fam_cstr = unsafe { CStr::from_ptr(family_str) };
    let fam_str = match fam_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in family_str: {}", e));
            return std::ptr::null_mut();
        }
    };

    let family = ModelFamily::from_str(fam_str);

    match sysreflect_core::storage::get_safe_zone_dir(family) {
        Ok(path) => {
            let path_str = path.to_string_lossy().into_owned();
            CString::new(path_str)
                .unwrap_or_else(|_| CString::new("").unwrap())
                .into_raw()
        }
        Err(e) => {
            set_last_error(format!("Failed to get safe zone dir: {}", e));
            std::ptr::null_mut()
        }
    }
}
