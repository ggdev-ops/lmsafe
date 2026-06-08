use std::ffi::CString;
use std::os::raw::c_char;
use crate::{SystemMonitor, MetricStatus};

#[no_mangle]
pub extern "C" fn sysreflect_new() -> *mut SystemMonitor {
    Box::into_raw(Box::new(SystemMonitor::new()))
}

#[no_mangle]
pub extern "C" fn sysreflect_free(ptr: *mut SystemMonitor) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn sysreflect_refresh(ptr: *mut SystemMonitor) {
    if ptr.is_null() { return; }
    let monitor = unsafe { &mut *ptr };
    monitor.refresh();
}

#[no_mangle]
pub extern "C" fn sysreflect_get_cpu_usage(ptr: *mut SystemMonitor) -> f32 {
    if ptr.is_null() { return -1.0; }
    let monitor = unsafe { &*ptr };
    match monitor.global_cpu_usage() {
        MetricStatus::Available(v) => v,
        MetricStatus::Restricted => -1.0,
        MetricStatus::Unavailable => -2.0,
    }
}

#[no_mangle]
pub extern "C" fn sysreflect_get_total_memory(ptr: *mut SystemMonitor) -> u64 {
    if ptr.is_null() { return 0; }
    let monitor = unsafe { &*ptr };
    match monitor.memory_info() {
        MetricStatus::Available((total, _)) => total,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn sysreflect_get_used_memory(ptr: *mut SystemMonitor) -> u64 {
    if ptr.is_null() { return 0; }
    let monitor = unsafe { &*ptr };
    match monitor.memory_info() {
        MetricStatus::Available((_, used)) => used,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn sysreflect_get_core_count(ptr: *mut SystemMonitor) -> usize {
    if ptr.is_null() { return 0; }
    let monitor = unsafe { &*ptr };
    monitor.core_count()
}

#[no_mangle]
pub extern "C" fn sysreflect_get_host_name(ptr: *mut SystemMonitor) -> *mut c_char {
    if ptr.is_null() { return std::ptr::null_mut(); }
    let monitor = unsafe { &*ptr };
    let (host, _, _, _) = monitor.host_info();
    CString::new(host).unwrap_or_else(|_| CString::new("Invalid").unwrap()).into_raw()
}

#[no_mangle]
pub extern "C" fn sysreflect_get_os_version(ptr: *mut SystemMonitor) -> *mut c_char {
    if ptr.is_null() { return std::ptr::null_mut(); }
    let monitor = unsafe { &*ptr };
    let (_, os, _, _) = monitor.host_info();
    CString::new(os).unwrap_or_else(|_| CString::new("Invalid").unwrap()).into_raw()
}

#[no_mangle]
pub extern "C" fn sysreflect_get_kernel_version(ptr: *mut SystemMonitor) -> *mut c_char {
    if ptr.is_null() { return std::ptr::null_mut(); }
    let monitor = unsafe { &*ptr };
    let (_, _, kernel, _) = monitor.host_info();
    CString::new(kernel).unwrap_or_else(|_| CString::new("Invalid").unwrap()).into_raw()
}

#[no_mangle]
pub extern "C" fn sysreflect_get_uptime(ptr: *mut SystemMonitor) -> u64 {
    if ptr.is_null() { return 0; }
    let monitor = unsafe { &*ptr };
    let (_, _, _, uptime) = monitor.host_info();
    uptime
}

#[no_mangle]
pub extern "C" fn sysreflect_get_net_rx(ptr: *mut SystemMonitor) -> u64 {
    if ptr.is_null() { return 0; }
    let monitor = unsafe { &*ptr };
    match monitor.network_throughput() {
        MetricStatus::Available((rx, _)) => rx,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn sysreflect_get_net_tx(ptr: *mut SystemMonitor) -> u64 {
    if ptr.is_null() { return 0; }
    let monitor = unsafe { &*ptr };
    match monitor.network_throughput() {
        MetricStatus::Available((_, tx)) => tx,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn sysreflect_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn test_ffi_lifecycle() {
        unsafe {
            let monitor = sysreflect_new();
            assert!(!monitor.is_null());
            
            sysreflect_refresh(monitor);
            
            let _cpu = sysreflect_get_cpu_usage(monitor);
            assert!(_cpu >= -2.0);
            
            let _mem = sysreflect_get_total_memory(monitor);
            
            let host_ptr = sysreflect_get_host_name(monitor);
            assert!(!host_ptr.is_null());
            let host = CStr::from_ptr(host_ptr).to_string_lossy();
            assert!(!host.is_empty());
            sysreflect_free_string(host_ptr);
            
            sysreflect_free(monitor);
        }
    }

    #[test]
    fn test_ffi_null_safety() {
        sysreflect_refresh(std::ptr::null_mut());
        assert_eq!(sysreflect_get_cpu_usage(std::ptr::null_mut()), -1.0);
        assert_eq!(sysreflect_get_total_memory(std::ptr::null_mut()), 0);
        assert!(sysreflect_get_host_name(std::ptr::null_mut()).is_null());
    }
}
