use sysreflect_core::*;

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
    assert_ne!(host, "");
    assert_ne!(os, "");
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
fn test_format_bytes() {
    assert_eq!(format_bytes(0), "0 B");
    assert_eq!(format_bytes(512), "512 B");
    assert_eq!(format_bytes(1024), "1.00 KB");
    assert_eq!(format_bytes(1536), "1.50 KB");
    assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
    assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
}
