pub mod ffi;
use serde::Serialize;
use sysinfo::{System, Networks, Components, Disks};

#[derive(Serialize)]
pub enum MetricStatus<T> {
    Available(T),
    Restricted,
    Unavailable,
}

#[derive(Serialize)]
pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub cpu_usage: f32,
    pub memory: u64,
}

pub struct SystemMonitor {
    sys: System,
    networks: Networks,
    components: Components,
    disks: Disks,
}

impl SystemMonitor {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        
        Self {
            sys,
            networks: Networks::new_with_refreshed_list(),
            components: Components::new_with_refreshed_list(),
            disks: Disks::new_with_refreshed_list(),
        }
    }

    pub fn refresh(&mut self) {
        self.sys.refresh_all();
        self.networks.refresh(true);
        self.components.refresh(true);
        self.disks.refresh(true);
    }

    pub fn global_cpu_usage(&self) -> MetricStatus<f32> {
        let usage = self.sys.global_cpu_usage();
        MetricStatus::Available(usage)
    }

    pub fn memory_info(&self) -> MetricStatus<(u64, u64)> {
        let total = self.sys.total_memory();
        let used = self.sys.used_memory();
        
        if total == 0 {
            // Try fallback via libc if possible (simplified here)
            MetricStatus::Restricted
        } else {
            MetricStatus::Available((total, used))
        }
    }

    pub fn core_count(&self) -> usize {
        let count = self.sys.cpus().len();
        if count == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(0)
        } else {
            count
        }
    }

    pub fn host_info(&self) -> (String, String, String, u64) {
        (
            System::host_name().unwrap_or_else(|| "Unknown".to_string()),
            System::os_version().unwrap_or_else(|| "Unknown".to_string()),
            System::kernel_version().unwrap_or_else(|| "Restricted".to_string()),
            System::uptime(),
        )
    }

    pub fn top_processes_by_cpu(&self, limit: usize) -> MetricStatus<Vec<ProcessInfo>> {
        let procs = self.sys.processes();
        if procs.is_empty() {
            return MetricStatus::Restricted;
        }

        let mut procs_refs: Vec<_> = procs.values().collect();
        procs_refs.sort_by(|a, b| b.cpu_usage().partial_cmp(&a.cpu_usage()).unwrap_or(std::cmp::Ordering::Equal));

        let list: Vec<ProcessInfo> = procs_refs.into_iter()
            .take(limit)
            .map(|process| ProcessInfo {
                pid: process.pid().as_u32(),
                name: process.name().to_string_lossy().into_owned(),
                cpu_usage: process.cpu_usage(),
                memory: process.memory(),
            })
            .collect();

        MetricStatus::Available(list)
    }

    pub fn network_throughput(&self) -> MetricStatus<(u64, u64)> {
        if self.networks.is_empty() {
            return MetricStatus::Restricted;
        }
        let mut rx = 0;
        let mut tx = 0;
        for (_, data) in &self.networks {
            rx += data.received();
            tx += data.transmitted();
        }
        MetricStatus::Available((rx, tx))
    }
}

impl Default for SystemMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_init() {
        let monitor = SystemMonitor::new();
        assert!(monitor.core_count() > 0);
    }

    #[test]
    fn test_host_info() {
        let monitor = SystemMonitor::new();
        let (host, os, kernel, _) = monitor.host_info();
        assert!(!host.is_empty());
        assert!(!os.is_empty());
        assert!(!kernel.is_empty());
    }

    #[test]
    fn test_memory_info() {
        let monitor = SystemMonitor::new();
        match monitor.memory_info() {
            MetricStatus::Available((total, _)) => {
                assert!(total > 0);
            }
            MetricStatus::Restricted => {
                // Restricted is allowed on some platforms but let's see
            }
            MetricStatus::Unavailable => panic!("Memory info unavailable"),
        }
    }

    #[test]
    fn test_refresh() {
        let mut monitor = SystemMonitor::new();
        monitor.refresh();
        // Just ensuring it doesn't panic
    }
}


