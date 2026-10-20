use serde::Serialize;
use sysinfo::{Components, Disks, Networks, System};
use tracing::{debug, info, warn};

/// Human-readable byte count (KB / MB / GB). Shared by every lmsafe binary so
/// the same number is always spelled the same way.
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    // Precision loss only appears above 2^53 bytes (~9 PB), far beyond any
    // host this reads; the two-decimal output cannot show the difference.
    #[expect(
        clippy::cast_precision_loss,
        reason = "byte counts are converted to f64 for unit division; \
                  precision loss starts at ~9 PB"
    )]
    let kb = bytes as f64 / 1024.0;
    let mb = kb / 1024.0;
    let gb = mb / 1024.0;

    if gb >= 1.0 {
        format!("{gb:.2} GB")
    } else if mb >= 1.0 {
        format!("{mb:.2} MB")
    } else if kb >= 1.0 {
        format!("{kb:.2} KB")
    } else {
        format!("{bytes} B")
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// macOS / Apple-Silicon unified-memory helper
//
// On macOS ARM64 the GPU and CPU share the same DRAM pool.  sysinfo's
// `used_memory()` only tracks app + wired memory; it completely misses Metal /
// GPU allocations that LLMs write into.  This causes `available` to be
// over-estimated, producing false ACCEPTED verdicts.
//
// We use `host_statistics64` (the same kernel call that `vm_stat` and
// `fastfetch` make) to obtain the true number of free + speculative pages,
// which reflects pressure from *every* client of unified memory.
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod macos_unified_mem {
    use std::mem::MaybeUninit;

    // The vm_statistics64 struct layout (from <mach/vm_statistics.h>).
    // The real struct is 416 bytes (HOST_VM_INFO64_COUNT = 104 u32s).
    // We only need the first 15 fields (through speculative_count), which occupy
    // 92 bytes. We pad to 152 bytes (38 u32s) to satisfy host_minimum_size.
    #[repr(C)]
    struct VmStatistics64 {
        free_count: u32,
        active_count: u32,
        inactive_count: u32,
        wire_count: u32,
        zero_fill_count: u64,
        reactivations: u64,
        pageins: u64,
        pageouts: u64,
        faults: u64,
        cow_faults: u64,
        lookups: u64,
        hits: u64,
        purges: u64,
        purgeable_count: u32,
        speculative_count: u32,
        _reserved: [u32; 14],
    }

    unsafe extern "C" {
        fn mach_host_self() -> libc::c_uint;
        fn host_statistics64(
            host_priv: libc::c_uint,
            flavor: libc::c_int,
            host_info_out: *mut libc::c_int,
            host_info_outCnt: *mut libc::c_uint,
        ) -> libc::c_int;
        fn host_page_size(host: libc::c_uint, out: *mut libc::vm_size_t) -> libc::c_int;
        fn sysctlbyname(
            name: *const libc::c_char,
            oldp: *mut libc::c_void,
            oldlenp: *mut libc::size_t,
            newp: *mut libc::c_void,
            newlen: libc::size_t,
        ) -> libc::c_int;
    }

    const HOST_VM_INFO64: libc::c_int = 4;
    const HOST_VM_INFO64_COUNT: libc::c_uint = 38; // We only read the first 38 u32s (152 bytes)

    /// Returns the true physical RAM in bytes via `sysctl hw.memsize`.
    /// This is what fastfetch / `system_profiler` report — the actual installed memory,
    /// not the sum of page states which can undercount on Apple Silicon.
    fn physical_memory_bytes() -> Option<u64> {
        unsafe {
            let name = c"hw.memsize".as_ptr();
            let mut size: u64 = 0;
            let mut len = std::mem::size_of::<u64>();
            let rc = sysctlbyname(
                name,
                (&raw mut size).cast::<libc::c_void>(),
                &raw mut len,
                std::ptr::null_mut(),
                0,
            );
            if rc != 0 || size == 0 {
                return None;
            }
            Some(size)
        }
    }

    /// Returns `(total_bytes, used_bytes)` via the Mach kernel interface.
    /// `total` uses `sysctl hw.memsize` (true physical RAM).
    /// `used` = total − (free + inactive + speculative) pages × `page_size`.
    ///
    /// On macOS, "inactive" pages are recently-used pages that are reclaimable
    /// but not yet freed. Activity Monitor / fastfetch count them as available.
    /// Excluding them (using only free + speculative) overestimates "used" by ~25%.
    pub fn unified_memory_info() -> Option<(u64, u64)> {
        // Get true physical RAM via sysctl (constant, doesn't change).
        let total = physical_memory_bytes()?;

        unsafe {
            let host = mach_host_self();

            // Query page size.
            let mut page_size: libc::vm_size_t = 0;
            if host_page_size(host, &raw mut page_size) != 0 || page_size == 0 {
                return None;
            }

            // Zero-init the whole struct so kernel-only trailing fields are safe.
            let mut stats_uninit: MaybeUninit<VmStatistics64> = MaybeUninit::zeroed();
            let mut count = HOST_VM_INFO64_COUNT;
            let kern_return = host_statistics64(
                host,
                HOST_VM_INFO64,
                stats_uninit.as_mut_ptr().cast::<libc::c_int>(),
                &raw mut count,
            );
            if kern_return != 0 {
                return None;
            }
            let stats = stats_uninit.assume_init();

            // "Available" pages on macOS = free + inactive + speculative.
            // Inactive pages are reclaimable but not yet freed; they count as
            // available in Activity Monitor and fastfetch.
            let available_pages = u64::from(stats.free_count)
                + u64::from(stats.inactive_count)
                + u64::from(stats.speculative_count);
            let ps = page_size as u64;
            let used = total.saturating_sub(available_pages * ps);

            Some((total, used))
        }
    }
}

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

impl Default for SystemMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemMonitor {
    #[tracing::instrument(skip_all)]
    pub fn new() -> Self {
        info!("Creating new SystemMonitor instance");
        let mut sys = System::new_all();
        sys.refresh_all();

        let monitor = Self {
            sys,
            networks: Networks::new_with_refreshed_list(),
            components: Components::new_with_refreshed_list(),
            disks: Disks::new_with_refreshed_list(),
        };

        debug!(
            "SystemMonitor created with {} CPU cores",
            monitor.core_count()
        );
        monitor
    }

    #[tracing::instrument(skip_all)]
    pub fn refresh(&mut self) {
        debug!("Refreshing system metrics");
        self.sys.refresh_all();
        self.networks.refresh(true);
        self.components.refresh(true);
        self.disks.refresh(true);
    }

    #[must_use]
    pub fn global_cpu_usage(&self) -> MetricStatus<f32> {
        let usage = self.sys.global_cpu_usage();
        MetricStatus::Available(usage)
    }

    pub fn memory_info(&self) -> MetricStatus<(u64, u64)> {
        // On Apple Silicon the GPU and CPU share the same unified DRAM pool.
        // Use the Mach kernel interface so that Metal / GPU pressure is
        // reflected in the "used" figure, just as fastfetch reports it.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some((total, used)) = macos_unified_mem::unified_memory_info() {
            debug!(
                "Apple Silicon unified memory: total={}, used={}",
                total, used
            );
            return MetricStatus::Available((total, used));
        }

        // Fallback for non-Apple-Silicon platforms (Linux, Windows, x86 Mac).
        let total = self.sys.total_memory();
        let used = self.sys.used_memory();

        if total == 0 {
            warn!("Memory info unavailable: total=0");
            MetricStatus::Restricted
        } else {
            debug!("Memory info: total={}, used={}", total, used);
            MetricStatus::Available((total, used))
        }
    }

    #[must_use]
    pub fn core_count(&self) -> usize {
        let count = self.sys.cpus().len();
        if count == 0 {
            std::thread::available_parallelism().map_or(0, std::num::NonZero::get)
        } else {
            count
        }
    }

    #[must_use]
    pub fn host_info(&self) -> (String, String, String, u64) {
        let os_name = System::name().unwrap_or_default();
        let os_version = System::os_version().unwrap_or_else(|| "Unknown".to_string());
        let os = if os_name.is_empty() {
            os_version
        } else {
            // sysinfo returns "Darwin" on macOS; use the marketing name instead.
            let display_name = if os_name == "Darwin" {
                "macOS"
            } else {
                &os_name
            };
            format!("{display_name} {os_version}")
        };
        (
            System::host_name().unwrap_or_else(|| "Unknown".to_string()),
            os,
            System::kernel_version().unwrap_or_else(|| "Restricted".to_string()),
            System::uptime(),
        )
    }

    #[must_use]
    pub fn cpu_brand(&self) -> String {
        self.sys
            .cpus()
            .first()
            .map_or_else(|| "Unknown".to_string(), |cpu| cpu.brand().to_string())
    }

    #[must_use]
    pub fn top_processes_by_cpu(&self, limit: usize) -> MetricStatus<Vec<ProcessInfo>> {
        let procs = self.sys.processes();
        if procs.is_empty() {
            return MetricStatus::Restricted;
        }

        let mut procs_refs: Vec<_> = procs.values().collect();
        procs_refs.sort_by(|a, b| {
            b.cpu_usage()
                .partial_cmp(&a.cpu_usage())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let list: Vec<ProcessInfo> = procs_refs
            .into_iter()
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

    #[must_use]
    pub fn network_throughput(&self) -> MetricStatus<(u64, u64)> {
        let mut rx = 0;
        let mut tx = 0;
        for (_, data) in &self.networks {
            rx += data.received();
            tx += data.transmitted();
        }

        if rx == 0 && tx == 0 {
            MetricStatus::Restricted
        } else {
            MetricStatus::Available((rx, tx))
        }
    }
}
