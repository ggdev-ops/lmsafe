//! Admission control — the single source of truth for
//! "is it safe to load this model into memory right now?"
//!
//! This rule used to exist in three places (sysreflect-core, lmsafe-cli,
//! lmsafe-tui) and the copies had already drifted apart. Every consumer now
//! asks this module and renders the answer.

use serde::Serialize;
use std::path::Path;
use sysreflect_core::{MetricStatus, SystemMonitor, format_bytes};
use sysreflect_model::{get_recursive_size, resolve_model_path};
use tracing::{info, warn};

/// Outcome of an admission check.
///
/// `Unknown` is deliberately distinct from `Rejected`: we could not measure
/// the system, so we do not claim it is safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Verdict {
    Accepted,
    Rejected,
    Unknown,
}

/// Everything a consumer needs to explain the verdict to a human.
#[derive(Debug, Clone, Serialize)]
pub struct AdmissionReport {
    pub verdict: Verdict,
    pub message: String,
    pub file_size: u64,
    pub required_bytes: u64,
    pub available_memory: Option<u64>,
    pub safety_margin: f64,
}

impl AdmissionReport {
    #[must_use]
    pub fn is_accepted(&self) -> bool {
        self.verdict == Verdict::Accepted
    }
}

/// Negative margin is invalid input; fail safe to the 1.2 default instead of
/// accidentally disabling the headroom gate. 0.0 is valid and disables it.
fn normalize_margin(safety_margin: f64) -> f64 {
    if safety_margin < 0.0 {
        1.2
    } else {
        safety_margin
    }
}

/// The rule itself. Pure: no I/O, no sampling — hand it the numbers.
///
/// 1. `file_size == 0` → `Rejected`. The model is missing, empty, or a path we
///    could not read. Comparing zero against free RAM and answering "accepted"
///    would be absurd, so this is checked first.
/// 2. Memory unavailable → `Unknown`. Missing data never becomes a green light.
/// 3. `available > file_size * margin` → `Accepted`, else `Rejected`.
pub fn assess_admission(
    available_memory: Option<u64>,
    file_size: u64,
    safety_margin: f64,
) -> AdmissionReport {
    let safety_margin = normalize_margin(safety_margin);
    // Note: this truncates toward zero, so 100 bytes at 1.15x requires 114,
    // not 115. Pre-existing behaviour across all former copies, kept until we
    // decide to ceil instead. If that call changes, this is the line.
    // f64 is used because the margin is fractional; precision loss and
    // truncation only diverge from exact arithmetic above 2^53 bytes.
    #[expect(
        clippy::cast_precision_loss,
        reason = "margin is fractional, so the product must be f64"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "result truncates toward zero; pending the ceil decision"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "f64 -> u64 for a size; inputs are non-negative file sizes"
    )]
    let required_bytes = (file_size as f64 * safety_margin) as u64;

    let report = if file_size == 0 {
        AdmissionReport {
            verdict: Verdict::Rejected,
            message: "Model size is zero (file missing, empty, or unreadable)".to_string(),
            file_size,
            required_bytes,
            available_memory,
            safety_margin,
        }
    } else {
        match available_memory {
            Some(available) if available > required_bytes => AdmissionReport {
                verdict: Verdict::Accepted,
                message: format!(
                    "Safe to load (Free RAM: {} > Required: {})",
                    format_bytes(available),
                    format_bytes(required_bytes)
                ),
                file_size,
                required_bytes,
                available_memory: Some(available),
                safety_margin,
            },
            Some(available) => AdmissionReport {
                verdict: Verdict::Rejected,
                message: format!(
                    "OOM Risk (Free RAM: {} < Required: {})",
                    format_bytes(available),
                    format_bytes(required_bytes)
                ),
                file_size,
                required_bytes,
                available_memory: Some(available),
                safety_margin,
            },
            None => AdmissionReport {
                verdict: Verdict::Unknown,
                message: "Telemetry restricted".to_string(),
                file_size,
                required_bytes,
                available_memory: None,
                safety_margin,
            },
        }
    };

    match report.verdict {
        Verdict::Accepted => info!(
            "Admission ACCEPTED: {} (available={:?}, required={})",
            report.message, report.available_memory, report.required_bytes
        ),
        Verdict::Rejected => warn!(
            "Admission REJECTED: {} (available={:?}, required={})",
            report.message, report.available_memory, report.required_bytes
        ),
        Verdict::Unknown => warn!(
            "Admission UNKNOWN: telemetry unavailable (required={})",
            report.required_bytes
        ),
    }

    report
}

/// Sample the live system, then apply the rule. This is the entry point the
/// CLI, the TUI and the FFI all funnel through.
#[must_use]
pub fn assess_with_monitor(
    monitor: &SystemMonitor,
    file_size: u64,
    safety_margin: f64,
) -> AdmissionReport {
    let available = match monitor.memory_info() {
        MetricStatus::Available((total, used)) => Some(total.saturating_sub(used)),
        MetricStatus::Restricted | MetricStatus::Unavailable => None,
    };
    assess_admission(available, file_size, safety_margin)
}

/// Resolve, size and assess a model path in one call.
pub fn assess_model_path<P: AsRef<Path>>(path: P, safety_margin: f64) -> AdmissionReport {
    let resolved = resolve_model_path(&path);
    let file_size = get_recursive_size(&resolved);
    let monitor = SystemMonitor::new();
    assess_with_monitor(&monitor, file_size, safety_margin)
}

/// Boolean convenience for the C FFI. `Unknown` maps to `false`: a caller that
/// cannot verify gets the conservative answer.
#[tracing::instrument(skip_all, fields(path = %path.as_ref().display(), safety_margin))]
pub fn model_is_safe_to_load<P: AsRef<Path>>(path: P, safety_margin: f64) -> bool {
    assess_model_path(path, safety_margin).is_accepted()
}
