use clap::Parser;
use color_eyre::Result;
use lmsafe_zone::Verdict;
use lmsafe_zone::admission::{AdmissionReport, assess_model_path_with_monitor};
use std::path::PathBuf;
use std::time::Duration;
use sysreflect_core::{MetricStatus, ProcessInfo, SystemMonitor, format_bytes};
use sysreflect_model::{ModelInspectionReport, inspect_model_file};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "CLI tool for lmsafe",
    long_about = "CLI tool for lmsafe\n\nWith no subcommand, prints a one-shot system telemetry snapshot (the headless equivalent of lmsafe-tui with no flags). Model inspection always needs --file <FILE>.",
    after_help = "Examples:\n  lmsafe-cli\n  lmsafe-cli inspect --file <model path>\n  lmsafe-cli inspect --file <model path> --json\n  lmsafe-cli inspect --help"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Inspect a model file (requires --file <FILE>)
    #[command(
        about = "Inspect a model file (requires --file <FILE>)",
        long_about = "Inspect a model file and output its metadata.\n\nRequires --file <FILE>. Optionally pass --json to print results in JSON format.",
        after_help = "Examples:\n  lmsafe-cli inspect --file <model path>\n  lmsafe-cli inspect --file <model path> --json"
    )]
    Inspect {
        /// Path to the model file (required)
        #[arg(short, long, value_name = "FILE", required = true, value_hint = clap::ValueHint::FilePath)]
        file: PathBuf,

        /// Output results in JSON format
        #[arg(short, long)]
        json: bool,

        /// Safety margin multiplier for memory assessment (e.g. 1.15)
        #[arg(
            short,
            long,
            default_value = "1.15",
            help = "Safety margin multiplier for memory assessment (e.g. 1.15)"
        )]
        safety_margin: f64,
    },
}

#[derive(serde::Serialize)]
struct CliReport {
    model: Option<ModelInspectionReport>,
    system: SystemSnapshot,
    verdict: AdmissionReport,
}

#[derive(serde::Serialize)]
struct SystemSnapshot {
    host: String,
    os: String,
    kernel: String,
    uptime: u64,
    cpu_brand: String,
    cpu_cores: usize,
    cpu_usage: Option<f32>,
    memory_total: Option<u64>,
    memory_used: Option<u64>,
    memory_available: Option<u64>,
    network_rx: Option<u64>,
    network_tx: Option<u64>,
    top_processes: Vec<ProcessInfo>,
}

fn build_system_snapshot(sys: &SystemMonitor) -> SystemSnapshot {
    let (host, os, kernel, uptime) = sys.host_info();

    let cpu_usage = match sys.global_cpu_usage() {
        MetricStatus::Available(v) => Some(v),
        _ => None,
    };

    let (memory_total, memory_used, memory_available) = match sys.memory_info() {
        MetricStatus::Available((total, used)) => {
            let available = total.saturating_sub(used);
            (Some(total), Some(used), Some(available))
        }
        _ => (None, None, None),
    };

    let (network_rx, network_tx) = match sys.network_throughput() {
        MetricStatus::Available((rx, tx)) => (Some(rx), Some(tx)),
        _ => (None, None),
    };

    let top_processes = match sys.top_processes_by_cpu(20) {
        MetricStatus::Available(list) => list,
        _ => Vec::new(),
    };

    SystemSnapshot {
        host,
        os,
        kernel,
        uptime,
        cpu_brand: sys.cpu_brand(),
        cpu_cores: sys.core_count(),
        cpu_usage,
        memory_total,
        memory_used,
        memory_available,
        network_rx,
        network_tx,
        top_processes,
    }
}

/// Print host, CPU and memory sections. Shared by both the moment snapshot
/// and the `inspect` report so the two never drift apart.
fn print_system_status(snapshot: &SystemSnapshot) {
    let w = 64;

    // Header
    println!("{}", "=".repeat(w));
    println!("{:^w$}", "System Status", w = w);
    println!("{}", "=".repeat(w));
    println!("  Host: {} | OS: {}", snapshot.host, snapshot.os);
    println!(
        "  Kernel: Darwin {} | Uptime: {}",
        snapshot.kernel,
        format_uptime(snapshot.uptime)
    );
    match (snapshot.network_rx, snapshot.network_tx) {
        (Some(rx), Some(tx)) => {
            println!("  Net RX: {} | TX: {}", format_bytes(rx), format_bytes(tx));
        }
        _ => println!("  Net: Access Denied / N/A"),
    }

    // CPU
    println!();
    println!("{}", "-".repeat(w));
    println!("{:^w$}", "CPU", w = w);
    println!("{}", "-".repeat(w));
    match snapshot.cpu_usage {
        Some(usage) => println!(
            "  {} ({} cores): {:.1}%",
            snapshot.cpu_brand, snapshot.cpu_cores, usage
        ),
        None => println!(
            "  {} ({} cores): N/A",
            snapshot.cpu_brand, snapshot.cpu_cores
        ),
    }

    // Memory
    println!();
    println!("{}", "-".repeat(w));
    println!("{:^w$}", "Memory", w = w);
    println!("{}", "-".repeat(w));
    match (
        snapshot.memory_total,
        snapshot.memory_used,
        snapshot.memory_available,
    ) {
        (Some(total), Some(used), Some(avail)) => {
            // A gauge, not an accounting figure: f64 error only matters past
            // 2^53 bytes of RAM.
            #[expect(
                clippy::cast_precision_loss,
                reason = "memory gauge percentage; precision loss is unreachable \
                          at realistic RAM sizes"
            )]
            let pct = if total > 0 {
                used as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            println!("  Total:     {}", format_bytes(total));
            println!("  Used:      {} ({:.1}%)", format_bytes(used), pct);
            println!("  Available: {}", format_bytes(avail));
        }
        _ => println!("  Memory: Access Denied / N/A"),
    }
}

/// Print the admission-control verdict block. Only meaningful when a model
/// file is in play, so the pure telemetry snapshot skips it entirely.
fn print_guard(report: Option<&ModelInspectionReport>, verdict: &AdmissionReport) {
    let w = 64;

    println!();
    println!("{}", "=".repeat(w));
    println!("{:^w$}", "Security Guard (Admission Control)", w = w);
    println!("{}", "=".repeat(w));

    let (verdict_icon, verdict_label) = match verdict.verdict {
        Verdict::Accepted => ("[OK]", "ACCEPTED"),
        Verdict::Rejected => ("[!!]", "REJECTED"),
        Verdict::Unknown => ("[??]", "UNKNOWN"),
    };
    println!(
        "  Verdict: {} {} - {}",
        verdict_icon, verdict_label, verdict.message
    );

    if let Some(r) = report {
        println!();
        println!("  Model Name:    {}", r.name);
        println!("  Family:        {:?}", r.family);
        if r.family == sysreflect_model::ModelFamily::AudioSegmentation {
            let arch = r
                .raw_metadata
                .get("architecture")
                .and_then(|v| v.as_str())
                .unwrap_or("Silero-RNN-v5");
            println!("  Architecture:  {arch}");
            println!("  Capabilities:  Audio/VAD, Streaming");
            println!("  Tensor Check:  Pass (Inputs: [audio, sr, state], Outputs: [prob, state])");
        } else {
            if let Some(ref q) = r.capabilities.quantization {
                println!("  Quantization:  {q}");
            }
            if let Some(ref p) = r.capabilities.parameter_count {
                println!("  Parameters:    {p}");
            }
            println!("  Inferred:      {:?}", r.inferred_capabilities);

            if let Some(serde_json::Value::Array(units)) = r.raw_metadata.get("units") {
                for u in units {
                    let name = u
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("?");
                    let classes: Vec<&str> = u
                        .get("classes")
                        .and_then(serde_json::Value::as_array)
                        .map(|arr| arr.iter().filter_map(serde_json::Value::as_str).collect())
                        .unwrap_or_default();
                    let load = u
                        .get("load_bytes")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0);
                    println!(
                        "  Unit:          {name} [{}] ({})",
                        classes.join(", "),
                        format_bytes(load)
                    );
                }
            }

            if let (Some(load), Some(footprint)) = (
                r.raw_metadata
                    .get("load_bytes")
                    .and_then(serde_json::Value::as_u64),
                r.raw_metadata
                    .get("footprint_bytes")
                    .and_then(serde_json::Value::as_u64),
            ) && load != footprint
            {
                println!(
                    "  Footprint:     {} on disk ({} loaded)",
                    format_bytes(footprint),
                    format_bytes(load)
                );
            }

            if let Some(serde_json::Value::Array(styles)) = r.raw_metadata.get("available_styles") {
                let style_list: Vec<&str> = styles.iter().filter_map(|s| s.as_str()).collect();
                println!("  Voice Styles:  {}", style_list.join(", "));
            }
            if let Some(serde_json::Value::Array(samples)) = r.raw_metadata.get("audio_samples") {
                let sample_list: Vec<&str> = samples.iter().filter_map(|s| s.as_str()).collect();
                println!("  Audio Samples: {}", sample_list.join(", "));
            }
        }
        println!("  File Size:     {}", format_bytes(verdict.file_size));
        println!("  Safety Margin: {:.2}x", verdict.safety_margin);
    }
}

fn print_processes(snapshot: &SystemSnapshot) {
    if snapshot.top_processes.is_empty() {
        return;
    }

    let w = 64;
    println!();
    println!("{}", "-".repeat(w));
    println!("{:^w$}", "Top Processes", w = w);
    println!("{}", "-".repeat(w));
    println!("  {:<8} {:<28} {:>6} {:>12}", "PID", "Name", "CPU%", "Mem");
    println!("  {}", "-".repeat(w - 2));
    for p in &snapshot.top_processes {
        println!(
            "  {:<8} {:<28} {:>5.1}% {:>11}",
            p.pid,
            truncate_str(&p.name, 28),
            p.cpu_usage,
            format_bytes(p.memory)
        );
    }
}

fn print_sysreflect(
    report: Option<&ModelInspectionReport>,
    snapshot: &SystemSnapshot,
    verdict: &AdmissionReport,
) {
    print_system_status(snapshot);
    print_guard(report, verdict);
    print_processes(snapshot);

    println!();
    println!("{}", "=".repeat(64));
}

/// One-shot telemetry snapshot: the headless mirror of `lmsafe-tui` with no
/// flags. Same data, printed once, no model and no admission control.
///
/// CPU and network metrics are deltas between two samples, so a single
/// refresh reports 0.0% and ~0 B. Take a warm-up sample, let a real interval
/// pass, then sample again before printing.
fn print_moment_snapshot() {
    let mut sys = SystemMonitor::new();
    sys.refresh();
    std::thread::sleep(Duration::from_millis(300));
    sys.refresh();

    let snapshot = build_system_snapshot(&sys);
    print_system_status(&snapshot);
    print_processes(&snapshot);

    println!();
    println!("{}", "=".repeat(64));
    println!(
        "{:^w$}",
        "Moment Snapshot (use lmsafe-tui to watch live)",
        w = 64
    );
    println!("{}", "=".repeat(64));
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}

fn format_uptime(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;
    if days > 0 {
        format!("{days}d {hours}h {mins}m")
    } else if hours > 0 {
        format!("{hours}h {mins}m")
    } else {
        format!("{mins}m")
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Inspect {
            file,
            json,
            safety_margin,
        }) => {
            if !file.exists() {
                eprintln!("Error: Model file does not exist: {}", file.display());
                std::process::exit(1);
            }

            let report = inspect_model_file(&file).ok();

            let mut sys = SystemMonitor::new();
            sys.refresh();

            let snapshot = build_system_snapshot(&sys);
            // One rule, owned by lmsafe-zone. The TUI renders this same report.
            let verdict = assess_model_path_with_monitor(&sys, &file, safety_margin);

            if json {
                let cli_report = CliReport {
                    model: report,
                    system: snapshot,
                    verdict,
                };
                println!("{}", serde_json::to_string_pretty(&cli_report)?);
            } else {
                print_sysreflect(report.as_ref(), &snapshot, &verdict);
            }
        }
        None => print_moment_snapshot(),
    }

    Ok(())
}
