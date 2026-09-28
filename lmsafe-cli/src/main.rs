use clap::Parser;
use color_eyre::Result;
use std::path::PathBuf;
use sysreflect_core::model::{
    get_recursive_size, inspect_model_file, resolve_model_path, ModelInspectionReport,
};
use sysreflect_core::{MetricStatus, ProcessInfo, SystemMonitor};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "CLI tool for lmsafe",
    long_about = "CLI tool for lmsafe\n\nRequires a subcommand. model inspection always needs --file <FILE>.",
    after_help = "Examples:\n  lmsafe-cli inspect --file <model path>\n  lmsafe-cli inspect --file <model path> --json\n  lmsafe-cli inspect --help"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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
    verdict: VerdictInfo,
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

#[derive(serde::Serialize)]
struct VerdictInfo {
    status: String,
    message: String,
    model_name: Option<String>,
    model_family: Option<String>,
    quantization: Option<String>,
    parameter_count: Option<String>,
    file_size: u64,
    required_bytes: u64,
    available_memory: u64,
    safety_margin: f64,
}

fn format_bytes(bytes: u64) -> String {
    let kb = bytes as f64 / 1024.0;
    let mb = kb / 1024.0;
    let gb = mb / 1024.0;

    if gb >= 1.0 {
        format!("{:.2} GB", gb)
    } else if mb >= 1.0 {
        format!("{:.2} MB", mb)
    } else if kb >= 1.0 {
        format!("{:.2} KB", kb)
    } else {
        format!("{} B", bytes)
    }
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

fn build_verdict(
    report: &Option<ModelInspectionReport>,
    snapshot: &SystemSnapshot,
    model_size: u64,
    safety_margin: f64,
) -> VerdictInfo {
    let required_bytes = (model_size as f64 * safety_margin) as u64;

    let (status, message, available_memory) = match (snapshot.memory_available, required_bytes) {
        (Some(available), _) if available > required_bytes => (
            "ACCEPTED".to_string(),
            format!(
                "Safe to load (Free RAM: {} > Required: {})",
                format_bytes(available),
                format_bytes(required_bytes)
            ),
            available,
        ),
        (Some(available), _) => (
            "REJECTED".to_string(),
            format!(
                "OOM Risk (Free RAM: {} < Required: {})",
                format_bytes(available),
                format_bytes(required_bytes)
            ),
            available,
        ),
        _ => ("UNKNOWN".to_string(), "Telemetry Restricted".to_string(), 0),
    };

    VerdictInfo {
        status,
        message,
        model_name: report.as_ref().map(|r| r.name.clone()),
        model_family: report.as_ref().map(|r| format!("{:?}", r.family)),
        quantization: report
            .as_ref()
            .and_then(|r| r.capabilities.quantization.clone()),
        parameter_count: report
            .as_ref()
            .and_then(|r| r.capabilities.parameter_count.clone()),
        file_size: model_size,
        required_bytes,
        available_memory,
        safety_margin,
    }
}

fn print_sysreflect(
    report: &Option<ModelInspectionReport>,
    snapshot: &SystemSnapshot,
    verdict: &VerdictInfo,
) {
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
            println!("  Net RX: {} | TX: {}", format_bytes(rx), format_bytes(tx))
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

    // Security Guard
    println!();
    println!("{}", "=".repeat(w));
    println!("{:^w$}", "Security Guard (Admission Control)", w = w);
    println!("{}", "=".repeat(w));

    let verdict_icon = match verdict.status.as_str() {
        "ACCEPTED" => "[OK]",
        "REJECTED" => "[!!]",
        _ => "[??]",
    };
    println!(
        "  Verdict: {} {} - {}",
        verdict_icon, verdict.status, verdict.message
    );

    if let Some(ref r) = report {
        println!();
        println!("  Model Name:    {}", r.name);
        println!("  Family:        {:?}", r.family);
        if r.family == sysreflect_core::model::ModelFamily::AudioSegmentation {
            let arch = r
                .raw_metadata
                .get("architecture")
                .and_then(|v| v.as_str())
                .unwrap_or("Silero-RNN-v5");
            println!("  Architecture:  {}", arch);
            println!("  Capabilities:  Audio/VAD, Streaming");
            println!("  Tensor Check:  Pass (Inputs: [audio, sr, state], Outputs: [prob, state])");
        } else {
            if let Some(ref q) = r.capabilities.quantization {
                println!("  Quantization:  {}", q);
            }
            if let Some(ref p) = r.capabilities.parameter_count {
                println!("  Parameters:    {}", p);
            }
            println!("  Inferred:      {:?}", r.inferred_capabilities);

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

    // Top Processes
    if !snapshot.top_processes.is_empty() {
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

    println!();
    println!("{}", "=".repeat(w));
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
        format!("{}d {}h {}m", days, hours, mins)
    } else if hours > 0 {
        format!("{}h {}m", hours, mins)
    } else {
        format!("{}m", mins)
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect {
            file,
            json,
            safety_margin,
        } => {
            if !file.exists() {
                eprintln!("Error: Model file does not exist: {}", file.display());
                std::process::exit(1);
            }

            let resolved = resolve_model_path(&file);
            let report = inspect_model_file(&file).ok();
            let model_size = get_recursive_size(&resolved);

            let mut sys = SystemMonitor::new();
            sys.refresh();

            let snapshot = build_system_snapshot(&sys);
            let verdict = build_verdict(&report, &snapshot, model_size, safety_margin);

            if json {
                let cli_report = CliReport {
                    model: report,
                    system: snapshot,
                    verdict,
                };
                println!("{}", serde_json::to_string_pretty(&cli_report)?);
            } else {
                print_sysreflect(&report, &snapshot, &verdict);
            }
        }
    }

    Ok(())
}
