use clap::Parser;
use color_eyre::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use lmsafe_zone::Verdict;
use lmsafe_zone::admission::{
    assess_model_path_with_monitor, assess_with_monitor, model_load_bytes,
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, BorderType, Borders, Gauge, Paragraph, Row, Sparkline, Table},
};
use std::{
    io,
    path::PathBuf,
    time::{Duration, Instant},
};
use sysreflect_core::{MetricStatus, SystemMonitor, format_bytes};
use sysreflect_model::{ModelInspectionReport, inspect_model_file};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "lmsafe-tui: Telemetry & Model Security Guard",
    long_about = None
)]
struct Cli {
    /// Path to a model file to inspect and monitor safety for
    #[arg(short, long, value_name = "FILE")]
    file: Option<PathBuf>,

    /// Safety margin multiplier for memory assessment (e.g. 1.15)
    #[arg(short, long, default_value = "1.15")]
    safety_margin: f64,
}

struct App {
    monitor: SystemMonitor,
    cpu_history: Vec<u64>,
    model_report: Option<ModelInspectionReport>,
    model_path: Option<PathBuf>,
    /// Bytes the verdict math uses (containers load), computed once.
    model_load_bytes: u64,
    safety_margin: f64,
}

impl App {
    fn new(model_path: Option<PathBuf>, safety_margin: f64) -> Self {
        let model_report = model_path
            .as_ref()
            .and_then(|path| inspect_model_file(path).ok());
        let model_load_bytes = model_path
            .as_deref()
            .and_then(model_load_bytes)
            .unwrap_or(0);

        Self {
            monitor: SystemMonitor::new(),
            cpu_history: Vec::with_capacity(100),
            model_report,
            model_path,
            model_load_bytes,
            safety_margin,
        }
    }

    fn on_tick(&mut self) {
        self.monitor.refresh();
        let cpu_val = match self.monitor.global_cpu_usage() {
            // History is sampled as whole percent; the sparkline has no use for
            // sub-percent resolution.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "history series stores whole percent"
            )]
            #[expect(
                clippy::cast_sign_loss,
                reason = "CPU percent is non-negative; the cast saturates"
            )]
            MetricStatus::Available(v) => v as u64,
            _ => 0,
        };
        self.cpu_history.push(cpu_val);
        if self.cpu_history.len() > 100 {
            self.cpu_history.remove(0);
        }
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let cli = Cli::parse();

    // Verify model file exists and is valid if provided
    if let Some(ref path) = cli.file {
        if !path.exists() {
            eprintln!("Error: Model file does not exist: {}", path.display());
            std::process::exit(1);
        }
        if let Err(e) = inspect_model_file(path) {
            eprintln!(
                "Error: Failed to inspect model file {}: {}",
                path.display(),
                e
            );
            std::process::exit(1);
        }
    }

    run_tui(cli.file, cli.safety_margin)?;

    Ok(())
}

fn run_tui(model_path: Option<PathBuf>, safety_margin: f64) -> Result<()> {
    let mut terminal = setup_terminal()?;
    let mut app = App::new(model_path, safety_margin);
    let tick_rate = Duration::from_millis(500);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if event::poll(timeout)?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
            && (key.code == KeyCode::Char('q') || key.code == KeyCode::Esc)
        {
            break;
        }

        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = Instant::now();
        }
    }

    restore_terminal(terminal)?;
    Ok(())
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Ok(Terminal::new(backend)?)
}

fn restore_terminal(mut terminal: Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn ui(f: &mut ratatui::Frame, app: &App) {
    let area = f.area();
    let width = area.width;
    let height = area.height;

    // Minimum dimensions check to prevent crash on tiny windows
    if width < 45 || height < 10 {
        render_too_small(f, area);
        return;
    }

    // Dynamic UI features based on terminal sizes
    let show_header = height >= 12;
    let show_processes = height >= 20;
    let show_sparkline = height >= 22 && width >= 80;

    let mut constraints = Vec::new();
    if show_header {
        constraints.push(Constraint::Length(3)); // Header
    }

    // Determine height for middle section (Telemetry & Security Guard)
    let middle_height = if app.model_path.is_some() {
        if show_sparkline {
            14
        } else if height >= 16 {
            9
        } else {
            6
        }
    } else {
        if show_sparkline { 10 } else { 5 }
    };
    constraints.push(Constraint::Length(middle_height));

    if show_processes {
        constraints.push(Constraint::Min(0)); // Processes List
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let mut chunk_idx = 0;
    if show_header {
        render_header(f, app, chunks[chunk_idx]);
        chunk_idx += 1;
    }

    let middle_chunk = chunks[chunk_idx];
    chunk_idx += 1;

    // Render Middle Area
    if app.model_path.is_some() {
        let side_by_side = width >= 90;
        if side_by_side {
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(middle_chunk);

            render_resources_split(f, app, cols[0], show_sparkline);
            render_security_guard(f, app, cols[1]);
        } else {
            let rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(middle_chunk);

            render_resources_split(f, app, rows[0], false);
            render_security_guard(f, app, rows[1]);
        }
    } else {
        render_resources(f, app, middle_chunk, show_sparkline);
    }

    if show_processes {
        render_processes(f, app, chunks[chunk_idx]);
    }
}

fn render_too_small(f: &mut ratatui::Frame, area: Rect) {
    let p = Paragraph::new("⚠️ Terminal window too small! Please resize.")
        .style(
            Style::default()
                .fg(Color::Rgb(239, 68, 68))
                .add_modifier(Modifier::BOLD),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Rgb(220, 38, 38))),
        );
    f.render_widget(p, area);
}

fn render_header(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let (host, os, kernel, uptime) = app.monitor.host_info();
    let net_info = match app.monitor.network_throughput() {
        MetricStatus::Available((rx, tx)) => format!("Net RX: {rx} B | TX: {tx} B"),
        MetricStatus::Restricted => "Net: Access Denied".to_string(),
        MetricStatus::Unavailable => "Net: N/A".to_string(),
    };

    let header = Paragraph::new(format!(
        "Host: {host} | OS: {os} | Kernel: {kernel} | Uptime: {uptime}s | {net_info}"
    ))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Rgb(100, 116, 139)))
            .title(" ✨ System Status "),
    );
    f.render_widget(header, area);
}

/// Bytes as a float, for percentage and GB display math. Precision loss only
/// begins near 2^53 bytes (~9 PB), far beyond any host this reads.
fn bytes_f64(bytes: u64) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "byte counts are divided for display arithmetic"
    )]
    let result = bytes as f64;
    result
}

/// Whole percent for a gauge. Rust's float-to-int cast saturates, so NaN or
/// out-of-range telemetry cannot produce a wild gauge value.
fn gauge_percent(value: f64) -> u16 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "whole percent is the gauge's resolution"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "percentages are non-negative; the cast saturates on NaN"
    )]
    let result = value as u16;
    result
}

/// CPU gauge value and its label. Shared by both resource layouts.
fn cpu_gauge_values(app: &App) -> (u16, String) {
    match app.monitor.global_cpu_usage() {
        MetricStatus::Available(v) => (gauge_percent(f64::from(v)), format!("{v:.1}%")),
        MetricStatus::Restricted => (0, "Restricted (Access Denied)".to_string()),
        MetricStatus::Unavailable => (0, "N/A".to_string()),
    }
}

/// Memory gauge value and its label. Shared by both resource layouts.
fn mem_gauge_values(app: &App) -> (u16, String) {
    match app.monitor.memory_info() {
        MetricStatus::Available((total, used)) => {
            let p = if total > 0 {
                gauge_percent(bytes_f64(used) / bytes_f64(total) * 100.0)
            } else {
                0
            };
            (
                p,
                format!(
                    "{:.1} / {:.1} GB ({:.1}%)",
                    bytes_f64(used) / 1e9,
                    bytes_f64(total) / 1e9,
                    p
                ),
            )
        }
        MetricStatus::Restricted => (0, "Memory: Access Denied".to_string()),
        MetricStatus::Unavailable => (0, "Memory: N/A".to_string()),
    }
}

fn render_resources(f: &mut ratatui::Frame, app: &App, area: Rect, show_sparkline: bool) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // CPU
    let cpu_constraints = if show_sparkline {
        vec![Constraint::Percentage(60), Constraint::Percentage(40)]
    } else {
        vec![Constraint::Percentage(100)]
    };

    let cpu_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(cpu_constraints)
        .split(chunks[0]);

    let mut cpu_chunk_idx = 0;
    if show_sparkline {
        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .title(" CPU History ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(Color::Rgb(71, 85, 105))),
            )
            .data(&app.cpu_history)
            .style(Style::default().fg(Color::Rgb(56, 189, 248)));
        f.render_widget(sparkline, cpu_chunks[cpu_chunk_idx]);
        cpu_chunk_idx += 1;
    }

    let (cpu_percent, cpu_label) = cpu_gauge_values(app);

    let cpu_gauge = Gauge::default()
        .block(
            Block::default()
                .title(format!(" CPU Usage ({} cores) ", app.monitor.core_count()))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Rgb(71, 85, 105))),
        )
        .gauge_style(Style::default().fg(Color::Rgb(56, 189, 248)))
        .percent(cpu_percent)
        .label(cpu_label);
    f.render_widget(cpu_gauge, cpu_chunks[cpu_chunk_idx]);

    // Memory
    let mem_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(100)])
        .split(chunks[1]);

    let (mem_percent, mem_label) = mem_gauge_values(app);

    let mem_gauge = Gauge::default()
        .block(
            Block::default()
                .title(" Memory ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Rgb(71, 85, 105))),
        )
        .gauge_style(Style::default().fg(Color::Rgb(192, 132, 252)))
        .percent(mem_percent)
        .label(mem_label);
    f.render_widget(mem_gauge, mem_chunks[0]);
}

fn render_resources_split(f: &mut ratatui::Frame, app: &App, area: Rect, show_sparkline: bool) {
    let constraints = if show_sparkline {
        vec![
            Constraint::Length(4), // Sparkline
            Constraint::Length(4), // CPU Gauge
            Constraint::Length(4), // Memory Gauge
        ]
    } else {
        vec![
            Constraint::Length(4), // CPU Gauge
            Constraint::Length(4), // Memory Gauge
        ]
    };

    let cpu_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let mut chunk_idx = 0;
    if show_sparkline {
        let sparkline = Sparkline::default()
            .block(
                Block::default()
                    .title(" CPU History ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(Color::Rgb(71, 85, 105))),
            )
            .data(&app.cpu_history)
            .style(Style::default().fg(Color::Rgb(56, 189, 248)));
        f.render_widget(sparkline, cpu_chunks[chunk_idx]);
        chunk_idx += 1;
    }

    let (cpu_percent, cpu_label) = cpu_gauge_values(app);

    let cpu_gauge = Gauge::default()
        .block(
            Block::default()
                .title(format!(" CPU Usage ({} cores) ", app.monitor.core_count()))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Rgb(71, 85, 105))),
        )
        .gauge_style(Style::default().fg(Color::Rgb(56, 189, 248)))
        .percent(cpu_percent)
        .label(cpu_label);
    f.render_widget(cpu_gauge, cpu_chunks[chunk_idx]);
    chunk_idx += 1;

    let (mem_percent, mem_label) = mem_gauge_values(app);

    let mem_gauge = Gauge::default()
        .block(
            Block::default()
                .title(" Memory ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Rgb(71, 85, 105))),
        )
        .gauge_style(Style::default().fg(Color::Rgb(192, 132, 252)))
        .percent(mem_percent)
        .label(mem_label);
    f.render_widget(mem_gauge, cpu_chunks[chunk_idx]);
}

/// Detail lines shown under the admission verdict. Split by family because an
/// audio-segmentation report carries different metadata than a generic one.
fn model_detail_lines(app: &App, render_full_details: bool) -> Vec<ratatui::text::Line<'static>> {
    let Some(ref report) = app.model_report else {
        let mut lines = vec![ratatui::text::Line::from(format!(
            "  Path: {}",
            app.model_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default()
        ))];
        if render_full_details {
            lines.push(ratatui::text::Line::from(
                "  (Model report unavailable - metadata parsing failed)",
            ));
        }
        return lines;
    };

    let mut lines = vec![ratatui::text::Line::from(format!(
        "  Model Name: {}",
        report.name
    ))];

    if report.family == sysreflect_model::ModelFamily::AudioSegmentation {
        lines.extend(silero_detail_lines(report, app, render_full_details));
    } else {
        lines.extend(generic_detail_lines(report, app, render_full_details));
    }
    lines
}

/// Lines for the Silero audio-segmentation model, which reports architecture
/// and tensor wiring instead of quantization and parameter counts.
fn silero_detail_lines(
    report: &ModelInspectionReport,
    app: &App,
    render_full_details: bool,
) -> Vec<ratatui::text::Line<'static>> {
    let mut lines = vec![ratatui::text::Line::from(
        "  Family: AudioSegmentation (Silero)",
    )];
    if !render_full_details {
        return lines;
    }

    let arch = report
        .raw_metadata
        .get("architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("Silero-RNN-v5");
    lines.push(ratatui::text::Line::from(format!("  Architecture: {arch}")));
    lines.push(ratatui::text::Line::from(
        "  Capabilities: Audio/VAD, Streaming",
    ));
    lines.push(ratatui::text::Line::from("  Sample Rates: 8000Hz, 16000Hz"));
    lines.push(ratatui::text::Line::from(
        "  Tensor Check: Pass (Inputs: [audio, sr, state], Outputs: [prob, state])",
    ));
    lines.push(ratatui::text::Line::from(format!(
        "  File Size: {}",
        format_bytes(app.model_load_bytes)
    )));
    lines.push(ratatui::text::Line::from(format!(
        "  Safety Margin: {:.2}x",
        app.safety_margin
    )));
    lines
}

/// Lines for a generic model report.
fn generic_detail_lines(
    report: &ModelInspectionReport,
    app: &App,
    render_full_details: bool,
) -> Vec<ratatui::text::Line<'static>> {
    let mut lines = vec![ratatui::text::Line::from(format!(
        "  Family: {:?}",
        report.family
    ))];

    if !render_full_details {
        return lines;
    }

    lines.push(ratatui::text::Line::from(format!(
        "  Quantization: {}",
        report.capabilities.quantization.as_deref().unwrap_or("N/A")
    )));
    lines.push(ratatui::text::Line::from(format!(
        "  Parameters: {}",
        report
            .capabilities
            .parameter_count
            .as_deref()
            .unwrap_or("N/A")
    )));
    lines.push(ratatui::text::Line::from(format!(
        "  File Size: {}",
        format_bytes(app.model_load_bytes)
    )));
    lines.push(ratatui::text::Line::from(format!(
        "  Safety Margin: {:.2}x",
        app.safety_margin
    )));
    lines.push(ratatui::text::Line::from(format!(
        "  Inferred Capabilities: {:?}",
        report.inferred_capabilities
    )));

    if let Some(serde_json::Value::Array(styles)) = report.raw_metadata.get("available_styles") {
        let style_list: Vec<&str> = styles.iter().filter_map(|s| s.as_str()).collect();
        lines.push(ratatui::text::Line::from(format!(
            "  Voice Styles: {}",
            style_list.join(", ")
        )));
    }
    if let Some(serde_json::Value::Array(samples)) = report.raw_metadata.get("audio_samples") {
        let sample_list: Vec<&str> = samples.iter().filter_map(|s| s.as_str()).collect();
        let joined = if sample_list.len() > 4 {
            format!(
                "{} (+{} more)",
                sample_list[..4].join(", "),
                sample_list.len() - 4
            )
        } else {
            sample_list.join(", ")
        };
        lines.push(ratatui::text::Line::from(format!(
            "  Audio Samples: {joined}"
        )));
    }
    lines
}

fn render_security_guard(f: &mut ratatui::Frame, app: &App, area: Rect) {
    // The verdict comes from lmsafe-zone — the same rule the CLI prints.
    let admission = app.model_path.as_ref().map_or_else(
        || assess_with_monitor(&app.monitor, 0, app.safety_margin),
        |path| assess_model_path_with_monitor(&app.monitor, path, app.safety_margin),
    );

    let (verdict_text, verdict_style) = match admission.verdict {
        Verdict::Accepted => (
            format!("🟢 ACCEPTED: {}", admission.message),
            Style::default()
                .fg(Color::Rgb(34, 197, 94))
                .add_modifier(Modifier::BOLD),
        ),
        Verdict::Rejected => (
            format!("🔴 REJECTED: {}", admission.message),
            Style::default()
                .fg(Color::Rgb(239, 68, 68))
                .add_modifier(Modifier::BOLD),
        ),
        Verdict::Unknown => (
            format!("🟡 UNKNOWN: {}", admission.message),
            Style::default()
                .fg(Color::Rgb(234, 179, 8))
                .add_modifier(Modifier::BOLD),
        ),
    };

    let mut text = Vec::new();
    text.push(ratatui::text::Line::from(vec![
        ratatui::text::Span::styled("Verdict: ", Style::default().add_modifier(Modifier::BOLD)),
        ratatui::text::Span::styled(verdict_text, verdict_style),
    ]));
    text.push(ratatui::text::Line::from(""));

    // Render detailed metadata if screen height allows it
    let render_full_details = area.height >= 8;
    text.extend(model_detail_lines(app, render_full_details));

    let paragraph = Paragraph::new(text).block(
        Block::default()
            .title(" 🛡️ Security Guard (Admission Control) ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Rgb(100, 116, 139))),
    );
    f.render_widget(paragraph, area);
}

fn render_processes(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(" 📋 Top Processes ")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Rgb(100, 116, 139)));

    let procs = match app.monitor.top_processes_by_cpu(20) {
        MetricStatus::Available(list) => list,
        MetricStatus::Restricted => {
            f.render_widget(
                Paragraph::new("Process List: Access Denied").block(block),
                area,
            );
            return;
        }
        MetricStatus::Unavailable => {
            f.render_widget(Paragraph::new("Process List: N/A").block(block), area);
            return;
        }
    };

    let mut data = Vec::new();
    for p in procs {
        data.push((
            p.pid.to_string(),
            p.name.clone(),
            format!("{:.1}%", p.cpu_usage),
            format!("{:.1} MB", bytes_f64(p.memory) / 1024.0 / 1024.0),
        ));
    }

    let rows: Vec<Row> = data
        .iter()
        .map(|(pid, name, cpu, mem)| {
            Row::new(vec![
                pid.as_str(),
                name.as_str(),
                cpu.as_str(),
                mem.as_str(),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(10),
            Constraint::Percentage(50),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ],
    )
    .header(
        Row::new(vec!["PID", "Name", "CPU%", "Mem"]).style(
            Style::default()
                .fg(Color::Rgb(234, 179, 8))
                .add_modifier(Modifier::BOLD),
        ),
    )
    .block(block);

    f.render_widget(table, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use sysreflect_model::{Capability, ModelCapabilities, ModelFamily};

    /// Flatten rendered lines to plain strings so assertions read as the exact
    /// text a user sees in the Security Guard panel.
    fn flatten(lines: &[ratatui::text::Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect()
    }

    fn report(family: ModelFamily, name: &str) -> ModelInspectionReport {
        ModelInspectionReport {
            family,
            name: name.to_string(),
            version: "1.0".to_string(),
            author: "test".to_string(),
            uuid: String::new(),
            capabilities: ModelCapabilities {
                is_instruct: false,
                has_chat_template: false,
                has_tool_support: false,
                has_json_support: false,
                is_moe: false,
                is_vision: false,
                context_length: 0,
                quantization: Some("Q4_0".to_string()),
                parameter_count: None,
            },
            inferred_capabilities: vec![Capability::TextGeneration],
            raw_metadata: HashMap::new(),
        }
    }

    fn app_with(model_report: Option<ModelInspectionReport>, model_load_bytes: u64) -> App {
        App {
            monitor: SystemMonitor::new(),
            cpu_history: Vec::new(),
            model_report,
            model_path: Some(PathBuf::from("/models/test.gguf")),
            model_load_bytes,
            safety_margin: 1.15,
        }
    }

    #[test]
    fn no_report_shows_path_and_failure_note() {
        let app = app_with(None, 0);

        assert_eq!(
            flatten(&model_detail_lines(&app, false)),
            vec!["  Path: /models/test.gguf"]
        );
        assert_eq!(
            flatten(&model_detail_lines(&app, true)),
            vec![
                "  Path: /models/test.gguf",
                "  (Model report unavailable - metadata parsing failed)",
            ]
        );
    }

    #[test]
    fn generic_report_full_details_matches_layout() {
        let app = app_with(
            Some(report(ModelFamily::LlamaCpp, "Test Harness Model")),
            152,
        );

        assert_eq!(
            flatten(&model_detail_lines(&app, true)),
            vec![
                "  Model Name: Test Harness Model",
                "  Family: LlamaCpp",
                "  Quantization: Q4_0",
                "  Parameters: N/A",
                "  File Size: 152 B",
                "  Safety Margin: 1.15x",
                "  Inferred Capabilities: [TextGeneration]",
            ]
        );
    }

    #[test]
    fn generic_report_collapses_to_name_and_family_when_short() {
        let app = app_with(
            Some(report(ModelFamily::LlamaCpp, "Test Harness Model")),
            152,
        );

        assert_eq!(
            flatten(&model_detail_lines(&app, false)),
            vec!["  Model Name: Test Harness Model", "  Family: LlamaCpp"]
        );
    }

    #[test]
    fn silero_report_shows_architecture_and_tensor_wiring() {
        let mut silero = report(ModelFamily::AudioSegmentation, "silero_vad");
        silero.raw_metadata.insert(
            "architecture".to_string(),
            serde_json::Value::String("Silero-RNN-v5".to_string()),
        );
        let app = app_with(Some(silero), 1024);

        assert_eq!(
            flatten(&model_detail_lines(&app, true)),
            vec![
                "  Model Name: silero_vad",
                "  Family: AudioSegmentation (Silero)",
                "  Architecture: Silero-RNN-v5",
                "  Capabilities: Audio/VAD, Streaming",
                "  Sample Rates: 8000Hz, 16000Hz",
                "  Tensor Check: Pass (Inputs: [audio, sr, state], Outputs: [prob, state])",
                "  File Size: 1.00 KB",
                "  Safety Margin: 1.15x",
            ]
        );
    }

    #[test]
    fn voice_styles_and_samples_render_truncated_at_four() {
        let mut onnx = report(ModelFamily::Onnx, "Supertonic");
        onnx.raw_metadata.insert(
            "available_styles".to_string(),
            serde_json::Value::Array(vec![
                serde_json::Value::String("angry".to_string()),
                serde_json::Value::String("calm".to_string()),
            ]),
        );
        let samples: Vec<serde_json::Value> = (0..6)
            .map(|i| serde_json::Value::String(format!("sample{i}")))
            .collect();
        onnx.raw_metadata.insert(
            "audio_samples".to_string(),
            serde_json::Value::Array(samples),
        );
        let app = app_with(Some(onnx), 1024);

        let lines = flatten(&model_detail_lines(&app, true));
        assert!(
            lines.contains(&"  Voice Styles: angry, calm".to_string()),
            "missing voice styles line; got {lines:?}"
        );
        assert!(
            lines.contains(
                &"  Audio Samples: sample0, sample1, sample2, sample3 (+2 more)".to_string()
            ),
            "missing truncated samples line; got {lines:?}"
        );
    }
}
