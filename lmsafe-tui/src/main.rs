use clap::{Parser, Subcommand, ValueEnum};
use color_eyre::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Gauge, Paragraph, Row, Table, Sparkline},
    Terminal,
};
use serde::Serialize;
use std::{io, time::{Duration, Instant}, thread};
use sysreflect::{SystemMonitor, MetricStatus, ProcessInfo};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the interactive TUI monitor
    Tui,
    /// Run a continuous non-interactive monitor
    Monitor {
        /// Output format
        #[arg(short, long, value_enum, default_value_t = Format::Json)]
        format: Format,
        /// Update interval in seconds
        #[arg(short, long, default_value_t = 1)]
        interval: u64,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Format {
    Json,
    Kv,
}

#[derive(Serialize)]
struct MonitorOutput {
    timestamp: u64,
    hostname: String,
    cpu_usage: MetricStatus<f32>,
    memory: MetricStatus<(u64, u64)>,
    top_processes: MetricStatus<Vec<ProcessInfo>>,
    network: MetricStatus<(u64, u64)>,
}

struct App {
    monitor: SystemMonitor,
    cpu_history: Vec<u64>,
}

impl App {
    fn new() -> Self {
        Self {
            monitor: SystemMonitor::new(),
            cpu_history: Vec::with_capacity(100),
        }
    }

    fn on_tick(&mut self) {
        self.monitor.refresh();
        let cpu_val = match self.monitor.global_cpu_usage() {
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

    match cli.command {
        Commands::Tui => run_tui()?,
        Commands::Monitor { format, interval } => run_monitor(format, interval)?,
    }

    Ok(())
}

fn run_tui() -> Result<()> {
    let mut terminal = setup_terminal()?;
    let mut app = App::new();
    let tick_rate = Duration::from_millis(500);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && (key.code == KeyCode::Char('q') || key.code == KeyCode::Esc) {
                    break;
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = Instant::now();
        }
    }

    restore_terminal(terminal)?;
    Ok(())
}

fn run_monitor(format: Format, interval: u64) -> Result<()> {
    let mut monitor = SystemMonitor::new();
    let sleep_duration = Duration::from_secs(interval);

    loop {
        monitor.refresh();
        let (host, _, _, _) = monitor.host_info();
        
        let output = MonitorOutput {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            hostname: host,
            cpu_usage: monitor.global_cpu_usage(),
            memory: monitor.memory_info(),
            top_processes: monitor.top_processes_by_cpu(5),
            network: monitor.network_throughput(),
        };

        match format {
            Format::Json => {
                println!("{}", serde_json::to_string(&output)?);
            }
            Format::Kv => {
                let cpu = match output.cpu_usage {
                    MetricStatus::Available(v) => format!("{:.1}%", v),
                    _ => "N/A".to_string(),
                };
                let mem = match output.memory {
                    MetricStatus::Available((t, u)) => format!("{:.1}/{:.1} GB", u as f64 / 1e9, t as f64 / 1e9),
                    _ => "N/A".to_string(),
                };
                println!("ts={} host={} cpu={} mem={}", output.timestamp, output.hostname, cpu, mem);
            }
        }

        thread::sleep(sleep_duration);
    }
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
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Length(7), // CPU & Memory Area
            Constraint::Min(0),    // Process List
        ])
        .split(f.area());

    render_header(f, app, chunks[0]);
    render_resources(f, app, chunks[1]);
    render_processes(f, app, chunks[2]);
}

fn render_header(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let (host, os, kernel, uptime) = app.monitor.host_info();
    let net_info = match app.monitor.network_throughput() {
        MetricStatus::Available((rx, tx)) => format!("Net RX: {} B | TX: {} B", rx, tx),
        MetricStatus::Restricted => "Net: Access Denied".to_string(),
        MetricStatus::Unavailable => "Net: N/A".to_string(),
    };

    let header = Paragraph::new(format!(
        "Host: {} | OS: {} | Kernel: {} | Uptime: {}s | {}",
        host, os, kernel, uptime, net_info
    ))
    .block(Block::default().borders(Borders::ALL).title(" System Status "));
    f.render_widget(header, area);
}

fn render_resources(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // CPU
    let cpu_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(chunks[0]);

    let sparkline = Sparkline::default()
        .block(Block::default().title(" CPU History ").borders(Borders::ALL))
        .data(&app.cpu_history)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sparkline, cpu_chunks[0]);

    let (cpu_percent, cpu_label) = match app.monitor.global_cpu_usage() {
        MetricStatus::Available(v) => (v as u16, format!("{:.1}%", v)),
        MetricStatus::Restricted => (0, "Restricted (Access Denied)".to_string()),
        MetricStatus::Unavailable => (0, "N/A".to_string()),
    };
    
    let cpu_gauge = Gauge::default()
        .block(Block::default().title(format!(" CPU Usage ({} cores) ", app.monitor.core_count())).borders(Borders::ALL))
        .gauge_style(Style::default().fg(Color::Cyan))
        .percent(cpu_percent)
        .label(cpu_label);
    f.render_widget(cpu_gauge, cpu_chunks[1]);

    // Memory
    let mem_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(100)])
        .split(chunks[1]);

    let (mem_percent, mem_label) = match app.monitor.memory_info() {
        MetricStatus::Available((total, used)) => {
            let p = if total > 0 { (used as f64 / total as f64 * 100.0) as u16 } else { 0 };
            (p, format!("{:.1} / {:.1} GB ({:.1}%)", used as f64 / 1e9, total as f64 / 1e9, p))
        },
        MetricStatus::Restricted => (0, "Memory: Access Denied".to_string()),
        MetricStatus::Unavailable => (0, "Memory: N/A".to_string()),
    };

    let mem_gauge = Gauge::default()
        .block(Block::default().title(" Memory ").borders(Borders::ALL))
        .gauge_style(Style::default().fg(Color::Magenta))
        .percent(mem_percent)
        .label(mem_label);
    f.render_widget(mem_gauge, mem_chunks[0]);
}

fn render_processes(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let block = Block::default().title(" Top Processes ").borders(Borders::ALL);
    
    let procs = match app.monitor.top_processes_by_cpu(20) {
        MetricStatus::Available(list) => list,
        MetricStatus::Restricted => {
            f.render_widget(Paragraph::new("Process List: Access Denied").block(block), area);
            return;
        },
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
            format!("{:.1} MB", p.memory as f64 / 1024.0 / 1024.0),
        ));
    }

    let rows: Vec<Row> = data.iter()
        .map(|(pid, name, cpu, mem)| {
            Row::new(vec![pid.as_str(), name.as_str(), cpu.as_str(), mem.as_str()])
        })
        .collect();

    let table = Table::new(rows, [
        Constraint::Percentage(10),
        Constraint::Percentage(50),
        Constraint::Percentage(20),
        Constraint::Percentage(20),
    ])
    .header(Row::new(vec!["PID", "Name", "CPU%", "Mem"])
        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)))
    .block(block);

    f.render_widget(table, area);
}
