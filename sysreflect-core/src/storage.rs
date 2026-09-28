use crate::model::ModelFamily;
use serde::Serialize;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use sysinfo::Disks;
use tracing::{debug, info, warn};

static BASE_STORAGE_PATH: OnceLock<PathBuf> = OnceLock::new();
static SANDBOXED: OnceLock<bool> = OnceLock::new();

fn is_sandboxed() -> bool {
    *SANDBOXED.get().unwrap_or(&false)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaFamily {
    Video,
    Image,
}

impl MediaFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            MediaFamily::Video => "video",
            MediaFamily::Image => "image",
        }
    }

    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "mp4" | "mov" | "m4v" | "avi" | "mkv" | "webm" => Some(MediaFamily::Video),
            "jpg" | "jpeg" | "png" | "webp" | "gif" | "heic" => Some(MediaFamily::Image),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MediaInfo {
    pub file_name: String,
    pub size_bytes: u64,
}

#[tracing::instrument(skip_all, fields(base_path = %base_path.as_ref().display(), sandboxed))]
pub fn init_storage<P: AsRef<Path>>(base_path: P, sandboxed: bool) -> io::Result<()> {
    let path = base_path.as_ref().to_path_buf();
    info!(
        "Initializing storage at: {} (sandboxed={})",
        path.display(),
        sandboxed
    );

    fs::create_dir_all(&path)?;

    SANDBOXED.set(sandboxed).ok();

    match BASE_STORAGE_PATH.set(path.clone()) {
        Ok(()) => {
            debug!("Storage path set successfully");
            Ok(())
        }
        Err(_) => {
            warn!("Storage already initialized at: {}", path.display());
            Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "Storage already initialized",
            ))
        }
    }
}

pub fn get_base_path() -> PathBuf {
    BASE_STORAGE_PATH.get().cloned().unwrap_or_else(|| {
        let mut fallback = std::env::temp_dir();
        fallback.push("lmsafe_default_storage");
        fallback
    })
}

pub fn get_safe_zone_dir(family: ModelFamily) -> io::Result<PathBuf> {
    let mut path = get_base_path();
    path.push("models");
    path.push(family.as_str());
    fs::create_dir_all(&path)?;
    Ok(path)
}

pub fn get_media_safe_zone_dir(family: MediaFamily) -> io::Result<PathBuf> {
    let mut path = get_base_path();
    path.push("media");
    path.push(family.as_str());
    fs::create_dir_all(&path)?;
    Ok(path)
}

/// Checks if free disk space is sufficient to copy a file into SafeZone
pub fn is_storage_safe_for_duplication<P: AsRef<Path>>(source_path: P, safety_margin: f64) -> bool {
    let source = source_path.as_ref();
    if !source.exists() || !source.is_file() {
        return false;
    }

    let file_size = match source.metadata() {
        Ok(meta) => meta.len(),
        Err(_) => return false,
    };

    let required_space = (file_size as f64 * safety_margin) as u64;

    let base_dir = get_base_path();
    let disks = Disks::new_with_refreshed_list();

    // Match disk mount point with base storage path
    let available_space = disks
        .iter()
        .filter(|disk| base_dir.starts_with(disk.mount_point()))
        .map(|disk| disk.available_space())
        .max()
        .unwrap_or_else(|| {
            // Fallback: Check first disk
            disks
                .first()
                .map(|d| d.available_space())
                .unwrap_or(u64::MAX)
        });

    available_space >= required_space
}

#[tracing::instrument(skip_all, fields(source_path = %source_path.as_ref().display(), family = ?family, safety_margin))]
pub fn copy_to_safe_zone<P: AsRef<Path>>(
    source_path: P,
    family: ModelFamily,
    safety_margin: f64,
) -> io::Result<PathBuf> {
    let source = source_path.as_ref();
    info!(
        "Storing model in safe zone: {} -> {:?} (sandboxed={})",
        source.display(),
        family,
        is_sandboxed()
    );

    if !source.exists() || !source.is_file() {
        warn!(
            "Source file does not exist or is not a file: {}",
            source.display()
        );
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Source file does not exist or is not a file",
        ));
    }

    let dest_dir = get_safe_zone_dir(family)?;

    if is_sandboxed() {
        // Android (sandboxed): copy the file — can't trust original path stays accessible
        if !is_storage_safe_for_duplication(source, safety_margin) {
            warn!("Insufficient disk space for file duplication");
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "Insufficient free disk space in SafeZone for file duplication",
            ));
        }

        let file_name = source.file_name().ok_or_else(|| {
            warn!("Invalid source filename: {}", source.display());
            io::Error::new(io::ErrorKind::InvalidInput, "Invalid source filename")
        })?;

        let dest_path = dest_dir.join(file_name);

        debug!("Copying {} -> {}", source.display(), dest_path.display());
        fs::copy(source, &dest_path)?;

        info!("Model copied successfully to: {}", dest_path.display());
        Ok(dest_path)
    } else {
        // Desktop / Linux / macOS (unsandboxed): path IS the safe zone. Store reference only.
        let path_file = dest_dir.join("_model_path");
        debug!(
            "Storing path reference: {} -> {}",
            source.display(),
            path_file.display()
        );
        fs::write(&path_file, source.to_string_lossy().as_bytes())?;

        info!("Model path stored: {}", source.display());
        Ok(source.to_path_buf())
    }
}

pub fn copy_media_to_safe_zone<P: AsRef<Path>>(
    source_path: P,
    family: MediaFamily,
    safety_margin: f64,
) -> io::Result<PathBuf> {
    let source = source_path.as_ref();
    if !source.exists() || !source.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Source media file does not exist or is not a file",
        ));
    }

    // Security Guard Storage Check
    if !is_storage_safe_for_duplication(source, safety_margin) {
        return Err(io::Error::new(
            io::ErrorKind::OutOfMemory,
            "Insufficient free disk space in SafeZone for media file duplication",
        ));
    }

    let file_name = source
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid source filename"))?;
    let dest_dir = get_media_safe_zone_dir(family)?;
    let dest_path = dest_dir.join(file_name);

    fs::copy(source, &dest_path)?;
    Ok(dest_path)
}

pub fn clean_safe_zone(family: ModelFamily) -> io::Result<()> {
    let dir = get_safe_zone_dir(family)?;
    // Also remove _model_path reference (path-only mode)
    let path_file = dir.join("_model_path");
    if path_file.exists() {
        fs::remove_file(&path_file)?;
    }
    clean_directory(&dir)
}

pub fn clean_media_safe_zone(family: MediaFamily) -> io::Result<()> {
    let dir = get_media_safe_zone_dir(family)?;
    clean_directory(&dir)
}

pub fn list_media_safe_zone(family: MediaFamily) -> io::Result<Vec<MediaInfo>> {
    let dir = get_media_safe_zone_dir(family)?;
    let mut files = Vec::new();
    if dir.exists() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let metadata = fs::metadata(&path)?;
                let file_name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                files.push(MediaInfo {
                    file_name,
                    size_bytes: metadata.len(),
                });
            }
        }
    }
    Ok(files)
}

pub fn remove_media_from_safe_zone(family: MediaFamily, file_name: &str) -> io::Result<()> {
    let dir = get_media_safe_zone_dir(family)?;
    let target = dir.join(file_name);

    // Prevent path traversal: canonicalize both and ensure target is inside the dir
    let canonical_dir = fs::canonicalize(&dir).unwrap_or(dir.clone());
    let canonical_target = match fs::canonicalize(&target) {
        Ok(p) => p,
        Err(_) => {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Media file not found in SafeZone",
            ))
        }
    };
    if !canonical_target.starts_with(&canonical_dir) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Path traversal detected",
        ));
    }

    if !canonical_target.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Media file not found in SafeZone",
        ));
    }

    fs::remove_file(&canonical_target)?;
    info!(
        "Removed media from SafeZone: {}",
        canonical_target.display()
    );
    Ok(())
}

fn clean_directory(dir: &Path) -> io::Result<()> {
    if dir.exists() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                fs::remove_file(path)?;
            } else if path.is_dir() {
                fs::remove_dir_all(path)?;
            }
        }
    }
    Ok(())
}

pub fn get_first_stored_model(family: ModelFamily) -> io::Result<Option<PathBuf>> {
    let dir = get_safe_zone_dir(family)?;
    if dir.exists() {
        // Path-only mode (desktop/linux/macOS): check _model_path reference first
        let path_file = dir.join("_model_path");
        if path_file.exists() {
            if let Ok(path_str) = fs::read_to_string(&path_file) {
                let model_path = PathBuf::from(path_str.trim());
                if model_path.exists() {
                    debug!("Found stored model path: {}", model_path.display());
                    return Ok(Some(model_path));
                }
                warn!("Stored path no longer exists: {}", model_path.display());
            }
        }
        // Copy mode (Android): scan for actual model files
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.is_file() && path.file_name().map(|n| n != "_model_path").unwrap_or(true) {
                return Ok(Some(path));
            }
        }
    }
    Ok(None)
}
