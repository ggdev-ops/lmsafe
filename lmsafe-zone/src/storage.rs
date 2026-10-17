use serde::Serialize;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use sysinfo::Disks;
use sysreflect_model::ModelFamily;
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
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            MediaFamily::Video => "video",
            MediaFamily::Image => "image",
        }
    }

    #[must_use]
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

/// Create the storage root and record it as the process-wide base path.
///
/// # Errors
///
/// * [`ErrorKind::AlreadyExists`](io::ErrorKind::AlreadyExists) if storage was
///   already initialized in this process.
/// * IO errors from creating the directory.
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

    if let Ok(()) = BASE_STORAGE_PATH.set(path.clone()) {
        debug!("Storage path set successfully");
        Ok(())
    } else {
        warn!("Storage already initialized at: {}", path.display());
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "Storage already initialized",
        ))
    }
}

pub fn get_base_path() -> PathBuf {
    BASE_STORAGE_PATH.get().cloned().unwrap_or_else(|| {
        let mut fallback = std::env::temp_dir();
        fallback.push("lmsafe_default_storage");
        fallback
    })
}

/// Resolve (and create) the model safe-zone directory for a family.
///
/// # Errors
///
/// IO errors from creating the directory.
pub fn get_safe_zone_dir(family: ModelFamily) -> io::Result<PathBuf> {
    let mut path = get_base_path();
    path.push("models");
    path.push(family.as_str());
    fs::create_dir_all(&path)?;
    Ok(path)
}

/// Resolve (and create) the media safe-zone directory for a family.
///
/// # Errors
///
/// IO errors from creating the directory.
pub fn get_media_safe_zone_dir(family: MediaFamily) -> io::Result<PathBuf> {
    let mut path = get_base_path();
    path.push("media");
    path.push(family.as_str());
    fs::create_dir_all(&path)?;
    Ok(path)
}

/// Checks if free disk space is sufficient to copy a file into `SafeZone`
pub fn is_storage_safe_for_duplication<P: AsRef<Path>>(source_path: P, safety_margin: f64) -> bool {
    let source = source_path.as_ref();
    if !source.exists() || !source.is_file() {
        return false;
    }

    let file_size = match source.metadata() {
        Ok(meta) => meta.len(),
        Err(_) => return false,
    };

    // Same f64 margin arithmetic as `admission::assess_admission`; the two
    // must agree on what "required space" means.
    #[expect(
        clippy::cast_precision_loss,
        reason = "margin is fractional, so the product must be f64"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "result truncates toward zero; matches the admission rule"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "f64 -> u64 for a size; inputs are non-negative file sizes"
    )]
    let required_space = (file_size as f64 * safety_margin) as u64;

    let base_dir = get_base_path();
    let disks = Disks::new_with_refreshed_list();

    // Match disk mount point with base storage path
    let available_space = disks
        .iter()
        .filter(|disk| base_dir.starts_with(disk.mount_point()))
        .map(sysinfo::Disk::available_space)
        .max()
        .unwrap_or_else(|| {
            // Fallback: Check first disk
            disks
                .first()
                .map_or(u64::MAX, sysinfo::Disk::available_space)
        });

    available_space >= required_space
}

/// Copy a model file into the safe zone for its family.
///
/// # Errors
///
/// * [`ErrorKind::NotFound`](io::ErrorKind::NotFound) if the source is missing
///   or not a regular file.
/// * [`ErrorKind::OutOfMemory`](io::ErrorKind::OutOfMemory) when free disk space
///   is below the required size plus safety margin.
/// * [`ErrorKind::InvalidInput`](io::ErrorKind::InvalidInput) when the source
///   has no usable file name.
/// * IO errors from creating directories or copying the file.
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

/// Copy a media file into the safe zone for its family.
///
/// # Errors
///
/// * [`ErrorKind::NotFound`](io::ErrorKind::NotFound) if the source is missing
///   or not a regular file.
/// * [`ErrorKind::OutOfMemory`](io::ErrorKind::OutOfMemory) when free disk space
///   is below the required size plus safety margin.
/// * [`ErrorKind::InvalidInput`](io::ErrorKind::InvalidInput) when the source
///   has no usable file name.
/// * IO errors from creating directories or copying the file.
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

/// Remove every stored model artifact for a family.
///
/// # Errors
///
/// IO errors from resolving the directory or deleting its contents.
pub fn clean_safe_zone(family: ModelFamily) -> io::Result<()> {
    let dir = get_safe_zone_dir(family)?;
    // Also remove _model_path reference (path-only mode)
    let path_file = dir.join("_model_path");
    if path_file.exists() {
        fs::remove_file(&path_file)?;
    }
    clean_directory(&dir)
}

/// Remove every stored media artifact for a family.
///
/// # Errors
///
/// IO errors from resolving the directory or deleting its contents.
pub fn clean_media_safe_zone(family: MediaFamily) -> io::Result<()> {
    let dir = get_media_safe_zone_dir(family)?;
    clean_directory(&dir)
}

/// List the media files held in a family's safe zone.
///
/// # Errors
///
/// IO errors from resolving the directory or reading its entries.
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

/// Delete a single media file from a family's safe zone.
///
/// # Errors
///
/// * [`ErrorKind::NotFound`](io::ErrorKind::NotFound) when the file is absent.
/// * [`ErrorKind::InvalidInput`](io::ErrorKind::InvalidInput) when the name
///   escapes the safe zone (path traversal).
/// * IO errors from resolving or deleting the file.
pub fn remove_media_from_safe_zone(family: MediaFamily, file_name: &str) -> io::Result<()> {
    let dir = get_media_safe_zone_dir(family)?;
    let target = dir.join(file_name);

    // Prevent path traversal: canonicalize both and ensure target is inside the dir
    let canonical_dir = fs::canonicalize(&dir).unwrap_or(dir.clone());
    let Ok(canonical_target) = fs::canonicalize(&target) else {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Media file not found in SafeZone",
        ));
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

/// Return the first model stored for a family, or `None` if the zone is empty.
///
/// Prefers the `_model_path` reference (path-only mode on desktop) and falls
/// back to scanning for a copied file (Android).
///
/// # Errors
///
/// IO errors from resolving the directory or reading its entries.
pub fn get_first_stored_model(family: ModelFamily) -> io::Result<Option<PathBuf>> {
    let dir = get_safe_zone_dir(family)?;
    if dir.exists() {
        // Path-only mode (desktop/linux/macOS): check _model_path reference first
        let path_file = dir.join("_model_path");
        if path_file.exists()
            && let Ok(path_str) = fs::read_to_string(&path_file)
        {
            let model_path = PathBuf::from(path_str.trim());
            if model_path.exists() {
                debug!("Found stored model path: {}", model_path.display());
                return Ok(Some(model_path));
            }
            warn!("Stored path no longer exists: {}", model_path.display());
        }
        // Copy mode (Android): scan for actual model files
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.is_file() && path.file_name().is_none_or(|n| n != "_model_path") {
                return Ok(Some(path));
            }
        }
    }
    Ok(None)
}
