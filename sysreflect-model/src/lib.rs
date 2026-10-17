use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ModelFamily {
    #[serde(rename = "llama-cpp")]
    LlamaCpp,
    #[serde(rename = "whisper-cpp")]
    WhisperCpp,
    #[serde(rename = "litert-lm")]
    LiteRtLm,
    #[serde(rename = "onnx")]
    Onnx,
    #[serde(rename = "audio-segmentation")]
    AudioSegmentation,
    #[serde(rename = "safetensors")]
    SafeTensors,
    #[serde(rename = "unknown")]
    Unknown,
}

impl ModelFamily {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelFamily::LlamaCpp => "llama-cpp",
            ModelFamily::WhisperCpp => "whisper-cpp",
            ModelFamily::LiteRtLm => "litert-lm",
            ModelFamily::Onnx => "onnx",
            ModelFamily::AudioSegmentation => "audio-segmentation",
            ModelFamily::SafeTensors => "safetensors",
            ModelFamily::Unknown => "unknown",
        }
    }

    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s {
            "llama-cpp" => ModelFamily::LlamaCpp,
            "whisper-cpp" => ModelFamily::WhisperCpp,
            "litert-lm" => ModelFamily::LiteRtLm,
            "onnx" => ModelFamily::Onnx,
            "audio-segmentation" => ModelFamily::AudioSegmentation,
            "safetensors" => ModelFamily::SafeTensors,
            _ => ModelFamily::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Capability {
    #[serde(rename = "TEXT_GENERATION")]
    TextGeneration,
    #[serde(rename = "SPEECH_TO_TEXT")]
    SpeechToText,
    #[serde(rename = "TOOLS")]
    Tools,
    #[serde(rename = "AUDIO_INPUT")]
    AudioInput,
    #[serde(rename = "AUDIO_OUTPUT")]
    AudioOutput,
    #[serde(rename = "MULTIMODAL")]
    Multimodal,
    #[serde(rename = "VISION")]
    Vision,
    #[serde(rename = "TEXT_TO_SPEECH")]
    TextToSpeech,
}

/// Capability flags for a model. Serialized to JSON and consumed through the
/// C FFI (Android), so the field shape is a public contract.
#[expect(
    clippy::struct_excessive_bools,
    reason = "these bools are distinct JSON fields on the FFI contract; \
              collapsing them into an enum/bitflags would break Android callers"
)]
#[derive(Debug, Serialize)]
pub struct ModelCapabilities {
    pub is_instruct: bool,
    pub has_chat_template: bool,
    pub has_tool_support: bool,
    pub has_json_support: bool,
    pub is_moe: bool,
    pub is_vision: bool,
    pub context_length: u64,
    pub quantization: Option<String>,
    pub parameter_count: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelInspectionReport {
    pub family: ModelFamily,
    pub name: String,
    pub version: String,
    pub author: String,
    pub uuid: String,
    pub capabilities: ModelCapabilities,
    pub inferred_capabilities: Vec<Capability>,
    pub raw_metadata: HashMap<String, Value>,
}

impl ModelInspectionReport {
    pub fn infer_capabilities(&mut self) {
        let mut caps = Vec::new();
        match self.family {
            ModelFamily::LlamaCpp => {
                caps.push(Capability::TextGeneration);
                if self.capabilities.has_tool_support {
                    caps.push(Capability::Tools);
                }
                if self.capabilities.is_vision {
                    caps.push(Capability::Vision);
                    caps.push(Capability::Multimodal);
                }
            }
            ModelFamily::WhisperCpp => {
                caps.push(Capability::SpeechToText);
                caps.push(Capability::AudioInput);
            }
            ModelFamily::AudioSegmentation => {
                caps.push(Capability::AudioInput);
            }
            ModelFamily::LiteRtLm => {
                caps.push(Capability::TextGeneration);
                if self.capabilities.has_tool_support {
                    caps.push(Capability::Tools);
                }
            }
            ModelFamily::Onnx | ModelFamily::SafeTensors => {
                caps.extend(self.inferred_capabilities.clone());
            }
            ModelFamily::Unknown => {}
        }
        self.inferred_capabilities = caps;
    }
}

/// Resolves Hugging Face hub cache directory (with snapshots/refs) to the active snapshot path.
pub fn resolve_model_path<P: AsRef<Path>>(path: P) -> std::path::PathBuf {
    let p = path.as_ref();
    if p.is_dir() {
        let snapshots_dir = p.join("snapshots");
        if snapshots_dir.is_dir() {
            // Check refs/main
            let ref_main = p.join("refs/main");
            if let Ok(hash) = fs::read_to_string(&ref_main) {
                let target = snapshots_dir.join(hash.trim());
                if target.is_dir() {
                    return target;
                }
            }
            // Fallback: take the first snapshot folder found
            if let Ok(entries) = fs::read_dir(&snapshots_dir) {
                for entry in entries.flatten() {
                    let sp = entry.path();
                    if sp.is_dir() {
                        return sp;
                    }
                }
            }
        }
    }
    p.to_path_buf()
}

// ==========================================
// 1. Model Family Detector
// ==========================================

/// True if any direct child of `dir` carries the given file extension.
fn dir_has_extension(dir: &Path, ext: &str) -> bool {
    fs::read_dir(dir).is_ok_and(|entries| {
        entries
            .flatten()
            .any(|entry| entry.path().extension().is_some_and(|e| e == ext))
    })
}

/// Infer a family from a model *directory*. Returns `None` when nothing
/// recognizable is present.
fn probe_directory_family(path: &Path) -> Option<ModelFamily> {
    debug!("Path is a directory, checking for directory-based model families");

    let config_path = path.join("config.json");
    let tts_path = path.join("onnx/tts.json");
    if config_path.exists() && tts_path.exists() {
        debug!("Found config.json and onnx/tts.json - detected as ONNX");
        return Some(ModelFamily::Onnx);
    }

    if dir_has_extension(path, "onnx") {
        debug!("Found .onnx file in directory - detected as ONNX");
        return Some(ModelFamily::Onnx);
    }

    let onnx_dir = path.join("onnx");
    if onnx_dir.is_dir() && dir_has_extension(&onnx_dir, "onnx") {
        debug!("Found .onnx file in onnx/ subdirectory - detected as ONNX");
        return Some(ModelFamily::Onnx);
    }

    if dir_has_extension(path, "safetensors") {
        debug!("Found .safetensors file in directory - detected as SafeTensors");
        return Some(ModelFamily::SafeTensors);
    }

    debug!("No recognized model files found in directory");
    None
}

#[tracing::instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn detect_model_family<P: AsRef<Path>>(path: P) -> ModelFamily {
    let resolved_path = resolve_model_path(path.as_ref());
    let path = resolved_path.as_path();
    info!("Detecting model family for: {}", path.display());

    // A directory is never read as a file, so its family comes purely from
    // which well-known artifacts it holds.
    if path.is_dir() {
        return probe_directory_family(path).unwrap_or(ModelFamily::Unknown);
    }

    if path.extension().is_some_and(|e| e == "onnx") {
        debug!("File extension is .onnx - detected as ONNX");
        return ModelFamily::Onnx;
    }

    if path.extension().is_some_and(|e| e == "safetensors") {
        debug!("File extension is .safetensors - detected as SafeTensors");
        return ModelFamily::SafeTensors;
    }

    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            warn!("Failed to open file {}: {}", path.display(), e);
            return ModelFamily::Unknown;
        }
    };

    let mut header = [0u8; 8];
    let bytes_read = match file.read(&mut header) {
        Ok(n) => n,
        Err(e) => {
            warn!("Failed to read header from {}: {}", path.display(), e);
            return ModelFamily::Unknown;
        }
    };

    if bytes_read < 4 {
        debug!("File too small ({} bytes) to determine family", bytes_read);
        return ModelFamily::Unknown;
    }

    let magic4 = &header[0..4];
    let magic8 = &header[0..bytes_read.min(8)];

    // LITERTLM magic
    if magic8 == b"LITERTLM" {
        debug!("Magic bytes match LITERTLM");
        return ModelFamily::LiteRtLm;
    }

    // GGUF magic
    if magic4 == b"GGUF" {
        debug!("Magic bytes match GGUF");
        return ModelFamily::LlamaCpp;
    }

    // GGML Whisper magics (including little-endian order: lmgg, fmgg, jmgg)
    if magic4 == b"ggml"
        || magic4 == b"ggmf"
        || magic4 == b"ggjt"
        || magic4 == b"lmgg"
        || magic4 == b"fmgg"
        || magic4 == b"jmgg"
    {
        debug!("Magic bytes match GGML/Whisper");
        return ModelFamily::WhisperCpp;
    }

    // TFLite magic "TFL3" at offset 4
    if bytes_read >= 8 {
        let tfl_magic = &header[4..8];
        if tfl_magic == b"TFL3" {
            debug!("Magic bytes match TFL3 (LiteRT)");
            return ModelFamily::LiteRtLm;
        }
    }

    // Safetensors magic check: first 8 bytes as u64 is N, and next byte is '{'
    if bytes_read == 8 {
        let n = u64::from_le_bytes(header);
        if n > 0 && n < 100_000_000 {
            let mut first_char = [0u8; 1];
            if file.read_exact(&mut first_char).is_ok() && first_char[0] == b'{' {
                debug!("Safetensors header pattern detected");
                return ModelFamily::SafeTensors;
            }
        }
    }

    debug!("No recognized magic bytes found");
    ModelFamily::Unknown
}

// ==========================================
// 2. GGUF Parser
// ==========================================
struct GgufParser<R: Read + Seek> {
    reader: R,
    skip_keys: HashSet<String>,
}

impl<R: Read + Seek> GgufParser<R> {
    fn new(reader: R) -> Self {
        let mut skip_keys = HashSet::new();
        skip_keys.insert("tokenizer.ggml.tokens".to_string());
        skip_keys.insert("tokenizer.ggml.scores".to_string());
        skip_keys.insert("tokenizer.ggml.token_type".to_string());
        skip_keys.insert("tokenizer.ggml.merges".to_string());
        Self { reader, skip_keys }
    }

    fn read_u32(&mut self) -> std::io::Result<u32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> std::io::Result<u64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_string(&mut self) -> std::io::Result<String> {
        let len = self.read_u64()?;
        // A corrupt header can claim a length no address space can hold.
        let len = usize::try_from(len).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "GGUF string length does not fit in usize",
            )
        })?;
        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn skip_value(&mut self, val_type: u32) -> std::io::Result<()> {
        match val_type {
            0 | 1 | 7 => {
                self.reader.seek(SeekFrom::Current(1))?;
            } // u8, i8, bool
            2 | 3 => {
                self.reader.seek(SeekFrom::Current(2))?;
            } // u16, i16
            4..=6 => {
                self.reader.seek(SeekFrom::Current(4))?;
            } // u32, i32, f32
            10..=12 => {
                self.reader.seek(SeekFrom::Current(8))?;
            } // u64, i64, f64
            8 => {
                // String
                let len = self.read_u64()?;
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "seek offset; a length above i64::MAX is rejected by the reader"
                )]
                let len = len as i64;
                self.reader.seek(SeekFrom::Current(len))?;
            }
            9 => {
                // Array
                let elem_type = self.read_u32()?;
                let count = self.read_u64()?;
                for _ in 0..count {
                    self.skip_value(elem_type)?;
                }
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Unknown value type",
                ));
            }
        }
        Ok(())
    }

    fn parse_value(&mut self, val_type: u32) -> std::io::Result<Value> {
        let val = match val_type {
            0 => {
                let mut buf = [0u8; 1];
                self.reader.read_exact(&mut buf)?;
                Value::Number(buf[0].into())
            }
            1 => {
                let mut buf = [0u8; 1];
                self.reader.read_exact(&mut buf)?;
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "GGUF encodes i8 values as a raw byte; reinterpreting is the format"
                )]
                let signed = buf[0] as i8;
                Value::Number(signed.into())
            }
            2 => {
                let mut buf = [0u8; 2];
                self.reader.read_exact(&mut buf)?;
                Value::Number(u16::from_le_bytes(buf).into())
            }
            3 => {
                let mut buf = [0u8; 2];
                self.reader.read_exact(&mut buf)?;
                Value::Number(i16::from_le_bytes(buf).into())
            }
            4 => Value::Number(self.read_u32()?.into()),
            5 => {
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "GGUF i32 values are stored as u32 bits; same width, no data lost"
                )]
                let val = self.read_u32()? as i32;
                Value::Number(val.into())
            }
            6 => {
                let mut buf = [0u8; 4];
                self.reader.read_exact(&mut buf)?;
                let f = f32::from_le_bytes(buf);
                serde_json::Number::from_f64(f64::from(f)).map_or(Value::Null, Value::Number)
            }
            7 => {
                let mut buf = [0u8; 1];
                self.reader.read_exact(&mut buf)?;
                Value::Bool(buf[0] != 0)
            }
            8 => Value::String(self.read_string()?),
            9 => {
                let elem_type = self.read_u32()?;
                let count = self.read_u64()?;
                if count > 1000 {
                    for _ in 0..count {
                        self.skip_value(elem_type)?;
                    }
                    Value::String(format!(
                        "Array(type={elem_type}, count={count}) /* skipped */"
                    ))
                } else {
                    let count = usize::try_from(count).unwrap_or(0);
                    let mut arr = Vec::with_capacity(count);
                    for _ in 0..count {
                        arr.push(self.parse_value(elem_type)?);
                    }
                    Value::Array(arr)
                }
            }
            10 => Value::Number(self.read_u64()?.into()),
            11 => {
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "GGUF i64 values are stored as u64 bits; same width, no data lost"
                )]
                let val = self.read_u64()? as i64;
                Value::Number(val.into())
            }
            12 => {
                let mut buf = [0u8; 8];
                self.reader.read_exact(&mut buf)?;
                let f = f64::from_le_bytes(buf);
                serde_json::Number::from_f64(f).map_or(Value::Null, Value::Number)
            }
            _ => Value::Null,
        };
        Ok(val)
    }

    fn parse(&mut self) -> std::io::Result<HashMap<String, Value>> {
        // GGUF magic is already read
        let _version = self.read_u32()?;
        let _tensor_count = self.read_u64()?;
        let kv_count = self.read_u64()?;

        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = self.read_string()?;
            let val_type = self.read_u32()?;
            if self.skip_keys.contains(&key) {
                self.skip_value(val_type)?;
            } else {
                let value = self.parse_value(val_type)?;
                metadata.insert(key, value);
            }
        }
        Ok(metadata)
    }
}

// ==========================================
// 3. LiteRT-LM FlatBuffers/Protobuf Scanner
// ==========================================
struct FlatBufferReader<'a> {
    data: &'a [u8],
}

impl<'a> FlatBufferReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    fn read_int(&self, offset: usize) -> i32 {
        if offset + 4 > self.data.len() {
            return 0;
        }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.data[offset..offset + 4]);
        i32::from_le_bytes(buf)
    }

    fn read_short(&self, offset: usize) -> u16 {
        if offset + 2 > self.data.len() {
            return 0;
        }
        let mut buf = [0u8; 2];
        buf.copy_from_slice(&self.data[offset..offset + 2]);
        u16::from_le_bytes(buf)
    }

    fn read_long(&self, offset: usize) -> i64 {
        if offset + 8 > self.data.len() {
            return 0;
        }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&self.data[offset..offset + 8]);
        i64::from_le_bytes(buf)
    }

    /// `FlatBuffer` offsets are signed i32. A negative value means the file is
    /// corrupt; map it to the reader's `0` = "absent" sentinel instead of
    /// letting it wrap into a huge offset.
    fn as_offset(value: i32) -> usize {
        usize::try_from(value).unwrap_or(0)
    }

    fn get_root(&self) -> usize {
        Self::as_offset(self.read_int(0))
    }

    fn get_field_offset(&self, table_offset: usize, field_index: usize) -> usize {
        let table_i32 = i32::try_from(table_offset).unwrap_or(0);
        let vtable_offset = Self::as_offset(table_i32 - self.read_int(table_offset));
        if vtable_offset + 2 > self.data.len() {
            return 0;
        }
        let vtable_size = usize::from(self.read_short(vtable_offset));
        let field_offset_in_vtable = 4 + field_index * 2;
        if field_offset_in_vtable >= vtable_size {
            return 0;
        }
        let offset_in_table = usize::from(self.read_short(vtable_offset + field_offset_in_vtable));
        if offset_in_table == 0 {
            return 0;
        }
        table_offset + offset_in_table
    }

    fn get_indirect_offset(&self, table_offset: usize, field_index: usize) -> usize {
        let offset = self.get_field_offset(table_offset, field_index);
        if offset == 0 {
            return 0;
        }
        offset + Self::as_offset(self.read_int(offset))
    }

    fn get_byte_field(&self, table_offset: usize, field_index: usize, default: u8) -> u8 {
        let offset = self.get_field_offset(table_offset, field_index);
        if offset == 0 {
            default
        } else {
            self.data[offset]
        }
    }

    fn get_int_field(&self, table_offset: usize, field_index: usize, default: i32) -> i32 {
        let offset = self.get_field_offset(table_offset, field_index);
        if offset == 0 {
            default
        } else {
            self.read_int(offset)
        }
    }

    fn get_long_field(&self, table_offset: usize, field_index: usize, default: i64) -> i64 {
        let offset = self.get_field_offset(table_offset, field_index);
        if offset == 0 {
            default
        } else {
            self.read_long(offset)
        }
    }

    fn get_string_field(&self, table_offset: usize, field_index: usize) -> Option<String> {
        let mut offset = self.get_field_offset(table_offset, field_index);
        if offset == 0 {
            return None;
        }
        offset += Self::as_offset(self.read_int(offset));
        let length = Self::as_offset(self.read_int(offset));
        if offset + 4 + length > self.data.len() {
            return None;
        }
        let s = String::from_utf8_lossy(&self.data[offset + 4..offset + 4 + length]).into_owned();
        Some(s)
    }

    fn get_vector_size(&self, vector_offset: usize) -> usize {
        Self::as_offset(self.read_int(vector_offset))
    }

    fn get_vector_element_offset(&self, vector_offset: usize, index: usize) -> usize {
        let element_pos = vector_offset + 4 + index * 4;
        element_pos + Self::as_offset(self.read_int(element_pos))
    }

    fn get_union_value(
        &self,
        table_offset: usize,
        type_index: usize,
        value_index: usize,
    ) -> Option<Value> {
        let union_type = u8::try_from(self.get_int_field(table_offset, type_index, 0)).unwrap_or(0);
        let offset = self.get_indirect_offset(table_offset, value_index);
        if offset == 0 {
            return None;
        }

        let val = match union_type {
            1 => Value::Number(u64::from(self.data[offset]).into()), // UInt8
            2 => Value::Number(i64::from(self.data[offset]).into()), // Int8
            5 => {
                // FlatBuffers stores a UInt32 in an i32 slot: same width,
                // bit-for-bit reinterpretable.
                #[expect(
                    clippy::cast_sign_loss,
                    reason = "UInt32 lives in an i32 table slot; same width, no data lost"
                )]
                let bits = self.read_int(offset) as u32;
                Value::Number(u64::from(bits).into())
            }
            6 => Value::Number(i64::from(self.read_int(offset)).into()), // Int32
            9 => {
                // StringValue (a table with string field at index 0)
                self.get_string_field(offset, 0).map(Value::String)?
            }
            10 => {
                // UInt64 lives in an i64 table slot: same width, bit pattern
                // preserved.
                #[expect(
                    clippy::cast_sign_loss,
                    reason = "UInt64 lives in an i64 table slot; same width, no data lost"
                )]
                let bits = self.read_long(offset) as u64;
                Value::Number(bits.into())
            }
            11 => Value::Number(self.read_long(offset).into()), // Int64
            _ => return None,
        };
        Some(val)
    }
}

struct ProtobufScanner<'a> {
    data: &'a [u8],
    i: usize,
    last_wire_type: u8,
}

impl<'a> ProtobufScanner<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            i: 0,
            last_wire_type: 0,
        }
    }

    fn has_next(&self) -> bool {
        self.i < self.data.len()
    }

    fn read_varint(&mut self) -> u64 {
        let mut value = 0u64;
        let mut shift = 0;
        while self.i < self.data.len() {
            let b = u64::from(self.data[self.i]);
            self.i += 1;
            value |= (b & 0x7F) << shift;
            if (b & 0x80) == 0 {
                break;
            }
            shift += 7;
        }
        value
    }

    fn read_fixed32(&mut self) -> u32 {
        if self.i + 4 > self.data.len() {
            return 0;
        }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.data[self.i..self.i + 4]);
        self.i += 4;
        u32::from_le_bytes(buf)
    }

    fn read_string(&mut self) -> String {
        let length = usize::try_from(self.read_varint()).unwrap_or(0);
        if self.i + length > self.data.len() {
            return String::new();
        }
        let s = String::from_utf8_lossy(&self.data[self.i..self.i + length]).into_owned();
        self.i += length;
        s
    }

    fn read_bytes(&mut self, length: usize) -> &'a [u8] {
        let end = (self.i + length).min(self.data.len());
        let bytes = &self.data[self.i..end];
        self.i = end;
        bytes
    }

    fn next_tag(&mut self) -> (u32, u8) {
        let tag = u32::try_from(self.read_varint()).unwrap_or(0);
        self.last_wire_type = (tag & 0x7) as u8;
        let field_num = tag >> 3;
        (field_num, self.last_wire_type)
    }

    fn skip_field(&mut self) {
        match self.last_wire_type {
            0 => {
                self.read_varint();
            }
            1 => {
                self.i += 8;
            }
            2 => {
                let length = usize::try_from(self.read_varint()).unwrap_or(0);
                self.i += length;
            }
            5 => {
                self.i += 4;
            }
            _ => {
                self.i = self.data.len();
            } // Stop on unknown
        }
        if self.i > self.data.len() {
            self.i = self.data.len();
        }
    }
}

// ==========================================
// 4. Inspectors
// ==========================================

fn parse_llama_gguf<P: AsRef<Path>>(path: P) -> std::io::Result<ModelInspectionReport> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Skip magic GGUF
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;

    let mut gguf = GgufParser::new(reader);
    let raw = gguf.parse()?;

    // Helper macro to fetch strings/numbers
    let get_str = |k: &str| raw.get(k).and_then(|v| v.as_str()).map(String::from);
    let get_u64 = |k: &str| raw.get(k).and_then(serde_json::Value::as_u64);

    let name = get_str("general.name").unwrap_or_else(|| "Unknown Llama Model".to_string());
    let size_label = get_str("general.size_label");

    let arch = get_str("general.architecture").unwrap_or_else(|| "llama".to_string());
    let file_type = get_u64("general.file_type");
    let quant_label = file_type.and_then(|code| {
        // Map GGUF general.file_type code to labels
        match code {
            0 => Some("F32"),
            1 => Some("F16"),
            2 => Some("Q4_0"),
            3 => Some("Q4_1"),
            6 => Some("Q5_0"),
            7 => Some("Q5_1"),
            8 => Some("Q8_0"),
            10 => Some("Q2_K"),
            11 => Some("Q3_K_S"),
            12 => Some("Q3_K_M"),
            13 => Some("Q3_K_L"),
            14 => Some("Q4_K_S"),
            15 => Some("Q4_K_M"),
            16 => Some("Q5_K_S"),
            17 => Some("Q5_K_M"),
            18 => Some("Q6_K"),
            _ => None,
        }
        .map(String::from)
    });

    let context_key = format!("{arch}.context_length");
    let context_length = get_u64(&context_key).unwrap_or(0);

    let chat_template = get_str("tokenizer.chat_template").unwrap_or_default();

    let _is_llama3 = arch == "llama"
        && (chat_template.contains("header_id") || chat_template.contains("eot_id"));
    let _is_gemma = arch.starts_with("gemma");

    let inferred_capabilities = Vec::new();
    let family = ModelFamily::LlamaCpp;

    let is_instruct = get_str("general.finetune")
        .is_some_and(|f| f.to_lowercase().contains("instruct") || f.to_lowercase().contains("it"))
        || !chat_template.is_empty();
    let has_tool_support = chat_template.to_lowercase().contains("tool_calls")
        || chat_template.to_lowercase().contains("available_tools")
        || name.to_lowercase().contains("llama-3.1")
        || name.to_lowercase().contains("llama-3.2");

    let _block_key = format!("{arch}.block_count");
    let is_moe = get_u64(&format!("{arch}.expert_count")).unwrap_or(0) > 0;
    let is_vision = arch.to_lowercase().contains("clip");

    let param_count = size_label.or_else(|| {
        let re = regex::Regex::new(r"(?i)(\d+[BbMm])").ok()?;
        let cap = re.captures(&name)?;
        Some(cap[1].to_uppercase())
    });

    let caps = ModelCapabilities {
        is_instruct,
        has_chat_template: !chat_template.is_empty(),
        has_tool_support,
        has_json_support: has_tool_support || chat_template.to_lowercase().contains("json_schema"),
        is_moe,
        is_vision,
        context_length,
        quantization: quant_label,
        parameter_count: param_count,
    };

    Ok(ModelInspectionReport {
        inferred_capabilities,
        family,
        name: get_str("general.name").unwrap_or(name),
        version: get_str("general.version").unwrap_or_else(|| "v1.0".to_string()),
        author: get_str("general.author").unwrap_or_else(|| "Unknown".to_string()),
        uuid: get_str("general.uuid").unwrap_or_default(),
        capabilities: caps,
        raw_metadata: raw,
    })
}

/// Read a `LiteRT-LM` model (`FlatBuffer` header + embedded sections).
#[expect(
    clippy::too_many_lines,
    reason = "single-pass LiteRT-LM reader; splitting it requires parser tests first"
)]
fn parse_litertlm<P: AsRef<Path>>(path: P) -> std::io::Result<ModelInspectionReport> {
    let path = path.as_ref();
    let mut file = File::open(path)?;

    // Read LiteRT Magic header (32 bytes)
    let mut header = [0u8; 32];
    file.read_exact(&mut header)?;

    if &header[0..8] != b"LITERTLM" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid LiteRT-LM magic",
        ));
    }

    let major = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
    let minor = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
    let patch_version = u32::from_le_bytes([header[16], header[17], header[18], header[19]]);
    let header_end_offset = usize::try_from(u64::from_le_bytes([
        header[24], header[25], header[26], header[27], header[28], header[29], header[30],
        header[31],
    ]))
    .unwrap_or(0);

    // Read FlatBuffer header metadata
    let header_size = header_end_offset - 32;
    let mut header_data = vec![0u8; header_size];
    file.read_exact(&mut header_data)?;

    let fb = FlatBufferReader::new(&header_data);
    let root = fb.get_root();

    let mut system_meta = HashMap::new();
    let sys_meta_table = fb.get_indirect_offset(root, 0);
    if sys_meta_table != 0 {
        let entries_vector = fb.get_indirect_offset(sys_meta_table, 0);
        if entries_vector != 0 {
            let size = fb.get_vector_size(entries_vector);
            for i in 0..size {
                let kvp_table = fb.get_vector_element_offset(entries_vector, i);
                if let Some(key) = fb.get_string_field(kvp_table, 0)
                    && let Some(val) = fb.get_union_value(kvp_table, 1, 2)
                {
                    system_meta.insert(key, val);
                }
            }
        }
    }

    let mut proto_section = None;
    let section_meta_table = fb.get_indirect_offset(root, 1);
    if section_meta_table != 0 {
        let objects_vector = fb.get_indirect_offset(section_meta_table, 0);
        if objects_vector != 0 {
            let size = fb.get_vector_size(objects_vector);
            for i in 0..size {
                let section_table = fb.get_vector_element_offset(objects_vector, i);
                let data_type = fb.get_byte_field(section_table, 3, 0);
                if data_type == 5 {
                    // LlmMetadataProto
                    let begin = u64::try_from(fb.get_long_field(section_table, 1, 0)).unwrap_or(0);
                    let end = u64::try_from(fb.get_long_field(section_table, 2, 0)).unwrap_or(0);
                    proto_section = Some((begin, end));
                }
            }
        }
    }

    let mut jinja = None;
    let mut max_tokens = 0u64;
    let mut model_family: Option<String> = None;

    if let Some((begin, end)) = proto_section {
        file.seek(SeekFrom::Start(begin))?;
        // saturating: a corrupt header can list end < begin, which would
        // otherwise underflow and panic in debug builds.
        let len = usize::try_from(end.saturating_sub(begin)).unwrap_or(0);
        let mut proto_data = vec![0u8; len];
        file.read_exact(&mut proto_data)?;

        let mut scanner = ProtobufScanner::new(&proto_data);
        while scanner.has_next() {
            let (field, wire_type) = scanner.next_tag();
            match field {
                5 => {
                    max_tokens = if wire_type == 0 {
                        scanner.read_varint()
                    } else {
                        u64::from(scanner.read_fixed32())
                    };
                }
                6 => {
                    let length = usize::try_from(scanner.read_varint()).unwrap_or(0);
                    let bytes = scanner.read_bytes(length);
                    let mut sub_scanner = ProtobufScanner::new(bytes);
                    while sub_scanner.has_next() {
                        let (sub_field, _) = sub_scanner.next_tag();
                        model_family = match sub_field {
                            1 => Some("Generic"),
                            2 => Some("Gemma 3N"),
                            3 => Some("Function Gemma"),
                            4 => Some("Gemma 3"),
                            5 => Some("Qwen 3"),
                            7 => Some("Qwen 2.5"),
                            8 => Some("Gemma 4"),
                            9 => Some("Fast VLM"),
                            _ => model_family.as_deref(),
                        }
                        .map(String::from);
                        sub_scanner.skip_field();
                    }
                }
                7 => {
                    jinja = Some(scanner.read_string());
                }
                _ => scanner.skip_field(),
            }
        }
    }

    let name = system_meta
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("Gemma LiteRT")
        .to_string();
    let author = system_meta
        .get("author")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown")
        .to_string();
    let uuid = system_meta
        .get("uuid")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let chat_template = jinja.clone().unwrap_or_default();

    let is_instruct = system_meta.get("finetune").is_some_and(|v| {
        v.as_str()
            .is_some_and(|f| f.contains("instruct") || f.contains("it"))
    }) || !chat_template.is_empty();
    let has_tool_support = chat_template.to_lowercase().contains("tool_calls")
        || chat_template.to_lowercase().contains("available_tools")
        || name.to_lowercase().contains("gemma");

    let param_count = name
        .split('-')
        .find(|s| s.ends_with('b') || s.ends_with('B'))
        .map(str::to_uppercase);

    let caps = ModelCapabilities {
        is_instruct,
        has_chat_template: !chat_template.is_empty(),
        has_tool_support,
        has_json_support: has_tool_support || chat_template.to_lowercase().contains("json_schema"),
        is_moe: false,
        is_vision: model_family.as_deref().is_some_and(|f| f.contains("VLM")),
        context_length: max_tokens,
        quantization: None,
        parameter_count: param_count,
    };

    let mut raw = system_meta;
    raw.insert("major".to_string(), Value::Number(major.into()));
    raw.insert("minor".to_string(), Value::Number(minor.into()));
    raw.insert("patch".to_string(), Value::Number(patch_version.into()));
    raw.insert(
        "headerEndOffset".to_string(),
        Value::Number(header_end_offset.into()),
    );
    if let Some(j) = jinja {
        raw.insert("jinja_prompt_template".to_string(), Value::String(j));
    }
    if let Some(f) = model_family {
        raw.insert("model_family".to_string(), Value::String(f));
    }

    Ok(ModelInspectionReport {
        inferred_capabilities: Vec::new(),
        family: ModelFamily::LiteRtLm,
        name,
        version: format!("{major}.{minor}.{patch_version}"),
        author,
        uuid,
        capabilities: caps,
        raw_metadata: raw,
    })
}

fn extract_silero_info(filename: &str) -> (String, String) {
    let lowercase = filename.to_lowercase();

    // Find version part
    let mut version_opt = None;
    for prefix in &["-v", "_v", " v"] {
        if let Some(idx) = lowercase.find(prefix) {
            let version_part = &filename[idx + prefix.len()..];
            // Check if first char is digit
            if version_part
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit())
            {
                let end_idx = version_part
                    .find(|c: char| !c.is_numeric() && c != '.')
                    .unwrap_or(version_part.len());
                version_opt = Some(version_part[..end_idx].to_string());
                break;
            }
        }
    }

    let mut version = version_opt.unwrap_or_else(|| {
        if lowercase.contains("v5") {
            "5.0.0".to_string()
        } else if lowercase.contains("v4") {
            "4.0.0".to_string()
        } else if lowercase.contains("v3") {
            "3.0.0".to_string()
        } else if lowercase == "silero_vad.onnx" {
            "4.0.0".to_string()
        } else {
            "5.0.0".to_string()
        }
    });

    if version.ends_with('.') {
        version.pop();
    }

    let arch = if version.starts_with('6') {
        "Silero-RNN-v6".to_string()
    } else if version.starts_with('5') {
        "Silero-RNN-v5".to_string()
    } else if version.starts_with('4') {
        "Silero-RNN-v4".to_string()
    } else if version.starts_with('3') {
        "Silero-RNN-v3".to_string()
    } else {
        "Silero-RNN-v5".to_string()
    };

    (version, arch)
}

/// Read a Whisper/GGML model header and its hyperparameters.
#[expect(
    clippy::too_many_lines,
    reason = "single-pass GGML header reader; splitting it requires parser tests first"
)]
fn parse_whisper_ggml<P: AsRef<Path>>(path: P) -> std::io::Result<ModelInspectionReport> {
    let path_ref = path.as_ref();
    let filename = path_ref.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let is_silero_vad =
        filename.to_lowercase().contains("silero") || filename.to_lowercase().contains("vad");

    let mut file = File::open(path_ref)?;

    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?; // magic

    let read_int = |f: &mut File| -> std::io::Result<i32> {
        let mut b = [0u8; 4];
        f.read_exact(&mut b)?;
        Ok(i32::from_le_bytes(b))
    };

    let n_vocab = read_int(&mut file)?;
    let n_audio_ctx = read_int(&mut file)?;
    let n_audio_state = read_int(&mut file)?;
    let n_audio_head = read_int(&mut file)?;
    let n_audio_layer = read_int(&mut file)?;
    let n_text_ctx = read_int(&mut file)?;
    let n_text_state = read_int(&mut file)?;
    let n_text_head = read_int(&mut file)?;
    let n_text_layer = read_int(&mut file)?;
    let n_mels = read_int(&mut file)?;
    let ftype = read_int(&mut file)?;

    let size_label = match n_audio_layer {
        4 => "tiny",
        6 => "base",
        12 => "small",
        24 => "medium",
        32 => "large",
        _ => "custom",
    };

    let caps = ModelCapabilities {
        is_instruct: false,
        has_chat_template: false,
        has_tool_support: false,
        has_json_support: false,
        is_moe: false,
        is_vision: false,
        context_length: if is_silero_vad {
            0
        } else {
            u64::try_from(n_text_ctx).unwrap_or(0)
        },
        quantization: Some(format!("ftype:{ftype}")),
        parameter_count: Some(size_label.to_string()),
    };

    let mut raw = HashMap::new();
    raw.insert("n_vocab".to_string(), Value::Number(n_vocab.into()));
    raw.insert("n_audio_ctx".to_string(), Value::Number(n_audio_ctx.into()));
    raw.insert(
        "n_audio_state".to_string(),
        Value::Number(n_audio_state.into()),
    );
    raw.insert(
        "n_audio_head".to_string(),
        Value::Number(n_audio_head.into()),
    );
    raw.insert(
        "n_audio_layer".to_string(),
        Value::Number(n_audio_layer.into()),
    );
    raw.insert("n_text_ctx".to_string(), Value::Number(n_text_ctx.into()));
    raw.insert(
        "n_text_state".to_string(),
        Value::Number(n_text_state.into()),
    );
    raw.insert("n_text_head".to_string(), Value::Number(n_text_head.into()));
    raw.insert(
        "n_text_layer".to_string(),
        Value::Number(n_text_layer.into()),
    );
    raw.insert("n_mels".to_string(), Value::Number(n_mels.into()));
    raw.insert("ftype".to_string(), Value::Number(ftype.into()));

    let (version_str, arch_str) = if is_silero_vad {
        extract_silero_info(filename)
    } else {
        ("1.0".to_string(), String::new())
    };

    let model_name = if is_silero_vad {
        format!("Silero VAD v{version_str}")
    } else {
        format!("Whisper {size_label}")
    };

    let author = if is_silero_vad {
        "Silero".to_string()
    } else {
        "OpenAI".to_string()
    };

    if is_silero_vad {
        raw.insert(
            "general.architecture".to_string(),
            Value::String("silero".to_string()),
        );
        raw.insert(
            "general.author".to_string(),
            Value::String("Silero".to_string()),
        );
        raw.insert(
            "general.license".to_string(),
            Value::String("MIT".to_string()),
        );
        raw.insert(
            "silero.min_sample_rate".to_string(),
            Value::Number(8000.into()),
        );
        raw.insert(
            "silero.max_sample_rate".to_string(),
            Value::Number(16000.into()),
        );
        raw.insert(
            "silero.version".to_string(),
            Value::String(version_str.clone()),
        );
        raw.insert("architecture".to_string(), Value::String(arch_str));
        raw.insert("streaming".to_string(), Value::Bool(true));
        raw.insert("author".to_string(), Value::String("Silero".to_string()));
        raw.insert("license".to_string(), Value::String("MIT".to_string()));
        raw.insert("min_sample_rate".to_string(), Value::Number(8000.into()));
        raw.insert("max_sample_rate".to_string(), Value::Number(16000.into()));
        raw.insert("version".to_string(), Value::String(version_str.clone()));
    }

    Ok(ModelInspectionReport {
        inferred_capabilities: Vec::new(),
        family: if is_silero_vad {
            ModelFamily::AudioSegmentation
        } else {
            ModelFamily::WhisperCpp
        },
        name: model_name,
        version: if is_silero_vad {
            version_str
        } else {
            "v1.0".to_string()
        },
        author,
        uuid: String::new(),
        capabilities: caps,
        raw_metadata: raw,
    })
}

/// Infer ONNX capabilities from a model's name. Both the directory path and
/// the single-file path need this; they previously carried identical copies.
fn onnx_capabilities_from_name(name: &str) -> Vec<Capability> {
    let lower = name.to_lowercase();
    let mut caps = Vec::new();
    if lower.contains("decoder") || lower.contains("encoder") || lower.contains("llm") {
        caps.push(Capability::TextGeneration);
    }
    if lower.contains("whisper") || lower.contains("stt") || lower.contains("moonshine") {
        caps.push(Capability::SpeechToText);
        caps.push(Capability::AudioInput);
    }
    if lower.contains("tts") || lower.contains("speech") || lower.contains("vocoder") {
        caps.push(Capability::TextToSpeech);
        caps.push(Capability::AudioOutput);
    }
    if lower.contains("vad") || lower.contains("silero") {
        caps.push(Capability::AudioInput);
    }
    caps
}

/// Read an ONNX model's metadata. The body mixes three concerns — directory
/// layout probing, protobuf scanning, and capability inference — which is why
/// it exceeds the line limit. Splitting it safely needs parser tests that do
/// not exist yet; see `tests/model.rs` (it currently covers family detection
/// only). Keep this marker until those tests land.
#[expect(
    clippy::too_many_lines,
    reason = "single-pass ONNX reader; splitting it requires parser tests first"
)]
fn parse_onnx<P: AsRef<Path>>(path: P) -> std::io::Result<ModelInspectionReport> {
    let dir = path.as_ref();
    let path_ref = path.as_ref();
    let filename = path_ref.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let is_silero_vad =
        filename.to_lowercase().contains("silero") || filename.to_lowercase().contains("vad");

    let mut model_name = "Generic ONNX Model".to_string();
    let mut version = "1.0".to_string();
    let mut author = "Unknown ONNX Developer".to_string();
    let mut inferred_capabilities = Vec::new();
    let mut raw = HashMap::new();

    // Check directory structure for Supertonic (onnx model subset) layouts
    if dir.is_dir() {
        if let Some(name) = dir.file_name().and_then(|s| s.to_str()) {
            model_name = name.to_string();
        }

        let config_path = dir.join("config.json");
        let tts_path = dir.join("onnx/tts.json");
        let styles_path = dir.join("voice_styles");
        let audio_path = dir.join("audio_samples");

        let mut sample_rate = 0;
        let mut tts_version = "unknown".to_string();
        let mut split = "unknown".to_string();
        let mut available_styles = Vec::new();
        let mut audio_samples = Vec::new();
        let mut is_supertonic_layout = false;

        if tts_path.exists() {
            is_supertonic_layout = true;
            let content = fs::read_to_string(&tts_path)?;
            if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&content) {
                if let Some(ae) = map.get("ae").and_then(|v| v.as_object()) {
                    sample_rate = ae
                        .get("sample_rate")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(0);
                }
                if let Some(v) = map.get("tts_version").and_then(|v| v.as_str()) {
                    tts_version = v.to_string();
                }
                if let Some(s) = map.get("split").and_then(|v| v.as_str()) {
                    split = s.to_string();
                }
                if let Some(n) = map.get("model_name").and_then(|v| v.as_str()) {
                    model_name = n.to_string();
                }
            }
        }

        if config_path.exists() {
            let content = fs::read_to_string(&config_path)?;
            if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&content)
                && let Some(n) = map.get("model_name").and_then(|v| v.as_str())
            {
                model_name = n.to_string();
            }
        }

        if styles_path.is_dir()
            && let Ok(entries) = fs::read_dir(styles_path)
        {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().is_some_and(|e| e == "json")
                    && let Some(stem) = p.file_stem().and_then(|s| s.to_str())
                {
                    available_styles.push(stem.to_string());
                }
            }
        }

        if audio_path.is_dir()
            && let Ok(entries) = fs::read_dir(audio_path)
        {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().is_some_and(|e| e == "wav" || e == "mp3")
                    && let Some(name) = p.file_name().and_then(|n| n.to_str())
                {
                    audio_samples.push(name.to_string());
                }
            }
        }

        available_styles.sort();
        audio_samples.sort();

        if is_supertonic_layout {
            author = "Supertonic AI".to_string();
            version = tts_version;
            inferred_capabilities.push(Capability::TextToSpeech);
            inferred_capabilities.push(Capability::AudioOutput);

            raw.insert("split".to_string(), Value::String(split));
            raw.insert("sample_rate".to_string(), Value::Number(sample_rate.into()));
            raw.insert(
                "available_styles".to_string(),
                Value::Array(available_styles.into_iter().map(Value::String).collect()),
            );
            raw.insert(
                "audio_samples".to_string(),
                Value::Array(audio_samples.into_iter().map(Value::String).collect()),
            );
        } else {
            // General directory scan fallback
            inferred_capabilities = onnx_capabilities_from_name(&model_name);
        }
    } else {
        // Single flat file parsing
        if let Some(stem) = dir.file_stem().and_then(|s| s.to_str()) {
            model_name = stem.to_string();
        }

        inferred_capabilities = onnx_capabilities_from_name(&model_name);
    }

    let caps = ModelCapabilities {
        is_instruct: if is_silero_vad {
            false
        } else {
            !inferred_capabilities.is_empty()
        },
        has_chat_template: false,
        has_tool_support: false,
        has_json_support: false,
        is_moe: false,
        is_vision: if is_silero_vad {
            false
        } else {
            model_name.to_lowercase().contains("vision")
                || model_name.to_lowercase().contains("yolo")
                || model_name.to_lowercase().contains("resnet")
        },
        context_length: 0,
        quantization: None,
        parameter_count: if author == "Supertonic AI" {
            Some("99M".to_string())
        } else {
            None
        },
    };

    raw.insert(
        "path".to_string(),
        Value::String(dir.to_string_lossy().into_owned()),
    );

    let (version_str, arch_str) = if is_silero_vad {
        extract_silero_info(filename)
    } else {
        (version.clone(), String::new())
    };

    let final_name = if is_silero_vad {
        format!("Silero VAD v{version_str}")
    } else {
        model_name
    };

    let final_author = if is_silero_vad {
        "Silero".to_string()
    } else {
        author
    };

    let final_version = if is_silero_vad {
        version_str.clone()
    } else {
        version
    };

    if is_silero_vad {
        raw.insert(
            "general.architecture".to_string(),
            Value::String("silero".to_string()),
        );
        raw.insert(
            "general.author".to_string(),
            Value::String("Silero".to_string()),
        );
        raw.insert(
            "general.license".to_string(),
            Value::String("MIT".to_string()),
        );
        raw.insert(
            "silero.min_sample_rate".to_string(),
            Value::Number(8000.into()),
        );
        raw.insert(
            "silero.max_sample_rate".to_string(),
            Value::Number(16000.into()),
        );
        raw.insert(
            "silero.version".to_string(),
            Value::String(version_str.clone()),
        );
        raw.insert("architecture".to_string(), Value::String(arch_str));
        raw.insert("streaming".to_string(), Value::Bool(true));
        raw.insert("author".to_string(), Value::String("Silero".to_string()));
        raw.insert("license".to_string(), Value::String("MIT".to_string()));
        raw.insert("min_sample_rate".to_string(), Value::Number(8000.into()));
        raw.insert("max_sample_rate".to_string(), Value::Number(16000.into()));
        raw.insert("version".to_string(), Value::String(version_str));
    }

    Ok(ModelInspectionReport {
        inferred_capabilities,
        family: if is_silero_vad {
            ModelFamily::AudioSegmentation
        } else {
            ModelFamily::Onnx
        },
        name: final_name,
        version: final_version,
        author: final_author,
        uuid: String::new(),
        capabilities: caps,
        raw_metadata: raw,
    })
}

/// Read a `SafeTensors` file: magic, JSON header, and tensor shapes.
#[expect(
    clippy::too_many_lines,
    reason = "single-pass SafeTensors reader; splitting it requires parser tests first"
)]
fn parse_safetensors<P: AsRef<Path>>(path: P) -> std::io::Result<ModelInspectionReport> {
    let path = path.as_ref();
    let safetensors_file_path = if path.is_dir() {
        let mut target = None;
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().is_some_and(|e| e == "safetensors") {
                    target = Some(p);
                    break;
                }
            }
        }
        target.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No safetensors file found in directory",
            )
        })?
    } else {
        path.to_path_buf()
    };

    let file = File::open(&safetensors_file_path)?;
    let mut reader = BufReader::new(file);

    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes)?;
    let n = u64::from_le_bytes(len_bytes);

    if n == 0 || n > 100_000_000 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid safetensors header length",
        ));
    }

    let mut header_bytes = vec![0u8; usize::try_from(n).unwrap_or(0)];
    reader.read_exact(&mut header_bytes)?;

    let header_val: Value = serde_json::from_slice(&header_bytes).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse safetensors header JSON: {e}"),
        )
    })?;

    let metadata = header_val.get("__metadata__");

    let mut raw = HashMap::new();
    if let Some(meta_obj) = metadata.and_then(|m| m.as_object()) {
        for (k, v) in meta_obj {
            raw.insert(k.clone(), v.clone());
        }
    }

    // Try reading config.json for extra metadata
    let parent_dir = if path.is_dir() {
        Some(path)
    } else {
        path.parent()
    };
    let mut config_val: Option<Value> = None;
    if let Some(dir) = parent_dir {
        let config_json_path = dir.join("config.json");
        if config_json_path.exists()
            && let Ok(data) = fs::read_to_string(&config_json_path)
            && let Ok(val) = serde_json::from_str::<Value>(&data)
        {
            config_val = Some(val);
        }
    }

    if let Some(ref config) = config_val
        && let Some(obj) = config.as_object()
    {
        for (k, v) in obj {
            if !raw.contains_key(k) {
                raw.insert(k.clone(), v.clone());
            }
        }
    }

    let model_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Unknown Safetensors Model")
        .to_string();

    let mut name = raw
        .get("name")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| {
            raw.get("_name_or_path")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .unwrap_or(model_name);

    if name.contains('/')
        && let Some(last) = name.rsplit('/').next()
    {
        name = last.to_string();
    }

    let author = raw
        .get("author")
        .and_then(|v| v.as_str())
        .map_or_else(|| "Unknown".to_string(), String::from);

    let version = raw
        .get("version")
        .and_then(|v| v.as_str())
        .map_or_else(|| "v1.0".to_string(), String::from);

    let mut inferred_capabilities = Vec::new();
    let mut has_lm_head = false;
    let mut has_vision_keys = false;
    let mut has_audio_keys = false;

    if let Some(obj) = header_val.as_object() {
        for key in obj.keys() {
            if key == "__metadata__" {
                continue;
            }
            let lower_key = key.to_lowercase();
            if lower_key.contains("lm_head")
                || lower_key.contains("embed_tokens")
                || lower_key.contains("wte")
            {
                has_lm_head = true;
            }
            if lower_key.contains("vision") || lower_key.contains("patch_embed") {
                has_vision_keys = true;
            }
            if lower_key.contains("audio")
                || (lower_key.contains("encoder") && lower_key.contains("conv"))
            {
                has_audio_keys = true;
            }
        }
    }

    let mut model_type = String::new();
    let mut archs = Vec::new();
    if let Some(ref config) = config_val {
        if let Some(t) = config.get("model_type").and_then(|v| v.as_str()) {
            model_type = t.to_lowercase();
        }
        if let Some(arr) = config.get("architectures").and_then(|v| v.as_array()) {
            for v in arr {
                if let Some(s) = v.as_str() {
                    archs.push(s.to_lowercase());
                }
            }
        }
    }

    let lower_name = name.to_lowercase();
    let is_text_gen = has_lm_head
        || lower_name.contains("gpt")
        || lower_name.contains("llama")
        || lower_name.contains("gemma")
        || lower_name.contains("mistral")
        || model_type.contains("marian")
        || model_type.contains("llama")
        || model_type.contains("gpt")
        || archs.iter().any(|a| {
            a.contains("marian")
                || a.contains("causallm")
                || a.contains("lmval")
                || a.contains("model")
        });

    if is_text_gen {
        inferred_capabilities.push(Capability::TextGeneration);
    }
    if has_vision_keys
        || lower_name.contains("vision")
        || lower_name.contains("clip")
        || lower_name.contains("vit")
        || model_type.contains("vision")
        || model_type.contains("clip")
    {
        inferred_capabilities.push(Capability::Vision);
        if inferred_capabilities.contains(&Capability::TextGeneration) {
            inferred_capabilities.push(Capability::Multimodal);
        }
    }
    if has_audio_keys || lower_name.contains("whisper") || model_type.contains("whisper") {
        inferred_capabilities.push(Capability::SpeechToText);
        inferred_capabilities.push(Capability::AudioInput);
    }

    let quant_val = raw
        .get("quantization")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| {
            raw.get("torch_dtype")
                .and_then(|v| v.as_str())
                .map(String::from)
        });

    let family = ModelFamily::SafeTensors;

    let caps = ModelCapabilities {
        is_instruct: lower_name.contains("instruct")
            || lower_name.contains("it")
            || lower_name.contains("chat")
            || lower_name.contains("mt"),
        has_chat_template: raw.contains_key("chat_template")
            || raw.contains_key("tokenizer.chat_template"),
        has_tool_support: false,
        has_json_support: false,
        is_moe: lower_name.contains("moe") || lower_name.contains("mixtral"),
        is_vision: inferred_capabilities.contains(&Capability::Vision),
        context_length: raw
            .get("context_length")
            .and_then(serde_json::Value::as_u64)
            .or_else(|| {
                raw.get("max_position_embeddings")
                    .and_then(serde_json::Value::as_u64)
            })
            .or_else(|| raw.get("max_length").and_then(serde_json::Value::as_u64))
            .unwrap_or(0),
        quantization: quant_val,
        parameter_count: raw
            .get("parameter_count")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| {
                let re = regex::Regex::new(r"(?i)(\d+[BbMm])").ok()?;
                let cap = re.captures(&name)?;
                Some(cap[1].to_uppercase())
            }),
    };

    Ok(ModelInspectionReport {
        inferred_capabilities,
        family,
        name,
        version,
        author,
        uuid: raw
            .get("uuid")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        capabilities: caps,
        raw_metadata: raw,
    })
}

/// Inspect a model file and report its family, size, and inferred capabilities.
///
/// # Errors
///
/// * [`ErrorKind::NotFound`](std::io::ErrorKind::NotFound) when the path does
///   not exist.
/// * [`ErrorKind::InvalidData`](std::io::ErrorKind::InvalidData) when the
///   magic bytes or headers cannot be parsed (truncated or corrupt file).
/// * Other IO errors surface unchanged from opening or reading the file.
#[tracing::instrument(skip_all, fields(path = %path.as_ref().display()))]
pub fn inspect_model_file<P: AsRef<Path>>(path: P) -> std::io::Result<ModelInspectionReport> {
    let path_ref = path.as_ref();
    info!("Inspecting model file: {}", path_ref.display());

    // Missing files must surface as "not found" (matches the FFI error
    // classifier's ModelNotFound branch) instead of falling through to
    // magic-byte detection and being reported as "Unknown model family".
    if !path_ref.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Model not found: {}", path_ref.display()),
        ));
    }

    let family = detect_model_family(path_ref);
    debug!("Detected model family: {:?}", family);

    let mut report = match family {
        ModelFamily::LlamaCpp => {
            debug!("Parsing GGUF model");
            parse_llama_gguf(path_ref)?
        }
        ModelFamily::LiteRtLm => {
            debug!("Parsing LiteRT-LM model");
            parse_litertlm(path_ref)?
        }
        ModelFamily::WhisperCpp => {
            debug!("Parsing Whisper/GGML model");
            parse_whisper_ggml(path_ref)?
        }
        ModelFamily::Onnx => {
            debug!("Parsing ONNX model");
            parse_onnx(path_ref)?
        }
        ModelFamily::SafeTensors => {
            debug!("Parsing SafeTensors model");
            parse_safetensors(path_ref)?
        }
        ModelFamily::AudioSegmentation => {
            debug!("Parsing audio segmentation model");
            let extension = path_ref.extension().and_then(|e| e.to_str()).unwrap_or("");
            if extension == "onnx" {
                parse_onnx(path_ref)?
            } else {
                parse_whisper_ggml(path_ref)?
            }
        }
        ModelFamily::Unknown => {
            error!("Unknown model family for: {}", path_ref.display());
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Unknown model family",
            ));
        }
    };

    debug!("Inferring capabilities for model: {}", report.name);
    report.infer_capabilities();

    info!(
        "Model inspection complete: name={}, family={:?}, capabilities={:?}",
        report.name, report.family, report.inferred_capabilities
    );

    Ok(report)
}

// ==========================================
// 5. Model Sizing
// ==========================================
/// Total size of a model on disk: file size for a single file, or the
/// recursive sum for a directory layout (HF snapshots, ONNX folders).
pub fn get_recursive_size<P: AsRef<Path>>(path: P) -> u64 {
    let path = path.as_ref();
    if path.is_file() {
        return fs::metadata(path).map_or(0, |m| m.len());
    }
    let mut size = 0u64;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                size += fs::metadata(&p).map_or(0, |m| m.len());
            } else if p.is_dir() {
                size += get_recursive_size(&p);
            }
        }
    }
    size
}

// ==========================================
// 6. Unit Tests
// ==========================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_family_detection_stubs() {
        let temp = std::env::temp_dir();

        // Write mock GGUF file
        let gguf_path = temp.join("mock.gguf");
        fs::write(&gguf_path, b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00").unwrap();
        assert_eq!(detect_model_family(&gguf_path), ModelFamily::LlamaCpp);
        fs::remove_file(gguf_path).unwrap();

        // Write mock LiteRT file
        let litert_path = temp.join("mock.litertlm");
        fs::write(&litert_path, b"LITERTLM\x01\x00\x00\x00").unwrap();
        assert_eq!(detect_model_family(&litert_path), ModelFamily::LiteRtLm);
        fs::remove_file(litert_path).unwrap();

        // Write mock Safetensors file
        let safetensors_path = temp.join("mock.safetensors");
        let json_header = r#"{"__metadata__":{"name":"MockSafetensors","author":"Tester","version":"v2.0"},"lm_head.weight":{"dtype":"F16","shape":[10,10]}}"#;
        let json_bytes = json_header.as_bytes();
        let header_len = u64::try_from(json_bytes.len()).unwrap_or(0);
        let mut mock_data = Vec::new();
        mock_data.extend_from_slice(&header_len.to_le_bytes());
        mock_data.extend_from_slice(json_bytes);
        fs::write(&safetensors_path, &mock_data).unwrap();

        assert_eq!(
            detect_model_family(&safetensors_path),
            ModelFamily::SafeTensors
        );

        let report = inspect_model_file(&safetensors_path).unwrap();
        assert_eq!(report.family, ModelFamily::SafeTensors);
        assert_eq!(report.name, "MockSafetensors");
        assert_eq!(report.author, "Tester");
        assert_eq!(report.version, "v2.0");
        assert!(
            report
                .inferred_capabilities
                .contains(&Capability::TextGeneration)
        );

        fs::remove_file(safetensors_path).unwrap();
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MediaInspectionReport {
    pub file_name: String,
    pub format: String,
    pub size_bytes: u64,
    pub width: u32,
    pub height: u32,
    pub duration_ms: u64,
    pub frame_count: usize,
    pub is_supported: bool,
}

fn get_mp4_duration_ms(path: &std::path::Path) -> Option<(u64, usize)> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let file_size = file.metadata().ok()?.len();

    let search_limit = std::cmp::min(file_size, 10 * 1024 * 1024);
    let mut buffer = vec![0u8; usize::try_from(search_limit).unwrap_or(0)];
    file.read_exact(&mut buffer).ok()?;

    let mvhd_tag = b"mvhd";
    if let Some(pos) = buffer.windows(4).position(|window| window == mvhd_tag) {
        let data_start = pos + 4;
        if data_start + 32usize <= buffer.len() {
            let version = buffer[data_start];
            if version == 0 {
                let timescale = u64::from(u32::from_be_bytes([
                    buffer[data_start + 12],
                    buffer[data_start + 13],
                    buffer[data_start + 14],
                    buffer[data_start + 15],
                ]));
                let duration = u64::from(u32::from_be_bytes([
                    buffer[data_start + 16],
                    buffer[data_start + 17],
                    buffer[data_start + 18],
                    buffer[data_start + 19],
                ]));
                if let Some(duration_ms) = duration
                    .checked_mul(1000)
                    .and_then(|total| total.checked_div(timescale))
                {
                    let frame_count = usize::try_from((duration_ms * 30) / 1000)
                        .unwrap_or(1)
                        .max(1);
                    return Some((duration_ms, frame_count));
                }
            } else if version == 1 {
                let timescale = u64::from(u32::from_be_bytes([
                    buffer[data_start + 20],
                    buffer[data_start + 21],
                    buffer[data_start + 22],
                    buffer[data_start + 23],
                ]));
                let duration = u64::from_be_bytes([
                    buffer[data_start + 24],
                    buffer[data_start + 25],
                    buffer[data_start + 26],
                    buffer[data_start + 27],
                    buffer[data_start + 28],
                    buffer[data_start + 29],
                    buffer[data_start + 30],
                    buffer[data_start + 31],
                ]);
                if let Some(duration_ms) = duration
                    .checked_mul(1000)
                    .and_then(|total| total.checked_div(timescale))
                {
                    let frame_count = usize::try_from((duration_ms * 30) / 1000)
                        .unwrap_or(1)
                        .max(1);
                    return Some((duration_ms, frame_count));
                }
            }
        }
    }
    None
}

/// Inspect a media file (image/audio/video) and report its format and size.
///
/// # Errors
///
/// * A plain message when the path is not an existing regular file.
/// * The underlying IO error, as a string, when metadata cannot be read.
/// * `Unsupported media format: .{ext}` for any extension outside the
///   recognized image/audio/video set.
pub fn inspect_media_file<P: AsRef<Path>>(path: P) -> Result<MediaInspectionReport, String> {
    let p = path.as_ref();
    if !p.exists() || !p.is_file() {
        return Err("File does not exist or is not a file".to_string());
    }

    let file_name = p
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    let meta = fs::metadata(p).map_err(|e| e.to_string())?;
    let size_bytes = meta.len();

    let ext = p
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "mp4" | "mov" | "m4v" | "avi" | "mkv" | "webm" => {
            let (duration_ms, frame_count) = get_mp4_duration_ms(p).unwrap_or((10000, 300));
            Ok(MediaInspectionReport {
                file_name,
                format: format!("video/{ext}"),
                size_bytes,
                width: 1920,
                height: 1080,
                duration_ms,
                frame_count,
                is_supported: true,
            })
        }
        "jpg" | "jpeg" | "png" | "webp" | "gif" | "heic" => Ok(MediaInspectionReport {
            file_name,
            format: format!("image/{ext}"),
            size_bytes,
            width: 1080,
            height: 1080,
            duration_ms: 0,
            frame_count: 1,
            is_supported: true,
        }),
        _ => Err(format!("Unsupported media format: .{ext}")),
    }
}
