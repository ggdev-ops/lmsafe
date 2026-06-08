use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, BufReader};
use std::path::{Path, PathBuf};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ModelFamily {
    #[serde(rename = "llama-cpp")]
    LlamaCpp,
    #[serde(rename = "whisper-cpp")]
    WhisperCpp,
    #[serde(rename = "litert-lm")]
    LiteRtLm,
    #[serde(rename = "supertonic")]
    Supertonic,
    #[serde(rename = "unknown")]
    Unknown,
}

impl ModelFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelFamily::LlamaCpp => "llama-cpp",
            ModelFamily::WhisperCpp => "whisper-cpp",
            ModelFamily::LiteRtLm => "litert-lm",
            ModelFamily::Supertonic => "supertonic",
            ModelFamily::Unknown => "unknown",
        }
    }
}

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
    pub raw_metadata: HashMap<String, Value>,
}

// ==========================================
// 1. Model Family Detector
// ==========================================
pub fn detect_model_family<P: AsRef<Path>>(path: P) -> ModelFamily {
    let path = path.as_ref();
    
    // Check Supertonic directory structure
    if path.is_dir() {
        let config_path = path.join("config.json");
        let tts_path = path.join("onnx/tts.json");
        if config_path.exists() && tts_path.exists() {
            return ModelFamily::Supertonic;
        }
        return ModelFamily::Unknown;
    }

    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return ModelFamily::Unknown,
    };

    let mut header = [0u8; 8];
    let bytes_read = match file.read(&mut header) {
        Ok(n) => n,
        Err(_) => return ModelFamily::Unknown,
    };

    if bytes_read < 4 {
        return ModelFamily::Unknown;
    }

    let magic4 = &header[0..4];
    let magic8 = &header[0..bytes_read.min(8)];

    // LITERTLM magic
    if magic8 == b"LITERTLM" {
        return ModelFamily::LiteRtLm;
    }

    // GGUF magic
    if magic4 == b"GGUF" {
        return ModelFamily::LlamaCpp;
    }

    // GGML Whisper magics (including little-endian order: lmgg, fmgg, jmgg)
    if magic4 == b"ggml" || magic4 == b"ggmf" || magic4 == b"ggjt" ||
       magic4 == b"lmgg" || magic4 == b"fmgg" || magic4 == b"jmgg" {
        return ModelFamily::WhisperCpp;
    }

    // TFLite magic "TFL3" at offset 4
    if bytes_read >= 8 {
        let tfl_magic = &header[4..8];
        if tfl_magic == b"TFL3" {
            return ModelFamily::LiteRtLm;
        }
    }

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
        let mut buf = vec![0u8; len as usize];
        self.reader.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn skip_value(&mut self, val_type: u32) -> std::io::Result<()> {
        match val_type {
            0 | 1 | 7 => { self.reader.seek(SeekFrom::Current(1))?; } // u8, i8, bool
            2 | 3 => { self.reader.seek(SeekFrom::Current(2))?; }     // u16, i16
            4 | 5 | 6 => { self.reader.seek(SeekFrom::Current(4))?; } // u32, i32, f32
            10 | 11 | 12 => { self.reader.seek(SeekFrom::Current(8))?; } // u64, i64, f64
            8 => { // String
                let len = self.read_u64()?;
                self.reader.seek(SeekFrom::Current(len as i64))?;
            }
            9 => { // Array
                let elem_type = self.read_u32()?;
                let count = self.read_u64()?;
                for _ in 0..count {
                    self.skip_value(elem_type)?;
                }
            }
            _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unknown value type")),
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
                Value::Number((buf[0] as i8).into())
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
                let val = self.read_u32()? as i32;
                Value::Number(val.into())
            }
            6 => {
                let mut buf = [0u8; 4];
                self.reader.read_exact(&mut buf)?;
                let f = f32::from_le_bytes(buf);
                serde_json::Number::from_f64(f as f64)
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
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
                    Value::String(format!("Array(type={}, count={}) /* skipped */", elem_type, count))
                } else {
                    let mut arr = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        arr.push(self.parse_value(elem_type)?);
                    }
                    Value::Array(arr)
                }
            }
            10 => Value::Number(self.read_u64()?.into()),
            11 => {
                let val = self.read_u64()? as i64;
                Value::Number(val.into())
            }
            12 => {
                let mut buf = [0u8; 8];
                self.reader.read_exact(&mut buf)?;
                let f = f64::from_le_bytes(buf);
                serde_json::Number::from_f64(f)
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
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
        if offset + 4 > self.data.len() { return 0; }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.data[offset..offset+4]);
        i32::from_le_bytes(buf)
    }

    fn read_short(&self, offset: usize) -> u16 {
        if offset + 2 > self.data.len() { return 0; }
        let mut buf = [0u8; 2];
        buf.copy_from_slice(&self.data[offset..offset+2]);
        u16::from_le_bytes(buf)
    }

    fn read_long(&self, offset: usize) -> i64 {
        if offset + 8 > self.data.len() { return 0; }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&self.data[offset..offset+8]);
        i64::from_le_bytes(buf)
    }

    fn get_root(&self) -> usize {
        self.read_int(0) as usize
    }

    fn get_field_offset(&self, table_offset: usize, field_index: usize) -> usize {
        let vtable_offset = (table_offset as i32 - self.read_int(table_offset)) as usize;
        if vtable_offset + 2 > self.data.len() { return 0; }
        let vtable_size = self.read_short(vtable_offset) as usize;
        let field_offset_in_vtable = 4 + field_index * 2;
        if field_offset_in_vtable >= vtable_size { return 0; }
        let offset_in_table = self.read_short(vtable_offset + field_offset_in_vtable) as usize;
        if offset_in_table == 0 { return 0; }
        table_offset + offset_in_table
    }

    fn get_indirect_offset(&self, table_offset: usize, field_index: usize) -> usize {
        let offset = self.get_field_offset(table_offset, field_index);
        if offset == 0 { return 0; }
        offset + self.read_int(offset) as usize
    }

    fn get_byte_field(&self, table_offset: usize, field_index: usize, default: u8) -> u8 {
        let offset = self.get_field_offset(table_offset, field_index);
        if offset == 0 { default } else { self.data[offset] }
    }

    fn get_int_field(&self, table_offset: usize, field_index: usize, default: i32) -> i32 {
        let offset = self.get_field_offset(table_offset, field_index);
        if offset == 0 { default } else { self.read_int(offset) }
    }

    fn get_long_field(&self, table_offset: usize, field_index: usize, default: i64) -> i64 {
        let offset = self.get_field_offset(table_offset, field_index);
        if offset == 0 { default } else { self.read_long(offset) }
    }

    fn get_string_field(&self, table_offset: usize, field_index: usize) -> Option<String> {
        let mut offset = self.get_field_offset(table_offset, field_index);
        if offset == 0 { return None; }
        offset += self.read_int(offset) as usize;
        let length = self.read_int(offset) as usize;
        if offset + 4 + length > self.data.len() { return None; }
        let s = String::from_utf8_lossy(&self.data[offset+4..offset+4+length]).into_owned();
        Some(s)
    }

    fn get_vector_size(&self, vector_offset: usize) -> usize {
        self.read_int(vector_offset) as usize
    }

    fn get_vector_element_offset(&self, vector_offset: usize, index: usize) -> usize {
        let element_pos = vector_offset + 4 + index * 4;
        element_pos + self.read_int(element_pos) as usize
    }

    fn get_union_value(&self, table_offset: usize, type_index: usize, value_index: usize) -> Option<Value> {
        let union_type = self.get_int_field(table_offset, type_index, 0) as u8;
        let offset = self.get_indirect_offset(table_offset, value_index);
        if offset == 0 { return None; }
        
        let val = match union_type {
            1 => Value::Number((self.data[offset] as u64).into()), // UInt8
            2 => Value::Number((self.data[offset] as i64).into()), // Int8
            5 => Value::Number((self.read_int(offset) as u32 as u64).into()), // UInt32
            6 => Value::Number((self.read_int(offset) as i64).into()), // Int32
            9 => { // StringValue (a table with string field at index 0)
                self.get_string_field(offset, 0).map(Value::String)?
            }
            10 => Value::Number((self.read_long(offset) as u64).into()), // UInt64
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
        Self { data, i: 0, last_wire_type: 0 }
    }

    fn has_next(&self) -> bool {
        self.i < self.data.len()
    }

    fn read_varint(&mut self) -> u64 {
        let mut value = 0u64;
        let mut shift = 0;
        while self.i < self.data.len() {
            let b = self.data[self.i] as u64;
            self.i += 1;
            value |= (b & 0x7F) << shift;
            if (b & 0x80) == 0 { break; }
            shift += 7;
        }
        value
    }

    fn read_fixed32(&mut self) -> u32 {
        if self.i + 4 > self.data.len() { return 0; }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.data[self.i..self.i+4]);
        self.i += 4;
        u32::from_le_bytes(buf)
    }

    fn read_string(&mut self) -> String {
        let length = self.read_varint() as usize;
        if self.i + length > self.data.len() { return String::new(); }
        let s = String::from_utf8_lossy(&self.data[self.i..self.i+length]).into_owned();
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
        let tag = self.read_varint() as u32;
        self.last_wire_type = (tag & 0x7) as u8;
        let field_num = tag >> 3;
        (field_num, self.last_wire_type)
    }

    fn skip_field(&mut self) {
        match self.last_wire_type {
            0 => { self.read_varint(); }
            1 => { self.i += 8; }
            2 => {
                let length = self.read_varint() as usize;
                self.i += length;
            }
            5 => { self.i += 4; }
            _ => { self.i = self.data.len(); } // Stop on unknown
        }
        if self.i > self.data.len() { self.i = self.data.len(); }
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
    let get_u64 = |k: &str| raw.get(k).and_then(|v| v.as_u64());
    
    let name = get_str("general.name").unwrap_or_else(|| "Unknown Llama Model".to_string());
    let size_label = get_str("general.size_label");
    
    let arch = get_str("general.architecture").unwrap_or_else(|| "llama".to_string());
    let file_type = get_u64("general.file_type");
    let quant_label = file_type.and_then(|code| {
        // Map GGUF general.file_type code to labels
        match code {
            0 => Some("F32"), 1 => Some("F16"), 2 => Some("Q4_0"), 3 => Some("Q4_1"),
            6 => Some("Q5_0"), 7 => Some("Q5_1"), 8 => Some("Q8_0"), 10 => Some("Q2_K"),
            11 => Some("Q3_K_S"), 12 => Some("Q3_K_M"), 13 => Some("Q3_K_L"),
            14 => Some("Q4_K_S"), 15 => Some("Q4_K_M"), 16 => Some("Q5_K_S"),
            17 => Some("Q5_K_M"), 18 => Some("Q6_K"), _ => None,
        }.map(String::from)
    });

    let context_key = format!("{}.context_length", arch);
    let context_length = get_u64(&context_key).unwrap_or(0);

    let chat_template = get_str("tokenizer.chat_template").unwrap_or_default();
    
    let is_llama3 = arch == "llama" && (chat_template.contains("header_id") || chat_template.contains("eot_id"));
    let is_gemma = arch.starts_with("gemma");

    let is_instruct = get_str("general.finetune").map(|f| f.to_lowercase().contains("instruct") || f.to_lowercase().contains("it")).unwrap_or(false) || !chat_template.is_empty();
    let has_tool_support = chat_template.to_lowercase().contains("tool_calls") ||
                          chat_template.to_lowercase().contains("available_tools") ||
                          name.to_lowercase().contains("llama-3.1") ||
                          name.to_lowercase().contains("llama-3.2");

    let block_key = format!("{}.block_count", arch);
    let is_moe = get_u64(&format!("{}.expert_count", arch)).unwrap_or(0) > 0;
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
        family: ModelFamily::LlamaCpp,
        name: get_str("general.name").unwrap_or(name),
        version: get_str("general.version").unwrap_or_else(|| "v1.0".to_string()),
        author: get_str("general.author").unwrap_or_else(|| "Unknown".to_string()),
        uuid: get_str("general.uuid").unwrap_or_default(),
        capabilities: caps,
        raw_metadata: raw,
    })
}

fn parse_litertlm<P: AsRef<Path>>(path: P) -> std::io::Result<ModelInspectionReport> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    
    // Read LiteRT Magic header (32 bytes)
    let mut header = [0u8; 32];
    file.read_exact(&mut header)?;
    
    if &header[0..8] != b"LITERTLM" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid LiteRT-LM magic"));
    }

    let major = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
    let minor = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
    let patch = u32::from_le_bytes([header[16], header[17], header[18], header[19]]);
    let header_end_offset = u64::from_le_bytes([
        header[24], header[25], header[26], header[27],
        header[28], header[29], header[30], header[31],
    ]) as usize;

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
                if let Some(key) = fb.get_string_field(kvp_table, 0) {
                    if let Some(val) = fb.get_union_value(kvp_table, 1, 2) {
                        system_meta.insert(key, val);
                    }
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
                if data_type == 5 { // LlmMetadataProto
                    let begin = fb.get_long_field(section_table, 1, 0) as u64;
                    let end = fb.get_long_field(section_table, 2, 0) as u64;
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
        let len = (end - begin) as usize;
        let mut proto_data = vec![0u8; len];
        file.read_exact(&mut proto_data)?;

        let mut scanner = ProtobufScanner::new(&proto_data);
        while scanner.has_next() {
            let (field, wire_type) = scanner.next_tag();
            match field {
                5 => {
                    max_tokens = if wire_type == 0 { scanner.read_varint() } else { scanner.read_fixed32() as u64 };
                }
                6 => {
                    let length = scanner.read_varint() as usize;
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
                        }.map(String::from);
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

    let name = system_meta.get("name").and_then(|v| v.as_str()).unwrap_or("Gemma LiteRT").to_string();
    let author = system_meta.get("author").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string();
    let uuid = system_meta.get("uuid").and_then(|v| v.as_str()).unwrap_or("").to_string();

    let chat_template = jinja.clone().unwrap_or_default();
    
    let is_instruct = system_meta.get("finetune").map(|v| v.as_str().map(|f| f.contains("instruct") || f.contains("it")).unwrap_or(false)).unwrap_or(false) || !chat_template.is_empty();
    let has_tool_support = chat_template.to_lowercase().contains("tool_calls") ||
                          chat_template.to_lowercase().contains("available_tools") ||
                          name.to_lowercase().contains("gemma");

    let param_count = name.split('-').find(|s| s.ends_with('b') || s.ends_with('B')).map(|s| s.to_uppercase());

    let caps = ModelCapabilities {
        is_instruct,
        has_chat_template: !chat_template.is_empty(),
        has_tool_support,
        has_json_support: has_tool_support || chat_template.to_lowercase().contains("json_schema"),
        is_moe: false,
        is_vision: model_family.as_deref().map(|f| f.contains("VLM")).unwrap_or(false),
        context_length: max_tokens,
        quantization: None,
        parameter_count: param_count,
    };

    let mut raw = system_meta;
    raw.insert("major".to_string(), Value::Number(major.into()));
    raw.insert("minor".to_string(), Value::Number(minor.into()));
    raw.insert("patch".to_string(), Value::Number(patch.into()));
    raw.insert("headerEndOffset".to_string(), Value::Number(header_end_offset.into()));
    if let Some(j) = jinja {
        raw.insert("jinja_prompt_template".to_string(), Value::String(j));
    }
    if let Some(f) = model_family {
        raw.insert("model_family".to_string(), Value::String(f));
    }

    Ok(ModelInspectionReport {
        family: ModelFamily::LiteRtLm,
        name,
        version: format!("{}.{}.{}", major, minor, patch),
        author,
        uuid,
        capabilities: caps,
        raw_metadata: raw,
    })
}

fn parse_whisper_ggml<P: AsRef<Path>>(path: P) -> std::io::Result<ModelInspectionReport> {
    let mut file = File::open(path)?;
    
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
        context_length: n_text_ctx as u64,
        quantization: Some(format!("ftype:{}", ftype)),
        parameter_count: Some(size_label.to_string()),
    };

    let mut raw = HashMap::new();
    raw.insert("n_vocab".to_string(), Value::Number(n_vocab.into()));
    raw.insert("n_audio_layer".to_string(), Value::Number(n_audio_layer.into()));
    raw.insert("n_text_layer".to_string(), Value::Number(n_text_layer.into()));
    raw.insert("n_mels".to_string(), Value::Number(n_mels.into()));

    Ok(ModelInspectionReport {
        family: ModelFamily::WhisperCpp,
        name: format!("Whisper {}", size_label),
        version: "v1.0".to_string(),
        author: "OpenAI".to_string(),
        uuid: "".to_string(),
        capabilities: caps,
        raw_metadata: raw,
    })
}

fn parse_supertonic<P: AsRef<Path>>(path: P) -> std::io::Result<ModelInspectionReport> {
    let dir = path.as_ref();
    let config_path = dir.join("config.json");
    let tts_path = dir.join("onnx/tts.json");
    let styles_path = dir.join("voice_styles");

    let mut model_name = "Supertonic".to_string();
    let mut sample_rate = 0;
    let mut tts_version = "unknown".to_string();
    let mut split = "unknown".to_string();
    let mut available_styles = Vec::new();

    if tts_path.exists() {
        let content = fs::read_to_string(&tts_path)?;
        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&content) {
            if let Some(ae) = map.get("ae").and_then(|v| v.as_object()) {
                sample_rate = ae.get("sample_rate").and_then(|v| v.as_i64()).unwrap_or(0);
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
        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&content) {
            if let Some(n) = map.get("model_name").and_then(|v| v.as_str()) {
                model_name = n.to_string();
            }
        }
    }

    if styles_path.is_dir() {
        if let Ok(entries) = fs::read_dir(styles_path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().map(|e| e == "json").unwrap_or(false) {
                    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                        available_styles.push(stem.to_string());
                    }
                }
            }
        }
    }

    let caps = ModelCapabilities {
        is_instruct: true,
        has_chat_template: false,
        has_tool_support: false,
        has_json_support: false,
        is_moe: false,
        is_vision: false,
        context_length: 0,
        quantization: None,
        parameter_count: Some("99M".to_string()),
    };

    let mut raw = HashMap::new();
    raw.insert("split".to_string(), Value::String(split));
    raw.insert("sample_rate".to_string(), Value::Number(sample_rate.into()));
    raw.insert("available_styles".to_string(), Value::Array(available_styles.into_iter().map(Value::String).collect()));

    Ok(ModelInspectionReport {
        family: ModelFamily::Supertonic,
        name: model_name,
        version: tts_version,
        author: "Supertonic AI".to_string(),
        uuid: "".to_string(),
        capabilities: caps,
        raw_metadata: raw,
    })
}

pub fn inspect_model_file<P: AsRef<Path>>(path: P) -> std::io::Result<ModelInspectionReport> {
    let family = detect_model_family(&path);
    match family {
        ModelFamily::LlamaCpp => parse_llama_gguf(path),
        ModelFamily::LiteRtLm => parse_litertlm(path),
        ModelFamily::WhisperCpp => parse_whisper_ggml(path),
        ModelFamily::Supertonic => parse_supertonic(path),
        ModelFamily::Unknown => Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unknown model family")),
    }
}

// ==========================================
// 5. Memory Security Guard
// ==========================================
pub fn model_is_safe_to_load<P: AsRef<Path>>(path: P, safety_margin: f64) -> bool {
    let path = path.as_ref();
    
    // Get total size on disk
    let file_size = if path.is_dir() {
        // Enumerate directory size
        let mut size = 0u64;
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        size += meta.len();
                    }
                }
            }
        }
        size
    } else {
        fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    };

    if file_size == 0 {
        return false;
    }

    // Call sysinfo to get memory
    let mut sys = sysinfo::System::new_all();
    sys.refresh_all();

    let total = sys.total_memory();
    let used = sys.used_memory();
    
    if total == 0 {
        return true; // restricted/fallback sandbox context, assume safe
    }

    let available = total.saturating_sub(used);
    let required = (file_size as f64 * safety_margin) as u64;

    available > required
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
    }

    #[test]
    fn test_real_models_inspection() {
        let models_dir = Path::new("/home/gg/models");
        if !models_dir.exists() {
            return;
        }

        // Test Llama-3.2.3.gguf
        let llama_path = models_dir.join("Llama-3.2.3.gguf");
        if llama_path.exists() {
            let report = inspect_model_file(&llama_path).unwrap();
            assert_eq!(report.family, ModelFamily::LlamaCpp);
            assert!(report.capabilities.context_length > 0);
            println!("Llama inspect: {:?}", report.name);
        }

        // Test gemma-4-E2B-it.litertlm
        let gemma_path = models_dir.join("gemma-4-E2B-it.litertlm");
        if gemma_path.exists() {
            let report = inspect_model_file(&gemma_path).unwrap();
            assert_eq!(report.family, ModelFamily::LiteRtLm);
            assert_eq!(report.capabilities.is_vision, false);
            assert!(report.capabilities.has_chat_template);
            println!("Gemma inspect: {:?}", report.name);
        }

        // Test tiny.bin
        let whisper_path = models_dir.join("tiny.bin");
        if whisper_path.exists() {
            let report = inspect_model_file(&whisper_path).unwrap();
            assert_eq!(report.family, ModelFamily::WhisperCpp);
            assert_eq!(report.name, "Whisper tiny");
            println!("Whisper inspect: {:?}", report.name);
        }

        // Test supertonic-3
        let supertonic_path = models_dir.join("supertonic-3");
        if supertonic_path.exists() {
            let report = inspect_model_file(&supertonic_path).unwrap();
            assert_eq!(report.family, ModelFamily::Supertonic);
            println!("Supertonic inspect: {:?}", report.name);
        }
    }
}
