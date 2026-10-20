//! Minimal ONNX container reader: the graph *signature*.
//!
//! Extracts graph inputs, outputs and initializer shapes from an ONNX
//! protobuf file by seeking past weight data. No model is ever loaded:
//! a 460 MB decoder costs the same reads as a 100 KB VAD.
//!
//! Field numbers are verified against the descriptor of the installed onnx
//! package (onnx 1.23): `ModelProto.graph = 7`; `GraphProto` carries
//! `node = 1`, `initializer = 5`, `input = 11`, `output = 12`,
//! `value_info = 13`; `TensorProto` carries `dims = 1`, `data_type = 2`,
//! `name = 8`, `raw_data = 9`.
//! Nodes and initializers hold the weights — they are skipped by seeking.
//! `value_info` (intermediate shapes, megabytes on optimized models) is
//! skipped too: the scan stops at the first graph field after 12.

use serde::Serialize;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Largest sub-message the scanner will buffer. Graph input/output lists are
/// kilobytes; a bigger claimed length is a corrupt or hostile field.
const MAX_SUBMESSAGE: u64 = 8 * 1024 * 1024;

/// ONNX tensor element types (`TypeProto.Tensor.elem_type` numbers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OnnxDataType {
    Undefined,
    Float,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Int32,
    Int64,
    String,
    Bool,
    Float16,
    Double,
    Uint32,
    Uint64,
    Complex64,
    Complex128,
    Bfloat16,
}

impl OnnxDataType {
    #[must_use]
    pub fn from_elem_type(n: u64) -> Self {
        match n {
            // 0 (UNDEFINED) falls through to the wildcard below.
            1 => Self::Float,
            2 => Self::Uint8,
            3 => Self::Int8,
            4 => Self::Uint16,
            5 => Self::Int16,
            6 => Self::Int32,
            7 => Self::Int64,
            8 => Self::String,
            9 => Self::Bool,
            10 => Self::Float16,
            11 => Self::Double,
            12 => Self::Uint32,
            13 => Self::Uint64,
            14 => Self::Complex64,
            15 => Self::Complex128,
            16 => Self::Bfloat16,
            _ => Self::Undefined,
        }
    }
}

/// One dimension of a tensor: a concrete value, a symbolic parameter
/// (`batch_size`), or absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnnxDim {
    Value(i64),
    Param(String),
    Unknown,
}

/// A graph-level tensor (input, output) or an initializer header.
#[derive(Debug, Clone, PartialEq)]
pub struct OnnxTensor {
    pub name: String,
    pub dtype: OnnxDataType,
    pub dims: Vec<OnnxDim>,
}

impl OnnxTensor {
    /// Element count from known dimensions. 0 when any dimension is symbolic
    /// or missing — unknown is reported as unknown, never guessed.
    #[must_use]
    pub fn element_count(&self) -> u64 {
        let mut total: u64 = 1;
        for dim in &self.dims {
            let OnnxDim::Value(v) = dim else { return 0 };
            let Ok(v) = u64::try_from(*v) else { return 0 };
            total = total.saturating_mul(v);
        }
        total
    }
}

/// What a container's graph says it is. `Unknown` is a first-class answer:
/// an unmatched signature is reported, never guessed from a filename.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum OnnxModelClass {
    /// `input_ids` + `past_key_values.*` — standalone autoregressive decoder.
    DecoderOnlyLm,
    /// `input_ids` + `encoder_hidden_states` — cross-attention decoder
    /// (speech, translation, … — the task is not in the graph).
    Seq2SeqDecoder,
    /// `input_values` → `last_hidden_state` — audio encoder.
    AudioEncoder,
    /// `input` + `state` + `sr` → `output` + `stateN` — streaming VAD.
    Vad,
    /// any → `wav*` — waveform synthesis.
    Vocoder,
    /// `input_ids` → `last_hidden_state` — standalone transformer text
    /// encoder (the encoder half of a seq2seq pair, e.g. marian/nllb).
    TextEncoderTransformer,
    /// `text_ids` → `text_emb` — TTS text encoder.
    TextEncoder,
    /// `text_ids` → `duration` — TTS duration predictor.
    DurationPredictor,
    /// `noisy_latent`-driven iterative estimator — TTS latent estimator.
    TtsEstimator,
    Unknown,
}

impl OnnxModelClass {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DecoderOnlyLm => "decoder-only-lm",
            Self::Seq2SeqDecoder => "seq2seq-decoder",
            Self::AudioEncoder => "audio-encoder",
            Self::Vad => "vad",
            Self::Vocoder => "vocoder",
            Self::TextEncoderTransformer => "text-encoder",
            Self::TextEncoder => "tts-text-encoder",
            Self::DurationPredictor => "tts-duration-predictor",
            Self::TtsEstimator => "tts-estimator",
            Self::Unknown => "unknown",
        }
    }
}

/// The signature of one ONNX container.
#[derive(Debug, Default)]
pub struct OnnxGraphIo {
    pub inputs: Vec<OnnxTensor>,
    pub outputs: Vec<OnnxTensor>,
    pub initializers: Vec<OnnxTensor>,
}

impl OnnxGraphIo {
    /// Total parameter count across initializers (element counts summed).
    /// Initializers whose shapes contain symbolic dimensions contribute 0 —
    /// real weights always have concrete shapes.
    #[must_use]
    pub fn parameter_count(&self) -> u64 {
        self.initializers
            .iter()
            .map(OnnxTensor::element_count)
            .fold(0_u64, u64::saturating_add)
    }
}

// ---------------------------------------------------------------------------
// Protobuf wire scanning (file level)
// ---------------------------------------------------------------------------

fn invalid(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

/// Streaming scanner over the file: varints, tags, and — crucially — skips.
/// Weight-bearing fields are jumped over with seeks, never read.
struct FileScan<R: Read + Seek> {
    inner: R,
}

impl<R: Read + Seek> FileScan<R> {
    fn new(inner: R) -> Self {
        Self { inner }
    }

    /// One varint. `None` at clean EOF before the first byte.
    fn read_varint(&mut self) -> std::io::Result<Option<u64>> {
        let mut value: u64 = 0;
        let mut shift = 0_u32;
        let mut byte = [0_u8; 1];
        loop {
            match self.inner.read_exact(&mut byte) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && shift == 0 => {
                    return Ok(None);
                }
                Err(e) => return Err(e),
            }
            let b = byte[0];
            if shift >= 64 {
                return Err(invalid("varint longer than 64 bits"));
            }
            value |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(Some(value));
            }
            shift += 7;
        }
    }

    fn read_tag(&mut self) -> std::io::Result<Option<(u32, u8)>> {
        let Some(tag) = self.read_varint()? else {
            return Ok(None);
        };
        let field = u32::try_from(tag >> 3).map_err(|_| invalid("field number overflow"))?;
        let wire = u8::try_from(tag & 0x7).map_err(|_| invalid("wire type overflow"))?;
        if field == 0 {
            return Err(invalid("field number 0 is not valid protobuf"));
        }
        Ok(Some((field, wire)))
    }

    /// Jump past one field of the given wire type.
    fn skip_field(&mut self, wire: u8) -> std::io::Result<()> {
        match wire {
            0 => {
                self.read_varint()?;
            }
            1 => {
                self.inner.seek(SeekFrom::Current(8))?;
            }
            2 => {
                let len = self
                    .read_varint()?
                    .ok_or_else(|| invalid("truncated length"))?;
                self.seek_forward(len)?;
            }
            5 => {
                self.inner.seek(SeekFrom::Current(4))?;
            }
            _ => {
                return Err(invalid(
                    "unsupported wire type (groups are not used by ONNX)",
                ));
            }
        }
        Ok(())
    }

    fn seek_forward(&mut self, len: u64) -> std::io::Result<()> {
        let len_i64 = i64::try_from(len).map_err(|_| invalid("field length beyond i64 range"))?;
        self.inner.seek(SeekFrom::Current(len_i64))?;
        Ok(())
    }

    /// Read a bounded sub-message into memory. These are metadata fields
    /// (inputs, outputs) — never weights.
    fn read_submessage(&mut self) -> std::io::Result<Vec<u8>> {
        let len = self
            .read_varint()?
            .ok_or_else(|| invalid("truncated length"))?;
        if len > MAX_SUBMESSAGE {
            return Err(invalid("sub-message exceeds scan budget"));
        }
        let len = usize::try_from(len).map_err(|_| invalid("length does not fit usize"))?;
        let mut buf = vec![0_u8; len];
        self.inner.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn position(&mut self) -> std::io::Result<u64> {
        self.inner.stream_position()
    }
}

// ---------------------------------------------------------------------------
// Protobuf wire scanning (in-memory sub-messages)
// ---------------------------------------------------------------------------

/// Bounds-checked cursor over one buffered sub-message.
struct SliceScan<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> SliceScan<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn read_varint(&mut self) -> std::io::Result<u64> {
        let mut value: u64 = 0;
        let mut shift = 0_u32;
        loop {
            let Some(&b) = self.buf.get(self.pos) else {
                return Err(invalid("truncated varint"));
            };
            self.pos += 1;
            if shift >= 64 {
                return Err(invalid("varint longer than 64 bits"));
            }
            value |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
    }

    fn read_tag(&mut self) -> std::io::Result<(u32, u8)> {
        let tag = self.read_varint()?;
        let field = u32::try_from(tag >> 3).map_err(|_| invalid("field number overflow"))?;
        let wire = u8::try_from(tag & 0x7).map_err(|_| invalid("wire type overflow"))?;
        if field == 0 {
            return Err(invalid("field number 0 is not valid protobuf"));
        }
        Ok((field, wire))
    }

    fn read_bytes(&mut self) -> std::io::Result<&'a [u8]> {
        let len = self.read_varint()?;
        let len = usize::try_from(len).map_err(|_| invalid("length does not fit usize"))?;
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| invalid("length overflow"))?;
        let out = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| invalid("overrun"))?;
        self.pos = end;
        Ok(out)
    }

    fn skip_field(&mut self, wire: u8) -> std::io::Result<()> {
        match wire {
            0 => {
                self.read_varint()?;
            }
            1 => self.bump(8)?,
            2 => {
                self.read_bytes()?;
            }
            5 => self.bump(4)?,
            _ => return Err(invalid("unsupported wire type")),
        }
        Ok(())
    }

    fn bump(&mut self, n: usize) -> std::io::Result<()> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| invalid("length overflow"))?;
        if end > self.buf.len() {
            return Err(invalid("overrun"));
        }
        self.pos = end;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ONNX message shapes
// ---------------------------------------------------------------------------

fn parse_dim(buf: &[u8]) -> std::io::Result<OnnxDim> {
    let mut s = SliceScan::new(buf);
    let mut dim = OnnxDim::Unknown;
    while !s.at_end() {
        let (field, wire) = s.read_tag()?;
        match (field, wire) {
            (1, 0) => {
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "protobuf int64 varints are two's-complement; the cast is the format"
                )]
                let value = s.read_varint()? as i64;
                dim = OnnxDim::Value(value);
            }
            (2, 2) => {
                let bytes = s.read_bytes()?;
                dim = OnnxDim::Param(String::from_utf8_lossy(bytes).into_owned());
            }
            (_, w) => s.skip_field(w)?,
        }
    }
    Ok(dim)
}

fn parse_value_info(buf: &[u8]) -> std::io::Result<OnnxTensor> {
    let mut s = SliceScan::new(buf);
    let mut tensor = OnnxTensor {
        name: String::new(),
        dtype: OnnxDataType::Undefined,
        dims: Vec::new(),
    };
    while !s.at_end() {
        let (field, wire) = s.read_tag()?;
        match (field, wire) {
            (1, 2) => tensor.name = String::from_utf8_lossy(s.read_bytes()?).into_owned(),
            (2, 2) => parse_type_proto(s.read_bytes()?, &mut tensor)?,
            (_, w) => s.skip_field(w)?,
        }
    }
    Ok(tensor)
}

fn parse_type_proto(buf: &[u8], tensor: &mut OnnxTensor) -> std::io::Result<()> {
    let mut s = SliceScan::new(buf);
    while !s.at_end() {
        let (field, wire) = s.read_tag()?;
        match (field, wire) {
            (1, 2) => parse_tensor_type(s.read_bytes()?, tensor)?,
            (_, w) => s.skip_field(w)?,
        }
    }
    Ok(())
}

fn parse_tensor_type(buf: &[u8], tensor: &mut OnnxTensor) -> std::io::Result<()> {
    let mut s = SliceScan::new(buf);
    while !s.at_end() {
        let (field, wire) = s.read_tag()?;
        match (field, wire) {
            (1, 0) => tensor.dtype = OnnxDataType::from_elem_type(s.read_varint()?),
            (2, 2) => parse_shape(s.read_bytes()?, tensor)?,
            (_, w) => s.skip_field(w)?,
        }
    }
    Ok(())
}

fn parse_shape(buf: &[u8], tensor: &mut OnnxTensor) -> std::io::Result<()> {
    let mut s = SliceScan::new(buf);
    while !s.at_end() {
        let (field, wire) = s.read_tag()?;
        match (field, wire) {
            (1, 2) => tensor.dims.push(parse_dim(s.read_bytes()?)?),
            (_, w) => s.skip_field(w)?,
        }
    }
    Ok(())
}

/// `TensorProto` header fields only: `dims = 1` (packed or not),
/// `data_type = 2`, `name = 8`. `raw_data = 9` and friends are the weights —
/// skipped by seeking.
fn parse_initializer_header<R: Read + Seek>(
    scan: &mut FileScan<R>,
    len: u64,
) -> std::io::Result<OnnxTensor> {
    let end = scan
        .position()?
        .checked_add(len)
        .ok_or_else(|| invalid("initializer length overflow"))?;
    let mut tensor = OnnxTensor {
        name: String::new(),
        dtype: OnnxDataType::Undefined,
        dims: Vec::new(),
    };
    while scan.position()? < end {
        let (field, wire) = scan.read_tag()?.ok_or_else(|| invalid("truncated tag"))?;
        match (field, wire) {
            (1, 0) => {
                let raw = scan
                    .read_varint()?
                    .ok_or_else(|| invalid("truncated dim"))?;
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "protobuf int64 varints are two's-complement; the cast is the format"
                )]
                let value = raw as i64;
                tensor.dims.push(OnnxDim::Value(value));
            }
            (1, 2) => {
                let packed = scan.read_submessage()?;
                let mut s = SliceScan::new(&packed);
                while !s.at_end() {
                    #[expect(
                        clippy::cast_possible_wrap,
                        reason = "protobuf int64 varints are two's-complement; the cast is the format"
                    )]
                    let value = s.read_varint()? as i64;
                    tensor.dims.push(OnnxDim::Value(value));
                }
            }
            (2, 0) => {
                let raw = scan
                    .read_varint()?
                    .ok_or_else(|| invalid("truncated dtype"))?;
                tensor.dtype = OnnxDataType::from_elem_type(raw);
            }
            (8, 2) => {
                let bytes = scan.read_submessage()?;
                tensor.name = String::from_utf8_lossy(&bytes).into_owned();
            }
            (_, w) => scan.skip_field(w)?,
        }
    }
    Ok(tensor)
}

/// Scan one `GraphProto` window: `input` (11), `output` (12), initializer
/// headers (5). `node` (1) and everything after the outputs (`value_info`
/// and friends) are skipped by seeking.
fn parse_graph_window<R: Read + Seek>(
    scan: &mut FileScan<R>,
    len: u64,
) -> std::io::Result<OnnxGraphIo> {
    let end = scan
        .position()?
        .checked_add(len)
        .ok_or_else(|| invalid("graph length overflow"))?;
    let mut graph = OnnxGraphIo::default();
    while scan.position()? < end {
        let Some((field, wire)) = scan.read_tag()? else {
            break;
        };
        match (field, wire) {
            (5, 2) => {
                let len = scan.read_varint()?.ok_or_else(|| invalid("trunc"))?;
                graph
                    .initializers
                    .push(parse_initializer_header(scan, len)?);
            }
            (11, 2) => {
                let buf = scan.read_submessage()?;
                graph.inputs.push(parse_value_info(&buf)?);
            }
            (12, 2) => {
                let buf = scan.read_submessage()?;
                graph.outputs.push(parse_value_info(&buf)?);
            }
            // value_info and anything later: not part of the signature.
            (f, _) if f > 12 => break,
            (_, w) => scan.skip_field(w)?,
        }
    }
    Ok(graph)
}

/// Read the graph signature of an ONNX container without loading it.
///
/// # Errors
///
/// Returns `Err` for unreadable files, corrupt wire data, or a file with no
/// graph. Never panics: a corrupt file yields an `Err` the caller reports as
/// an unreadable signature.
pub fn read_onnx_graph<P: AsRef<Path>>(path: P) -> std::io::Result<OnnxGraphIo> {
    let file = File::open(path)?;
    let mut scan = FileScan::new(file);
    while let Some((field, wire)) = scan.read_tag()? {
        if field == 7 && wire == 2 {
            let len = scan
                .read_varint()?
                .ok_or_else(|| invalid("truncated graph"))?;
            return parse_graph_window(&mut scan, len);
        }
        scan.skip_field(wire)?;
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "no graph found in ONNX container",
    ))
}

// ---------------------------------------------------------------------------
// Signature → model class
// ---------------------------------------------------------------------------

/// Classify a container from its graph signature alone. No filename, no
/// directory name — the graph is the only witness.
#[must_use]
pub fn classify(inputs: &[OnnxTensor], outputs: &[OnnxTensor]) -> OnnxModelClass {
    let named = |ts: &[OnnxTensor], name: &str| ts.iter().any(|t| t.name == name);
    let prefixed = |ts: &[OnnxTensor], prefix: &str| ts.iter().any(|t| t.name.starts_with(prefix));

    if named(inputs, "input_ids") {
        // Cross-attention first: merged seq2seq decoders carry both
        // encoder_hidden_states and past_key_values inputs.
        if prefixed(inputs, "encoder_hidden_states") {
            return OnnxModelClass::Seq2SeqDecoder;
        }
        if named(outputs, "last_hidden_state") {
            return OnnxModelClass::TextEncoderTransformer;
        }
        if prefixed(inputs, "past_key_values") {
            return OnnxModelClass::DecoderOnlyLm;
        }
        return OnnxModelClass::Unknown;
    }
    if named(inputs, "input_values") && named(outputs, "last_hidden_state") {
        return OnnxModelClass::AudioEncoder;
    }
    if named(inputs, "input")
        && named(inputs, "state")
        && named(inputs, "sr")
        && named(outputs, "output")
        && named(outputs, "stateN")
    {
        return OnnxModelClass::Vad;
    }
    if prefixed(outputs, "wav") {
        return OnnxModelClass::Vocoder;
    }
    if named(inputs, "text_ids") && named(outputs, "text_emb") {
        return OnnxModelClass::TextEncoder;
    }
    if named(inputs, "text_ids") && named(outputs, "duration") {
        return OnnxModelClass::DurationPredictor;
    }
    if named(inputs, "noisy_latent") {
        return OnnxModelClass::TtsEstimator;
    }
    OnnxModelClass::Unknown
}
