//! ONNX graph-signature reader: crafted wire fixtures for the parser logic,
//! plus — when the model drive is mounted — the real containers that served
//! as ground truth (cross-checked against `onnx_info.py`).

use std::fs;
use std::path::PathBuf;
use sysreflect_model::onnx_graph::{
    OnnxDataType, OnnxDim, OnnxModelClass, OnnxTensor, classify, read_onnx_graph,
};

// ---------------------------------------------------------------------------
// Minimal protobuf builders
// ---------------------------------------------------------------------------

fn push_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = u8::try_from(v & 0x7F).expect("masked to 7 bits");
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn tag(out: &mut Vec<u8>, field: u32, wire: u8) {
    push_varint(out, (u64::from(field) << 3) | u64::from(wire));
}

fn len_field(out: &mut Vec<u8>, field: u32, payload: &[u8]) {
    tag(out, field, 2);
    push_varint(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

fn varint_field(out: &mut Vec<u8>, field: u32, v: u64) {
    tag(out, field, 0);
    push_varint(out, v);
}

fn str_field(out: &mut Vec<u8>, field: u32, s: &str) {
    len_field(out, field, s.as_bytes());
}

fn dim_value(v: i64) -> Vec<u8> {
    let mut out = Vec::new();
    #[expect(
        clippy::cast_sign_loss,
        reason = "protobuf int64 varints are two's-complement; the cast is the format"
    )]
    let v = v as u64;
    varint_field(&mut out, 1, v);
    out
}

fn dim_param(p: &str) -> Vec<u8> {
    let mut out = Vec::new();
    str_field(&mut out, 2, p);
    out
}

/// TypeProto.Tensor: elem_type = 1, shape = 2.
fn tensor_type(elem: u64, dims: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    varint_field(&mut out, 1, elem);
    let mut shape = Vec::new();
    for d in dims {
        len_field(&mut shape, 1, d);
    }
    len_field(&mut out, 2, &shape);
    out
}

/// TypeProto: tensor_type = 1 (message).
fn type_proto(elem: u64, dims: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    len_field(&mut out, 1, &tensor_type(elem, dims));
    out
}

/// ValueInfoProto: name = 1, type = 2.
fn value_info(name: &str, elem: u64, dims: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    str_field(&mut out, 1, name);
    len_field(&mut out, 2, &type_proto(elem, dims));
    out
}

fn initializer(name: &str, dtype: u64, dims: &[i64], blob_bytes: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for d in dims {
        #[expect(
            clippy::cast_sign_loss,
            reason = "protobuf int64 varints are two's-complement; the cast is the format"
        )]
        let d = *d as u64;
        varint_field(&mut out, 1, d); // TensorProto.dims = 1
    }
    varint_field(&mut out, 2, dtype); // TensorProto.data_type = 2
    str_field(&mut out, 8, name); // TensorProto.name = 8
    len_field(&mut out, 9, &vec![0_u8; blob_bytes]); // raw_data: weights, skipped
    out
}

/// GraphProto with initializers first, then inputs, then outputs — the order
/// real exporters produce. Field numbers: initializer = 5, input = 11,
/// output = 12.
fn graph_of(inputs: &[Vec<u8>], outputs: &[Vec<u8>], inits: &[Vec<u8>]) -> Vec<u8> {
    let mut g = Vec::new();
    for i in inits {
        len_field(&mut g, 5, i);
    }
    for i in inputs {
        len_field(&mut g, 11, i);
    }
    for o in outputs {
        len_field(&mut g, 12, o);
    }
    g
}

/// ModelProto: ir_version = 1, producer_name = 2, graph = 7.
fn model_of(graph: &[u8]) -> Vec<u8> {
    let mut m = Vec::new();
    varint_field(&mut m, 1, 8); // ir_version
    str_field(&mut m, 2, "test-producer"); // producer_name
    len_field(&mut m, 7, graph); // graph
    m
}

fn write_temp(name: &str, bytes: &[u8]) -> PathBuf {
    let p = std::env::temp_dir().join(format!("onnx_graph_test_{name}.onnx"));
    fs::write(&p, bytes).expect("write temp fixture");
    p
}

fn tensor(name: &str) -> OnnxTensor {
    OnnxTensor {
        name: name.to_string(),
        dtype: OnnxDataType::Float,
        dims: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Wire-logic tests
// ---------------------------------------------------------------------------

#[test]
fn reads_inputs_outputs_and_initializer_shapes() {
    let graph = graph_of(
        &[
            value_info(
                "input_ids",
                7,
                &[dim_param("batch_size"), dim_param("sequence_length")],
            ),
            value_info("attention_mask", 7, &[dim_param("batch"), dim_param("seq")]),
        ],
        &[value_info(
            "logits",
            1,
            &[dim_param("batch"), dim_param("seq"), dim_value(32000)],
        )],
        &[initializer("layer0.weight", 1, &[2048, 512], 4096)],
    );
    let path = write_temp("basic", &model_of(&graph));

    let io = read_onnx_graph(&path).expect("valid model");
    assert_eq!(io.inputs.len(), 2);
    assert_eq!(io.inputs[0].name, "input_ids");
    assert_eq!(io.inputs[0].dtype, OnnxDataType::Int64);
    assert_eq!(
        io.inputs[0].dims,
        vec![
            OnnxDim::Param("batch_size".into()),
            OnnxDim::Param("sequence_length".into())
        ]
    );
    assert_eq!(io.outputs[0].name, "logits");
    assert_eq!(io.outputs[0].dims[2], OnnxDim::Value(32000));
    assert_eq!(io.initializers.len(), 1);
    assert_eq!(io.initializers[0].name, "layer0.weight");
    assert_eq!(io.parameter_count(), 2048 * 512);
}

#[test]
fn skips_multi_megabyte_weight_blobs_and_still_finds_outputs() {
    let graph = graph_of(
        &[value_info(
            "latent",
            1,
            &[dim_param("batch"), dim_value(144), dim_param("len")],
        )],
        &[value_info(
            "wav_tts",
            1,
            &[dim_param("samples"), dim_value(1)],
        )],
        &[initializer("big.weight", 1, &[65536, 64], 2 * 1024 * 1024)],
    );
    let path = write_temp("bigblob", &model_of(&graph));

    let io = read_onnx_graph(&path).expect("valid model");
    assert_eq!(io.outputs[0].name, "wav_tts");
    assert_eq!(classify(&io.inputs, &io.outputs), OnnxModelClass::Vocoder);
}

#[test]
fn value_info_after_outputs_is_stopped_at() {
    // value_info = field 13, after outputs; a correct reader stops and never
    // touches it.
    let mut graph = graph_of(
        &[value_info("input", 1, &[dim_param("a"), dim_param("b")])],
        &[value_info("output", 1, &[dim_param("c"), dim_value(1)])],
        &[],
    );
    len_field(
        &mut graph,
        13,
        &value_info("intermediate", 1, &[dim_value(1)]),
    );
    len_field(&mut graph, 13, &vec![0_u8; 64 * 1024]);
    let path = write_temp("valueinfo", &model_of(&graph));

    let io = read_onnx_graph(&path).expect("valid model");
    assert_eq!(io.inputs.len(), 1);
    assert_eq!(io.outputs.len(), 1);
}

#[test]
fn empty_garbage_and_truncated_files_are_errors_not_panics() {
    for (name, bytes) in [
        ("empty", vec![]),
        ("garbage", vec![0xFF; 512]),
        (
            "truncated",
            model_of(&graph_of(&[value_info("x", 1, &[])], &[], &[]))[..12].to_vec(),
        ),
    ] {
        let path = write_temp(name, &bytes);
        assert!(
            read_onnx_graph(&path).is_err(),
            "{name}: corrupt input must not be reported as a graph"
        );
    }
}

// ---------------------------------------------------------------------------
// Signature → class (the nine reference containers)
// ---------------------------------------------------------------------------

#[test]
fn classifies_reference_signatures() {
    let case = |inputs: &[&str], outputs: &[&str]| {
        let ins: Vec<OnnxTensor> = inputs.iter().map(|n| tensor(n)).collect();
        let outs: Vec<OnnxTensor> = outputs.iter().map(|n| tensor(n)).collect();
        classify(&ins, &outs)
    };

    // Qwen1.5-0.5B-Chat exported as decoder-only.
    assert_eq!(
        case(
            &[
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_values.0.key",
                "past_key_values.0.value"
            ],
            &["logits", "present.0.key"],
        ),
        OnnxModelClass::DecoderOnlyLm
    );
    // Marian merged decoder: cross-attention AND kv-cache inputs.
    assert_eq!(
        case(
            &[
                "encoder_attention_mask",
                "input_ids",
                "encoder_hidden_states",
                "past_key_values.0.decoder.key"
            ],
            &["logits", "present.0.decoder.key"],
        ),
        OnnxModelClass::Seq2SeqDecoder
    );
    // Moonshine decoder.
    assert_eq!(
        case(
            &["input_ids", "encoder_hidden_states"],
            &["logits", "present.0.decoder.key"]
        ),
        OnnxModelClass::Seq2SeqDecoder
    );
    // Marian/NLLB standalone text encoder (encoder half of a seq2seq pair).
    assert_eq!(
        case(&["input_ids", "attention_mask"], &["last_hidden_state"]),
        OnnxModelClass::TextEncoderTransformer
    );
    // Moonshine encoder.
    assert_eq!(
        case(&["input_values"], &["last_hidden_state"]),
        OnnxModelClass::AudioEncoder
    );
    // Silero VAD.
    assert_eq!(
        case(&["input", "state", "sr"], &["output", "stateN"]),
        OnnxModelClass::Vad
    );
    // Supertonic pipeline parts.
    assert_eq!(case(&["latent"], &["wav_tts"]), OnnxModelClass::Vocoder);
    assert_eq!(
        case(&["text_ids", "style_ttl", "text_mask"], &["text_emb"]),
        OnnxModelClass::TextEncoder
    );
    assert_eq!(
        case(&["text_ids", "style_dp", "text_mask"], &["duration"]),
        OnnxModelClass::DurationPredictor
    );
    assert_eq!(
        case(
            &[
                "noisy_latent",
                "text_emb",
                "style_ttl",
                "latent_mask",
                "text_mask",
                "current_step",
                "total_step"
            ],
            &["latent_out"],
        ),
        OnnxModelClass::TtsEstimator
    );
    // Honest unknown: no name in this table gets to claim a class.
    assert_eq!(case(&["foo"], &["bar"]), OnnxModelClass::Unknown);
    // input_ids alone (no cache, no cross-attention) stays unknown too.
    assert_eq!(case(&["input_ids"], &["logits"]), OnnxModelClass::Unknown);
}

// ---------------------------------------------------------------------------
// Real containers (skipped when the model drive is not mounted)
// ---------------------------------------------------------------------------

fn real(relative: &str) -> Option<PathBuf> {
    // sysreflect-model/ -> repo root is one level up.
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../Models/onnx/assets")
        .join(relative);
    p.exists().then_some(p)
}

fn class_of(relative: &str) -> Option<OnnxModelClass> {
    let path = real(relative)?;
    let io = read_onnx_graph(&path).expect("real container must parse");
    Some(classify(&io.inputs, &io.outputs))
}

#[test]
fn real_containers_classify_as_reference_does() {
    let expectations = [
        ("llm/model.onnx", OnnxModelClass::DecoderOnlyLm),
        ("stt/moonshine_encoder.onnx", OnnxModelClass::AudioEncoder),
        ("stt/moonshine_decoder.onnx", OnnxModelClass::Seq2SeqDecoder),
        ("stt/vad/silero_vad.onnx", OnnxModelClass::Vad),
        (
            "supertonic-3/onnx/text_encoder.onnx",
            OnnxModelClass::TextEncoder,
        ),
        (
            "supertonic-3/onnx/duration_predictor.onnx",
            OnnxModelClass::DurationPredictor,
        ),
        (
            "supertonic-3/onnx/vector_estimator.onnx",
            OnnxModelClass::TtsEstimator,
        ),
        ("supertonic-3/onnx/vocoder.onnx", OnnxModelClass::Vocoder),
        (
            "translation/marian/decoder_model_merged.onnx",
            OnnxModelClass::Seq2SeqDecoder,
        ),
        (
            "translation/marian/encoder_model.onnx",
            OnnxModelClass::TextEncoderTransformer,
        ),
        (
            "translation/nllb/encoder_model.onnx",
            OnnxModelClass::TextEncoderTransformer,
        ),
        (
            "translation/nllb/decoder_with_past_model.onnx",
            OnnxModelClass::Seq2SeqDecoder,
        ),
        ("vad/silero_vad.onnx", OnnxModelClass::Vad),
    ];
    let mut checked = 0;
    for (path, expected) in expectations {
        match class_of(path) {
            Some(class) => {
                assert_eq!(class, expected, "{path}");
                checked += 1;
            }
            None => eprintln!("model drive not mounted; skipping {path}"),
        }
    }
    if checked > 0 {
        // Reference-verified parameter count from onnx_info.py:
        // moonshine encoder = 20,153,268 parameters.
        let path = real("stt/moonshine_encoder.onnx").expect("checked above");
        let io = read_onnx_graph(&path).expect("parses");
        assert_eq!(io.parameter_count(), 20_153_268);
    }
}
