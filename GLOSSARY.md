# lmsafe

Admission control and quarantined storage for local hosts. lmsafe answers two
questions before any file is trusted — *what is this file?* and *can this
system hold it?* — and provides the lanes hosts load from. It never runs what
it holds.

This glossary is the shared vocabulary between Developer and User — when a term
here is used differently in conversation, this file wins until we renegotiate
it. A consumer's own glossary wins within that consumer's project; no consumer
is part of lmsafe's vocabulary.

## Language

### The chain

**Host**:
The program that calls lmsafe through the FFI. Any host; lmsafe names and
assumes none.
_Avoid_: app, caller, engine

**Inspector**:
The metadata layer: what is this file (sysreflect-model), what is the system
doing (sysreflect-core). Every gate consults it.
_Avoid_: detector, parser, monitor

**SafeZone**:
Quarantined storage where vetted files live, organized in lanes. The host only
ever loads from here.
_Avoid_: cache, models folder

**Lane**:
A SafeZone's kind of cargo — models, media, or whatever a project next needs
(documents, private storage). The lane fixes the gate, the inspection report,
and the lifecycle operations. Lanes grow with need; the zone is not tied to
any one kind of thing.
_Avoid_: category, bucket

**Gate**:
An lmsafe check that must pass before a file is trusted — at load AND at
send. Two exist: the RAM gate and the copy gate.
_Avoid_: validation, check

**Verdict**:
A gate's answer: Accepted, Rejected, or Unknown.
_Avoid_: result, status

**Admission**:
The single rule both gates consult — file size × safety margin against free
resources. Derived once, never reimplemented.
_Avoid_: policy, decision, logic

**Safety margin**:
The headroom multiplier a gate requires over a file's size; `0` means gates
are off. A Developer tunable.
_Avoid_: buffer, padding

### Files

**Container**:
The file format a model ships in — GGUF, GGML, LiteRT-LM, ONNX, SafeTensors.
One model can ship in several containers; one ONNX directory can hold several
containers. An ONNX container is classified by its graph signature (inputs,
outputs, initializer shapes) read straight from the protobuf — never by
filename or directory name.
_Avoid_: engine

**Unit**:
What an engine loads together and what one admission verdict describes: an
ONNX path is one unit (a single `.onnx`, or a directory of them like
`supertonic-3/onnx`) or a group of units (`assets/` → llm, stt,
supertonic-3, translation/{marian, nllb}, vad). Discovered by walking the
path for containers — not by names.
_Avoid_: model folder, bundle

**Model**:
The network inside a container (llama, whisper, moonshine, supertonic),
identified by the container's own metadata — never by filename.
_Avoid_: weights, AI

**Model family**:
The engine format a model belongs to (llama.cpp, whisper.cpp). Determines
where it may be stored and loaded.
_Avoid_: type, format

**Inspection report**:
What the inspector returns about one file before it is trusted. Its shape
belongs to the lane: models report name, family, capabilities.
_Avoid_: model info, header

**Media report**:
The media lane's inspection report for one file — dimensions, duration, frame
count, and whether the file is supported.
_Avoid_: metadata

**Tensor check**:
An inspection report line proving a model's graph inputs and outputs match
what its family claims (silero VAD: audio, sr, state → prob, state).
_Avoid_: validation

**Package**:
A directory shipping one model as many files, inspected as one unit —
moonshine (encoder + decoder + tokenizer + VAD), supertonic (config + voice
styles + samples).
_Avoid_: folder, bundle

**Capability**:
An inspection report claim of what a model can do — text generation,
speech-to-text, text-to-speech, audio.
_Avoid_: feature, skill

**Media**:
The SafeZone lane for user videos and images — inspected for dimensions,
duration, and support; gated by free disk space, never by RAM.
_Avoid_: assets, uploads

### Telemetry

**System reflection**:
The telemetry layer — what the machine is doing: CPU, memory, network,
processes, uptime. Gates consult it for free resources.
_Avoid_: monitoring, metrics

**Metric status**:
A metric's truthfulness — Available with values, or Restricted because the
system refused to disclose. Zero activity is Available, never Restricted.
_Avoid_: error, missing

**Restricted**:
A metric the system declined to disclose (permissions). It never means idle.
_Avoid_: denied, blocked

### Scope

**Engine**:
The host's runtime that owns a loaded model. lmsafe never loads, runs, or
chooses engines — the word belongs to the host.
_Avoid_ (in lmsafe): runtime, backend, adapter
