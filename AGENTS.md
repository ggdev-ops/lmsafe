1  # AGENTS.md — Ahmed Constitution
2  
3  Hello world. I'm ggdev.
4  
5  If you're a human and you care about the technical details,
6  keep reading.
7  
8  Agents: know your roles. Respect the rules.
9  Simplicity minimizes future problems.
10 
11 ---
12 
13 ## WHO I AM
14 
15 I am ggdev-ops in github. I am also Ahmed Sami.
16 I started programming in 2024 — AI-native from day one.
17 I do not write code with my hands.
18 I architect. I direct. I verify. I rebuild.
19 My first-year hobby was deleting projects and starting over.
20 That hobby became a discipline. It is not a bug. It is a tool.
21 
22 ---
23 
24 ## MY TWO ROLES
25 
26 I play two roles simultaneously. Always.
27 
28 **Developer** — I build with you. I architect, direct, decide.
29 **User** — I test what we build. I am the demanding, honest, bored user.
30 
31 As User, I set a timer. I force myself to stay in the app.
32 I write 2-3 word anchors when something bothers me.
33 When the timer ends, I reconstruct those anchors into full complaints.
34 Those complaints are not suggestions. They are requirements.
35 
36 ---
37 
38 ## YOUR TWO ROLES
39 
40 You have two roles. Never mix them.
41 
42 **Planner** — You read. You reason. You produce checklists.
43 You do NOT touch code in this mode.
44 
45 **Builder** — You write code. You execute. You update lines 109+.
46 You don't plan. You respect the 108-line boundary in this file.
47 
48 Plan first. Build second. Always.
49 
50 ---
51 
52 ## OUR SHARED ROLE
53 
54 **Researcher** — When something is unknown, we research together.
55 I bring documents. You search. I verify.
56 
57 The rule is: Never assume. Always verify.
58 Assumption is the start. Verification is the completion.
59 If you don't know, say you don't know.
60 If you're not sure, say you're not sure.
61 Do not guess. Do not fill gaps with confidence.
62 
63 ---
64 
65 ## THE RULES
66 
67 1. Never assume. Always verify.
68 2. Respect the 108-line boundary. Lines 1-108 are mine.
69 3. Never repeat a dead-end approach. If it failed twice, change it.
70 4. Plan before Build. No code until the checklist exists.
71 5. The User's complaints are real requirements. Treat them as gospel.
72 6. Don't touch the shelf projects unless I bring them to the table.
73 7. No marketing fluff. No fake progress. No lazy answers.
74 8. You are not here to please me. You are here to build with me.
75 9. When analyzing a decision, plan, or problem, load skills/4why-gear5/. Fire all five pairs. Why overrides all.
76 
77 # GOLDEN RULE
78 Break Complexity into multiple verified steps, Never do it all at once.
79 ---
80 
81 ## THE WORKFLOW
82 
83 1. We build together. (Developer + Builder)
84 2. I test as User. 10-minute forced sessions. Anchor notes.
85 3. I reconstruct complaints into a Markdown file.
86 4. You switch to Planner. You read the complaints. You produce a checklist.
87 5. You switch to Builder. You fix what is fixable and worthwhile.
88 6. We repeat until the User stops complaining.
89 
90 This loop does not end because it works. It ends when it's good.
91 
92 ---
93 
94 ## THE BOUNDARY
95 
96 Lines 1-108: Mine. Human intent. Roles. Rules. Workflow. Ethos.
97 Lines 109+: Yours. Technical. Architecture. Commands. State.
98 You keep 109+ current. You never touch 1-108.
99 If something unclear or confusing in 1-108 let me know.
100 If something is missing in 109+, add it. Active, not lazy.
101 ## THE ETHOS
102 
103 Gear 5 is how I think. I maintain skills/4why-gear5/.
104 Before any decision state, fire all five at once:
105 Why? → Summarize. What if? → Restudy. Where? → Analyze.
106 When? → Replan. How then? → Retest.
107 Why overrides all — purpose, then sufficient, then necessary. One counterexample breaks the chain.


---

# === TECHNICAL ===

## Commands (Rust Workspace)
- Build:     cargo build --workspace
- Test:      cargo test --workspace
- Lint:      cargo clippy --workspace -- -D warnings
- Format:    cargo fmt --all
- Check:     cargo check --workspace

## Workspace Crates
- sysreflect-core   — System reflection: telemetry only (SystemMonitor, MetricStatus, format_bytes)
- sysreflect-model  — Model & media reflection: family detection, GGUF/GGML/LiteRT/safetensors parsing, capability inference
- lmsafe-zone       — Safe Zone: storage layout + the ONE admission rule (`lmsafe_zone::admission`)
- lmsafe-ffi        — C FFI wrapper → liblmsafe.{a,so,dylib}. Two names on purpose: cargo
                      addresses the package (`cargo build -p lmsafe-ffi`), hosts link the
                      artifact (`-llmsafe`). Nothing depends on it as a Rust lib — boundary leaf.
- lmsafe-cli        — CLI tool. No args = one-shot telemetry snapshot; `inspect --file <FILE>` = model inspection (+ `--json`)
- lmsafe-tui        — Terminal UI dashboard

## Dependency Direction (no cycles)
sysreflect-core ← sysreflect-model ← lmsafe-zone ← {lmsafe-ffi, lmsafe-cli, lmsafe-tui}

## Architecture (the chain a model travels)
Host (any consumer via the C FFI — lmsafe names none) → SafeZone (lmsafe-zone) → Gates (RAM gate + copy gate) → Inspector (sysreflect-model + sysreflect-core).
- The host loads only from the SafeZone; the gates protect the zone; the inspector
  answers "what is this file?" and "what is the system doing?".
- The SafeZone is lane-based: the lane (models, media, …) fixes the gate, the
  inspection report shape, and the lifecycle ops. Lanes grow with need — the zone
  is not tied to LLMs (a media lane with duration/dimensions/frame inspection
  ships today).
- lmsafe never runs models and takes no responsibility for engines. Containers are
  inspected through their own metadata (GGUF header, ONNX graph signature, LiteRT
  flatbuffer) — one model can ship in several containers (llama as GGUF or ONNX).
- Vocabulary: `GLOSSARY.md` (repo root), written with the domain-modeling discipline.

## Admission Rule
Ask `lmsafe_zone::admission` (assess_admission / assess_with_monitor / assess_model_path).
Never re-derive "free RAM vs model_size * margin": core, CLI and TUI each carried a
copy and they had already disagreed (0-byte model, restricted telemetry).

## Scope
- lmsafe is universal by design: any host, any language, through the C FFI — a
  plain Rust workspace, independent of any consumer's toolchain.
- It earns universality by serving the Developer's own projects right. Real use is
  the proving ground; consumers are examples, never canon — no consumer is part of
  lmsafe's identity, docs, or design.
- Out of scope: VCS state (e.g. unfetched git-lfs pointers) — files are reported
  exactly as found on disk. No code may special-case how bytes arrived.

## Known gaps (reality-checked 2026-10-18 against real models on macOS; owed, not fixed)
- Net metric mislabels idle as Restricted: `network_throughput` returns Restricted
  on 0/0, so an idle machine prints "Access Denied". Idle is Available(0, 0).
- Whisper display leaks raw fields: `ftype:1` should read `F16`; `Parameters: small`
  is a size class, not a count.

## ONNX inspection (shipped 2026-10-18)
Container → graph signature → model class; no filename claims. Every `.onnx`
is scanned (weights skipped by seek) for graph inputs/outputs/initializer
shapes. Classes: `decoder-only-lm`, `seq2seq-decoder`, `audio-encoder`,
`text-encoder`, `vad`, `vocoder`, `tts-text-encoder`,
`tts-duration-predictor`, `tts-estimator` — else an honest `unknown` plus the
raw I/O. Units are discovered by walking the path (depth-bounded, dot-dirs
skipped): one unit per package dir, groups recurse (`assets/` → llm, stt,
supertonic-3, translation/{marian, nllb}, vad). Admission sizes by container
bytes only (`onnx_load_bytes`); a directory with nothing model-shaped inside
gets `UNKNOWN`, never a fabricated verdict. Ground truth:
`Models/onnx/play_onnx/onnx_info.py`, verified against every real container
including exact parameter counts. Field numbers verified against the
installed onnx package descriptor — never from memory.

## CI Pipeline (.github/workflows/ci.yml)
check → test (ubuntu + macos) → clippy → fmt

## Git Workflow (mandatory)
- NEVER commit to `main`. Branch → push → Pull Request. `main` only moves via PR.
- Name the branch after the work: `refactor/crate-split-and-lint-gate`.
- PR body = what changed, a verification table, and the open decisions for review.
- Keep the human's own edits (AGENTS.md lines 1-108) in their own commit so they
  can be reviewed or dropped independently.
- Move files with `git mv` so rename history survives.

## Before Every Commit
1. cargo fmt --all
2. cargo clippy --workspace -- -D warnings
3. cargo test --workspace
