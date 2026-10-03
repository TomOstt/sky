# Part XIII — LLM runtime: from a model name to tokens

This part specifies everything between `sky run hf://org/model "prompt"` and the text on the screen, except the kernels themselves:
- the Hugging Face client (`sky-hub`);
- config normalization and the architecture builders (`sky-llm::config`, `sky-llm::arch`);
- weight binding for MLX safetensors and GGUF, including the relayout cache;
- the tokenizer, chat templates and stop tokens;
- sampling on the GPU;
- the streaming detokenizer;
- the generation loop and its speculation hooks;
- the OpenAI-compatible server;
- the `.skybin` ahead-of-time bundle;
- the command line.

Kernels, the TIR compiler and the decode program itself are in [Part VII](part07-tensor-ir.md) to [Part XII](part12-quantization.md), [Part XVI](part16-decode.md) and [Part XX](part20-persistent-execution.md). TTFT is in [Part XVIII](part18-ttft.md) and memory in [Part XIX](part19-memory.md).

Labels: `[VERIFIED: source]`, `[LIKELY: reason]`, `[EXPERIMENTAL: what decides]`, `[MOONSHOT]`. Research notes are cited as (R9 §5) and ship as appendices, for example [research/R9-formats-methodology.md](research/R9-formats-methodology.md).

## 1. Summary

| Area | Decision | Why |
|---|---|---|
| Model names | `hf://[type/]org/name[@rev][:QUANT | /path]` | huggingface_hub's own URI form plus llama.cpp's `:QUANT` suffix (R9 §5.5) |
| Cache | Share `~/.cache/huggingface/hub`; never write the shared Xet blob store; take the same `flock` as Python clients | Users already have the files; the store is read-only and keyed by Xet hash (R9 §5.4) |
| Network | Offline first: a pinned commit with a complete snapshot makes zero network calls | Cold start, privacy, reproducibility |
| Provenance | Every run records repo, requested and resolved revision, and each file's size and sha256 | Required by the benchmark protocol (R9 §5.5, §8) |
| Config | Normalize both `rope_scaling` and `rope_parameters`; fail on unknown keys | mlx-lm silently drops unknown keys and loads a re-saved Llama 3.1 with θ = 1e4 and no scaling (R9 §3.1) |
| Builders | One Rust builder per family in `sky-llm::arch`, emitting TIR (D7) | Fast path to first tokens; Sky-source models later must produce identical TIR |
| Weights | GGUF mapped directly; MLX safetensors through a page-aligned relayout cache (`.skyweights`), proven lossless | MLX files are unaligned, and unaligned GPU loads return wrong data silently (R9 §1.3) |
| Tokenizer | `tokenizers` 0.23.2, pure Rust (fancy-regex), plus a golden-id gate | 168/168 identical to Python on an adversarial corpus (R9 §4.2) |
| Templates | `minijinja` 2.24.0 with `pycompat`, a Python-exact `tojson` and `preserve_order` | 30/30 renderable cases byte-identical to transformers (R9 §4.3) |
| Stop tokens | The union of config, generation_config and tokenizer_config (and GGUF) | Qwen3 stops on two ids; mlx-community repos lack `generation_config.json` (R9 §4.1) |
| Sampling | On the GPU, in the static decode program; keyed Gumbel noise from Philox; filters in a declared order | Exact speculation at any temperature (D10, R8 §3.2); no host round trip per token |
| Loop | GPU-resident: the GPU writes the token, position and stop flag; the CPU pre-commits steps and never waits per token | A CPU wait leaves the GPU idle ≈ 105 µs per token on the M2 Pro (R6 B3) |
| Output | A CPU thread detokenizes on UTF-8 boundaries | Qwen3 tokens can end inside a code point (R9 §4.2) |
| Plans | `sky run` caches the compiled plan as validated data only (kernel choices, parameters, offsets); no MSL and no binaries. `sky compile` writes an explicit `.skybin` build artifact that may also carry MSL and a Metal binary archive | Spec non-negotiable 3: caches are data-only (D19). Removes compile work from TTFT (Part XVIII) |

## 2. The end-to-end flow

```text
 sky run hf://mlx-community/Qwen3-32B-4bit "Explain RoPE."
   │
   │ 1 RESOLVE          sky-hub: HubUri → Snapshot{commit, files[path, size, sha256]}
   │                    offline first · shared HF cache · flock · provenance record
   │
   │ 2 PLAN LOOKUP      an explicit .skybin (sky compile output) if given, else the plan cache
   │                    key = (model hashes, Sky build, GPU, run config)
   │      hit ─────────► plan as validated data: kernel choices, parameters, offsets, memory plan
   │                     → MSL regenerated deterministically → pipelines from Metal's OS shader
   │                       cache (a .skybin may instead carry a Metal binary archive)
   │      miss ─► 3 NORMALIZE   config.json + generation_config + tokenizer_config → ModelSpec
   │              4 BUILD       sky-llm::arch::<family>::build(&ModelSpec) → TIR graph + parameter table
   │              5 COMPILE     sky-tir → sky-kernel → sky-metal (Parts VII–XI, XIX) → plan (cached as data)
   │
   ├──────────── in parallel ─────────────────────────────────────────────────────────────┐
   │ 6 BIND WEIGHTS     MLX safetensors → .skyweights (first run only) → mmap → no-copy     │
   │                    MTLBuffers, one region per layer group; GGUF → mmap directly       │
   │                    → residency sets → requestResidency                               │
   │ 7 TEXT ASSETS      tokenizer.json → Tokenizer · template → ChatTemplate · StopSet     │
   ├──────────────────────────────────────────────────────────────────────────────────────┘
   │
   │ 8 PROMPT           messages → render template → ids (one-BOS rule) → prefix-cache lookup
   │
   │ 9 PREFILL          chunk programs over uncached ids; the last chunk also computes the
   │                    logits of the last position and samples token 1 on the GPU
   │
   │10 DECODE           static step program, pre-committed D steps ahead; per step:
   │                    layers → lm_head → sampler → token ring, pos + 1, stop flag
   │                    MTLSharedEvent(value = step) ──────────────────────────────┐
   │                                                                               ▼
   │11 STREAM           CPU thread: token ring → StreamDetokenizer (UTF-8 boundaries) → stdout / SSE
   │                    text-level stop strings and cancellation feed back as a host flag
```

Steps 6 and 7 run on separate threads while step 2 loads pipelines, so cold start costs the maximum of the three, not the sum ([Part XVIII](part18-ttft.md) §4).

## 3. `sky-hub`: names, cache, downloads, pinning

### 3.1 The `hf://` grammar

```text
hub_uri   := "hf://" [ repo_type "/" ] repo_id [ "@" revision ] [ selector ]
repo_type := "models"                      ; "datasets" and "spaces" are rejected with a message
repo_id   := org "/" name
revision  := branch | tag | commit40 | commit_short | "refs/pr/" digits | pct_encoded
selector  := ":" QUANT                     ; GGUF repos only; never together with a path
           | "/" path                      ; one file inside the repo
QUANT     := [A-Za-z0-9_]+                 ; matched as the regex  QUANT[.-]  against GGUF file names
```

Rules, each chosen to remove an ambiguity:
- The grammar is huggingface_hub's `hf://[<type>/]<id>[@<rev>][/<path>]` (`utils/_hf_uris.py:259-319`) plus a llama.cpp-style `:QUANT`. Git refs cannot contain `:`, so `:QUANT` is unambiguous. `[LIKELY: R9 §5.5]`
- A revision may contain `/` only in the form `refs/pr/<n>`. Any other revision with `/` (a branch `feature/x`) must be written %-encoded (`@feature%2Fx`); otherwise `@feature/x/config.json` could be a revision or a path. The error message prints the encoded form.
- `:QUANT` must select exactly one GGUF file, or one split set (`…-00001-of-0000N.gguf`). Zero or several matches are an error that lists the candidates. llama.cpp instead falls back to Q4_K_M, then Q8_0, then the first GGUF (R9 §5.4); a silent fallback changes the bytes per token, so Sky does not copy it.
- `:QUANT` on an MLX repo is an error ("this repo is already one quantization; choose the repo").
- An argument that starts with `hf://` is always a hub name. A local path with that spelling is written `./hf://…`.
- Benchmark mode (`sky bench llm`, `--pinned`) requires a 40-hex `@rev` and an exact file name, because `main` moves (R9 §5.4: llama.cpp `-hf` always resolves `main`; mlx-lm's CLIs have no `--revision`).

Examples:

| Name | Meaning |
|---|---|
| `hf://mlx-community/Qwen3-32B-4bit` | latest locally known commit of `main`; offline if the snapshot is complete |
| `hf://mlx-community/Qwen3-32B-4bit@<40hex>` | exactly that commit |
| `hf://bartowski/Qwen_Qwen3-32B-GGUF:Q4_0` | the one file matching `Q4_0[.-]` |
| `hf://Qwen/Qwen3-32B@refs/pr/12` | an open pull request's head |
| `hf://org/model/model-00001-of-00004.safetensors` | one file (for `sky pull`; `sky run` needs a whole model) |

Types (`crates/sky-hub/src/uri.rs`):

```rust
pub struct HubUri { pub repo_id: RepoId, pub revision: Option<Revision>, pub selector: Selector }
pub enum Revision { Commit([u8; 20]), Short(String), Ref(String), PullRequest(u32) }
pub enum Selector { Model, Quant(String), Path(RepoPath) }   // RepoPath rejects "..", absolute, empty segments
pub fn parse(s: &str) -> Result<HubUri, UriError>;           // never panics; UriError names the byte offset
```

### 3.2 Where the cache is

Sky reads and writes the standard Hugging Face hub cache, in the same order Python does (`constants.py`): `HF_HUB_CACHE`, then `HUGGINGFACE_HUB_CACHE`, then `$HF_HOME/hub`, then `~/.cache/huggingface/hub`. It does not invent a `SKY_*` variable for the hub cache (R9 §5.5).

Layout `[VERIFIED: R9 §5.4]`:

```text
hub/
  models--{org}--{name}/
    refs/<branch>                       40-hex commit
    blobs/<etag>                        payload (sha256 for LFS/Xet files, git SHA-1 for plain files)
                                        — or, since huggingface_hub 1.32, a symlink into the shared store
    snapshots/<commit>/<path>           symlink → ../../blobs/<etag>
    trees/<commit>.json                 tree cache (size, blob_id, lfs_sha256, xet_hash)
    .no_exist/<commit>/<path>
  blobs/<xet[:2]>/<xet_hash>            shared Xet store: read-only (0444), keyed by Xet hash, not sha256
  blobs/.huggingface-shared-blobs       marker
  .locks/models--{org}--{name}/<etag>.lock
```

Sky's own derived files (relayout caches, plan caches, prompt caches; all in data-only formats, D19) live under one root, `~/Library/Caches/sky/` (override `SKY_CACHE_DIR`), next to the existing tuning database `~/Library/Caches/sky/tune.db` (R1 §5.7). R9 §1.3 suggested `~/.cache/sky/`; we keep the location Sky already uses, so there is one Sky cache root. Large directories are named `*.noindex` so Spotlight skips them `[LIKELY: R9 §6.5]`.

**The hub cache is an external store of symlinks.** Every `snapshots/<commit>/<path>` is a symlink into `blobs/`, which may itself be a symlink into the shared store. `sky-project` refuses symlinks for project imports (spec, "Imports and the project root"); that rule governs Sky source, not this store (D19). `sky-hub` reads the cache as an external store with its own rules: it resolves each link with `realpath`, requires the target to stay inside the hub cache root, opens the target read-only by handle, and checks its size (and, on first use, its hash) against the snapshot record. A link that leaves the root is an error.

### 3.3 Offline-first resolution

`sky_hub::resolve(&HubUri, &ResolveOptions) -> Result<Snapshot, HubError>`:

| Input | Network calls | Rule |
|---|---|---|
| 40-hex commit, snapshot complete | 0 | Complete means every file of the required set (§3.4) exists under `snapshots/<commit>/` and its size matches Sky's lock record or `trees/<commit>.json` |
| Branch or tag, `--offline` or `HF_HUB_OFFLINE=1` | 0 | Read `refs/<rev>`; fail if absent |
| Branch or tag, no flag, snapshot of `refs/<rev>` complete | 0 | **Default.** Use the local commit and print it. `--update` or `sky pull` checks the network |
| Anything else | 1 | `GET /api/models/{repo}/revision/{rev}?blobs=true` returns the commit and every sibling's size and `lfs.sha256` (R9 §5.1) |

The third row is a deliberate difference from huggingface_hub, which resolves the revision over the network whenever it is online `[LIKELY: snapshot_download resolves the revision through the API unless offline; R9 §5.4]`. A user who wants the newest `main` asks for it. This removes the network from cold start and makes runs reproducible by default.

The required file set (R9 §5.5):
- **Safetensors repos:** `config.json`; `generation_config.json` if the repo has it; `tokenizer.json` (else `tokenizer.model`); `tokenizer_config.json`; `special_tokens_map.json`; `chat_template.jinja` or `chat_template.json` if present; and exactly the shards listed in `model.safetensors.index.json` (or the single `model.safetensors`).
- **GGUF repos:** the selected quant and all its split parts. The tokenizer and template come from GGUF metadata unless the user adds `--tokenizer hf://<base repo>`.
- Never: `vocab.json`, `merges.txt` and other slow-tokenizer files. mlx-community repos ship them alongside `tokenizer.json` `[VERIFIED: listing of the local snapshot of mlx-community/Qwen3-8B-4bit]`.

### 3.4 Downloads and locking

The download path (`crates/sky-hub/src/download.rs`) follows R9 §5.5 step 5 exactly:

1. Take `flock(2)` on `.locks/models--{org}--{name}/<etag>.lock`. That is the lock file Python's `filelock` takes, so Sky and Python clients exclude each other `[LIKELY: filelock's Unix lock is fcntl.flock; test XIII-T2 checks it against a concurrent hf_hub_download]`.
2. If `blobs/<etag>` exists as a file or a symlink, reuse it.
3. Otherwise download to `blobs/<etag>.incomplete` with resumable, parallel HTTP Range requests. On the Xet bridge, 8 parallel ranges ran at 45.4 MB/s against 23.7 MB/s for one stream and 29.6 MB/s for hf_xet `[EXPERIMENTAL: one 409 MB sample on one link, R9 §5.3]`.
4. Verify the sha256 (LFS/Xet) or git SHA-1 (plain files) against `X-Linked-Etag`.
5. Rename atomically, then create the `snapshots/<commit>/<path>` symlink and, if resolved from a branch, write `refs/<branch>`.
6. **Never write `hub/blobs/` (the shared store).** Its payloads are read-only and shared across repos; Sky maps them `PROT_READ`. Sky writes per-repo blobs, as llama.cpp and the hf-hub crate do (no dedup, no corruption; R9 §5.4).

No Xet client in v1. The `hf-hub` 1.0.0 crate hard-depends on hf-xet and brings 356 packages (R9 §4.4, §5.3). Sky uses a small blocking HTTPS client and verifies hashes itself.

Security rules (R9 §5.2, §5.5):
- Token sources: `HF_TOKEN`, then `HUGGING_FACE_HUB_TOKEN`, then `HF_TOKEN_PATH` (default `$HF_HOME/token`). OIDC and Colab sources are not supported in v1.
- The token is never logged and is stripped on cross-host redirects (`file_download.py:1817-1821`).
- Object ids must be 64 hex characters (gated repos return 64 `*` characters).
- Repo paths may not be absolute or contain `..` (`hf-cache.cpp:164-173`).
- Error mapping: 401 with `X-Error-Code: GatedRepo` → "accept the license on the Hub and set HF_TOKEN"; 401 otherwise → "no such repo, or it is private" (the two are indistinguishable); 404 `RevisionNotFound` / `EntryNotFound` → the missing name.

Unauthenticated rate limits are 3,000 resolves and 500 API calls per 300 s (R9 §5.1). Sky makes at most one API call per resolution.

### 3.5 Pinning and provenance

Every `Snapshot` carries a `Provenance` (R9 §5.5 item 7, §8):

```rust
pub struct Provenance {
    pub repo_id: String, pub requested_revision: Option<String>, pub resolved_commit: [u8; 20],
    pub endpoint: String, pub offline: bool,
    pub files: Vec<FileRecord>,           // path, size, sha256 or git_sha1, xet_hash
}
```

- `sky run --lock` and `sky pull` write `sky-models.lock` (JSON, one entry per model name) next to `sky.toml` when run inside a Sky project, or under `~/Library/Caches/sky/locks/` otherwise. A later `sky run` of the same name uses the locked commit and verifies sizes; `sky verify` re-hashes through symlinks.
- Hashes of shared-store files are computed through the symlinks (`stat -L`); `du` on a repo directory reports 4.3 MB for a 4.6 GB model and must never be used for sizes (R9 §5.4).
- The `.skybin` manifest (§13) and every `sky bench llm` result JSON embed the provenance.

## 4. Config normalization (`sky-llm::config`)

### 4.1 The problem

| Drift | Effect elsewhere | Source |
|---|---|---|
| transformers 5 moves `rope_theta` and `rope_scaling` into `rope_parameters`, renames `torch_dtype` to `dtype`, adds `layer_types`, and nests Gemma 3's RoPE per layer type | mlx-lm drops unknown keys without a warning (`models/base.py:13-21`): a re-saved Llama 3.1 config loads with θ = 1e4 and no llama3 scaling; a Gemma 3 global RoPE scale becomes 1.0 instead of 0.125. Only 16 of 135 mlx-lm model files read `rope_parameters` | `[VERIFIED: R9 §3.1; R9-work/arch/R9-arch.md §2]` |
| `type` vs `rope_type` | both occur (DeepSeek-V3 and Phi-4-mini use `type`) | `[VERIFIED: R9-arch §2]` |
| Inert keys | Qwen's `sliding_window` does nothing unless `use_sliding_window` is true (R1-Distill-Qwen-7B ships 4096); Phi-4-mini ships `sliding_window` 262144 | `[VERIFIED: R9 §3.1]` |
| Missing keys | mlx-community's Gemma-3-27B text config has 10 keys; defaults live in transformers' config classes; mlx-lm's llama/qwen2 default to tied embeddings where HF's `LlamaConfig` defaults to untied | `[VERIFIED: R9 §3.1]` |
| The same fact in two places | Qwen3.6/3.8-27B carry `partial_rotary_factor` both inside `rope_parameters` and at the top of `text_config`; `tie_word_embeddings` appears at the wrapper level and in `text_config` | `[VERIFIED: configs in R8-work/configs/]` |
| New keys per release | Qwen3.6 adds `output_gate_type: "swish"` to the Qwen3.5 schema | `[VERIFIED: same configs]` |

### 4.2 The normalizer

`sky_llm::config::normalize(raw: &RawConfigs) -> Result<(ModelSpec, NormalizationReport), ConfigError>`, where `RawConfigs` holds the parsed `config.json`, `generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json` and, for GGUF, the metadata key/values.

Steps, in order:
1. **Unwrap multimodal wrappers.** If `text_config` exists (Gemma 3, Mistral Small 3.1, Qwen3.5/3.6/3.8), take it. A key present both at the wrapper level and in `text_config` must have the same value, or normalization fails. `vision_config` and vision token ids are recorded as "skipped: multimodal is out of scope in year 1" (spine §1).
2. **Pick the family** from `model_type` and `architectures`. An unknown family is an error listing the supported ones (§5.2).
3. **RoPE.** Read the legacy schema (`rope_theta`, `rope_scaling{type | rope_type, …}`, `partial_rotary_factor`, `rope_interleave`) and the transformers-5 schema (`rope_parameters`, flat or keyed by layer type) into one `RopeSpec` per layer type. If both schemas are present they must agree.
4. **Layer types.** Take `layer_types` if present; otherwise derive them from the family rule (Gemma 3: layer *i* is global when (i+1) mod 6 = 0; gpt-oss: alternating sliding/full, sliding first; Qwen3.5: `full_attention_interval`). If both exist they must agree.
5. **Sliding windows** are applied only where the family's semantics apply them; inert keys produce a `CFG-002` note.
6. **Defaults.** A missing key with a default in the pinned transformers config class (5.18.0) gets it and a `CFG-001` note naming the class and version. A missing key with no default is an error.
7. **Unknown keys** are an error, `CFG-003`, naming the key, its value and the family. Each family declares the keys it consumes and a short list of ignorable keys with a reason (`initializer_range`, `attention_dropout`, `use_cache`, `transformers_version`, `router_aux_loss_coef`, `output_router_logits`, `unsloth_fixed`, …). `pretraining_tp` is accepted only with value 1. `--allow-config-key=<k>` accepts one more key and records it in the provenance.
8. **dtype**: `torch_dtype` and `dtype` must agree when both exist.
9. **Quantization**: `quantization` and its `quantization_config` copy must be identical (mlx-lm keeps the copy "for the HF model tree", R9 §1.4); per-module overrides are carried to weight binding (§6.3). Legacy `quant_method` values are accepted for `mxfp4` (gpt-oss) only; `awq`, `gptq`, `bitnet` and `compressed-tensors` are errors in year 1.
10. **Stop and start tokens** become a `StopSet` and a BOS policy (§8.4).

```rust
pub struct ModelSpec {
    pub family: Family, pub dtype: DType, pub hidden: u32, pub layers: u32,
    pub layer_types: Vec<LayerType>,            // Full, Sliding{window}, LinearDeltaNet, ...
    pub attn: AttnSpec,                          // n_q, n_kv, head_dim, scale, qk_norm, biases, sinks, softcap
    pub rope: Vec<RopeSpec>,                     // indexed by LayerType
    pub mlp: MlpSpec,                            // SwiGLU | GeGLUTanh | SwiGLUOai{alpha, limit}; fused layout
    pub moe: Option<MoeSpec>,                    // experts, top_k, routing rule, shared experts, biases
    pub norm: NormSpec,                          // eps, (1+w), sandwich
    pub embed: EmbedSpec,                        // vocab rows, tied, scale (Gemma: sqrt(hidden) rounded to bf16)
    pub max_position: u32, pub quant: QuantSpec, pub stops: StopSet, pub bos: BosPolicy,
}
pub struct RopeSpec { pub theta: f64, pub kind: RopeKind, pub partial: f32, pub interleaved: bool }
pub enum RopeKind {
    Default, Linear{factor: f64},
    Llama3{factor: f64, low: f64, high: f64, original: u32},
    Yarn{factor: f64, original: u32, beta_fast: f64, beta_slow: f64, attn_factor: f64, truncate: bool},
    LongRope{short: Vec<f64>, long: Vec<f64>, original: u32, attn_factor: f64, select: LongRopeSelect},
}
```

Where runtimes disagree with HF numerically, `ModelSpec` follows HF, and the choice is visible in `sky explain --config`:

| Case | HF | mlx-lm / llama.cpp | Sky |
|---|---|---|---|
| gpt-oss YaRN `truncate=false` | honored | both floor/ceil anyway (`rope_utils.py:150-153`; `ggml.c:4482-4490`) | honored; correction range [8.09, 17.40], not [8, 18] |
| Phi LongRoPE short/long choice | by current length | llama.cpp by `n_ctx`; mlx-lm always `long` | by current length (`LongRopeSelect::CurrentLength`); `--longrope-select=long` reproduces mlx-lm |
| Gemma 3 embedding scale | √hidden rounded to bf16 (73.32 → 73.5) | mlx-lm same; llama.cpp fp32 | bf16-rounded |
| Gemma 3 attention scale | `query_pre_attn_scalar`^-½ | llama.cpp hard-codes by layer count | from the key |
| Qwen3 YaRN | opt-in in config | — | never enabled implicitly; `--context` above `max_position_embeddings` is an error that names the YaRN option, because the model card says static YaRN hurts short texts (R9-arch §2) |

All rows are `[VERIFIED: R9 §3.1, R9-arch §2]`.

## 5. Architecture builders (`sky-llm::arch`)

### 5.1 The builder interface

```rust
pub trait ArchBuilder {
    fn family(&self) -> Family;
    fn consumed_config_keys(&self) -> &'static [&'static str];
    fn ignorable_config_keys(&self) -> &'static [(&'static str, &'static str)];   // (key, reason)
    fn parameter_names(&self, spec: &ModelSpec) -> Vec<ParamDecl>;               // HF name → TIR param, logical shape
    fn build(&self, spec: &ModelSpec, g: &mut TirBuilder) -> Result<ModelGraph, BuildError>;
}
```

- `build` emits the L1 TIR graph of one forward pass with symbolic token count `T` and context `C`, plus the state values (KV, DeltaNet state) as typed TIR state ([Part VII](part07-tensor-ir.md)). Decode, verify and prefill programs are specializations of this one graph ([Part VIII](part08-graph-optimizer.md)).
- Layers are emitted once as a TIR function called `layers` times. All layers of a model share shapes, so the scheduler and autotuner work on one block (D11).
- `parameter_names` drives weight binding (§6). Every checkpoint tensor must be consumed exactly once, or appear on the builder's explicit skip list with a reason (`mtp.*` when MTP is off; vision towers; legacy `rotary_emb.inv_freq`). An unconsumed tensor is an error, so a renamed tensor cannot be ignored silently.
- D7's second front door (`models/qwen3.sky`) must produce byte-identical L1 text for the same `ModelSpec` (test XIII-T5).

### 5.2 Support matrix

Tiers follow R9 §3.2. Gates are from the spine (§6).

| Family | Checkpoints | Builder | What it adds beyond the Qwen3 tier 0 | Gate |
|---|---|---|---|---|
| **Qwen3 dense** | 0.6B–32B (MLX 4/8-bit, GGUF Q4_0/Q4_K_M/Q8_0, bf16) | `arch/qwen3.rs` | Tier 0 itself: per-head QK RMSNorm before RoPE, decoupled head_dim (32B: 64 × 128 = 8192 vs hidden 5120), GQA 8 (32B) / 4 (8B), NeoX RoPE θ 1e6 with fp32 angles, untied head at 8B/32B, tied at ≤ 4B | G2 (0.6B logits), G3 (8B tokens), G5 (32B) |
| Qwen2.5, R1-Distill-Qwen | 0.5B–32B | `arch/qwen2.rs` (shares code with qwen3) | q/k/v bias (hard-coded on, no config key); GQA 5 and 7; θ 1e4 for R1-Distill-Qwen-7B; eps 1e-5 for R1-32B; BOS 151646 in tokenizer and generation_config vs 151643 in config | G7 |
| **Qwen3-MoE** | 30B-A3B, 235B-A22B | `arch/qwen3_moe.rs` | 128 experts, top-8, no shared expert; softmax in fp32, then top-k, then renormalize (`norm_topk_prob`); experts stored unstacked (18,867 tensors for 30B-A3B) and stacked to `[E, out, in]` by the relayout | G7 |
| **Qwen3.5 / 3.6 / 3.8 hybrid** (D10's second headline) | 27B dense; 35B-A3B MoE | `arch/qwen3_5.rs` | 48 Gated DeltaNet layers (16 QK heads × 128, 48 V heads × 128, conv kernel 4, fp32 recurrent state ≈ 151 MB); 16 gated full-attention layers (24 Q / 4 KV heads, head_dim 256, output gate, partial RoPE 0.25, θ 1e7); interleaved M-RoPE sections [11, 11, 10]; one MTP layer (`mtp.fc` [5120, 10240]) sharing embedding and head; vocab 248,320; multimodal wrapper | G7, first |
| Llama 3.x | 3.1-8B/70B, 3.2-1B/3B, 3.3-70B | `arch/llama.rs` | llama3 RoPE remap (×8 or ×32); tied embeddings at 3.2; `attention_bias` also gates the o_proj bias; GQA 3 (3.2-3B); head_dim 64 (3.2-1B) | G7 |
| Mistral family | 7B v0.3, Ministral 8B, Mistral Small 3.x | `arch/llama.rs` (mistral variant) | θ up to 1e9 (fp32 angles needed); `layer_types` with a 32,768 window that is inert up to 32K; Mistral Small 3.1's text config lacks the tie key (mlx-lm forces untied) | G7 |
| Gemma 3 | 1B, 4B, 27B | `arch/gemma3.rs` | sandwich norms with (1+w) RMSNorm (the GGUF converter pre-adds the 1); GeGLU-tanh; embedding scale √hidden rounded to bf16; `query_pre_attn_scalar` (168 at 27B); 5 local : 1 global, windows 512/1024, θ 1e4 local and 1e6 + linear ×8 global; MQA at 1B; weights under `language_model.model.*`; no `lm_head`; run in bf16 (mlx-lm clips the residual in fp16) | G7 |
| Phi-4 | 14B; Phi-4-mini 3.8B | `arch/phi3.rs` | fused `qkv_proj` and `gate_up_proj` split in halves; LongRoPE with 48 short/long factors and attention factor 1.190 (mini); partial rotary 0.75 (mini) | G7 |
| gpt-oss | 20b, 120b | `arch/gpt_oss.rs` | MXFP4 experts (`*_blocks` U8 `[E, out, in/32, 16]`, `*_scales` U8 E8M0, low nibble = even element); learned per-head attention sinks; alternating sliding-128 / full layers; YaRN ×32 with `truncate=false`; clamped SwiGLU-OAI (α = 1.702, limit 7, "+1" on up); `gate_up` interleaved even/odd; q/k/v/o, router and expert biases; top-4 of 32 or 128 experts with softmax over the top-k; harmony chat format | G7 |
| DeepSeek-V3 / R1 | 671B | none in year 1 | MLA (a 576-value latent cache, 1,152 B per token per layer), interleaved decoupled RoPE on 64 of 192 dims, YaRN ×40 with softmax scale ×1.874, sigmoid routing with `e_score_correction_bias`, 8 groups → top-4 groups, 1 shared expert, FP8 e4m3 with 128×128 block scales, an MTP layer that mlx-lm drops | not planned |

All family facts: `[VERIFIED: R9 §3 and R9-arch §2; Qwen3.5/3.6/3.8 from R8 §1.7 and the configs in R8-work/configs/]`.

Notes on the matrix:
- **DeepSeek.** 671 B parameters at 4.5 bits per weight is about 377 GB (arithmetic). It fits only on a 512 GB M5 Ultra (R6), not on any M5 Max. Its MLA, FP8 and group routing are tier 2. The R1-Distill models run on the qwen2 and llama builders. mlx-lm also ships DeepSeek V3.2/V4.1 sparse-attention variants (R9 §3.2), which we have not studied.
- **Qwen3.5 M-RoPE.** For text-only input all three position components are equal, so interleaved M-RoPE should reduce to ordinary RoPE on the rotated 64 of 256 dimensions with a permuted frequency assignment `[LIKELY: the three sections share one position id per token]`. Test XIII-T20 checks it against transformers before any other work on that builder.
- **MTP and DeltaNet in other runtimes.** mlx-lm drops MTP weights at load (`qwen3_5.py:350`) and cannot speculate on hybrid models because their caches are not trimmable (mlx-lm#1446) `[VERIFIED: R4 §5.1]`. Sky keeps the MTP layer and checkpoints the recurrent state per draft position ([Part XXI](part21-speculative.md); R8 §3.5).

### 5.3 Correctness gates per builder

Each builder ships when it passes, on its smallest model:
1. the tokenizer gate (§7.3);
2. teacher-forced logits against HF transformers in fp32 on the reference interpreter (G2 style), then on Metal;
3. on identical weights, KLD(Sky ‖ other engine) ≤ 0.1 × KLD(other engine ‖ BF16) and ≥ 99% top-1 agreement (D13);
4. a 128-token greedy smoke test where every divergence is a near-tie (top-2 margin < 0.1) (R9 §7.3 G3).

## 6. Weight binding (`sky-llm::load`, `sky-quant`)

### 6.1 Safetensors as written by MLX

| Fact | Value | Label |
|---|---|---|
| Header | 8-byte length N, then N bytes of JSON; limit 100 MB | `[VERIFIED: R9 §1.1]` |
| MLX writer | no header padding; tensors in `unordered_map` order; data order ≠ header order | `[VERIFIED: mlx/io/safetensors.cpp:220-275; R9 §1.2]` |
| mlx-community/Qwen3-8B-4bit | header 103,454 B; data starts at 103,462 ≡ 6 (mod 8); **0 of 907 tensors 4-byte aligned**; relative offsets all 16-aligned | `[VERIFIED: R9 §1.2]` |
| mlx-community/Qwen3-0.6B-4bit | data starts at 78,296 ≡ 8 (mod 16), aligned only by luck | `[VERIFIED: R9 §1.2]` |
| HF writer | header padded to 8 B; tensors sorted by dtype size; at most 8-byte alignment | `[VERIFIED: tensor.rs:218-262; R9 §1.1]` |
| GPU behaviour | a `uint` bound at offset 2 or 6 silently returns the word at the rounded-down address; only `MTL_DEBUG_LAYER=1` catches it | `[VERIFIED on M2 Pro: R9 §1.3]` |
| Metal rules | `bytesNoCopy` needs a page-aligned pointer and length (16 KiB pages) within one VM region; a `device` buffer offset must be aligned to the element size | `[VERIFIED: Apple docs, R9 §1.3, R6 A4]` |

So mlx-community files cannot be used zero-copy. MLX itself copies: `Load::eval_cpu` allocates a new Metal buffer and `pread`s into it on 32 MiB batches (`mlx/backend/common/load.cpp:30-55`; R3 §4). Its maintainers declined an mmap path (mlx#4249: "At the moment we are not considering using mmap"; R3-issues G3).

### 6.2 The relayout cache (`.skyweights`, D9)

On first use of an MLX checkpoint, Sky writes a Sky-native file once and maps it on every later run. The file format and its proof obligations are owned by [Part XII](part12-quantization.md) §5; this section states what the runtime needs from it.

```text
~/Library/Caches/sky/weights/<key>.skyweights             (SKY_WEIGHTS_CACHE overrides the directory; Part XII §5)
  key = sha256( sorted source shard sha256s ‖ relayout recipe id ‖ .skyweights format version )

<key>.skyweights
  [0, 16 KiB)        header: magic "SKYW", version, provenance (repo, commit, shard sha256s),
                     layout-id, Sky version, region table, tensor table
  region 0           embedding (+ its scales/biases)                    16 KiB-aligned
  region 1..G        layer groups (layers grouped so that no group exceeds 5% of the GPU working set)
  region G+1         final norm + lm_head
  every tensor       ≥ 256-byte aligned inside its region (128-byte minimum for Metal tensor operands, R6 A3)
```

Rules:
- **Lossless, never re-quantized (D5).** The relayout may concatenate rows (q/k/v into one QKV matrix; gate and up interleaved for a fused SwiGLU epilogue), split fused tensors (Phi halves, gpt-oss even/odd), stack experts, and move scales and biases next to the weights a kernel reads together. All of these are permutations of bytes or nibbles, so they are invertible. MLX groups run along the input dimension, so concatenating output rows never splits a group.
- **The proof is a test, not a claim.** `sky-quant` test `relayout_roundtrip_bit_exact`: for every tensor, `unrelayout(relayout(bytes)) == bytes`, and the f32 dequantization of the original and the relayout are bitwise equal on the reference interpreter. `sky verify --weights` runs it on a real model.
- **Build cost.** One full read and one full write: 18.43 GB each for Qwen3-32B. The read is verified against the shard sha256 as it streams, so the hash costs no extra I/O. The source sha256 is normally free: per-repo blobs are named by their sha256 etag (R9 §5.4).
- **Atomic and shared.** Built to `<key>.skyweights.partial` under `flock`, fsynced, renamed. Several processes map the same file.
- **Disk cost.** A second copy: about 18.4 GB for Qwen3-32B (R9 §6.7 budgets it). `sky cache ls` and `sky cache gc --keep 50GB` manage it; the cache is never cleaned silently.
- **First-run strategy** is decided by measurement ([Part XVIII](part18-ttft.md) §4.3): either build before the first token, or load the first run through an aligned in-memory copy (as MLX does) and write the file in the background.
- **Integrity on load** checks the header, the file size and the source key, not a full re-hash, which would cost a full read at every start.

GGUF needs none of this. Its tensors are 32-byte aligned relative to the data section (`ggml/src/gguf.cpp:622-636, 780-815`), which suffices for `uint4` loads, and Sky maps it the way llama.cpp does: page-align the region down and up and wrap it with `newBufferWithBytesNoCopy`; split into views no larger than `maxBufferLength` (8.88 GiB on the 16 GB M2 Pro; R6 A4) `[VERIFIED: ggml-metal-device.m:2176-2270; R5 §1.5]`. Sky's views follow layer boundaries rather than fixed sizes, so residency can be managed per layer group ([Part XIX](part19-memory.md) §7).

### 6.3 MLX quantization: layout and inference rules

`[VERIFIED: mlx/ops.cpp:5122-5233, mlx_lm/utils.py:380-560; checked on real data in R9 §1.4]`

- `w = scale · q + bias`; eight 4-bit values per little-endian `uint32`, element *j* in bits [4j, 4j+4); group 64 along the input dimension; scales and biases in the weight dtype (bf16 here).
- **50.25% of Qwen3-8B-4bit's 128 M scales are negative.** The quantizer pins the larger-magnitude edge of each group exactly, so when the scale is negative the bias is the group *maximum*. Kernels and tests must not assume a positive scale or bias = minimum. Test `negative_scale_groups` builds synthetic groups of both signs.
- Tensor names: `p.weight` U32 `[out, in·bits/32]`, `p.scales` and `p.biases` `[out, in/group]`. A Linear's own bias is `p.bias`. Both names occur in one file (gpt-oss; Qwen2.5 4-bit q_proj), and the binder never confuses them.
- Modes: `affine` (bits 2–8, group 32/64/128), `mxfp4` (group 32, E8M0 scales, no biases), `nvfp4` (group 16, E4M3 scales), `mxfp8`.
- Parameters per module, in order:
  1. a per-module override: a key equal to the module path holding a dict or `false` (`"model.layers.3.mlp.down_proj": {"group_size": 64, "bits": 6}`);
  2. the global `quantization` dict (`mode` absent means affine);
  3. for modules with `.scales` but no entry, inference from shapes exactly as `infer_quant_config` does: group = `in_dims / scales.shape[-1]`, bits = `weight.shape[-1] · 32 / in_dims`; mode affine unless the scales are `uint8`, in which case (4, 16) is nvfp4, (4, 32) mxfp4 and (8, 32) mxfp8.
  4. anything ambiguous is an error naming the tensor.
- MLX mxfp4 and GGUF MXFP4 differ in layout (separate scale tensor and sequential nibbles vs E8M0 interleaved with 16 bytes of split-half nibbles) and convert losslessly (R9 §1.4).

### 6.4 GGUF specifics

`[VERIFIED: R9 §2]`

- Sky writes its own parser (~200 lines), because the existing crates are weak or stale (R9 §4.4). It enforces what the reference reader enforces: magic, version 3, power-of-two `general.alignment` (default 32), each tensor's offset equal to the padded sum of the earlier sizes, no duplicate names, `ne[0]` a multiple of the block size.
- **Files labelled "Q4_0" are mixes.** bartowski's Qwen3-8B Q4_0 has 249 Q4_0, 4 Q4_1 (`ffn_down`, layers 0–3, from the imatrix) and a Q6_K `output.weight`; 4.67 bpw. Qwen's Q4_K_M is 4.90 bpw. The binder records the format of every tensor, and bytes per token are computed from the actual binding and printed with every speed (R9 §7.6).
- **Per-file layout flags.** llama.cpp's converter permutes q/k for Llama and Mistral and then runs NORM (interleaved) RoPE; Qwen, Phi, Gemma 3 and gpt-oss run NEOX (`conversion/llama.py:17-20, 176-181, 260-264`; `src/llama-model.cpp:3061-3186`). GGUF Gemma norms already include the +1 (`conversion/gemma.py:131-132`). The RoPE layout is a property of the file, not the model, so `WeightBinding` carries a `rope_layout` flag and the TIR graph uses it.
- **Tokenizer metadata.** `tokenizer.ggml.model = "gpt2"`, `tokenizer.ggml.pre = "qwen2"` selects a hard-coded regex in llama.cpp; `tokens` includes padding entries; one EOS only (§8.4).
- Split files (`-00001-of-0000N.gguf`) are bound as one model.

### 6.5 Binding

```rust
pub struct WeightBinding {
    pub tensors: Vec<BoundTensor>,      // TIR param id → (region, offset, nbytes, QFormat, physical layout)
    pub regions: Vec<MappedRegion>,     // file, page-aligned [start, end), MTLBuffer view, residency group
    pub rope_layout: RopeLayout,        // NeoX | Interleaved (GGUF Llama/Mistral)
    pub bytes_per_decode_token: u64,    // all linear weights + LM head + one embedding row
    pub bits_per_weight: f64,
}
```

- Tied checkpoints that also store `lm_head.weight` (the original Qwen3-0.6B and 1.7B do; R8 §1.6) bind one copy, after checking that the two are byte-identical. If they differ, binding fails.
- The embedding has more rows than the tokenizer has ids (Qwen3: 151,936 rows for 151,669 ids; 267 padding rows; R9 §4.1). The binder records the tokenizer's id count; the sampler masks the padding rows (§9.4).
- `bytes_per_decode_token` and `bits_per_weight` come from the binding, not from the config, and are printed next to every speed (D5). For mlx-community/Qwen3-32B-4bit the file total is 18,429,667,328 B and the bytes per decode token are 17.99 GB (R9 §2.3, `roofline.py`).

## 7. The tokenizer (`sky-llm::text::tokenizer`)

### 7.1 The crate

```toml
tokenizers = { version = "=0.23.2", default-features = false, features = ["unstable_wasm"] }
```

- With default features, `tokenizers` pulls C code (Oniguruma, esaxx) (R9 §4.4). The pure-Rust build selects fancy-regex. That is the configuration R9 tested: `default-features = false` plus the `unstable_wasm` feature (`R9-work/tok/rs/Cargo.toml`).
- Result: **168/168 identical** to the Python wheel (Oniguruma) on 6 tokenizers × 28 adversarial strings, built with cargo 1.84.1 `[VERIFIED: R9 §4.2]`. Sky pins Rust ≥ 1.85 for its other crates.
- The feature is named for WebAssembly. Before upgrading (1.0.0-rc.2 exists, 2026-09-21), re-run the gate and check whether a plainly named fancy-regex feature exists `[LIKELY: the naming is a packaging accident, not a semantic difference]`.
- The Qwen3 regex `\s+(?!\S)` needs a backtracking engine, so the default Rust `regex` crate cannot be used (R9 §4.1).

### 7.2 What the Qwen3 tokenizer is

`[VERIFIED: R9 §4.1]`
- Byte-level BPE, `byte_fallback=false`, `ignore_merges=false`; 151,643 vocabulary entries, 151,387 merges stored as `[a, b]` pairs (older Qwen2.5 files use `"a b"` strings; the parser accepts both).
- Normalizer NFC; pre-tokenizer `Split(regex, Isolated)` → `ByteLevel(add_prefix_space=false, use_regex=false)`; decoder ByteLevel; digits split one at a time.
- 26 added tokens (151643–151668): 14 `special=true` (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, vision tokens) and 12 `special=false` (`<tool_call>` 151657, `<think>` 151667, `</think>` 151668, fim tokens).
- `tokenizer.json` is 11,422,654 bytes for mlx-community/Qwen3-8B-4bit `[VERIFIED: listing of the local HF cache]`. It is parsed on a background thread at start (Part XVIII §4.4).

### 7.3 The golden-id gate

- **Fixtures** come from the raw Python `tokenizers` library (pinned 0.23.2), generated by `tools/gen_golden_ids.py` and committed as JSON with the sha256 of each `tokenizer.json`. They are never generated with transformers v5 `AutoTokenizer` for repos that declare `LlamaTokenizer(Fast)`: v5 aliases `LlamaTokenizerFast` to `LlamaTokenizer` and turns "The quick brown fox" into "Thequickbrownfox" (13 ids instead of 11) for DeepSeek-R1-Distill-Llama-8B and its mlx-community copy (`tokenization_llama.py:149`) `[VERIFIED: R9 §4.2]`.
- **Corpus:** R9's 28 strings (whitespace runs, digits, contractions, CJK and RTL, ZWJ emoji and flags, NFC and NFD forms, zero-width and control characters, code, URLs, literal special-token strings), plus every benchmark prompt.
- **Pass:** identical ids for every (tokenizer, string). It runs in CI for Qwen3, Llama 3.1, gpt-oss, Phi-4, Gemma 3 and Mistral (`crates/sky-llm/tests/tokenizer_golden.rs`). A nightly job adds a differential fuzz against Python on random Unicode.
- **llama.cpp as a second oracle** for GGUF-only models: expect it to differ on NFD input, because llama.cpp skips the NFC normalizer ("café naïve Å" in NFD: 5 ids in HF, 11 in llama.cpp; NFD Hangul 3 vs 8) and selects its regex by a fingerprint of the vocabulary rather than reading it (R9 §4.2).

### 7.4 Pitfalls the wrapper handles

| Pitfall | Rule in Sky | Source |
|---|---|---|
| NFD input | Normalize as `tokenizer.json` says (NFC for Qwen). Sky matches HF and differs from llama.cpp on NFD text; benchmarks exchange ids, so engines are compared on identical ids | R9 §4.2, §9 Q7 |
| `ignore_merges=true` (Llama 3.1) | Honored from the model config; 588 vocabulary tokens are reachable only with it | R9 §4.2 |
| Mistral-Small-3.2's Hub `tokenizer.json` carries the Llama-3 regex instead of Tekken's | Detect by family and refuse with a message; offer the corrected tokenizer as an explicit option (transformers' `fix_mistral_regex`) | R9 §4.1 |
| SentencePiece `.model` vs `tokenizer.json` disagree (Gemma 3: 0/28 differ; Mistral 7B v0.3: 4/28, all with leading whitespace) | Use `tokenizer.json` when present; `tokenizer.model` only if it is absent, with a warning | R9 §4.2 |
| `clean_up_tokenization_spaces` | Never applied; Sky compares and stores ids, not strings | R9 §4.2 |
| Tables rebuilt per request | The id→bytes table and the special-token table are built once at load. mlx-lm 0.28.0–0.31.3 rebuilt its detokenizer map per call, adding 51–77 ms to every TTFT; fixed at mlx-lm HEAD | R4 §2.5, §9; mlx-lm#1435 |

## 8. Chat templates and stop tokens (`sky-llm::text::template`)

### 8.1 Engine and configuration

```toml
minijinja = { version = "=2.24.0", features = ["json", "loop_controls", "preserve_order", "unicode"] }
minijinja-contrib = { version = "=2.24.0", features = ["pycompat"] }
```

The environment reproduces transformers' renderer (`chat_template_utils.py:487-500`):
- `trim_blocks` and `lstrip_blocks` on; `loop_controls`;
- unknown Python string and dict methods through `minijinja_contrib::pycompat::unknown_method_callback`. Without it, the Qwen3 template fails on `.startswith` and R1-Distill on `.split` (R9 §4.3);
- `raise_exception(msg)` and `strftime_now(fmt)` as globals;
- a **Python-exact `tojson`** filter replacing minijinja's: separators `", "` and `": "`, `ensure_ascii=False`, no HTML escaping, the `indent`, `separators`, `sort_keys` and `ensure_ascii` keyword arguments, and Python's float repr (`1e-05`, not `0.00001`; scientific form when the decimal exponent is below −4 or at least 16). minijinja's built-in `tojson` breaks 5 tool-calling cases (compact separators, escaped `<` and `&`) (R9 §4.3);
- `preserve_order`, without which map keys come out sorted;
- the `{% generation %}` / `{% endgeneration %}` tags, which mark assistant spans for training masks, are removed from the template source before parsing (minijinja does not support them; R9 §4.3).

Result with this configuration: **all 30 renderable cases byte-identical** to transformers (10 templates × 3 conversations, plus Qwen3 with thinking on and off; the 3 remaining cases raise identically in both engines) `[VERIFIED: R9 §4.3]`. Known residual difference: printing a whole list renders JSON in minijinja and a Python repr in jinja2. A template that does this is detected by a render self-test and reported.

### 8.2 Which template

Order of precedence `[VERIFIED: tokenization_utils_base.py:1784; R9 §4.3]`: `chat_template.jinja` (and named templates in `additional_chat_templates/`), then `chat_template.json`, then `tokenizer_config.json["chat_template"]`. For GGUF, `tokenizer.chat_template`; publishers patch these (the unsloth Qwen3-0.6B GGUF template is 4,905 characters against 4,614 in bartowski's 8B file, with one semantic change; R9 §2.4, §4.3), so Sky prints the template's sha256 and source in `sky explain --template`. gpt-oss has no template in `tokenizer_config.json` and uses the harmony format; the `openai-harmony` crate (0.0.8) is the test oracle for its rendering (R9 §4.4).

### 8.3 Variables that change the prompt

| Variable | Behaviour in other runtimes | Sky |
|---|---|---|
| **Date** | gpt-oss injects "Current date: <today>"; Mistral-Small-3.2's default system prompt calls `strftime_now`; Llama 3.1 hard-codes "26 Jul 2024" (R9 §4.3) | `strftime_now` formats the date given by `--template-date YYYY-MM-DD`, else today's local date. `sky bench llm` always pins it and records it (`chat_template.date_pinned`, R9 §8) |
| **Thinking (Qwen3)** | Undefined `enable_thinking` means thinking on; mlx-lm passes `enable_thinking=True` by default (`tokenizer_utils.py:393`); `enable_thinking=false` appends `<think>\n\n</think>\n\n` | Pass nothing unless `--think` / `--no-think` is given; then pass the variable only if the template's AST references it (otherwise a warning). The server accepts `chat_template_kwargs`. Benchmarks use non-thinking mode (R9 §7.1). The model card's advice not to use greedy decoding in thinking mode is printed when both are requested (R8 §1.1) |
| **R1-Distill** | the template forces thinking (ends with `<｜Assistant｜><think>\n`) | rendered as is; `--no-think` is an error for these templates |
| **Tools** | Qwen3, gpt-oss and Mistral templates render tool schemas; Gemma 3 rejects the tool role; Mistral needs 9-character tool-call ids | passed through; template errors are shown with the template's message |

### 8.4 BOS, EOS and special tokens in user text

**One BOS.** Encoding a rendered template with `add_special_tokens=True` gives `[128000, 128000, …]` for Llama 3.1, `[2, 2, …]` for Gemma 3 and `[1, 1, …]` for Mistral-Small-3.2; the raw `tokenizers` library ignores `tokenizer_config`'s `add_bos_token` (R9 §4.2). Sky's rule:
- the chat path encodes the rendered text with `add_special_tokens = false`;
- the raw path (`sky run --raw`) encodes with `add_special_tokens = true`, so the post-processor adds BOS where the model wants one;
- in both paths the result must start with exactly one BOS if the model uses one (Llama, Gemma, Mistral) and none otherwise (Qwen); two leading BOS ids are an error, not a warning.

**EOS is a union.** `StopSet = config.eos_token_id ∪ generation_config.eos_token_id ∪ tokenizer_config.eos_token ∪ GGUF eos/eot ids`. For the Qwen originals that is {151645, 151643}. `mlx-community/Qwen3-8B-4bit` has no `generation_config.json`, so the union is {151645}, which is also what mlx-lm stops on; Sky then prints that the base model also stops on 151643, rather than adding it silently (R9 §4.1). The GGUF carries a single EOS (R9 §2.4). Throughput runs ignore EOS in every engine (R9 §4.1).

**Special tokens typed by a user.** By default HF turns a typed `<|im_end|>` into control id 151645, which lets message content forge turns (R9 §4.2). llama.cpp's Jinja engine marks strings from message content so they cannot become special tokens (R9 §4.3). Sky:
- `sky run` with a local prompt keeps HF's behaviour, so the golden tests hold;
- `sky serve` (untrusted input) escapes content: before rendering, each occurrence of a `special=true` literal in message content is replaced by a private-use sentinel registered as an added token; after tokenization each sentinel's id is replaced by the ids of the literal encoded as plain text (`split_special_tokens=true`). `[EXPERIMENTAL: the escaping test corpus decides; templates that transform content with string methods are covered by rendering both ways and comparing structure]`.

## 9. Sampling (`sky-llm::sample`; GPU kernels in the skeleton library)

### 9.1 Configuration

```rust
pub struct SamplerConfig {
    pub temperature: f32,            // 0 = greedy
    pub top_k: u32, pub top_p: f32, pub min_p: f32,
    pub repetition_penalty: f32, pub repetition_window: u32,     // HF: whole sequence; mlx-lm: last 20
    pub presence_penalty: f32, pub frequency_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub seed: u64,
    pub order: SamplerOrder,         // Hf (default) | MlxLm | LlamaCpp
}
```

Defaults come from `generation_config.json` when present (Qwen3: `do_sample` true, temperature 0.6, top_k 20, top_p 0.95; Llama 3.1: 0.6 and top_p 0.9) `[VERIFIED: the generation configs in R9-work/arch/configs/]` and are overridden by flags or request fields.

**The order of filters differs between engines, and the order changes the distribution:**

| Engine | Order | Source |
|---|---|---|
| transformers | repetition penalty → temperature → top-k → top-p → min-p → sample | `[VERIFIED: transformers 5.18.0 generation/utils.py:1453, 1590-1604]` |
| mlx-lm | top-p → min-p → XTC → top-k on untempered log-probabilities, then `categorical(logits / T)` | `[VERIFIED: sample_utils.py:55-64, 272-273; R4 §2.4]` |
| llama.cpp | penalties → DRY → top-n-σ → top-k → typical → top-p → min-p → XTC → temperature → sample | `[VERIFIED: R5 §8 D]` |

Sky implements HF's order by default and the other two as compatibility profiles. Consequences:
- sampled outputs are never compared across engines; the speed protocol uses argmax (R9 §7.3 G4);
- "distribution-identical" (spine §3) is a claim about Sky's speculative path against Sky's plain path, which share one order.

### 9.2 On the GPU

Sampling runs inside the static decode program ([Part XVI](part16-decode.md)), after the LM head, on fp32 logits. Parameters live in a `SamplerParams` buffer, so a request with new parameters does not rebuild the plan.

| Stage | Kernel design | Determinism |
|---|---|---|
| penalties | a GPU-resident `u32` count per vocabulary id and a ring of the last `repetition_window` ids, updated by the sampler when it emits a token; repetition is sign-aware (divide positive logits, multiply negative ones), presence and frequency are additive | integer state; exact |
| temperature | multiply by 1/T in fp32 | exact |
| top-k (k ≤ 1024) | each threadgroup keeps its local top-k (ties by lower id); one threadgroup merges G·k candidates | fixed merge order |
| top-p | on the top-k candidates when k > 0; with k = 0, a bisection on the probability threshold, one full-vocabulary reduction per step (≈ 20 passes over 152K floats), instead of mlx-lm's full O(V log V) sort per token (R4 §2.4) | fixed reduction tree; boundary ties resolved by id |
| min-p | one max reduction, then a mask below `min_p · p_max` | exact |
| padding ids | ids at or above the tokenizer's id count are masked (Qwen3: 151,669–151,935) | exact |
| selection | **keyed Gumbel-max**: argmax over ids of `z/T + g(seed, position, id)`; masked ids are −∞; ties go to the lowest id | bit-reproducible for a seed |
| greedy | argmax over fp32 logits, ties to the lowest id; no log-softmax (mlx-lm computes a full logsumexp even for greedy: 0.17 ms per token on 8B; R4 §2.4) | exact |

**Keyed Gumbel noise.** `g = −log(−log(u))`, with `u` from Philox-4×32-10 (Salmon et al., SC'11) keyed by (seed, request stream id) with counter (absolute position, id / 4, 0, 0): four uniforms per call, 24-bit mantissas, mapped into the open interval (0, 1). The logarithms use precise (not fast-math) `log` in fp32. Because the noise depends on the position and not on how many tokens a pass processes, the sampled sequence is a function of (prompt, seed) alone. With batch-invariant logits (D6), a speculative run that accepts a draft token only if it equals `argmax(z/T + g)` emits the same bytes as the plain run, for any drafter and any temperature `[VERIFIED theory: Daliri et al., arXiv 2408.07978; SGLang ships seeded-hash Gumbel sampling; R8 §3.2. EXPERIMENTAL for Sky: test XIII-T15]`. Masked Gumbel-max samples exactly the renormalized filtered distribution, so filters and keyed noise compose.

Logprobs for the API (`logprobs`, `top_logprobs` ≤ 20) are the log-softmax of the unfiltered, untempered fp32 logits, computed on the GPU only when requested.

llama.cpp also has GPU samplers, off by default for generation and on for drafting (`common/common.h:298, 334`); even then the token returns to the host before the next `llama_decode` (R5 §6). mlx-lm samples on the GPU inside its lazy graph (R4 §2.4). Sky's difference is not where sampling runs but that the sampled token never leaves the GPU before the next step reads it (§11).

## 10. The streaming detokenizer (`sky-llm::text::detok`)

Byte-level BPE tokens can end in the middle of a UTF-8 code point. Qwen3 id 61804 is `b' \xf0\x9f\x91'`; decoding a ZWJ family emoji plus a flag token by token yields 11 U+FFFD characters (R9 §4.2). mlx-lm holds output back when it ends in U+FFFD (`tokenizer_utils.py:249`); llama-server validates UTF-8 (`server-common.cpp:903`).

```rust
pub struct StreamDetokenizer<'t> {
    table: &'t IdToBytes,          // built once per model (§7.4)
    pending: Vec<u8>,              // at most 3 bytes of an incomplete code point
    at_start: bool,                // SentencePiece: strip one leading space only at sequence start
    skip_special: bool,            // special=true tokens are not shown; <think> (special=false) is text
}
impl StreamDetokenizer<'_> {
    pub fn push(&mut self, id: u32, out: &mut String);   // appends only complete code points
    pub fn finish(&mut self, out: &mut String);          // flushes; an incomplete tail becomes U+FFFD
}
```

- Byte-level BPE maps each token to bytes with the GPT-2 byte table. SentencePiece pipelines (Gemma, Mistral v0.3) map `▁` to a space, byte-fallback tokens `<0xNN>` to raw bytes, and strip the first leading space only at the start of the sequence (`prepend_scheme=first`).
- `push` appends the token's bytes and emits the longest valid UTF-8 prefix. `std::str::from_utf8`'s error tells the two cases apart: an incomplete tail (`error_len() == None`) is held, and an invalid sequence (`Some(n)`, possible with byte fallback) is emitted as U+FFFD.
- If the pinned `tokenizers` release provides a streaming decoder (`DecodeStream`), it may replace this only if it passes the same tests `[LIKELY: upstream added one in 2024; check 0.23.2]`.
- **Property test** (`detok_stream_equals_decode`): for every prefix of every test sequence, the concatenation of `push` outputs plus `finish` equals `tokenizer.decode(ids, skip_special)`, and no emitted chunk contains U+FFFD unless the full decode does. Sequences: the R9 corpus, emoji/ZWJ/flags, CJK, and random ids.
- Reasoning and tool calls are split on **token ids**, not strings: Qwen3 `<think>` 151667 / `</think>` 151668 and `<tool_call>` 151657.

## 11. The generation loop (`sky-llm::generate`)

This section is the runtime contract. The decode program's kernels are in [Part XVI](part16-decode.md); replay mechanics (direct encoding of the static plan each step; ICBs only where the GPU chooses the command count) are in [Part XX](part20-persistent-execution.md).

### 11.1 State that lives on the GPU

```rust
#[repr(C)] pub struct StepState {            // one per session, in a shared-storage buffer
    pub pos: u32,                             // absolute position of the next token
    pub step: u32,                            // decode steps executed
    pub n_new: u32, pub max_new: u32,         // generated so far / limit
    pub stopped: u32, pub stop_reason: u32,   // 0 running; EOS, MaxTokens, ContextFull, HostStop
    pub last_token: u32,
    pub host_stop: u32,                       // written by the CPU (cancellation, stop strings)
}
// plus: token ring u32[4096]; SamplerParams; penalty counts u32[vocab]; RNG key; StopSet bitmap
```

Each decode step reads `StepState`, runs the layers, the LM head and the sampler, writes the token to `ring[step % 4096]`, sets `last_token`, increments `pos`, `step` and `n_new`, and evaluates the stop conditions **on the GPU**: EOS-set membership (a bitmap over the vocabulary), `max_new`, the context limit, and `host_stop`.

### 11.2 Submission

- The prefill program's last chunk samples token 1 and initializes `StepState`, so decode steps are committed right behind prefill with no host round trip ([Part XVIII](part18-ttft.md) §4.6).
- The CPU keeps D steps committed ahead. On the M2 Pro the GPU idles 104.6 µs (95–135) between command buffers when the CPU waits for one before committing the next, and 0.4 µs when both are committed up front `[VERIFIED: R6 B3]`.
- **One command buffer per step** is the default (SPINE D9): depth comes from the D pre-committed command buffers above (Metal 4 commits arrays of command buffers; R6 A3), which [Part XX](part20-persistent-execution.md) §5 calls its K steps in flight. For Qwen3-32B (≈ 34 ms per step at the planning bandwidth; `roofline.py`) that is the only mode. Packing several steps into one command buffer remains an option for a 0.6B draft model (Part XX); it is chosen by time, not by count: at most ⌊T_cb / t_step⌋ steps per buffer, with T_cb a configured bound. The bound exists because macOS kills long GPU work while a display is active (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`), the limit is undocumented, and llama.cpp works around it with an undocumented environment variable `[VERIFIED: R6 A4]`. `[EXPERIMENTAL: T_cb is set from a watchdog test on the benchmark Mac; start at 100 ms]`
- **After a stop, no later committed step reads weights.** The mechanism is Part XX's choice among: a guard at the top of each kernel (`if (state->stopped) return;`), a step prologue that writes zero threadgroup counts into indirect-dispatch arguments, or GPU-encoded ICB slots that are reset (≈ 0.15 µs per empty slot on the M2 Pro; R6 B4). The test is the same for all three (XIII-T14).
- The host commits more steps while `n_new + in_flight < max_new` and the ring has room (ring capacity minus unconsumed tokens).

### 11.3 Delivery to the CPU

- Each step signals an `MTLSharedEvent` with value = step. The streamer thread waits on it, reads the new ring entries and runs the detokenizer. On the M2 Pro, a Metal 4 `signalEvent` is seen by the CPU 84.8 µs after commit `[VERIFIED: R6 B3]`; this latency is off the GPU's critical path.
- **Text-level stop strings** (OpenAI `stop`) are matched on the CPU after detokenization (an Aho-Corasick automaton over the stop strings, as mlx-lm's server does over token sequences, `generate.py:855-926`; R4 §2.5). On a match the CPU sets `host_stop`, stops committing, and discards tokens after the match. At most D − 1 committed steps are wasted. CPU writes to a shared buffer are seen by a running command buffer only on a best-effort basis, so the hard guarantee is the next submission boundary `[LIKELY: no documented visibility guarantee for host writes during a command buffer]`.
- Cancellation (Ctrl-C, client disconnect) uses the same path.
- KV entries written by discarded steps are harmless: positions are overwritten by the next request or truncated ([Part XIV](part14-kv-cache.md)).

### 11.4 What the CPU does per token

Nothing on the critical path: it reads finished tokens and keeps the queue topped up. Per-token CPU work is then detokenization and I/O, which mlx-lm's measurements already show to be microseconds (R4 §2.5). Compare:
- mlx-lm builds and encodes ≈ 1,225 kernels per 64-layer token in Python and C++ (R3 §7.F); its pipelining hides this for 32B but caps small models (R3 §8.2), and its Python loop drew ≈ 7 W of CPU during 0.6B decode (R9 §6.3);
- llama.cpp re-encodes ≈ 724 dispatches per 8B token and synchronizes with the GPU before every sample (R5 §6).

## 12. Speculation hooks

The algorithms are in [Part XXI](part21-speculative.md). The runtime provides these interfaces:

```rust
pub trait Drafter {
    fn kind(&self) -> DrafterKind;   // PromptLookup | SuffixTree | DraftModel | Eagle3 | Mtp
    fn propose(&mut self, ctx: &DraftContext, budget: DraftBudget) -> DraftTree;  // tokens + parent links
}
pub struct VerifyProgram { pub width: u32, pub tree_mask: bool }   // compiled per width bucket: 2, 4, 8, 16, 32
```

- **Verify programs** are static specializations of the decode program at widths M = 1 + γ ([Part XVI](part16-decode.md)). They use the same batch-invariant kernels as decode, so position *t*'s logits are bit-identical whether computed in a verify pass or a plain step (D6).
- **Accept/reject on the GPU.** Greedy: accept while the draft equals the argmax. Sampled: accept while the draft equals the keyed-Gumbel sample at that position, which keeps byte-identity with plain decoding; or standard rejection sampling (accept with min(1, p/q)), which accepts more than mlx-lm's and llama.cpp's exact-match rule but is identical only in distribution (D10, R8 §3.1). The kernel writes the accepted count and the bonus token into `StepState`.
- **Penalties inside verify.** The penalty counts for position *t + i* include the draft tokens *t … t + i − 1*. Positions after the first mismatch are discarded, so this prefix computation is exact.
- **Rollback.** KV is preallocated and indexed by position, so rejecting tokens resets `pos`; nothing is copied. Hybrid models checkpoint the DeltaNet recurrent state at each draft position and restore the accepted one (R8 §3.5 rule 4). The memory plan reserves those checkpoints ([Part XIX](part19-memory.md) §2).
- **EAGLE-3 taps.** The verify program writes the hidden states of the three tapped target layers into a planned buffer that the drafter reads ([Part XXI](part21-speculative.md)).
- **mlx-lm's loop, for contrast.** One full CPU synchronization per draft/verify round and no pipelining across rounds (R4 §2.3); speculative greedy output diverges from plain greedy (#1470).

## 13. The plan cache and the `.skybin` bundle

Two artifacts hold the result of steps 2–13 of [Part VI](part06-architecture.md) §5, and they differ in kind (D19, Part VI §8.9):

| | Plan cache | `.skybin` |
|---|---|---|
| Written by | `sky run`, implicitly, under `~/Library/Caches/sky/plans/<key>/` | `sky compile -o m.skybin`, explicitly |
| Class | a cache: **data only** (spec non-negotiable 3) | a build artifact, like a `sky build` executable |
| Holds | kernel choices, skeleton parameters, buffer offsets, dispatch and barrier lists, the memory plan, text-asset hashes; bounded and validated on load like `sky-tune`'s database | everything the plan cache holds, plus MSL sources, optionally a Metal binary archive and a pipeline script, and optionally the weights |
| Compiled code | none. MSL is regenerated deterministically from the plan on load, and Metal's OS shader cache supplies the compiled pipelines (llama.cpp's later starts take 14–48 ms this way; R5 §1.6) | the archive, if present, for this GPU and OS build |

The rest of this section describes the `.skybin`; the plan cache is its data subset.

### 13.1 Contents

A single file: a 16 KiB header with a section table (name, offset, length, sha256), then 16 KiB-aligned sections so each can be mapped.

| Section | Contents |
|---|---|
| `manifest.json` | format version; Sky version and git sha; the cache keys (§13.2); provenance (§3.5); creation time |
| `spec.json` | the normalized `ModelSpec` and its `NormalizationReport` |
| `programs.bin` | decode, verify (per width), prefill (per chunk bucket) and sampler programs: dispatch lists, pipeline ids, buffer bindings as (arena, offset) or (weight region, offset), threadgroup sizes, barrier list, uniform layouts ([Part XX](part20-persistent-execution.md)) |
| `memplan.json` | the static memory plan ([Part XIX](part19-memory.md)) |
| `msl/` | every kernel's MSL source; always present in a `.skybin` (never in the plan cache) |
| `archive/<gpu>-<os-build>.bin` | optional (`sky compile --archive`): a Metal binary archive harvested on the compiling machine (`MTL4Compiler` with an `MTL4PipelineDataSetSerializer`, `serializeAsArchiveAndFlush(url:)`; `MTLBinaryArchive` on Metal 3). Archives may hold IR, GPU-specific binaries or both, and Metal 3 and 4 both load them `[VERIFIED: Apple docs, R6 A3]` |
| `pipelines.script` | the serialized pipeline script (`serializeAsPipelinesScript()`) for the offline binary generator, which needs Xcode (R6 A3), for building archives for other machines |
| `text/` | `tokenizer.json` (verbatim, hashed), the chat template source(s), `StopSet`, BOS policy, sampler defaults |
| `tune.db` | the `sky-tune` entries the plan used |
| weights | **not embedded by default**: a reference to the `.skyweights` (or GGUF) by source key and layout id; `--embed-weights` produces a self-contained file for distribution |

### 13.2 Cache keys and invalidation

The key is layered, so a change invalidates only what depends on it:

| Key | Covers | When it changes |
|---|---|---|
| `K_model` | weight shard sha256s, normalized-config hash, tokenizer and template hashes | rebuild everything |
| `K_sky` | Sky version, compiler git sha, TIR and kernel-IR format versions | rebuild everything |
| `K_target` | GPU family and architecture name, core count, neural accelerators present, Metal feature set | re-plan kernels and tuning; keep spec and text |
| `K_os` | macOS build and Metal compiler version | a `.skybin`'s archive is ignored and pipelines compile from `msl/`; the plan stays valid. The plan cache holds no compiled code, so it is unaffected, except that tuning entries keyed by OS and Metal compiler version may trigger a re-plan ([Part X](part10-autotuner.md)) |
| `K_config` | max context, batch, KV format, numerics mode, `-O` level, speculation config, prefill chunk buckets | re-plan programs and memory |
| `K_tune` | hash of the tuning entries used | re-plan at `-O2`/`-O3` if better entries exist |

A `.skybin` built on another machine still runs: the plan is checked against `K_target`; the archive is ignored if `K_os` or `K_target` differ, and pipelines compile from `msl/` once; after that Metal's OS shader cache serves them. Nothing is harvested into an implicit cache. Bit-reproducibility is claimed only per (chip, OS, Sky build, tuning entry) (D6), and `sky explain` prints which of these the current run matches.

The plan cache is data, and its sizes and offsets are validated before use, as `sky-tune`'s database is (spec, "The database is data, never code"); a record that fails validation is ignored and rebuilt. A `.skybin` contains code (MSL, possibly binaries): it is a build artifact, trusted exactly as much as an executable from the same source, and `sky run` says so when the file was not produced locally.

## 14. The OpenAI-compatible server (`sky-serve`)

Year 1 serves one generation at a time (spine §1). Interface:

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | `messages`, `tools`, `stream`, `temperature`, `top_p`, `max_tokens` / `max_completion_tokens`, `stop`, `seed`, `n = 1` only, `logprobs` / `top_logprobs`; extensions `top_k`, `min_p`, `repetition_penalty`, `chat_template_kwargs`, `cache_prompt` |
| `POST /v1/completions` | `prompt` as text **or as an array of token ids**, which the benchmark's server mode needs (R9 §7.4 mode B) |
| `GET /v1/models` | the loaded model with its provenance |
| `POST /tokenize`, `POST /detokenize` | for clients and the harness |
| `GET /health` | |

Behaviour:
- Streaming uses server-sent events. Usage includes `prompt_tokens_details.cached_tokens`. A `timings` object (`prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, `cache_n`) mirrors llama-server's (R9 §6.1).
- `cache_prompt` defaults to true, as in llama-server (R9 §6.1); TTFT benchmarks set it to false.
- Reasoning goes in `reasoning_content` and tool calls in `tool_calls`, both split on token ids (§10). Parsers exist for Qwen3 and gpt-oss in year 1.
- Requests wait in a FIFO queue on a blocking channel. mlx_lm.server busy-polls its queue and pins a core at ≈ 100% while idle (mlx-lm#1500); test XIII-T18 asserts that an idle `sky serve` uses under 1% of a core.
- Message content is escaped for special-token literals (§8.4).
- It binds 127.0.0.1 by default, has an optional bearer key, and limits request bodies and prompt lengths (≤ the planned context).
- mlx-lm has prompt-cache gaps the server must not copy: per-request `chat_template_kwargs` bypass its cache (#1803), streamed generations were reported not to store entries (#1802), and each cache hit duplicates the conversation's KV on the first write (#1395, mechanism in R4 §4.1). Sky's prefix cache is in [Part XVIII](part18-ttft.md) §6.

## 15. The command line

Today's subcommands are `check`, `run`, `build`, `repl`, `hw` and `tune` (`crates/sky-cli/src/main.rs:71-136`) `[VERIFIED: read on main a99e7e3]`. The LLM path adds the following. `sky run` dispatches to it when its argument starts with `hf://`, ends in `.skybin` or `.gguf`, or is a directory containing `config.json`; otherwise it runs a Sky program as today.

| Command | Purpose | Key options |
|---|---|---|
| `sky run <model> ["prompt"]` | generate; interactive chat if no prompt | `--system`, `--raw`, `--think` / `--no-think`, `--max-tokens`, `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--seed`, `--context N`, `--kv f16|bf16|int8`, `--numerics inference|portable|relaxed`, `--speculate off|auto|lookup|draft:<model>|eagle3:<repo>`, `--offline`, `--update`, `--stats`, `--json` |
| `sky compile <model> -o m.skybin` | ahead-of-time plan (a build artifact, §13) | `--context`, `--kv`, `-O1|-O2|-O3`, `--autotune overnight`, `--archive`, `--embed-weights`, `--contract "<measure> <= <bound>"` ([Part XIX](part19-memory.md) §11) |
| `sky bench llm <model>` | the measurement protocol of [Part XXIII](part23-benchmarks.md) | `--scenario S1…S7`, `--engines sky,mlx-lm,llama.cpp`, `--pinned`, `--json <dir>` |
| `sky explain <model>` | the record of every decision | `--config`, `--weights`, `--template`, `--kernels`, `--fusion`, `--memory`, `--ttft` |
| `sky serve <model>` | the server (§14) | `--host`, `--port`, `--keep-alive` ([Part XVIII](part18-ttft.md) §5) |
| `sky pull <model>` | fetch and verify, then build the relayout cache | `--revision`, `--no-relayout` |
| `sky cache ls|gc` and `sky verify` | manage and check Sky's derived files | `--keep <size>`, `--weights` |

`sky build` (an executable from a Sky program, R1 §1.1) and `sky compile` (a `.skybin` for a model) are different things; each one's error message names the other when it is given the wrong kind of input.

`--stats` prints, after generation: TTFT, prefill tok/s, decode tok/s, weight bytes per token, bits per weight, KV format, context depth, and whether the prefix cache hit. These are the numbers R9 §7.6 requires next to every speed.

## 16. Implementation tasks

| ID | Task | Crate / files | Deliverable | Acceptance test |
|---|---|---|---|---|
| XIII-T1 | `hf://` grammar and cache discovery | `sky-hub/src/{uri.rs,cache.rs}` | `parse`, `CacheRoot::discover` | Table-driven tests over every example and error in §3.1; cache discovery honors the four variables in order; a fixture cache with shared-store symlinks resolves through the links |
| XIII-T2 | Offline-first resolution and downloads | `sky-hub/src/{resolve.rs,download.rs,lock.rs}` | `resolve`, `fetch_file` with Range, resume and hash checks | A pinned, complete snapshot makes 0 network calls (mock server counts requests); a concurrent `hf_hub_download` of the same file blocks on the same lock; no file is ever created under `hub/blobs/` |
| XIII-T3 | Provenance and the models lock | `sky-hub/src/provenance.rs` | `Provenance`, `sky-models.lock` | Round-trip; `sky verify` detects a modified byte in a shard through two symlinks |
| XIII-T4 | Config normalization | `sky-llm/src/config/{mod.rs,rope.rs,families.rs}` | `normalize`, `NormalizationReport` | The 31 configs of R9-work/arch/configs plus their transformers-5 re-saves normalize to equal `ModelSpec`s; the Llama 3.1 re-save has θ 5e5 and llama3 ×8; Gemma 3's global scale is 0.125; an unknown key fails with `CFG-003` |
| XIII-T5 | Qwen3 dense builder | `sky-llm/src/arch/qwen3.rs` | `build` → TIR | G2: Qwen3-0.6B logits match HF transformers fp32 within tolerance on the reference interpreter; later, identical L1 text from `models/qwen3.sky` (D7) |
| XIII-T6 | Safetensors reader and MLX quantization inference | `sky-llm/src/load/{safetensors.rs,quant_infer.rs}` | header validation as `Metadata::validate`; `infer_quant` | Every tensor of mlx-community Qwen3-0.6B/8B-4bit classified; `negative_scale_groups` passes; an ambiguous module fails by name |
| XIII-T7 | Relayout cache | `sky-quant/src/relayout.rs`, `sky-llm/src/load/skyweights.rs` | `.skyweights` writer and mapper | `relayout_roundtrip_bit_exact` on all tensors of Qwen3-8B-4bit; every tensor ≥ 256-B aligned, regions 16 KiB-aligned; a GPU read of every region equals the CPU dequantization |
| XIII-T8 | GGUF parser and direct mapping | `sky-llm/src/load/gguf.rs` | parser, layout flags, views | Reproduces R9's three `gguf_inspect` outputs (tensor counts, types, offsets, bpw 4.67 and 4.90); Llama GGUF gets `RopeLayout::Interleaved` |
| XIII-T9 | Tokenizer wrapper and golden gate | `sky-llm/src/text/tokenizer.rs`, `tests/tokenizer_golden.rs`, `tools/gen_golden_ids.py` | wrapper; fixtures for 6 tokenizers | 168/168 identical ids; nightly differential fuzz job green |
| XIII-T10 | Chat templates | `sky-llm/src/text/template.rs` | environment, `py_tojson`, generation-tag stripping | The 30 R9 cases render byte-identically; Qwen3 with thinking on/off; a pinned `strftime_now` |
| XIII-T11 | Stop set, BOS rule, content escaping | `sky-llm/src/text/{eos.rs,bos.rs,escape.rs}` | `StopSet`, `BosPolicy`, escaping | Double-BOS cases for Llama 3.1, Gemma 3, Mistral Small 3.2 produce exactly one BOS; Qwen3 {151645, 151643}; a typed `<|im_end|>` in `sky serve` never produces id 151645 |
| XIII-T12 | GPU sampler and keyed Gumbel | `sky-kernel` sampler skeleton, `sky-llm/src/sample.rs` | kernels + CPU reference | Chi-square test against the CPU reference over 10⁶ draws for each filter; same seed gives the same tokens across runs; ties go to the lowest id |
| XIII-T13 | Streaming detokenizer | `sky-llm/src/text/detok.rs` | `StreamDetokenizer` | `detok_stream_equals_decode` on the corpus and the Qwen3 id-61804 emoji case; no U+FFFD in chunks |
| XIII-T14 | GPU-resident generation loop | `sky-llm/src/generate.rs`, `sky-metal` events | `Session`, `StepState`, ring, pre-commit | Instrumented run: zero CPU waits between decode steps; after EOS no later step reads weights (GPU counters or timing); cancellation stops within one submission |
| XIII-T15 | Speculation hooks | `sky-llm/src/spec/mod.rs` | `Drafter`, `VerifyProgram`, accept/reject, rollback | With a random "drafter", output is byte-identical to plain decoding at T = 0, 0.6 and 1.0 over 100 prompts × draft lengths 1–16; mlx-lm#1470's script as an adversarial test |
| XIII-T16 | Plan cache and `.skybin` | `sky-llm/src/{plan_cache.rs,skybin.rs}` | data-only plan cache; `.skybin` writer, reader, key checks | The plan cache contains no MSL or binary (a test scans every record type); changing each key component invalidates exactly what §13.2 says; a corrupted offset is rejected without a panic (fuzzed); a second `sky run` regenerates byte-identical MSL from the cached plan |
| XIII-T17 | CLI | `sky-cli/src/llm/{run,compile,bench,explain,pull,cache}.rs` | the commands of §15 | `sky run hf://mlx-community/Qwen3-8B-4bit "…"` produces tokens (G3); `--stats` prints all R9-required fields |
| XIII-T18 | Server | `crates/sky-serve` | endpoints of §14 | OpenAI client conformance on streamed and non-streamed chat; ids in `/v1/completions`; idle CPU < 1% of a core over 60 s |
| XIII-T19 | G7 builders | `sky-llm/src/arch/{qwen2,qwen3_moe,llama,gemma3,phi3,gpt_oss}.rs` | one builder per row of §5.2 | Each passes §5.3 on its smallest checkpoint |
| XIII-T20 | Hybrid builder | `sky-llm/src/arch/qwen3_5.rs` | DeltaNet, gated attention, M-RoPE, MTP | Text-only M-RoPE equals transformers' output on the reference interpreter; Qwen3.6-27B passes §5.3 |

## 17. Measurements that decide

- **Tokenizer exactness** on the corpus and on every benchmark prompt: 100% or no speed number is published (D13).
- **Template byte-identity** on the 30 cases and on every new model family's template.
- **Config corpus**: the share of real-world configs that normalize without `--allow-config-key`. A high failure rate means the ignorable lists are too short, not that the rule is wrong.
- **Relayout build time** against MLX's `pread` load time for the same file ([Part XVIII](part18-ttft.md) §4.3). This decides whether the first run builds the cache before the first token.
- **Host overhead per token**: CPU time per decode step and GPU idle time between steps, from Metal timestamps. Target: no idle gap attributable to the CPU.
- **Server overhead**: TTFT measured at the socket minus TTFT measured in process (R9 §7.4 modes A and B).
- **Sampler cost**: GPU time of the sampler stage per token at top-k = 20 and at top-p without top-k, against the step time.

## 18. Risks and negative results

- **The Hugging Face ecosystem moves under us.** huggingface_hub 1.32 changed the cache two weeks before this report; 2.0.0 replaced the HTTP stack; transformers accepts both majors (R9 §5.4). Mitigation: pinned versions in tests, a nightly job against the newest hub, and never writing the shared store.
- **Strict config parsing will fail on new releases.** Qwen3.6 added `output_gate_type` to Qwen3.5's schema. Failing loudly is the intent; the cost is a release of Sky per new key. `--allow-config-key` is the escape hatch, and it is recorded.
- **Template engines diverge on new templates.** Byte-identity is shown for 30 cases, not for every template. Every new family adds its template to the gate.
- **Sampler order makes sampled output engine-specific.** This is not fixable; it is why the protocol uses argmax and why exactness claims are against Sky's own plain path.
- **The relayout cache doubles disk use** for MLX models (≈ 18.4 GB for Qwen3-32B) and its first build costs a full read and write. If users object, the fallback is the aligned in-memory copy, as MLX does, at MLX's load cost.
- **Content escaping changes token ids** relative to HF for prompts that contain special-token literals. This is intended for the server and documented; local `sky run` keeps HF behaviour.
- **GGUF-only models have no `tokenizer.json`.** Reconstructing the tokenizer from GGUF metadata reproduces llama.cpp's choices (its fingerprinted regex, no NFC), not HF's; the gate then runs against llama.cpp only.
- **Watchdog limits are undocumented.** If they are shorter than one 32B decode step, single-step command buffers are the only option; that is the current design for 32B anyway.
