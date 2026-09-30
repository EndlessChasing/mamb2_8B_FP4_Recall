# Resurface on repaired FP4 G16 base V1, native FP16 state

## Objective and fixed quality decision

Train a fresh Resurface adapter on the exact base-only FP4 G16 repair selected
at step448, whose completed official WikiText-2 test PPL was
**6.717854132348465**. Freeze all repaired base parameters. The adapter is
useful if full paired official WT2 test PPL is **no higher than its unadapted
repaired base**, normal CONFIRM MK improves, and the fixed-seed paired
bootstrap 95% lower bound for that MK gain is positive. Report these conditions
separately. Do not relax the conditions or select a different checkpoint using
test or CONFIRM results. Removing the adapter must exactly restore every
baseline window NLL and all generated sequences, native forwards/hooks and
reset/cache probes.

## Exact repaired base and provenance

- Pure Mamba2-8B, 56 blocks, 8,236,999,680 parameters, 507 base tensors.
- Selected standalone checkpoint on GPU:
  `/home/horde/mamb2_8B_W4A16_Recall_StateQuant/artifacts/fp4_base_repair_v1/formal/step0448/model`.
- Repaired packed manifest SHA256:
  `2d2208b690c2ec0def90578d8590cc122bb1e4542b0d49ae4417a48f72b0b73c`.
- Selected base training core SHA256:
  `d5fb50f2e045b6202a5f38b353ceab7ae47b21c7cc88e8cb398b46c342632c6e`.
- Completed base-repair validation selection SHA256:
  `2f1ff559f3e907b3803bcc6d7036d39dab88c11dbb5e69f1de57e2dff5ca452c`.
- Completed base-repair full three-arm test comparison SHA256:
  `c3a38223055d53603a64302b0acdbd8e236baa402d67e6927a126848ab21522a`.
- Original source checkpoint SHA256:
  `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.
- Original parent packed manifest SHA256:
  `4e7fb60f5d03c61e63324a6c25847014a26f9aa0c8d5a8c922d0f367e514fe07`.
- Original conversion receipt SHA256:
  `55de4189a7e04c9564a48b0fbb5f37428d3d1bb5b6274831d878b41ef4686c9b`.

Use the frozen `fp4_repaired_checkpoint.load_repaired_model` standalone loader,
which checks actual packed shards and decoded tensor hashes. The expected
507 values come from the **repaired manifest** `decoded_tensor_sha256`.
The old conversion ledger describes the original quantization and cannot
replace the 393 updated FP16 value hashes. All 114 FP4 matrices and the 393
repaired FP16 tensors are frozen during this adapter experiment. Check actual
student/teacher content at endpoints, native structure and independent storage.

Both separately loaded student and teacher use this same repaired FP4 base.
The teacher has no adapter. Training and PPL use native parallel SSD prefill
and native backward, with no StateQuant controller or surrogate state-gradient
replacement. MK uses native cached prefill followed by recurrent decoding;
inspect actual convolution and SSM cache tensors and require FP16. Pin the
existing singleton warp16 RMSNorm policy before loading: TF32 off, highest
FP32 matmul precision and eight CPU threads. Bind all actual imported local
source files and protocol before GPU work, ignoring nonexistent virtual labels.

## Fresh training and fixed candidate

- Numeric TRAIN manifest SHA256:
  `451b8703c21120667ef0ea272c21a10d4cd4561779a7b9663773ae47981600c0`.
  One fixed shuffled pass over 1,536 examples, seed2026092803.
- Official WT2 TRAIN prose manifest SHA256:
  `facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d`;
  stored448x2048-token TRAIN file SHA256:
  `e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233`.
  Use the established 512-token prose segment schedule from the manifest.
- Fresh soft-gated post-D Resurface initialization: V=0, g=1, router_w=0,
  router_b=-4. Train only 224 adapter tensors / 1,154,104 FP32 master
  parameters. No previous adapter checkpoint initializes this run.
- Numeric answer-token CE coefficient1; prose CE0.5, unadapted repaired-base
  teacher-to-student KL0.5, prose gate closure penalty3, temperature1.
  Full256K vocabulary; staged64-token logits chunks.
- AdamW betas(0.9,0.999), epsilon1e-8, weight decay0. Mixing LR1e-4,
  router LR3e-4, cosine multiplier1 to0.1 over1,536 successful updates.
  Gradient clip1. GradScaler starts1024; at most8 overflow retries, each
  keeps the same data/LR and skips optimizer/master updates.
- Complete and discard one-update smoke, with native train/deployment parity
  at128 and512 tokens, identity initialization, finite gradients, independent
  teacher/student storage, FP16 export/readback and native restoration.
  Formal training restarts with a fresh adapter, optimizer and scaler.
- **1,536 successful formal updates**, progress checkpoints384/768/1152/1536.
  Only the final step1536 FP16 export is a quality candidate. Earlier
  checkpoints are recovery artifacts, without PPL/MK selection.
- Check all507 frozen base values at start/end, actual checkpoint readback,
  all224 final FP16 exports equal saved masters cast to FP16, optimizer/scaler
  readback and native export/removal parity at128/512 tokens.

## Full paired evaluation

Evaluate three arms on the exact same repaired weights and backend:
**baseline / final Resurface / adapter removed**.

- Official `Salesforce/wikitext`, `wikitext-2-raw-v1`, **test**, revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Original tokenizer SHA256:
  `5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`;
  no automatic BOS/EOS, two-newline document join.
- Full stream300,964 tokens, SHA256:
  `5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608`.
  All300,963 targets,147 reset windows of up to2048 targets, final1955 targets.
  Native SSD prefill has no persistent PPL cache. Full-vocabulary head GEMM,
  FP32 cross entropy in64-token chunks; pooled PPL=exp(totalNLL/totalTargets).
- Fixed CONFIRM manifest SHA256:
  `445f1ebae5cfeeafccc0b6c02f1a514d8d7858360637af9ec37ab5e48212a091`.
  Exactly384 normal and384 target-removed prompts, N16/N64 reported separately.
  Full-vocabulary greedy decoding, at most12 tokens, stop at EOS; match the
  first standalone six-digit answer. Pair by original prompt/token identity.
- MK paired bootstrap10,000 draws, seed20260928; report point gain and95% CI,
  improvements/regressions/unchanged. No test prompt used for training.
- Inspect112 request-owned FP16 cache tensors across56 layers: convolution
  4,587,520B + SSM117,440,512B =122,028,032B for batch1. Adapter adds no
  recurrent cache. Record reset/prefill/one-step probes and restored equality.
- Require exact adapter-removed147 window NLLs and768 generated-token sequences,
  base parameter identities/content, native forwards/hooks and cache probes.

## Storage and independent audit

Base encoded weight payload remains4,638,460,360B, actual119 weight files
4,638,539,864B. Native reference runtime expands weights to16,473,999,360B
of FP16 weights. Adapter FP16 tensor payload is2,308,208B; measure actual
serialized file bytes separately. Training masters/optimizer archives and
provenance metadata are separate storage scopes. This is a quality reference
runtime; no packed-resident FP4 kernel or throughput claim is inferred.

Independent CPU audit checks source/protocol/report/adapter hashes, repaired
base provenance, fixed training history, export/master readback receipts,
full raw-token/window identities, pooled PPL, all MK scoring, bootstrap,
removed-arm equality and storage accounting. GPU-host audit reads actual
research checkpoint/adapter files. Copy small evidence and adapter to Mac
for an independent CPU audit. CPU checks do not replay GPU logits.

WT2 validation/test and the CONFIRM family have prior project exposure;
they are not untouched project holdouts. Base repair used TRAIN and
validation-selected448 before its test; this new adapter has a fixed1536
candidate before evaluation. Pretraining contamination is unaudited.
This request authorizes training and verification. Publication is not requested.
