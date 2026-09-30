# FP4 G16 base repair V1: existing small tensors, no Resurface

## Objective and representation

The user requests the FP4 G16 base without Resurface to reach full official
WikiText-2 test PPL **strictly below 7.2**. The current independently audited
native-prefill FP4 G16 base test PPL is 7.520669691121164. The historical
native-prefill original FP16 control is 7.244532175944678. These are references;
the final same-process FP4 baseline and repaired candidate determine this run.

Keep the original pure Mamba2-8B architecture and 8,236,999,680 parameters.
Keep all 114 quantized matrices' E2M1 G16 codes, E4M3FN block scales and FP32
global scales byte-for-byte unchanged. Update only the **393 existing retained
FP16 tensors / 3,580,928 parameters / 7,161,856 bytes**: block/final norms,
gated norms, convolution weights/biases, A_log, D and dt_bias. Replace their
existing slots in the retained safetensors shard. No adapter, extra projection,
router, StateQuant controller, promoted weight or added inference operation.

The model still has 119 weight shards, logical weight payload 4,638,460,360
bytes and actual shard bytes 4,638,539,864. Training FP32 masters and optimizer
state are research workspaces; they are excluded from the inference checkpoint.
The native reference runtime expands the weights into FP16 residency.

## Frozen inputs and execution

- Parent packed manifest SHA256:
  `4e7fb60f5d03c61e63324a6c25847014a26f9aa0c8d5a8c922d0f367e514fe07`.
- Parent conversion receipt SHA256:
  `55de4189a7e04c9564a48b0fbb5f37428d3d1bb5b6274831d878b41ef4686c9b`.
- Original source checkpoint SHA256:
  `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.
- Original tokenizer SHA256:
  `5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
- Prose TRAIN manifest SHA256:
  `facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d`.
- Stored TRAIN tokens SHA256:
  `e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233`.
  Exactly 448 fixed 2,048-token windows, no inserted BOS/EOS.

Load the student from the strict original packed checkpoint. Load a separate
frozen original FP16 teacher, checking source and all parameter identities.
Bind all actual imported local source files before GPU computation, ignoring
only nonexistent generated module paths. Sources must remain unchanged.

Native differentiable SSD prefill/backward, model eval mode, no state-controller
replacement. Functional native blocks cast independent FP32 small-tensor
masters to FP16 inside checkpoint recomputation. Deployment uses those actual
FP16 values in the original parameter slots and original native forward path.
TF32 off, highest FP32 matmul precision, eight CPU threads and the established
fixed singleton warp16 RMSNorm policy, applied before model loading.

## Training and predeclared candidate selection

- Fresh initialization from parent retained FP16 values; no Resurface state.
- Discard a one-update smoke, including native/functional and checkpoint
  gradient checks, FP16 export/native parity, finite nonzero gradients and
  restoration. The smoke must not initialize formal optimizer state.
- **896 successful updates**, two complete passes over all 448 TRAIN windows.
  Seed 2026093001; two successive CPU-generator randperm(448) draws.
  The independently prepared canonical CPU schedule has SHA256
  `6c0e280e623dc5e78ce478bac006b9844d73a730b18f03a34b70649f7a2eae07`;
  both training and the Torch-free auditor require that exact schedule.
- Each update uses all **2,047 next-token targets** in a stored window.
  Full 256K vocabulary, FP16 head GEMM and FP32 CE/KL in 64-token chunks.
  One staged hidden derivative and one backbone backward per update.
- Objective: real-text CE coefficient **1.0**, detached original-FP16-teacher
  to student KL coefficient **0.1**, temperature 1. No numeric recall objective.
- AdamW LR 1e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay zero;
  cosine multiplier 1 to 0.1 over successful updates; global gradient clip 1.
- GradScaler starts at 1,024. At most eight overflow retries; each retries
  the same window, halves the loss scale and performs no parameter/optimizer
  update. Stop on nonfinite forward/master/FP16 cast or exceeded retry budget.
  Check native negative finite A and positive finite dt_bias softplus;
  introduce no new clamps or altered SSM equations.
- Immutable export candidates at **224, 448, 672 and 896** successful updates.
  Each writes a checkpoint-core training report before its standalone repaired
  checkpoint export, avoiding self-referential report hashes.
- Evaluate each candidate on **full WT2 validation**, native prefill, 130 reset
  windows / 264,764 targets. Select the lowest pooled validation NLL; exact
  ties select the earlier step. Finish all 896 updates before final selection.
- Evaluate only that selected candidate on full WT2 test. Test/CONFIRM/MK
  never influence fitting or checkpoint selection within this V1 run.

## Final verification

The final evaluator independently reloads the actual selected repaired
checkpoint, original packed baseline and original FP16 source control, all
under the same frozen numerical backend policy. The older original-source
record did not pin that full policy; its value above is historical context.
For all arms, use full official
`Salesforce/wikitext`, `wikitext-2-raw-v1`, **test**, pinned revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, two-newline document join and
original tokenizer without automatic special tokens.

Require 300,964 tokens, token SHA256
`5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608`,
all 300,963 targets, 147 windows of at most 2,048 targets including the final
1,955-target partial window. Native parallel prefill resets per window;
no persistent PPL cache or tokenwise carry quantization. Compute
PPL = exp(total NLL / total targets), with FP32 full-vocabulary logits.

Required evidence: complete 896-update TRAIN-only history, fixed validation
selection, candidate training/export binding, native FP16 export parity,
all original 114 FP4 matrices and 118 FP4 shard identities unchanged,
393 retained-tensor geometry/dtype/region hashes and equal payload/shard sizes,
all 507 decoded candidate hashes and original native model structure,
full test window/token identities and independently recomputed PPL arithmetic.
Smoke and final forward/export restoration must exactly recover the original
parent. Independent CPU auditing checks reports, files, schedule and arithmetic;
it does not replay GPU logits. Report **PPL < 7.2** and integrity separately.
Do not relax the target or infer success from TRAIN loss, validation or MSE.

## Scope

WT2 test and validation have prior project exposure; they are not untouched
project holdouts. This run's selection is validation-only and its final test
candidate is fixed before test. Original pretraining contamination is unaudited.
MK is not a candidate-selection criterion and has not been inferred from PPL.
This request authorizes research training and verification, not publication.
