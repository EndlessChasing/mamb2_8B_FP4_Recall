# Mamb2_8B_FP4_Recall: audited paired results

**Full official WikiText-2 TEST PPL is 6.717746660814748 with final Resurface,
versus6.717800294314242 for its paired repaired base. Normal CONFIRM recall
is 353/384(91.9270833%), versus131/384(34.1145833%). All four fixed quality
conditions and both combined CPU audits pass.**

This portable result document describes the exact repaired FP4 G16 + native
FP16-state model in this bundle. Relative evidence links resolve from the
bundle or source repository. It is derived from the frozen research results
document SHA256
`738075edd8d6d1edc3ca88dd325c8c36b6567b445b9f7f1e77c9b634b4dc7c9a`;
that original document remains unchanged and is recorded as source provenance.
The [frozen adapter protocol](FP4_REPAIRED_S16_RESURFACE_V1_PROTOCOL.md)
and [base-repair protocol](FP4_BASE_REPAIR_V1_PROTOCOL.md) govern the experiments.

## Final same-process comparison

| Arm | Full WT2 TEST PPL | Pooled target NLL | Normal MK / 384 | Target-removed / 384 |
| --- | ---: | ---: | ---: | ---: |
| [Adapter off](../evidence/baseline.json) | 6.717800294314242 | 573262.5137534142 | 131(34.1145833%) | 0 |
| [Final Resurface on](../evidence/resurface.json) | **6.717746660814748** | **573260.1109187603** | **353(91.9270833%)** | 0 |
| [Adapter removed](../evidence/adapter_removed.json) | 6.717800294314242 | 573262.5137534142 | 131(34.1145833%) | 0 |

| CONFIRM stratum | Adapter off | Final Resurface | Adapter removed |
| --- | ---: | ---: | ---: |
| N16 normal / 192 | 101(52.6041667%) | **191(99.4791667%)** | 101(52.6041667%) |
| N64 normal / 192 | 30(15.625%) | **162(84.375%)** | 30(15.625%) |
| N16 target removed / 192 | 0 | 0 | 0 |
| N64 target removed / 192 | 0 | 0 | 0 |

Normal paired gain is 222/384 = **57.8125 percentage points**. There are 225
improvements, 3 regressions and 156 unchanged cases. The paired10,000-draw
bootstrap, seed 20260928, gives95% interval
`[0.5260416666666666, 0.6276041666666666]`, or
**[52.6041667,62.7604167] percentage points**.

PPL is nearly unchanged: absolute change-0.000053633499494054604,
relative change-0.0007983788910648215%. Pooled NLL is lower by 2.40283465385437,
or0.000007983820781472707 nats per target. Of147 windows, 83 have lower NLL
and 64 have higher NLL. The pooled no-worse gate passes; this tiny score change
does not establish a broad language-model improvement.

| Predeclared condition | Outcome |
| --- | --- |
| Full test adapted PPL no higher than the paired repaired-base PPL | PASS |
| Positive normal-MK correct-count gain | PASS: +222 |
| Paired95% lower confidence bound strictly positive | PASS |
| Exact adapter-removed147 window scores and 768 generated sequences | PASS |

The [full comparison](../evidence/comparison.json) records the individual
quality flags, restoration receipt, package/weight identity and memory scope.
All147 baseline/removed window NLLs, PPLs and target counts are exactly equal.
All768 generated-token sequences, outputs, predictions, scores and cache
receipts restore exactly. Native forwards/hooks, frozen parameter identities,
reset/prefill/one-step hidden/cache probes and repeated-reset checks pass.

## Data and execution protocol

PPL uses official `Salesforce/wikitext`, `wikitext-2-raw-v1`, **test**, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, two-newline document joins,
the original tokenizer and no automatic BOS/EOS. All300,963 next-token targets
are scored in 147 reset windows of up to2,048 targets, including the final
1,955 targets. The300,964-token stream has SHA256
`5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608`.
Pooled PPL is `exp(total NLL / 300963)`. Native parallel SSD prefill resets
each window; full-vocabulary head GEMMs feed FP32 cross entropy in 64-token
chunks. There is no persistent PPL cache or StateQuant requantization.

MK uses exactly 768 CONFIRM prompts:384 normal +384 with the queried binding
removed. N16 and N64 each have192 prompts per condition. Greedy decoding
emits at most12 tokens, stops at EOS, and checks the first standalone six-digit
answer. The canonical CONFIRM SHA256 is
`445f1ebae5cfeeafccc0b6c02f1a514d8d7858360637af9ec37ab5e48212a091`.
Target-removed matching is a removed-value control, not successful retrieval
from the prompt. Its zero result does not prove general hallucination resistance.

The GPU run used NVIDIA RTX PRO6000 Blackwell Server Edition, Python3.10.12,
Torch2.11.0+cu128, CUDA12.8, mamba-ssm2.3.2.post1, Triton3.6.0,
NumPy1.26.4, datasets4.8.5 and sentencepiece0.2.1. Singleton warp16 RMSNorm
was pinned before model loading; TF32 was off for matmul and cuDNN, FP32
matmul precision was `highest`, and eight CPU threads were used. Backend
and imported source identities stayed fixed through the paired run.

## Fixed repaired base and fresh adapter

The base is pure Mamba2-8B:56 blocks, 8,236,999,680 parameters and 507 tensors.
The114 large matrices use E2M1 FP4 codes with 16-element input-axis groups,
E4M3FN block scales and FP32 matrix scales. The393 existing small tensors
are FP16. [Base repair provenance](../evidence/repair_training_core.json)
binds TRAIN finetuning of those 393 tensors at selected step 448; the 114 FP4
matrices and 118 FP4 shard files retain their original identities and storage.

Base repair completed 896 successful updates over two fixed passes through 448
WT2 TRAIN windows. The fixed224/448/672/896 candidate pool chose448 by
minimum full-validation NLL, before its final test. That earlier
[base-only test comparison](../evidence/repair_test_comparison.json) recorded
PPL6.717854132348465; this adapter experiment independently reloads and
rescores the same base at 6.717800294314242. The gains above use only the new
same-process pair. The original FP16 control in base repair was an unfinetuned
source reference, not a matched post-training control.

All114 FP4 and 393 repaired FP16 tensors were frozen during fresh Resurface
training. The separate teacher loads the same repaired, unadapted base.
Only224 adapter tensors /1,154,104 parameters are learned. Fresh post-D soft
gates initialize V=0, g=1, router_w=0, router_b=-4. Numeric TRAIN CE1,
WT2 TRAIN prose CE0.5, repaired-base teacher KL0.5 and prose gate closure3
use native SSD forward/backward, without a StateQuant controller or surrogate
state-gradient replacement.

The [formal training report](../evidence/training_report.json) records1536
successful updates /1542 attempts /6 overflow retries and checkpoints384,
768,1152 and 1536. Only the final1536 export entered quality evaluation;
no PPL/MK adapter checkpoint selection occurred. All224 final FP16 tensors
equal saved FP32 masters cast to FP16; optimizer/scaler/checkpoint roundtrip
receipts pass. Independent teacher/student storage, 507 actual value checks
at both endpoints, initial/export/removal parity at 128/512 tokens and native
student/teacher restoration pass.

The initial smoke stopped before loading a model or performing an update:
backend pinning's Quack hardware query initialized CUDA before a source-closure
ordering assertion. The corrected trainer binds source first and rechecks it
after backend setup, with protocol/objective/loader unchanged. A fresh
[one-update retry](../evidence/smoke_report.json) passed1 update /1 attempt /
0 overflows and 128/512 parity; its export was discarded. Formal training
then restarted with fresh masters, optimizer and scaler.

Formal elapsed time is 1780.7513492107391 seconds (`finished_unix-started_unix`),
including loading/export/parity work. Full paired evaluation took
1505.3940229415894 seconds. These are job durations, not isolated throughput.

## Storage and CUDA memory

| Explicit component | Bytes |
| --- | ---: |
| Encoded base weight tensor payload | 4,638,460,360 |
| Actual119 base weight safetensors files | 4,638,539,864 |
| Final adapter FP16 tensor payload | 2,308,208 |
| Final serialized adapter | 2,381,119 |
| Base + adapter encoded tensor payload | 4,640,768,568 |
| Actual119 weights + adapter file | 4,640,920,983 |
| Decoded resident FP16 base weights | 16,473,999,360 |
| Batch1 MK native SSM cache | 117,440,512 |
| Batch1 MK native convolution cache | 4,587,520 |
| Total batch1 MK cache, 112 distinct FP16 tensors across56 layers | 122,028,032 |
| Added adapter recurrent cache | 0 |

The combined file total excludes metadata, tokenizer, code, licenses and
evidence. Decoded base weights, one MK cache and adapter FP16 payload sum
to16,598,335,600 bytes; this is a component sum, not total CUDA memory.
PPL's native SSD internal accumulator precision is separately scoped from
the inspected FP16 MK caches. The native reference expands all weights to
resident FP16; file compression does not demonstrate packed-VRAM inference.

| Observed CUDA scope | Peak allocated bytes | Peak reserved bytes |
| --- | ---: | ---: |
| Formal training, two expanded bases and adapter training | 34,600,171,520 | 34,781,265,920 |
| Evaluation baseline | 16,953,243,136 | 16,968,056,832 |
| Evaluation Resurface | 16,864,286,208 | 16,970,153,984 |
| Evaluation adapter removed | 16,849,382,912 | 16,970,153,984 |

Evaluation peaks are sequential one-process measurements containing temporary
buffers and allocator history. Their differences do not isolate adapter memory
or show an adapter-driven VRAM reduction. No throughput or ASIC result is inferred.

## Bound evidence and independent audit

| Compact evidence | Bytes | SHA256 |
| --- | ---: | --- |
| [Comparison](../evidence/comparison.json) | 235,203 | `f16227b149e1aa9c7113a35cd56ed1f95849ae6bdc6b7e1460d077a4847d91cb` |
| [Baseline](../evidence/baseline.json) | 3,868,006 | `ec407fad97262a991415c0cb1855d9ea1e67d118981a7f6ae7809561ee0b2a82` |
| [Resurface](../evidence/resurface.json) | 3,912,805 | `04047a78097be9412d4038886749d12721866d282828fa64e5c321608e1225a7` |
| [Removed](../evidence/adapter_removed.json) | 3,868,013 | `6e8a801ebe2632cae6426e66a22dea6f4ef7ea00a52ffec2fc727680ece91be8` |
| [Formal training](../evidence/training_report.json) | 1,084,073 | `84c4f5751726e035300f42ddaa63eae8faa748f192e3392a0c65626e42dda9ce` |
| [Training audit](../evidence/training_audit.json) | 1,905 | `a73d1a265608d203d05342c67b3790593e522bafda404654dd3c97a6364941a1` |
| [GPU-host combined CPU audit](../evidence/audit_gpu.json) | 3,935 | `d71975e3b80e5e4c5e7fda8fc54cc7eb98c09c1fb17c2b2531316b6ed4b89a4b` |
| [Mac combined CPU audit](../evidence/audit_mac.json) | 3,934 | `f3600b884bb60268dd8bf27c9651fe21cb8b93daac3046011b9e007577e15dae` |

The repaired weight manifest is
`2d2208b690c2ec0def90578d8590cc122bb1e4542b0d49ae4417a48f72b0b73c`.
The final adapter is
`dca3cda5a9ab8a665abd1a355c6e5b294acee023d2a1a5a8e20f4e84c9674580`.
The fixed [base selection](../evidence/repair_selection.json) is
`2f1ff559f3e907b3803bcc6d7036d39dab88c11dbb5e69f1de57e2dff5ca452c`.
The adapter protocol is
`9592a0dddcaa5c7d000422e5ac902b49df362781e4a4d9f668d99cb849cf8a7f`;
trainer/evaluator/auditor sources are bound by the reports and release manifest.

Both combined audits check complete reports, adapter/checkpoint file hashes,
TRAIN schedule/retry accounting, repaired507 identities, native parity/cache
receipts, saved token/window identities, all PPL/MK arithmetic, bootstrap
and exact removed-arm restoration. GPU-host audit used NumPy1.26.4;
Mac used NumPy2.3.5 and reproduced the same confidence interval. Neither
CPU audit imports Torch or recomputes GPU logits/hidden states. The GPU
evaluator streamed actual packed shards through the strict loader and checked
all 507 decoded values at load, after each arm and at completion; CPU audits
bind those recorded execution proofs. Torch's recorded shuffle is checked for
completeness and retry consistency, not independently regenerated.

Optimizer/checkpoint archives, full prose/token corpora and private job logs
are not part of the model download. The published audits record archive checks
already completed on the research artifacts; the download alone cannot repeat
the full training-archive audit. [Minimal reproduction](MINIMAL_REPRODUCTION.md)
provides complete-file verification, one-case strict recall smoke, prompt
generation and optional fresh full three-arm PPL/MK replay.

## Interpretation and licenses

WT2 validation/test and CONFIRM have prior project exposure, so they are not
untouched project holdouts. TRAIN optimization and fixed candidate selection
are separate from evaluation, but independent auditing does not remove that
historical exposure. Pretraining contamination is unaudited. These results
apply to the fixed measured protocol; longer contexts and other benchmarks
are unmeasured.

Read [LICENSES.md](../LICENSES.md) and [third-party notices](../THIRD_PARTY_NOTICES.md):
the modified base is Apache-2.0, while the project runtime and this new adapter
are GPL-3.0. No Quamba or research-only quantized checkpoint is used.
