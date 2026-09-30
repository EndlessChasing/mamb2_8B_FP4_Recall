# Mamb2_8B_FP4_Recall

**Full official WikiText-2 TEST PPL: 6.717746660814748 with Resurface, versus 6.717800294314242 without it. Normal CONFIRM MK: 353/384 (91.9270833%), versus 131/384 (34.1145833%).**

A member of the **Mamb2_8B_Recall** family: a pure Mamba2-8B base with FP4 G16
weights, repaired existing FP16 parameters, native FP16 recurrent state, and a
fresh Resurface adapter. There is **no StateQuant controller** in this model.

## Audited paired quality

| Same-process configuration | Full WT2 test PPL | Normal MK / 384 | Target-removed MK / 384 |
| --- | ---: | ---: | ---: |
| Repaired FP4 G16 base, no adapter | 6.717800294314242 | 131 (34.1145833%) | 0 |
| Same frozen base + final Resurface | **6.717746660814748** | **353 (91.9270833%)** | 0 |
| Same base after adapter removal | 6.717800294314242 | 131 (34.1145833%) | 0 |

| Normal MK population | Adapter off | Final Resurface on |
| --- | ---: | ---: |
| N16, 192 cases | 101/192 (52.6041667%) | **191/192 (99.4791667%)** |
| N64, 192 cases | 30/192 (15.625%) | **162/192 (84.375%)** |

The normal-MK gain is **222/384 = 57.8125 percentage points**, with a paired
bootstrap 95% interval of **52.6041667 to 62.7604167 percentage points**
(10,000 draws, seed 20260928). There are 225 improvements, 3 regressions and 156
unchanged normal pairs. Target-removed matches are 0/192 for each N in both
configurations. Measured PPL changes by only-0.0007983788910648215%; the
predeclared no-worse PPL gate passes. All147 window scores and 768 generated
sequences, native hooks/forwards and reset/cache probes restore exactly after
adapter removal. Both combined CPU audits pass; they check recorded evidence
and arithmetic without replaying GPU logits.

**Protocol:** official `Salesforce/wikitext`, `wikitext-2-raw-v1`, **test**,
revision `b08601e04326c79dfdd32d625aee71d232d685c3`: **147 reset windows /
300,963 targets**, final1,955-target partial window included. PPL uses native
parallel SSD prefill and pooled token NLL, full vocabulary, FP32 cross entropy
in 64-token chunks, no automatic BOS/EOS and no persistent request cache.
Recall uses 768 CONFIRM prompts (384 normal +384 target removed), greedy
at most12 tokens with EOS stop and first standalone six-digit answer matching.

**Data scope:** WT2 validation/test and the CONFIRM family have prior project
exposure. These are not untouched project holdouts. Base repair used TRAIN
and validation selection; the fresh adapter was TRAIN-only and its1536-step
candidate was fixed before evaluation. Independent report audits do not make
the evaluation data independent. Pretraining contamination is unaudited.

## Storage and native reference runtime

| Accounting scope | Exact bytes |
| --- | ---: |
| Base encoded weight tensor payload | 4,638,460,360 |
| Actual119 base safetensors files | 4,638,539,864 |
| Adapter FP16 tensor payload, 1,154,104 parameters | 2,308,208 |
| Serialized final adapter file | 2,381,119 |
| Actual119 weight files + adapter file | **4,640,920,983** |
| Decoded resident FP16 base weights | **16,473,999,360** |
| Batch1 native MK SSM + convolution caches | **122,028,032** |
| Additional adapter recurrent cache | **0** |

The combined file total excludes metadata, tokenizer, source, licenses and
evidence. The complete download is therefore larger. The native quality and
inference implementation expands packed weights to resident FP16. It needs
activations, request buffers, logits and allocator reserve beyond the table.
The stored4.641GB core files do not establish4.641GB VRAM inference.
Batch1 MK inspects112 distinct FP16 cache tensors across56 layers:117,440,512
SSM bytes +4,587,520 convolution bytes. PPL follows the native SSD scan path;
its internal accumulator precision is not a claim that every intermediate is
FP16. No packed-resident FP4 kernel, measured throughput benefit or ASIC
implementation is provided.

## What was trained

The base has 56 blocks, 8,236,999,680 parameters and 507 tensors. Its114 large
matrices use E2M1 FP4 codes in 16-element input-axis groups, E4M3FN block scales
and FP32 global scales. Independent weight-only conversion retains393 small
FP16 tensors. This is an NVFP4-style representation with a custom range search,
not a claim of hardware-native NVFP4 execution.

Base-only repair then trained those 393 existing FP16 tensors (3,580,928
parameters), leaving all 114 FP4 matrices and their118 shard files unchanged.
Two fixed passes over 448 WT2 TRAIN windows give896 successful updates. The
fixed224/448/672/896 pool selected step 448 by minimum full validation NLL.
Its earlier full test PPL was6.717854132348465; the current paired control
is 6.717800294314242. The original FP16 source used in base repair was an
unfinetuned reference, not a matched post-training control.

For this fresh post-D Resurface run, **all 114 FP4 matrices and 393 repaired
FP16 tensors were frozen**. Only224 adapter tensors /1,154,104 parameters
were trained, starting from V=0, g=1, router_w=0, router_b=-4. Numeric TRAIN CE1,
WT2 TRAIN prose CE0.5, separate unadapted repaired-base teacher KL0.5 and prose
gate closure3 used native SSD forward/backward. Formal training completed
1536 successful updates /1542 attempts /6 overflow retries. Only the final
1536-step FP16 export was a quality candidate; no PPL/MK checkpoint selection
was used. Export/readback, native128/512 parity and frozen507 identities pass.
Adapters from other family variants are not interchangeable.

## Download and run

The GitHub source checkout includes code and compact evidence; full weights are
release assets. Download all 119 safetensors assets and the runtime archive:

```bash
gh release download v0.1.0-fp4g16-repaired-s16-resurface \
  --repo EndlessChasing/mamb2_8B_FP4_Recall --pattern '*-runtime.tar.gz'
tar -xzf mamba2-8b-fp4g16-repaired-s16-resurface-v1-runtime.tar.gz
gh release download v0.1.0-fp4g16-repaired-s16-resurface \
  --repo EndlessChasing/mamb2_8B_FP4_Recall --pattern '*.safetensors' \
  --dir mamba2-8b-fp4g16-repaired-s16-resurface-v1/weights
cd mamba2-8b-fp4g16-repaired-s16-resurface-v1
```

```bash
python scripts/infer_fp4_repaired_s16_resurface_v1.py --bundle . --verify-only
python scripts/infer_fp4_repaired_s16_resurface_v1.py --bundle . --smoke-check
python scripts/infer_fp4_repaired_s16_resurface_v1.py --bundle . \
  --prompt 'The capital of France is' --max-new-tokens 64
```

`--verify-only` is standard-library CPU verification. `--smoke-check` strictly
replays one published normal recall sample, not the full benchmark.
`--without-adapter` with a prompt runs the same repaired base alone. The
custom source loads directly; there is no `AutoModel.from_pretrained` support.
Generation needs the pinned CUDA/Mamba/Triton environment and resident decoded
FP16 weights. Do not install the bundle itself with pip or create a venv inside it.


The [HF snapshot](https://huggingface.co/EndlessChasing/Mamb2_8B_FP4_Recall/tree/v0.1.0-fp4g16-repaired-s16-resurface) contains the complete identical bundle
under `release/`. See [minimal reproduction](docs/MINIMAL_REPRODUCTION.md)
for the pinned installation, HF download and optional full three-arm replay.

## Identity and license scope

The modified base is derived from the Apache-2.0
[NVIDIA pure Mamba2 checkpoint](https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/b915550c63ba9359f88f44d1f6a600d85af27302).
The modified base weights are Apache-2.0; inherited/project runtime and this
new Resurface adapter are GPL-3.0. The preserved StateQuant source reference
retains Apache-2.0, but no StateQuant execution or state configuration is used.
No Quamba quantized checkpoint or research-only weight package is used.

- Repaired packed manifest: `2d2208b690c2ec0def90578d8590cc122bb1e4542b0d49ae4417a48f72b0b73c`.
- Final adapter: `dca3cda5a9ab8a665abd1a355c6e5b294acee023d2a1a5a8e20f4e84c9674580`.
- Paired full comparison: `f16227b149e1aa9c7113a35cd56ed1f95849ae6bdc6b7e1460d077a4847d91cb`.
- Independent Mac combined audit: `f3600b884bb60268dd8bf27c9651fe21cb8b93daac3046011b9e007577e15dae`.

The base weight and adapter licenses apply to their identified artifacts;
preserve their notices and included source when redistributing. These independent
modifications do not imply endorsement by the original authors.

See [portable full results](docs/RESULTS.md), [LICENSES.md](LICENSES.md),
[third-party notices](THIRD_PARTY_NOTICES.md) and the exact [evidence](evidence/).
