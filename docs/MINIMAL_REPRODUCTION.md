# Verify and run Mamb2_8B_FP4_Recall

The complete bundle is sufficient for inference. The original NVIDIA checkpoint
is not required. The Python CLI uses the included custom source directly and
loads the packed weights into resident FP16; it has no Transformers
`AutoModel.from_pretrained` entry point or packed-resident FP4 kernel.

## Download a complete pinned bundle

### Hugging Face

```bash
hf download EndlessChasing/Mamb2_8B_FP4_Recall \
  --revision v0.1.0-fp4g16-repaired-s16-resurface \
  --local-dir Mamb2_8B_FP4_Recall
cd Mamb2_8B_FP4_Recall/release
```

### GitHub Release

Download all 119 weight assets and the runtime archive into a fresh directory:

```bash
gh release download v0.1.0-fp4g16-repaired-s16-resurface \
  --repo EndlessChasing/mamb2_8B_FP4_Recall \
  --pattern '*-runtime.tar.gz'
tar -xzf mamba2-8b-fp4g16-repaired-s16-resurface-v1-runtime.tar.gz
gh release download v0.1.0-fp4g16-repaired-s16-resurface \
  --repo EndlessChasing/mamb2_8B_FP4_Recall \
  --pattern '*.safetensors' \
  --dir mamba2-8b-fp4g16-repaired-s16-resurface-v1/weights
cd mamba2-8b-fp4g16-repaired-s16-resurface-v1
```

The runtime archive includes weight metadata, adapter, tokenizer, custom source,
license texts and compact evidence. A source-only Git checkout does not contain
the 119 large weight files. The HF snapshot's `release/` is the same complete
bundle content as the assembled GitHub download.

## CPU-only complete-file verification

From the complete bundle root:

```bash
python scripts/infer_fp4_repaired_s16_resurface_v1.py --bundle . --verify-only
```

This uses Python's standard library, checks the declared inventory and file
SHA256 identities, and does not load Torch or initialize CUDA. It verifies the
files; it does not execute the model or remeasure PPL/MK.

## Compatible CUDA environment

The measured reference uses Linux, Python3.10.12, PyTorch2.11.0+cu128,
CUDA12.8, Triton3.6.0, mamba-ssm2.3.2.post1, NumPy1.26.4,
datasets4.8.5 and sentencepiece0.2.1, on NVIDIA RTX PRO6000 Blackwell
Server Edition. Create the environment **beside the bundle**, preserving its
strict inventory. The CLI loads its included source directly; installing the
bundle itself with pip is unnecessary.

With Python3.10 and a matching CUDA build toolchain:

```bash
python3.10 -m venv ../fp4-recall-env
source ../fp4-recall-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'triton==3.6.0' 'numpy==1.26.4' \
  'sentencepiece==0.2.1' 'einops==0.8.2' 'safetensors==0.8.0' \
  'datasets==4.8.5' 'huggingface-hub==0.36.2' 'packaging==26.3' 'ninja==1.13.2'
python -m pip install --no-build-isolation \
  'mamba-ssm @ git+https://github.com/state-spaces/mamba.git@e9594ce1c732d97440f0332fdc43170a2294dbfa'
```

Mamba may compile an extension for the installed PyTorch/CUDA ABI. The runtime
checks bound package versions, external kernel source hashes and the singleton
warp16 RMSNorm policy, with TF32 off. Version strings alone do not guarantee
exact numerical replay on another stack. The complete base expands to
16,473,999,360 bytes of FP16 weight tensors; request buffers and activations
need additional CUDA memory.

## One published recall sample and generation

```bash
python scripts/infer_fp4_repaired_s16_resurface_v1.py --bundle . --smoke-check
python scripts/infer_fp4_repaired_s16_resurface_v1.py --bundle . \
  --prompt 'The capital of France is' --max-new-tokens 64
python scripts/infer_fp4_repaired_s16_resurface_v1.py --bundle . \
  --without-adapter --prompt 'The capital of France is' --max-new-tokens 64
```

`--smoke-check` replays the first published normal Resurface case and checks its
prompt-token identity, generated-token sequence and actual native FP16 cache
receipt. It covers one sample, not the full147-window PPL or768-case MK run.
`--without-adapter` uses the same repaired base without the added adapter.

## Optional full paired quality replay

The frozen evaluator and all its bound source files are included. From the
bundle root, copy the exact small training report and adapter to a fresh sibling
directory; leave the complete verified bundle unchanged:

```bash
mkdir ../fp4-recall-replay-formal
cp evidence/training_report.json ../fp4-recall-replay-formal/report.json
cp adapter_fp16.pt ../fp4-recall-replay-formal/adapter_fp16.pt
python scripts/run_fp4_repaired_s16_resurface_v1.py \
  --packed-dir weights --source-dir tokenizer \
  --repair-selection evidence/repair_selection.json \
  --repair-test-comparison evidence/repair_test_comparison.json \
  --training-report ../fp4-recall-replay-formal/report.json \
  --training-audit evidence/training_audit.json \
  --out-dir ../fp4-recall-replay-evaluation
```

This downloads the pinned WikiText test split and evaluates all three arms:
baseline, final Resurface and adapter removed. It measures147 reset windows /
300,963 targets and 768 CONFIRM cases, then checks exact restoration and the
predeclared gates. Its saved token file and outputs are written outside the
bundle. Use a fresh output directory and the recorded backend. A new run is a
new measurement; the reported values are bound to the published evidence.

The shipped CPU audit receipts describe checks already completed on GPU-host
and Mac copies. Full training-checkpoint/archive audit reproduction additionally
needs the four progress checkpoint/optimizer archives, which are research
artifacts excluded from the model download. The shipped reports and hashes
do not replay those tensor archives or GPU logits.

Read [measured results](RESULTS.md), [license scope](../LICENSES.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md) for the accounting and provenance.
