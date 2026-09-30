#!/usr/bin/env python3
"""Generate with the complete repaired FP4 G16/native-FP16 Resurface bundle.

Real packed weights are expanded into the original FP16 Mamba2 runtime.
No source checkpoint, StateQuant controller or extra state configuration is used.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import contextlib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_fp4_repaired_s16_resurface_v1 import verify_bundle, need, sha_file

FORMAT = 'MAMBA2_FP4_G16_REPAIRED_S16_RESURFACE_INFERENCE_V1'


def digest(values):
    raw = (json.dumps(values, sort_keys=True, separators=(',', ':')) + '\n').encode()
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--prompt')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--verify-only', action='store_true', help='Full file/shard hash verification with the standard library; no Torch/CUDA')
    parser.add_argument('--without-adapter', action='store_true', help='Use the identical repaired base without the final Resurface adapter')
    parser.add_argument('--smoke-check', action='store_true', help='Replay one released normal CONFIRM generated-token sequence exactly')
    parser.add_argument('--out', type=Path, help='Fresh receipt path outside the sealed bundle')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest = verify_bundle(bundle)
    if args.out:
        need(not args.out.exists() and not args.out.is_symlink()
             and not args.out.resolve().is_relative_to(bundle), 'Fresh receipt path outside sealed bundle required')
    if args.verify_only:
        result = dict(format=FORMAT, complete=True, verified=True,
            independent_full_file_and_weight_region_hashes=True,
            bundle_manifest_sha256=sha_file(bundle / 'manifest.json'),
            candidate_ppl=manifest['candidate_ppl'], baseline_ppl=manifest['baseline_ppl'],
            torch_imported='torch' in sys.modules, cuda_initialized=False, scope=manifest['scope'])
        need(result['torch_imported'] is False, 'Verification-only mode must not import Torch')
    else:
        need(ROOT.resolve() == bundle, 'Run the released bundle/scripts/infer entrypoint so all executed local sources come from the verified bundle')
        if args.max_new_tokens <= 0 or args.max_new_tokens > 4096 or (args.prompt is None and not args.smoke_check):
            parser.error('Provide --prompt or --smoke-check and 1 to4096 max-new-tokens')
        if args.smoke_check and (args.without_adapter or args.prompt is not None):
            parser.error('--smoke-check validates the released adapted case without a custom prompt')
        import torch
        from mamba2_recall import runtime, resurface_native as native, fp4
        from mamba2_recall.fp4_repaired_checkpoint import load_repaired_model
        from evaluate_resurface_more import pin_replay_backend, check_replay_backend
        from run_fp4_repaired_s16_resurface_v1 import cache_receipt
        from run_recall_native_prefill_wt2_test_v1 import (
            native_snapshot, check_native_snapshot, base_identities, check_base_identity)
        need(torch.cuda.is_available(), 'Recorded CUDA/Mamba/Triton execution stack required')
        torch.set_num_threads(8)
        torch.manual_seed(2026092803)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        policy = pin_replay_backend()
        need(policy == manifest['backend_policy'], 'Measured external source/version/pinned backend differs')
        torch.cuda.manual_seed_all(2026092803)
        tokenizer = runtime.SentencePieceTokenizer(bundle / 'tokenizer')
        expected_case = None
        if args.smoke_check:
            arm = json.loads((bundle / 'evidence/resurface.json').read_text())
            expected_case = next(row for row in arm['mk']['rows'] if row['condition'] == 'normal')
            args.prompt = expected_case['prompt']
            args.max_new_tokens = len(expected_case['generated_ids'])
        prompt_ids = tokenizer.encode(args.prompt)
        need(0 < len(prompt_ids) <= 4096, 'Prompt must contain1 to4096 tokens')
        if expected_case:
            need(runtime.token_digest(prompt_ids) == expected_case['prompt_token_sha256_int64le'],
                 'Released smoke prompt-token identity differs')
        model = load_repaired_model(bundle / 'weights', device='cuda', chunk_rows=512)
        weight_manifest = json.loads((bundle / 'weights/weight_manifest.json').read_text())
        expected_hashes = weight_manifest['decoded_tensor_sha256']
        need(len(expected_hashes) == 507 and model._fp4_repaired_receipt['decoded_tensor_sha256'] == expected_hashes
             and model._fp4_repaired_receipt['packed_checkpoint_manifest_sha256'] == manifest['repaired_weight_manifest_sha256'],
             'Actual repaired507 decode differs from published base')
        identities, snapshot = base_identities(model), native_snapshot(model)
        training = json.loads((bundle / 'evidence/training_report.json').read_text())
        payload = native.read_fp16(bundle / 'adapter_fp16.pt', expected_binding=manifest['adapter_binding'])
        adapter_hashes = {n: native.tensor_hash(v) for n, v in payload['tensors'].items()}
        need(payload['gate_mode'] == 'soft' and adapter_hashes == training['adapter']['tensor_sha256']
             and len(adapter_hashes) == 224, 'Actual final224 adapter tensors differ')
        manager = (contextlib.nullcontext() if args.without_adapter else native.install_fp16(
            model, bundle / 'adapter_fp16.pt', expected_binding=manifest['adapter_binding'],
            expected_base_hashes=expected_hashes))
        generated = []
        with torch.inference_mode(), manager as bank:
            tokens = torch.tensor([prompt_ids], dtype=torch.long, device='cuda')
            cache = runtime.make_cache(model, len(prompt_ids) + args.max_new_tokens, state_dtype=torch.float16)
            hidden = model.backbone(tokens, inference_params=cache)[:, -1:]
            storage = cache_receipt(cache, torch, native)
            for position in range(args.max_new_tokens):
                logits = model.lm_head(hidden[:, -1:])
                need(bool(torch.isfinite(logits).all()), 'Nonfinite native generation logits')
                token = int(logits.argmax(-1).item())
                generated.append(token)
                if token == tokenizer.eos_token_id:
                    break
                # Keep the published MK algorithm, including the final unused
                # recurrent update when the generation budget is exhausted.
                cache.seqlen_offset = len(prompt_ids) + position
                hidden = model.backbone(torch.tensor([[token]], device='cuda'), inference_params=cache)
            need(cache_receipt(cache, torch, native) == storage, 'Native cache dtype/storage changed in decoding')
            if bank is not None:
                need({n: native.tensor_hash(v) for n, v in bank.masters.items()} == adapter_hashes,
                     'Deployed adapter changed')
                bank.assert_base_frozen(check_values=False)
            identity_check = check_base_identity(model, identities)
            backend_check = check_replay_backend(policy)
        restored = check_native_snapshot(snapshot)
        if expected_case:
            need(generated == expected_case['generated_ids'], 'One released generated-token sequence does not exactly replay')
            need(storage == expected_case['actual_cache'], 'Released native FP16 cache receipt differs')
        result = dict(format=FORMAT, complete=True, prompt=args.prompt,
            completion=tokenizer.decode(generated), generated_token_ids=generated,
            prompt_tokens=len(prompt_ids), adapter_active=not args.without_adapter,
            smoke_case_id=None if expected_case is None else expected_case['id'],
            published_mk_generation_exact=expected_case is not None,
            execution='native parallel SSD prefill then native recurrent cached decoding',
            statequant_controller_installed=False, actual_native_cache=storage,
            additional_adapter_recurrent_bytes=0,
            repaired507_decoded_hashes_verified=True, decoded_tensor_ledger_sha256=digest(expected_hashes),
            repaired_weight_manifest_sha256=manifest['repaired_weight_manifest_sha256'],
            adapter_tensor_count=0 if args.without_adapter else 224,
            adapter_fp16_payload_bytes=0 if args.without_adapter else 2308208,
            adapter_tensor_ledger_sha256=None if args.without_adapter else digest(adapter_hashes),
            frozen_base_identity=identity_check, native_restoration=restored,
            backend_policy_check=backend_check, packed_resident_kernel=False,
            decoded_resident_weight_bytes=manifest['decoded_resident_weight_bytes'],
            bundle_manifest_sha256=sha_file(bundle / 'manifest.json'),
            scope='Exact weights/adapter plus one generated sequence for smoke; not a repeat of full PPL/MK quality evaluation')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
