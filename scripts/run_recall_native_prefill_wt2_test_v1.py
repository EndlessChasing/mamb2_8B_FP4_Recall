#!/usr/bin/env python3
"""Frozen published FP16/INT4 Recall checkpoints: complete official WT2 test PPL."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/RECALL_NATIVE_PREFILL_WT2_TEST_V1_PROTOCOL.md'
PINS = ROOT / 'docs/RECALL_NATIVE_PREFILL_WT2_TEST_V1_PINS.json'
PINS_SHA256 = '7e85a47cad0733ac386205dd91438f9df0b64c3c92706529c8109f5e428f40aa'
FORMAT = 'RECALL_NATIVE_PREFILL_WT2_TEST_V1'
ARMS = ('without_resurface', 'resurface')
PACKAGES = {'torch': '2.11.0+cu128', 'mamba-ssm': '2.3.2.post1',
            'numpy': '1.26.4', 'triton': '3.6.0', 'datasets': '4.8.5',
            'sentencepiece': '0.2.1'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def put(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def preflight(kind, release, adapter, w4_dir):
    need(sha(PINS) == PINS_SHA256, 'Frozen test pins changed')
    pins = json.loads(PINS.read_text())
    fixed = pins['models'][kind]
    for relative, expected in fixed['frozen_files'].items():
        need(sha(release / relative) == expected, 'Released file changed: ' + relative)
    need(sha(adapter) == fixed['adapter']['sha256'], 'Published adapter changed')
    need(adapter.stat().st_size == fixed['adapter']['bytes'], 'Adapter size differs')
    train_relative = next(n for n in fixed['frozen_files'] if n.endswith('_train.json'))
    train = json.loads((release / train_relative).read_text())
    need(train['complete'] and train['successful_updates'] == 1536 and
         train['binding'] == fixed['adapter_binding'] and train['adapter'] == fixed['adapter'],
         'Published training checkpoint binding differs')
    if kind == 'w4':
        need(w4_dir is not None, 'W4 requires --w4-dir')
        need(sha(w4_dir / 'manifest.json') == fixed['w4_manifest_sha256'], 'Published W4 manifest differs')
    else:
        need(w4_dir is None, 'FP16 must not supply a W4 package')
    return pins, fixed


def native_snapshot(model):
    return [(module, module.forward.__func__ if hasattr(module.forward, '__func__') else module.forward,
             {name: tuple(getattr(module, name).items()) for name in
              ('_forward_pre_hooks', '_forward_hooks', '_backward_hooks')})
            for module in model.modules()]


def check_native_snapshot(snapshot):
    for module, forward, hooks in snapshot:
        now = module.forward.__func__ if hasattr(module.forward, '__func__') else module.forward
        need(now is forward, 'Native forward method was replaced')
        for name, entries in hooks.items():
            need(tuple(getattr(module, name).items()) == entries, 'Native hook ownership was not restored')
    return {'complete': True, 'modules': len(snapshot), 'forward_methods_and_hooks_exact': True}


def base_identities(model):
    return {n: (id(p), p.data_ptr(), p._version) for n, p in model.named_parameters()}


def check_base_identity(model, before):
    now = dict(model.named_parameters())
    need(set(now) == set(before), 'Base parameter inventory changed')
    for name, p in now.items():
        need((id(p), p.data_ptr(), p._version) == before[name] and not p.requires_grad and p.grad is None,
             'Base identity/version/gradient changed: ' + name)
    return {'complete': True, 'tensors': len(now), 'parameters': sum(p.numel() for p in now.values()),
            'identity_version_gradients_unchanged': True}


def source_ledger(runtime, source_dir, torch):
    # Independent expected bytes from verified BF16 source, not the loaded GPU model.
    state = runtime.load_source_state(source_dir)
    ledger = {}
    for name, value in state.items():
        digest = hashlib.sha256()
        if value.ndim == 0:
            digest.update(value.half().contiguous().numpy().tobytes())
        else:
            for first in range(0, value.shape[0], 128):
                digest.update(value[first:first+128].to(dtype=torch.float16).contiguous().numpy().tobytes())
        ledger[name] = digest.hexdigest()
    del state
    return ledger


def check_content(model, expected, native):
    actual = {n: native.tensor_hash(p) for n, p in model.named_parameters()}
    need(actual == expected, 'Actual loaded FP16 weights differ from fixed source/package')
    return {'complete': True, 'tensors': len(actual), 'actual_content_checked': True,
            'actual_tensor_sha256': actual}


def adapter_receipt(values, fixed, native):
    hashes = {name: native.tensor_hash(value) for name, value in values.items()}
    payload = sum(value.numel() * value.element_size() for value in values.values())
    need(hashes == fixed['adapter']['tensor_sha256'] and payload == fixed['adapter']['payload_bytes'],
         'Actual adapter tensors differ from published export')
    return {'complete': True, 'tensor_sha256': hashes, 'payload_bytes': payload,
            'parameters': sum(value.numel() for value in values.values()), 'tensor_count': len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-kind', choices=('fp16', 'w4'), required=True)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--w4-dir', type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    release = args.release_root.resolve()
    fixed = json.loads(PINS.read_text())['models'][args.model_kind]
    adapter = args.adapter or release / fixed['adapter_path']
    pins, fixed = preflight(args.model_kind, release, adapter, args.w4_dir)
    if args.preflight_only:
        print(json.dumps({'complete': True, 'model_kind': args.model_kind, 'pins_sha256': sha(PINS),
                          'verified_files': len(fixed['frozen_files']), 'adapter_sha256': sha(adapter),
                          'ppl_execution': 'prefill', 'torch_imported': 'torch' in sys.modules,
                          'cuda_initialized': False}, indent=2))
        return
    need(args.out_dir is not None and not args.out_dir.exists(), 'Fresh --out-dir required')
    # Import exactly one published package per process, never the newer research runtime.
    need('mamba2_recall' not in sys.modules, 'A different runtime was already imported')
    sys.path.insert(0, str(release))
    from mamba2_recall import runtime, evaluation, calibration, resurface_native as native
    import torch
    need(all(Path(m.__file__).resolve().parent == release / 'mamba2_recall'
             for m in (runtime, evaluation, calibration, native)), 'Runtime import escaped frozen release')
    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    need(packages == PACKAGES and torch.version.cuda == '12.8', 'Recorded release backend versions required')
    need(torch.cuda.is_available(), 'CUDA quality reference required')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    args.out_dir.mkdir(parents=True)
    started = time.time()
    report = {'format': FORMAT, 'complete': False, 'model_kind': args.model_kind,
              'protocol_sha256': sha(PROTOCOL), 'pins_sha256': sha(PINS), 'runner_sha256': sha(__file__),
              'release_binding': fixed, 'reports': {}, 'checkpoint_frozen_before_test': True,
              'test_used_for_training_or_selection': False,
              'project_test_text_seen_in_earlier_2p7b_experiments': True,
              'scope': 'Additional official WT2 test PPL of fixed published checkpoints; native parallel SSD prefill. '
                       'Base pretraining contamination and cross-split text duplicates not audited.',
              'environment': runtime.environment_receipt(),
              'backend': {'packages': packages, 'cuda_runtime': torch.version.cuda, 'num_threads': torch.get_num_threads(),
                          'float32_matmul_precision': torch.get_float32_matmul_precision(),
                          'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                          'tf32_cudnn': torch.backends.cudnn.allow_tf32,
                          'fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                          'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}}
    path = args.out_dir / 'comparison.json'
    put(path, report)
    try:
        tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
        ids, dataset = calibration.load_wikitext_tokens(tokenizer, 'test')
        need(dataset == pins['test_dataset'], 'Pinned official WT2 test identity differs')
        windows = evaluation.ppl_windows(ids, 2048)
        need(len(windows) == pins['window_count'] and len(ids)-1 == pins['target_tokens'] and
             sum(len(w)-1 for _, w in windows) == len(ids)-1, 'Full test targets omitted or repeated')
        raw = ids.numpy().astype('<i8', copy=False).tobytes()
        (args.out_dir / 'tokens.int64le').write_bytes(raw)
        need(hashlib.sha256(raw).hexdigest() == dataset['token_stream_sha256_int64le'], 'Saved tokens differ')
        payload = native.read_fp16(adapter, expected_binding=fixed['adapter_binding'])
        need(payload['gate_mode'] == 'soft', 'Published gate mode changed')
        report['published_adapter_content'] = adapter_receipt(payload['tensors'], fixed, native)
        report.update(dataset=dataset, tokens_file={'file': 'tokens.int64le', 'bytes': len(raw),
                     'sha256': sha(args.out_dir / 'tokens.int64le')},
                     windowing={'targets_per_full_window': 2048, 'logits_chunk_tokens': 64,
                                'state_reset_each_window': True, 'final_partial_included': True,
                                'state_requantized_each_token': False, 'execution': 'prefill',
                                'window_count': len(windows), 'target_tokens': len(ids)-1})
        if args.model_kind == 'fp16':
            expected = source_ledger(runtime, args.source_dir, torch)
            model = runtime.load_source_model(args.source_dir)
            provenance = {'method': 'Verified pinned BF16 source cast to FP16 in bounded CPU row chunks',
                          'source_checkpoint_sha256': fixed['source_checkpoint_sha256']}
        else:
            from mamba2_recall.w4 import load_w4_model
            manifest = json.loads((args.w4_dir / 'manifest.json').read_text())
            expected = {name: entry['decoded_sha256'] for name, entry in manifest['tensors'].items()}
            # Preserve exact manifest bytes for an independently checkable expected ledger.
            (args.out_dir / 'w4_manifest.json').write_bytes((args.w4_dir / 'manifest.json').read_bytes())
            model = load_w4_model(args.w4_dir)
            provenance = {'method': 'Pinned published W4 manifest decoded FP16 hashes; original strict loader',
                          'manifest_sha256': fixed['w4_manifest_sha256'], 'manifest_file': 'w4_manifest.json'}
        need(len(expected) == 507 and sum(p.numel() for p in model.parameters()) == 8236999680,
             'Original 507-tensor pure Mamba2-8B inventory differs')
        put(args.out_dir / 'weight_ledger.json', {'provenance': provenance, 'tensor_sha256': expected})
        report.update(weight_ledger={'file': 'weight_ledger.json', 'sha256': sha(args.out_dir / 'weight_ledger.json')},
                      weight_provenance=provenance, package_receipt=model._package_receipt,
                      initial_weight_check=check_content(model, expected, native))
        snapshot = native_snapshot(model)
        identities = base_identities(model)
        probe = windows[0][1][:128].cuda()[None]
        with torch.inference_mode():
            original_probe = native.tensor_hash(model.backbone(probe))
        report['unadapted_probe_hidden_sha256'] = original_probe
        put(path, report)
        rows = {}
        for arm in ARMS:
            adapted = arm == 'resurface'
            manager = (native.install_fp16(model, adapter, expected_binding=fixed['adapter_binding'],
                       expected_base_hashes=expected) if adapted else contextlib.nullcontext())
            torch.cuda.reset_peak_memory_stats()
            with manager as bank:
                row = {'format': FORMAT, 'arm': arm, 'model_kind': args.model_kind, 'complete': False,
                       'adapter_loaded': adapted, 'adapter_sha256': fixed['adapter']['sha256'] if adapted else None,
                       'dataset': dataset, 'ppl': evaluation.evaluate_ppl(model, windows, execution='prefill', logits_chunk=64)}
                if adapted:
                    row['loaded_adapter_content'] = adapter_receipt(bank.masters, fixed, native)
                    row['frozen_base_check'] = bank.assert_base_frozen(check_values=True)
                row['base_identity'] = check_base_identity(model, identities)
            row.update(complete=True, native_restoration=check_native_snapshot(snapshot),
                       cuda_memory={'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                                    'scope': 'Resident decoded FP16 weights plus runtime allocations; not encoded model size.'})
            arm_path = args.out_dir / (arm + '.json')
            put(arm_path, row)
            rows[arm] = row
            report['reports'][arm] = {'file': arm_path.name, 'sha256': sha(arm_path), 'ppl': row['ppl']['ppl']}
            put(path, report)
        with torch.inference_mode():
            restored_probe = native.tensor_hash(model.backbone(probe))
        need(restored_probe == original_probe, 'Adapter removal failed bitwise native prefill hidden-state replay')
        report['adapter_removal_probe'] = {'complete': True, 'hidden_sha256': restored_probe,
                                          'matches_unadapted_bitwise': True, 'tokens': 128}
        report['final_native_restoration'] = check_native_snapshot(snapshot)
        report['final_base_identity'] = check_base_identity(model, identities)
        report['final_weight_check'] = check_content(model, expected, native)
        left, right = (rows[a]['ppl'] for a in ARMS)
        need([(r['start'], r['target_tokens'], r['token_sha256_int64le']) for r in left['windows']] ==
             [(r['start'], r['target_tokens'], r['token_sha256_int64le']) for r in right['windows']],
             'Unmatched official test populations')
        report.update(complete=True, ppl={a: rows[a]['ppl']['ppl'] for a in ARMS},
                      ppl_relative_change=right['ppl']/left['ppl']-1,
                      matched_windows_improved=sum(r['nll'] < l['nll'] for l, r in zip(left['windows'], right['windows'])),
                      elapsed_seconds=time.time()-started)
        put(path, report)
        print(json.dumps({k: report[k] for k in ('complete', 'model_kind', 'ppl', 'ppl_relative_change')}, indent=2), flush=True)
    except Exception as error:
        report.update(complete=False, error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.time()-started)
        put(path, report)
        raise


if __name__ == '__main__':
    main()
