#!/usr/bin/env python3
"""Final-only repaired FP4 G16/native FP16 Resurface: full WT2 test and CONFIRM.

The base is the fixed448-step repaired checkpoint. Its 393 retained FP16
tensors are part of the frozen507 base, not inherited from the old conversion
ledger. Small preflight reads no Torch or large weight shards.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORMAT = 'FP4_REPAIRED_S16_RESURFACE_EVAL_V1'
TRAIN_FORMAT = 'FP4_REPAIRED_S16_RESURFACE_TRAIN_V1'
PROTOCOL_SHA = '9592a0dddcaa5c7d000422e5ac902b49df362781e4a4d9f668d99cb849cf8a7f'
LEDGER_SHA = '55de4189a7e04c9564a48b0fbb5f37428d3d1bb5b6274831d878b41ef4686c9b'
ORIGINAL_MANIFEST_SHA = '4e7fb60f5d03c61e63324a6c25847014a26f9aa0c8d5a8c922d0f367e514fe07'
PACKED_MANIFEST_SHA = '2d2208b690c2ec0def90578d8590cc122bb1e4542b0d49ae4417a48f72b0b73c'
REPAIR_SELECTION_SHA = '2f1ff559f3e907b3803bcc6d7036d39dab88c11dbb5e69f1de57e2dff5ca452c'
REPAIR_COMPARISON_SHA = 'c3a38223055d53603a64302b0acdbd8e236baa402d67e6927a126848ab21522a'
REPAIR_CORE_SHA = 'd5fb50f2e045b6202a5f38b353ceab7ae47b21c7cc88e8cb398b46c342632c6e'
TOKENIZER_SHA = '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09'
TEST_TOKEN_SHA = '5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608'
TEST_TEXT_SHA = '696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83'
DATA_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
CONFIRM_SHA = '445f1ebae5cfeeafccc0b6c02f1a514d8d7858360637af9ec37ab5e48212a091'
ARMS = ('baseline', 'resurface', 'adapter_removed')
PACKAGES = {'torch': '2.11.0+cu128', 'mamba-ssm': '2.3.2.post1', 'numpy': '1.26.4',
            'triton': '3.6.0', 'datasets': '4.8.5', 'sentencepiece': '0.2.1'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def put(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def safe_source(name):
    need(isinstance(name, str) and not name.startswith('/') and '\\' not in name and ':' not in name,
         'Unsafe bound source filename')
    path = ROOT
    for part in name.split('/'):
        need(part not in ('', '.', '..'), 'Unsafe bound source component')
        path = path / part
        need(not path.is_symlink(), 'Bound source symlink forbidden')
    need(path.is_file(), 'Bound source missing: ' + name)
    return path


def code_hashes():
    names = ('scripts/run_fp4_repaired_s16_resurface_v1.py', 'docs/FP4_REPAIRED_S16_RESURFACE_V1_PROTOCOL.md',
             'mamba2_recall/fp4_repaired_checkpoint.py',
             'mamba2_recall/fp4_checkpoint.py',
             'mamba2_recall/fp4.py', 'mamba2_recall/runtime.py', 'mamba2_recall/evaluation.py',
             'mamba2_recall/calibration.py', 'mamba2_recall/resurface_native.py',
             'mamba2_recall/resurface_data.py', 'scripts/run_recall_native_prefill_wt2_test_v1.py')
    files = {safe_source(name).resolve() for name in names}
    for module in tuple(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if value:
            path = Path(value).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT) and path.is_file():
                files.add(path)
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}


def base_preflight(args):
    """Bind small repaired-base provenance without Torch or weight shard reads."""
    need(sha(args.protocol) == args.expected_protocol_sha256 == PROTOCOL_SHA, 'Frozen repaired V1 protocol changed')
    from mamba2_recall.fp4_repaired_checkpoint import read_repaired_metadata
    manifest, ledger = read_repaired_metadata(args.packed_dir, code_root=ROOT)
    need(sha(args.packed_dir / 'weight_manifest.json') == PACKED_MANIFEST_SHA
         and sha(args.packed_dir / 'parent_weight_manifest.json') == ORIGINAL_MANIFEST_SHA
         and sha(args.packed_dir / 'conversion_receipt.json') == LEDGER_SHA
         and sha(args.packed_dir / 'training_report.json') == REPAIR_CORE_SHA,
         'Different selected repaired checkpoint/core/original provenance')
    need(sha(args.repair_selection) == REPAIR_SELECTION_SHA
         and sha(args.repair_test_comparison) == REPAIR_COMPARISON_SHA,
         'Completed base-repair selection or test provenance changed')
    selection = json.loads(args.repair_selection.read_text())
    comparison = json.loads(args.repair_test_comparison.read_text())
    need(selection['complete'] is True and selection['format'] == 'FP4_BASE_REPAIR_SELECTION_V1'
         and selection['chosen_successful_updates'] == 448
         and selection['chosen_candidate_manifest_sha256'] == PACKED_MANIFEST_SHA
         and selection['test_used_for_selection'] is False
         and comparison['complete'] is True and comparison['format'] == 'FP4_BASE_REPAIR_PPL_EVAL_V1'
         and comparison['candidate_manifest_sha256'] == PACKED_MANIFEST_SHA
         and comparison['candidate_training_report_sha256'] == REPAIR_CORE_SHA
         and comparison['selection_receipt_sha256'] == REPAIR_SELECTION_SHA
         and comparison['candidate_successful_updates'] == 448 and comparison['integrity_pass'] is True,
         'This repaired base is not the completed fixed448-step export')
    expected = manifest['decoded_tensor_sha256']
    need(set(expected) == set(ledger['tensors']) and len(expected) == 507
         and all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) for value in expected.values()),
         'Incomplete repaired decoded507 ledger')
    need(all(expected[name] == row['decoded_sha256'] for name, row in ledger['tensors'].items()
             if row['kind'] == 'fp4_g16_e4m3'), 'Original114 FP4 matrices changed in repaired base')
    need(sum(expected[name] != row['decoded_sha256'] for name, row in ledger['tensors'].items()
             if row['kind'] == 'fp16') == 393, 'The fixed393 repaired FP16 identities were not used')
    provenance = dict(repair_selection_sha256=REPAIR_SELECTION_SHA,
        repair_test_comparison_sha256=REPAIR_COMPARISON_SHA, repair_training_core_sha256=REPAIR_CORE_SHA,
        repaired_packed_manifest_sha256=PACKED_MANIFEST_SHA, parent_packed_manifest_sha256=ORIGINAL_MANIFEST_SHA,
        original_conversion_ledger_sha256=LEDGER_SHA, repair_selected_successful_updates=448,
        repaired_checkpoint_format=manifest['format'])
    need('torch' not in sys.modules, 'Small metadata preflight unexpectedly imported Torch')
    return manifest, ledger, expected, provenance


def preflight(args):
    manifest, ledger, expected, provenance = base_preflight(args)
    need(args.training_report is not None and args.training_audit is not None,
         'Final adapter evaluation requires --training-report and --training-audit')
    training = json.loads(args.training_report.read_text())
    audit = json.loads(args.training_audit.read_text())
    need(training['format'] == TRAIN_FORMAT and training['complete'] is True and training['mode'] == 'formal'
         and training['successful_updates'] == 1536, 'Only complete final step1536 formal training is a candidate')
    need([row['successful_updates'] for row in training['checkpoints']] == [384,768,1152,1536]
         and training['checkpoints'][-1]['file'] == 'checkpoint_1536.pt'
         and training['final_checkpoint_export_check']['passed'] is True
         and training['final_checkpoint_export_check']['all_master_casts_equal_export'] is True
         and training['final_checkpoint_export_check']['final_only'] is True
         and training['adapter']['discarded'] is False, 'Final-only checkpoint/export provenance failed')
    need(audit['format'] == 'FP4_REPAIRED_S16_RESURFACE_CPU_AUDIT_V1'
         and audit['scope'] == 'training_only' and audit['passed'] is True and audit['complete'] is True and
         audit['training_report_sha256'] == sha(args.training_report), 'Independent final training audit binding failed')
    binding = training['binding']
    need(binding['protocol_sha256'] == PROTOCOL_SHA and binding['weight_format'] == 'fp4_g16_e4m3'
         and binding['state_mode'] == 'fp16_native_ssd' and
         binding['packed_manifest_sha256'] == sha(args.packed_dir / 'weight_manifest.json') and
         binding['conversion_ledger_sha256'] == LEDGER_SHA and binding['tokenizer_sha256'] == TOKENIZER_SHA
         and binding['repair_selection_sha256'] == REPAIR_SELECTION_SHA
         and binding['repair_test_comparison_sha256'] == REPAIR_COMPARISON_SHA
         and binding['repair_training_core_sha256'] == REPAIR_CORE_SHA
         and binding['parent_packed_manifest_sha256'] == ORIGINAL_MANIFEST_SHA
         and binding['repair_selected_successful_updates'] == 448
         and binding['repaired_checkpoint_format'] == manifest['format']
         and binding['teacher'] == 'separate same repaired FP4 G16 base, unadapted, native SSD FP16 state',
         'Final adapter was trained on a different frozen base/state/protocol')
    for name, digest in binding['code_sha256'].items():
        need(sha(safe_source(name)) == digest, 'Training-bound source changed: ' + name)
    decoded_digest = hashlib.sha256((json.dumps(expected,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
    need(binding['decoded_ledger_sha256'] == decoded_digest and binding['successful_updates'] == 1536
         and binding['planned_successful_updates'] == 1536
         and binding['checkpoint_selection'] == 'final successful update only; no quality selection',
         'Final candidate selection/decoded507 provenance differs')
    need(training['adapter']['file'] == 'adapter_fp16.pt', 'Final adapter must be the named export in the training directory')
    adapter = args.training_report.parent / training['adapter']['file']
    need(adapter.is_file() and not adapter.is_symlink()
         and sha(adapter) == training['adapter']['sha256'] and adapter.stat().st_size == training['adapter']['bytes'],
         'Final adapter file changed')
    adapter_names = {f'layer{i}.{kind}' for i in range(56) for kind in ('V_read','g_read','router_w','router_b')}
    need(training['adapter']['parameters'] == 1154104 and training['adapter']['payload_bytes'] == 2308208
         and set(training['adapter']['tensor_sha256']) == adapter_names
         and all(re.fullmatch(r'[0-9a-f]{64}', value) for value in training['adapter']['tensor_sha256'].values())
         and binding['gate_mode'] == 'soft', 'Final adapter geometry/payload/gate differs')
    need(manifest['format_name'] == 'fp4_g16_e4m3' and manifest['conversion_receipt_sha256'] == LEDGER_SHA
         and manifest['decoded_resident_weight_bytes'] == 16473999360 and
         manifest['logical_weight_payload_bytes'] == 4638460360 and manifest['packed_resident_kernel'] is False,
         'Packed FP4 model identity/accounting differs')
    return training, audit, adapter, manifest, ledger, expected, provenance


def backend(torch):
    return {'packages': {name: importlib.metadata.version(name) for name in PACKAGES},
            'cuda_runtime': torch.version.cuda, 'num_threads': torch.get_num_threads(),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic}


def cache_receipt(cache, torch, native, *, hashes=False):
    need(len(cache.key_value_memory_dict) == 56, 'Native cache does not cover56 layers')
    rows = {}
    conv_bytes = ssm_bytes = 0
    pointers = []
    for key, pair in sorted(cache.key_value_memory_dict.items()):
        need(len(pair) == 2, 'Unexpected native cache layout')
        row = {}
        for kind, value in zip(('conv', 'ssm'), pair):
            need(value.dtype == torch.float16 and value.element_size() == 2 and not value.is_meta,
                 'Actual native cache is not FP16')
            size = value.numel() * value.element_size()
            row[kind] = {'shape': list(value.shape), 'dtype': 'torch.float16', 'bytes': size}
            if hashes:
                row[kind]['sha256'] = native.tensor_hash(value)
            pointers.append(value.data_ptr())
            if kind == 'conv':
                conv_bytes += size
            else:
                ssm_bytes += size
        rows[str(key)] = row
    need(len(set(pointers)) == 112, 'Native request caches unexpectedly alias')
    need(conv_bytes == 4587520 and ssm_bytes == 117440512, 'Actual cache bytes do not match fixed Mamba2 geometry')
    return {'dtype': 'torch.float16', 'layers': 56, 'tensors': 112, 'conv_bytes': conv_bytes,
            'ssm_bytes': ssm_bytes, 'total_bytes': conv_bytes + ssm_bytes,
            'additional_adapter_recurrent_bytes': 0, 'request_owned_distinct_tensor_storage': True,
            **({'tensor_receipts': rows} if hashes else {})}


def reset_probe(model, ids, torch, runtime, native):
    with torch.inference_mode():
        prompt = ids[:128].to(device=next(model.parameters()).device)[None]
        cache = runtime.make_cache(model, 140, state_dtype=torch.float16)
        hidden = model.backbone(prompt, inference_params=cache)
        first = int(model.lm_head(hidden[:, -1:]).argmax(-1).item())
        receipt = {'prompt_tokens': 128, 'prefill_hidden_sha256': native.tensor_hash(hidden),
                   'prefill_cache': cache_receipt(cache, torch, native, hashes=True), 'first_generated_id': first}
        cache.seqlen_offset = prompt.shape[1]
        step = model.backbone(torch.tensor([[first]], device=prompt.device), inference_params=cache)
        receipt.update(step_hidden_sha256=native.tensor_hash(step),
                       step_cache=cache_receipt(cache, torch, native, hashes=True))
        del cache, hidden, step, prompt
        return receipt


def evaluate_mk(model, tokenizer, cases, torch, runtime, native):
    """The published greedy12-token algorithm, plus actual FP16 cache inspection."""
    rows = []
    started = time.time()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for index, case in enumerate(cases):
            ids = tokenizer.encode(case['prompt'])
            tokens = torch.tensor(ids, dtype=torch.long, device=device)[None]
            cache = runtime.make_cache(model, len(ids)+12, state_dtype=torch.float16)
            hidden = model.backbone(tokens, inference_params=cache)[:, -1:]
            storage = cache_receipt(cache, torch, native)
            generated = []
            for position in range(12):
                logits = model.lm_head(hidden[:, -1:])
                need(bool(torch.isfinite(logits).all()), 'Nonfinite native recall logits')
                token = int(logits.argmax(-1).item())
                generated.append(token)
                if token == tokenizer.eos_token_id:
                    break
                cache.seqlen_offset = len(ids)+position
                hidden = model.backbone(torch.tensor([[token]], device=device), inference_params=cache)
            need(cache_receipt(cache, torch, native) == storage, 'Native cache dtype/ownership/size changed in decoding')
            output = tokenizer.decode(generated)
            match = re.search(r'(?<!\d)\d{6}(?!\d)', output)
            prediction = match.group() if match else None
            rows.append({**case, 'prompt_tokens': len(ids), 'prompt_token_sha256_int64le': runtime.token_digest(ids),
                         'output': output, 'generated_ids': generated, 'prediction': prediction,
                         'correct': prediction == case['answer'], 'cache_bytes': storage['total_bytes'],
                         'actual_cache': storage})
            del cache, tokens, hidden, logits
            if index == 0 or (index+1) % 32 == 0:
                print(f'[native CONFIRM] {index+1}/{len(cases)} {time.time()-started:.1f}s', flush=True)
    summary = {}
    for condition in ('normal', 'target_removed'):
        selected = [r for r in rows if r['condition'] == condition]
        correct = sum(r['correct'] for r in selected)
        need(len(selected) == 384, 'Incomplete CONFIRM condition')
        summary[condition] = {'correct': correct, 'count': len(selected), 'accuracy': correct/len(selected)}
    cells = []
    for size in (16, 64):
        for template in (0, 1, 2):
            for condition in ('normal', 'target_removed'):
                selected = [r for r in rows if (r['N'],r['template'],r['condition']) == (size,template,condition)]
                cells.append({'N': size, 'template': template, 'condition': condition,
                              'correct': sum(r['correct'] for r in selected), 'count': len(selected)})
    by_n = []
    for size in (16, 64):
        for condition in ('normal', 'target_removed'):
            selected = [r for r in rows if (r['N'], r['condition']) == (size, condition)]
            need(len(selected) == 192, 'Incomplete CONFIRM N16/N64 population')
            correct = sum(r['correct'] for r in selected)
            by_n.append({'N': size, 'condition': condition, 'correct': correct,
                         'count': len(selected), 'accuracy': correct/len(selected)})
    return {'split': 'confirm', 'execution': 'native prefill then recurrent decoding', 'cache_dtype': 'torch.float16',
            'generation': 'greedy, at most12 tokens, stop at EOS', 'matching': 'first standalone six-digit integer equals answer',
            'summary': summary, 'summary_by_N': by_n, 'cells': cells, 'rows': rows,
            'elapsed_seconds': time.time()-started}


def compare(control, candidate):
    import numpy as np
    left = {r['id']: r for r in control['mk']['rows'] if r['condition'] == 'normal'}
    right = {r['id']: r for r in candidate['mk']['rows'] if r['condition'] == 'normal'}
    need(len(left) == len(right) == 384 and set(left) == set(right), 'Unpaired normal MK population')
    for name in left:
        need(all(left[name][k] == right[name][k] for k in ('answer','condition','prompt_token_sha256_int64le')), 'Changed paired MK input')
    changes = np.asarray([int(right[k]['correct'])-int(left[k]['correct']) for k in sorted(left)])
    rng = np.random.default_rng(20260928)
    samples = changes[rng.integers(0,384,size=(10000,384))].mean(axis=1)
    lo,hi = np.quantile(samples,[.025,.975])
    return {'control_arm': 'baseline', 'candidate_arm': 'resurface',
            'control_ppl': control['ppl']['ppl'], 'candidate_ppl': candidate['ppl']['ppl'],
            'ppl_relative_change': candidate['ppl']['ppl']/control['ppl']['ppl']-1,
            'control_normal_mk': control['mk']['summary']['normal'], 'candidate_normal_mk': candidate['mk']['summary']['normal'],
            'normal_mk_correct_delta': int(changes.sum()), 'normal_mk_accuracy_delta': float(changes.mean()),
            'paired_improvements': int((changes>0).sum()), 'paired_regressions': int((changes<0).sum()),
            'paired_unchanged': int((changes==0).sum()), 'normal_mk_paired_bootstrap_95ci': [float(lo),float(hi)],
            'bootstrap_draws': 10000, 'bootstrap_seed': 20260928,
            'control_target_removed_mk': control['mk']['summary']['target_removed'],
            'candidate_target_removed_mk': candidate['mk']['summary']['target_removed']}


def restoration(baseline, removed):
    need(baseline['ppl']['windows'] == removed['ppl']['windows'], 'Adapter removal changed per-window NLL/PPL/tokens')
    need(all(baseline['ppl'][k] == removed['ppl'][k] for k in ('ppl','nll','target_tokens')), 'Adapter removal changed aggregate PPL')
    need(len(baseline['mk']['rows']) == len(removed['mk']['rows']) == 768, 'Incomplete MK restoration population')
    for left,right in zip(baseline['mk']['rows'],removed['mk']['rows']):
        need(all(left[k] == right[k] for k in ('id','prompt','prompt_token_sha256_int64le','generated_ids','output','prediction','correct','cache_bytes','actual_cache')),
             'Adapter removal changed native recall: '+left['id'])
    need(baseline['cache_probe'] == removed['cache_probe'], 'Adapter removal changed reset hidden/cache bytes')
    return {'complete': True, 'per_window_nll_exact': True, 'aggregate_ppl_exact': True,
            'all_generated_ids_exact': True, 'all_predictions_exact': True, 'reset_hidden_and_cache_exact': True,
            'ppl_windows_repeated': 147, 'ppl_target_tokens_repeated': 300963, 'mk_cases_repeated': 768}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packed-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, help='Pinned tokenizer directory only; source weights are not opened')
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/FP4_REPAIRED_S16_RESURFACE_V1_PROTOCOL.md')
    parser.add_argument('--expected-protocol-sha256', default=PROTOCOL_SHA)
    parser.add_argument('--repair-selection', type=Path, required=True)
    parser.add_argument('--repair-test-comparison', type=Path, required=True)
    parser.add_argument('--training-report', type=Path)
    parser.add_argument('--training-audit', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--base-preflight-only', action='store_true', help='Before adapter training: verify only repaired base/provenance metadata, no Torch or weight shard reads')
    args = parser.parse_args()
    if args.base_preflight_only:
        manifest, ledger, expected, provenance = base_preflight(args)
        print(json.dumps({'complete': True, 'protocol_sha256': PROTOCOL_SHA,
                          'tensor_count': len(expected), 'base_provenance': provenance,
                          'torch_imported': 'torch' in sys.modules, 'cuda_initialized': False,
                          'large_shards_read': False,
                          'scope': 'Small repaired base/448-step core/provenance bindings; actual full loader verification is deferred to inference.'}, indent=2))
        return
    training,audit,adapter,manifest,ledger,expected,provenance = preflight(args)
    if args.preflight_only:
        print(json.dumps({'complete': True, 'protocol_sha256': PROTOCOL_SHA, 'training_report_sha256': sha(args.training_report),
                          'adapter_sha256': sha(adapter), 'tensor_count': len(expected),
                          'base_provenance': provenance, 'large_shards_read': False,
                          'torch_imported': 'torch' in sys.modules, 'cuda_initialized': False,
                          'scope': 'Small metadata/final-candidate binding; strict loader hashes all actual shards before inference.'}, indent=2))
        return
    need(args.source_dir is not None, '--source-dir pinned tokenizer directory required')
    need(args.out_dir is not None and not args.out_dir.exists(), 'Fresh --out-dir required')
    need(not args.out_dir.resolve().is_relative_to(args.packed_dir.resolve())
         and not args.out_dir.resolve().is_relative_to(args.training_report.parent.resolve()),
         'Evaluation output must not change sealed base/training inventories')
    import torch
    from mamba2_recall import runtime, calibration, evaluation, resurface_native as native, resurface_data as data, fp4
    from mamba2_recall.fp4_repaired_checkpoint import load_repaired_model
    from run_recall_native_prefill_wt2_test_v1 import (native_snapshot, check_native_snapshot, base_identities,
                                                      check_base_identity, check_content, adapter_receipt)
    from evaluate_resurface_more import pin_replay_backend, check_replay_backend
    need(torch.cuda.is_available(), 'CUDA quality reference required')
    need({n:importlib.metadata.version(n) for n in PACKAGES} == PACKAGES and torch.version.cuda == '12.8', 'Recorded native backend versions required')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    policy = backend(torch)
    replay_policy = pin_replay_backend()
    need(replay_policy == training['binding']['backend_policy'], 'Training/deployment frozen RMSNorm replay backend differs')
    args.out_dir.mkdir(parents=True)
    started = time.time()
    initial_code = code_hashes()
    report = {'format': FORMAT, 'complete': False, 'mode': 'full', 'protocol_sha256': PROTOCOL_SHA,
              'runner_sha256': sha(__file__), 'code_sha256': initial_code,
              'training_report_sha256': sha(args.training_report), 'training_audit_sha256': sha(args.training_audit),
              'training_binding': training['binding'], 'final_adapter': training['adapter'], 'reports': {},
              'base_provenance': provenance,
              'backend_before': policy, 'environment': runtime.environment_receipt(),
              'backend_policy': replay_policy,
              'weight_format': 'fp4_g16_e4m3', 'state_mode': 'fp16_native_ssd', 'statequant_controller_installed': False,
              'candidate_selection': 'Final successful update1536 only; no quality selection',
              'test_used_for_training_or_selection': False, 'project_test_and_confirm_previously_exposed': True,
              'publication_performed': False, 'quality_scope': 'Frozen native-prefill official WT2 test and existing CONFIRM; no untouched project holdout claim.'}
    path = args.out_dir/'comparison.json'
    put(path,report)
    try:
        tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
        need(tokenizer.sha256 == TOKENIZER_SHA, 'Pinned tokenizer differs')
        ids,dataset = calibration.load_wikitext_tokens(tokenizer,'test',DATA_REVISION)
        need(dataset['token_stream_sha256_int64le'] == TEST_TOKEN_SHA and dataset['text_sha256'] == TEST_TEXT_SHA and
             dataset['total_tokens'] == 300964 and dataset['split'] == 'test' and dataset['automatic_special_tokens'] is False,
             'Pinned complete WT2 test differs')
        windows = evaluation.ppl_windows(ids,2048)
        need(len(windows) == 147 and sum(len(w)-1 for _,w in windows) == 300963 and len(windows[-1][1])-1 == 1955,
             'Incomplete native-prefill test population')
        raw = ids.numpy().astype('<i8',copy=False).tobytes();(args.out_dir/'tokens.int64le').write_bytes(raw)
        need(hashlib.sha256(raw).hexdigest() == TEST_TOKEN_SHA, 'Exported test tokens differ')
        cases = data.generate_cases('confirm')
        need(len(cases)==768 and data.sha_bytes(data.canonical_bytes(cases)) == CONFIRM_SHA, 'Frozen CONFIRM cases differ')
        put(args.out_dir/'confirm_cases.json',cases)
        report.update(dataset=dataset,tokens_file={'file':'tokens.int64le','bytes':len(raw),'sha256':TEST_TOKEN_SHA},
                      confirm_cases={'file':'confirm_cases.json','sha256':sha(args.out_dir/'confirm_cases.json'),
                                     'canonical_sha256':CONFIRM_SHA,'normal_count':384,'target_removed_count':384},
                      windowing={'execution':'prefill','state_reset_each_window':True,'final_partial_included':True,
                                 'state_requantized_each_token':False,'persistent_ppl_cache_bytes':None,
                                 'window_count':147,'target_tokens':300963,'targets_per_full_window':2048,'logits_chunk_tokens':64})
        for name in ('weight_manifest.json','conversion_receipt.json','parent_weight_manifest.json'):
            (args.out_dir/name).write_bytes((args.packed_dir/name).read_bytes())
        for source, name in ((args.repair_selection, 'repair_selection.json'),
                             (args.repair_test_comparison, 'repair_test_comparison.json'),
                             (args.packed_dir/'training_report.json', 'repair_training_core.json')):
            shutil.copyfile(source, args.out_dir/name)
        report['base_provenance'].update(selection={'file':'repair_selection.json','sha256':REPAIR_SELECTION_SHA},
            test_comparison={'file':'repair_test_comparison.json','sha256':REPAIR_COMPARISON_SHA},
            training_core={'file':'repair_training_core.json','sha256':REPAIR_CORE_SHA},
            original_manifest={'file':'parent_weight_manifest.json','sha256':ORIGINAL_MANIFEST_SHA})
        put(args.out_dir/'weight_ledger.json',{'provenance':'Strict repaired FP4 G16 decoded507 hashes: original114 FP4 and fixed393 repaired FP16',
                                            'repaired_manifest_sha256':PACKED_MANIFEST_SHA,
                                            'conversion_receipt_sha256':LEDGER_SHA,'tensor_sha256':expected})
        report.update(packed_manifest={'file':'weight_manifest.json','sha256':sha(args.out_dir/'weight_manifest.json')},
                      conversion_ledger={'file':'conversion_receipt.json','sha256':LEDGER_SHA},
                      weight_ledger={'file':'weight_ledger.json','sha256':sha(args.out_dir/'weight_ledger.json')})
        model = load_repaired_model(args.packed_dir,device='cuda',chunk_rows=512)
        report['package_receipt'] = model._fp4_repaired_receipt
        need(report['package_receipt']['complete'] is True and report['package_receipt']['base_only'] is True
             and report['package_receipt']['packed_checkpoint_manifest_sha256'] == PACKED_MANIFEST_SHA
             and report['package_receipt']['decoded_tensor_sha256'] == expected
             and report['package_receipt']['training_binding']['training_report_sha256'] == REPAIR_CORE_SHA,
             'Strict actual repaired checkpoint reload receipt differs')
        report['initial_weight_check'] = check_content(model,expected,native)
        snapshot = native_snapshot(model);identities = base_identities(model)
        payload = native.read_fp16(adapter,expected_binding=training['binding'])
        need(payload['gate_mode']=='soft', 'Only frozen soft deployment adapter allowed')
        fixed={'adapter':training['adapter']}
        report['exported_adapter_content'] = adapter_receipt(payload['tensors'],fixed,native)
        put(path,report)
        rows={}
        for arm in ARMS:
            adapted=arm=='resurface'
            manager=(native.install_fp16(model,adapter,expected_binding=training['binding'],expected_base_hashes=expected)
                     if adapted else contextlib.nullcontext())
            torch.cuda.reset_peak_memory_stats()
            with manager as bank:
                probe=reset_probe(model,ids,torch,runtime,native)
                repeat=reset_probe(model,ids,torch,runtime,native)
                need(probe==repeat, 'Fresh native reset/cache probe differs within arm')
                row={'format':FORMAT,'arm':arm,'complete':False,'adapter_loaded':adapted,
                     'adapter_sha256':sha(adapter) if adapted else None,'dataset':dataset,'state_mode':'fp16_native_ssd',
                     'statequant_controller_installed':False,
                     'cache_probe':probe,'repeated_reset_probe':{'complete':True,'hidden_and_cache_exact':True},
                     'adapter_content':adapter_receipt(bank.masters,fixed,native) if adapted else
                                       {'complete':True,'tensor_sha256':{},'payload_bytes':0,'parameters':0,'tensor_count':0},
                     'ppl':evaluation.evaluate_ppl(model,windows,execution='prefill',logits_chunk=64)}
                need(row['ppl']['cache_bytes'] is None and row['ppl']['target_tokens']==300963,'Unexpected PPL carry/cache or coverage')
                row['mk']=evaluate_mk(model,tokenizer,cases,torch,runtime,native)
                row['base_identity']=check_base_identity(model,identities)
                row['actual_weight_check']=check_content(model,expected,native)
                if adapted:
                    row['frozen_base_check']=bank.assert_base_frozen(check_values=False)
                    row['adapter_content_after']=adapter_receipt(bank.masters,fixed,native)
                    need(row['adapter_content']==row['adapter_content_after'],'Final adapter changed in evaluation')
            row.update(complete=True,native_restoration=check_native_snapshot(snapshot),
                       backend_policy_check=check_replay_backend(replay_policy),
                       cuda_memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                                    'scope':'Resident decoded FP16 weights, native runtime and temporary request caches; not encoded model size.'})
            del bank, manager  # Release closed adapter tensors before the next arm's memory peak reset.
            row_path=args.out_dir/(arm+'.json');put(row_path,row);rows[arm]=row
            report['reports'][arm]={'file':row_path.name,'sha256':sha(row_path),'ppl':row['ppl']['ppl'],
                                    'normal_mk_correct':row['mk']['summary']['normal']['correct']}
            put(path,report)
        replay=restoration(rows['baseline'],rows['adapter_removed']);comparison=compare(rows['baseline'],rows['resurface'])
        checks={'adapted_ppl_no_worse':comparison['candidate_ppl']<=comparison['control_ppl'],
                'normal_mk_increases':comparison['normal_mk_correct_delta']>0,
                'paired95_lower_positive':comparison['normal_mk_paired_bootstrap_95ci'][0]>0,
                'all_baseline_nll_and_sequences_restored_exact':replay['complete']}
        report.update(comparison=comparison,restoration=replay,quality_checks=checks,quality_gate_pass=all(checks.values()),
                      final_native_restoration=check_native_snapshot(snapshot),final_base_identity=check_base_identity(model,identities),
                      final_weight_check=check_content(model,expected,native),backend_after=backend(torch),
                      backend_policy_final_check=check_replay_backend(replay_policy),
                      memory={'encoded_weight_payload_bytes':4638460360,'actual_weight_shard_bytes':manifest['shard_file_bytes'],
                              'expanded_fp16_weight_bytes':16473999360,'native_mk_ssm_bytes':117440512,
                              'native_mk_conv_bytes':4587520,'native_mk_cache_bytes':122028032,
                              'persistent_ppl_cache_bytes':None,'adapter_fp16_payload_bytes':2308208,
                              'adapter_file_bytes':adapter.stat().st_size,'additional_adapter_recurrent_bytes':0})
        need(report['backend_after']==policy and code_hashes()==initial_code and sha(args.protocol)==PROTOCOL_SHA,
             'Numerical policy or frozen sources changed during evaluation')
        need(sha(args.training_report)==report['training_report_sha256'] and sha(args.training_audit)==report['training_audit_sha256'] and
             sha(adapter)==training['adapter']['sha256'],'Final candidate provenance changed during evaluation')
        need(sha(args.repair_selection)==REPAIR_SELECTION_SHA and sha(args.repair_test_comparison)==REPAIR_COMPARISON_SHA
             and sha(args.packed_dir/'training_report.json')==REPAIR_CORE_SHA
             and sha(args.packed_dir/'weight_manifest.json')==PACKED_MANIFEST_SHA,
             'Fixed repaired base/provenance changed during evaluation')
        report.update(complete=True,elapsed_seconds=time.time()-started);put(path,report)
        print(json.dumps({'complete':True,'comparison':comparison,'quality_checks':checks},indent=2),flush=True)
    except Exception as exc:
        report.update(complete=False,error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-started)
        put(path,report)
        raise


if __name__=='__main__':
    main()
