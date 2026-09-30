#!/usr/bin/env python3
"""CPU report audit of fresh repaired FP4 G16 / native-FP16 Resurface V1.

Checks saved token identities, arithmetic, training/export provenance and
recorded GPU tensor/storage receipts. It does not replay model logits, load
the large packed weights, initialize CUDA, or turn a failed quality gate into
a pass. NumPy is used only to reproduce the fixed-seed paired bootstrap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import re
import struct
import sys
import tempfile
from copy import deepcopy

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_fp4_sq325_wt2_test_v1 import (file_in, is_digest, need, read, sha,
                                      TEST_TOKEN_SHA, TEST_TEXT_SHA, TOKENIZER_SHA,
                                      REVISION, TEST_TOKENS, WEIGHT_BYTES, ADAPTER_BYTES)
from mamba2_recall import resurface_data as data

FORMAT = 'FP4_REPAIRED_S16_RESURFACE_EVAL_V1'
TRAIN_FORMAT = 'FP4_REPAIRED_S16_RESURFACE_TRAIN_V1'
AUDIT_FORMAT = 'FP4_REPAIRED_S16_RESURFACE_CPU_AUDIT_V1'
PROTOCOL = ROOT / 'docs/FP4_REPAIRED_S16_RESURFACE_V1_PROTOCOL.md'
PROTOCOL_SHA = '9592a0dddcaa5c7d000422e5ac902b49df362781e4a4d9f668d99cb849cf8a7f'
ARMS = ('baseline', 'resurface', 'adapter_removed')
LEDGER_SHA = '55de4189a7e04c9564a48b0fbb5f37428d3d1bb5b6274831d878b41ef4686c9b'
SOURCE_SHA = '47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb'
PACKED_MANIFEST_SHA = '2d2208b690c2ec0def90578d8590cc122bb1e4542b0d49ae4417a48f72b0b73c'
PARENT_MANIFEST_SHA = '4e7fb60f5d03c61e63324a6c25847014a26f9aa0c8d5a8c922d0f367e514fe07'
REPAIR_SELECTION_SHA = '2f1ff559f3e907b3803bcc6d7036d39dab88c11dbb5e69f1de57e2dff5ca452c'
REPAIR_COMPARISON_SHA = 'c3a38223055d53603a64302b0acdbd8e236baa402d67e6927a126848ab21522a'
REPAIR_CORE_SHA = 'd5fb50f2e045b6202a5f38b353ceab7ae47b21c7cc88e8cb398b46c342632c6e'
REPAIRED_FORMAT = 'MAMBA2_FP4_G16_REPAIRED_CHECKPOINT_V1'
NUMERIC_TRAIN_SHA = '451b8703c21120667ef0ea272c21a10d4cd4561779a7b9663773ae47981600c0'
PROSE_MANIFEST_SHA = 'facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'
PROSE_FILE_SHA = 'e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233'
CONFIRM_SHA = '445f1ebae5cfeeafccc0b6c02f1a514d8d7858360637af9ec37ab5e48212a091'
CONV_BYTES = 4_587_520
SSM_BYTES = 117_440_512
CACHE_BYTES = CONV_BYTES + SSM_BYTES
WEIGHT_PAYLOAD_BYTES = 4_638_460_360
WEIGHT_FILE_BYTES = 4_638_539_864
OBJECTIVE = dict(mk_ce=1., mk_kl=0., mk_closure=0., prose_ce=.5,
                 prose_kl=.5, prose_closure=3., temperature=1., chunk_tokens=64)
PACKAGES = {'torch': '2.11.0+cu128', 'mamba-ssm': '2.3.2.post1', 'triton': '3.6.0'}


def close(actual, expected, label):
    need(type(actual) in (int, float) and math.isfinite(actual)
         and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label)


def code_check(code):
    need(isinstance(code, dict) and bool(code), 'Nonempty frozen code inventory required')
    for relative, expected in code.items():
        path = ROOT / relative
        need(not Path(relative).is_absolute() and path.resolve().is_relative_to(ROOT)
             and path.is_file() and not path.is_symlink() and is_digest(expected)
             and sha(path) == expected, 'Frozen source differs: ' + relative)


def ledger_digest(expected):
    raw = (json.dumps(expected, sort_keys=True, separators=(',', ':')) + '\n').encode()
    return hashlib.sha256(raw).hexdigest()


def repaired_metadata(args):
    need(sha(args.protocol) == PROTOCOL_SHA
         and (args.expected_protocol_sha256 in (None, PROTOCOL_SHA))
         and sha(args.conversion_ledger) == LEDGER_SHA
         and sha(args.packed_manifest) == PACKED_MANIFEST_SHA,
         'Immutable repaired protocol/packed metadata differs')
    ledger, manifest = read(args.conversion_ledger), read(args.packed_manifest)
    code_check(ledger['code_sha256']); code_check(manifest['code_sha256'])
    need(ledger['complete'] is True and ledger['source_checkpoint_sha256'] == SOURCE_SHA
         and ledger['tensor_count'] == 507 and ledger['parameter_count'] == 8236999680
         and ledger['format_name'] == 'fp4_g16_e4m3', 'Original FP4 geometry ledger differs')
    expected = manifest['decoded_tensor_sha256']
    need(set(expected) == set(ledger['tensors']) and len(expected) == 507
         and all(is_digest(value) for value in expected.values()), 'Repaired decoded 507 ledger differs')
    quantized = {name: row for name, row in ledger['tensors'].items() if row['kind'] == 'fp4_g16_e4m3'}
    retained = {name: row for name, row in ledger['tensors'].items() if row['kind'] == 'fp16'}
    need(len(quantized) == 114 and len(retained) == 393
         and sum(row['numel'] * 2 for row in retained.values()) == 7161856
         and all(expected[name] == row['decoded_sha256'] for name, row in quantized.items())
         and all(expected[name] != row['decoded_sha256'] for name, row in retained.items()),
         'Repair changed quantized tensors or promoted/grew retained weights')
    need(manifest['format'] == REPAIRED_FORMAT and manifest['complete'] is True
         and manifest['base_only'] is True and manifest['resurface_adapter'] is False
         and manifest['conversion_receipt_sha256'] == LEDGER_SHA
         and manifest['parent_manifest_sha256'] == PARENT_MANIFEST_SHA
         and manifest['source_checkpoint_sha256'] == SOURCE_SHA
         and manifest['decoded_resident_weight_bytes'] == WEIGHT_BYTES
         and manifest['logical_weight_payload_bytes'] == WEIGHT_PAYLOAD_BYTES
         and manifest['shard_file_bytes'] == WEIGHT_FILE_BYTES and manifest['packed_resident_kernel'] is False
         and manifest['tensor_count'] == 507 and manifest['parameter_count'] == 8236999680
         and manifest['fp4_tensor_count'] == 114 and manifest['retained_fp16_tensor_count'] == 393
         and manifest['retained_fp16_payload_bytes'] == 7161856 and len(manifest['files']) == 119,
         'Repaired payload/container/runtime scope differs')
    for path, digest in ((args.repair_selection, REPAIR_SELECTION_SHA),
                         (args.repair_test_comparison, REPAIR_COMPARISON_SHA),
                         (args.repair_training_core, REPAIR_CORE_SHA)):
        need(sha(path) == digest, 'Repaired base provenance file differs: ' + str(path))
    selection, comparison, core = (read(path) for path in
        (args.repair_selection, args.repair_test_comparison, args.repair_training_core))
    need(selection['format'] == 'FP4_BASE_REPAIR_SELECTION_V1' and selection['complete'] is True
         and selection['chosen_successful_updates'] == 448
         and selection['chosen_candidate_manifest_sha256'] == PACKED_MANIFEST_SHA
         and selection['test_used_for_selection'] is False
         and selection['criterion'] == 'minimum_full_validation_pooled_nll',
         'Different base-repair selected checkpoint')
    need(core['format'] == 'FP4_BASE_REPAIR_CHECKPOINT_CORE_V1' and core['complete'] is True
         and core['mode'] == 'formal' and core['successful_updates'] == 448
         and core['discarded'] is False and core['no_resurface_installed'] is True
         and {name: expected[name] for name in retained} == core['retained_tensor_sha256']
         and manifest['training_binding']['training_report_sha256'] == REPAIR_CORE_SHA
         and manifest['training_binding']['successful_updates'] == 448
         and manifest['training_binding']['discarded'] is False,
         'Repaired retained values do not match the selected base training core')
    need(comparison['format'] == 'FP4_BASE_REPAIR_PPL_EVAL_V1' and comparison['complete'] is True
         and comparison['mode'] == 'full' and comparison['split'] == 'test'
         and comparison['candidate_manifest_sha256'] == PACKED_MANIFEST_SHA
         and comparison['candidate_successful_updates'] == 448
         and comparison['candidate_training_report_sha256'] == REPAIR_CORE_SHA
         and comparison['selection_receipt_sha256'] == REPAIR_SELECTION_SHA
         and comparison['integrity_pass'] is True and comparison['test_used_for_training_or_selection'] is False
         and comparison['adapter_loaded'] is False and comparison['statequant_controller_installed'] is False
         and comparison['reports']['repaired']['ppl'] == 6.717854132348465
         and comparison['reports']['repaired']['target_tokens'] == 300963,
         'Repaired base full-test provenance differs')
    return manifest, ledger, expected


def package_check(package, manifest, expected):
    need(package == dict(format=REPAIRED_FORMAT, complete=True, base_only=True,
         source_checkpoint_required=False, serialization='safetensors_packed_checkpoint',
         packed_checkpoint_manifest_sha256=PACKED_MANIFEST_SHA, parent_manifest_sha256=PARENT_MANIFEST_SHA,
         conversion_receipt_sha256=LEDGER_SHA, logical_weight_payload_bytes=WEIGHT_PAYLOAD_BYTES,
         shard_file_bytes=WEIGHT_FILE_BYTES, decoded_resident_weight_bytes=WEIGHT_BYTES,
         tensor_count=507, decoded_tensor_sha256=expected, training_binding=manifest['training_binding'],
         packed_resident_kernel=False), 'Strict repaired packed-loader provenance differs')


def weight_check(receipt, expected):
    need(receipt['complete'] is True and receipt['passed'] is True
         and receipt['actual_content_checked'] is True and receipt['tensors'] == 507
         and receipt['parameters'] == 8236999680 and receipt['decoded_tensor_sha256'] == expected
         and receipt['decoded_ledger_sha256'] == ledger_digest(expected)
         and receipt['weight_payload_bytes'] == WEIGHT_BYTES,
         'Recorded actual content of the frozen 507 decoded FP16 weights differs')


def backend_check(policy, final):
    selected = dict(kwargs={}, num_warps=16, num_stages=3, num_ctas=1,
                    maxnreg=None, pre_hook_is_none=True)
    need(policy['package_versions'] == PACKAGES and policy['selected_config'] == selected
         and policy['singleton_config_bypasses_autotuning'] is True
         and policy['applied_before_first_model_load'] is True
         and policy['candidate_used_for_selection'] is False
         and policy['precision_flags']['tf32_matmul'] is False
         and policy['precision_flags']['tf32_cudnn'] is False
         and policy['precision_flags']['float32_matmul_precision'] == 'highest'
         and all(value is None for value in policy['determinism_environment'].values()),
         'Frozen native backend policy differs')
    need(final['singleton_config_unchanged'] is True and final['selected_config'] == selected
         and final['best_config'] in (None, selected), 'Native backend changed during measurement')


def restored(receipt):
    need(receipt == dict(complete=True, native_mixer_forwards_restored=True,
                         native_hooks_restored=True), 'Native training forwards/hooks not restored')


def adapter_file(directory, adapter, *, smoke=False):
    path = file_in(directory, adapter['file'])
    need(sha(path) == adapter['sha256'] and path.stat().st_size == adapter['bytes']
         and adapter['payload_bytes'] == ADAPTER_BYTES and adapter['parameters'] == 1154104
         and len(adapter['tensor_sha256']) == 224 and adapter['discarded'] is smoke
         and adapter['gate_mode'] == 'soft' and adapter['roundtrip_bitwise_equal'] is True,
         'Final FP16 adapter file/capacity differs')
    need(adapter['file'] == ('discarded_smoke_adapter_fp16.pt' if smoke else 'adapter_fp16.pt'),
         'Adapter filename differs from discarded-smoke/final-only export')
    expected_keys = {f'layer{index}.{field}' for index in range(56)
                     for field in ('V_read', 'g_read', 'router_w', 'router_b')}
    need(set(adapter['tensor_sha256']) == expected_keys
         and all(is_digest(value) for value in adapter['tensor_sha256'].values()),
         'Adapter 56-layer / 224-tensor inventory differs')
    return path


def training_check(args, expected):
    report = read(args.training_report)
    smoke_only = bool(args.smoke_only)
    limit = 1 if smoke_only else 1536
    need(report['format'] == TRAIN_FORMAT and report['complete'] is True
         and 'error' not in report and report['mode'] == ('smoke' if smoke_only else 'formal')
         and report['successful_updates'] == limit and report['attempts'] == len(report['history'])
         and type(report['overflows']) is int and 0 <= report['overflows'] <= 8
         and report['attempts'] - report['overflows'] == limit,
         'Discarded smoke/final formal training population differs')
    code_check(report['code_sha256'])
    binding = report['binding']
    need(binding['code_sha256'] == report['code_sha256']
         and binding['format'] == TRAIN_FORMAT and binding['protocol_sha256'] == sha(PROTOCOL)
         and binding['packed_manifest_sha256'] == sha(args.packed_manifest) == PACKED_MANIFEST_SHA
         and binding['conversion_ledger_sha256'] == sha(args.conversion_ledger) == LEDGER_SHA
         and binding['parent_packed_manifest_sha256'] == PARENT_MANIFEST_SHA
         and binding['repair_selection_sha256'] == sha(args.repair_selection) == REPAIR_SELECTION_SHA
         and binding['repair_test_comparison_sha256'] == sha(args.repair_test_comparison) == REPAIR_COMPARISON_SHA
         and binding['repair_training_core_sha256'] == sha(args.repair_training_core) == REPAIR_CORE_SHA
         and binding['repaired_checkpoint_format'] == REPAIRED_FORMAT
         and binding['repair_selected_successful_updates'] == 448
         and binding['decoded_ledger_sha256'] == ledger_digest(expected)
         and binding['source_checkpoint_sha256'] == SOURCE_SHA and binding['tokenizer_sha256'] == TOKENIZER_SHA
         and binding['weight_format'] == 'fp4_g16_e4m3' and binding['state_mode'] == 'fp16_native_ssd'
         and binding['student_execution'] == 'original native SSD prefill and backward; no state-controller replacement'
         and binding['teacher'] == 'separate same repaired FP4 G16 base, unadapted, native SSD FP16 state'
         and binding['numeric_train_manifest_sha256'] == NUMERIC_TRAIN_SHA
         and binding['prose_manifest_sha256'] == PROSE_MANIFEST_SHA
         and binding['prose_tokens_sha256'] == PROSE_FILE_SHA
         and binding['initial_adapter'] == 'fresh V=0,g=1,w=0,b=-4; no pretrained adapter'
         and binding['gate_mode'] == 'soft' and binding['objective'] == OBJECTIVE and binding['seed'] == 2026092803
         and binding['planned_successful_updates'] == binding['successful_updates'] == 1536
         and binding['checkpoint_selection'] == 'final successful update only; no quality selection',
         'Fresh TRAIN-only native-SSD binding differs')
    prose = read(ROOT / 'docs/prose_train_manifest.json')
    need(sha(ROOT / 'docs/prose_train_manifest.json') == PROSE_MANIFEST_SHA
         and prose['complete'] is True and prose['split'] == 'train' and prose['dataset']['split'] == 'train'
         and prose['evaluation_data_used'] is False and prose['heldout_used_for_fitting'] is False
         and prose['training_window_count'] == 448 and prose['seqlen'] == 2048
         and binding['prose_tokens_int64le_sha256'] == prose['training_tokens_sha256_int64le'],
         'Pinned prose TRAIN manifest scope differs')
    for key in ('fresh_initialization', 'initial_native_parity_128_512', 'export_native_parity_128_512',
                'removal_native_parity_128_512', 'parity_128_512_initial_and_export',
                'teacher_student_independent_storage', 'no_state_controller_installed'):
        need(report[key] is True, 'Native initialization/deployment proof missing: ' + key)
    if not smoke_only:
        need(args.smoke_report is not None, 'Formal audit requires a discarded smoke report')
        smoke_args = SimpleNamespace(**{**vars(args), 'training_report': args.smoke_report,
                                       'smoke_report': None, 'smoke_only': True})
        smoke = training_check(smoke_args, expected)
        need(smoke['binding'] == binding and smoke['code_sha256'] == report['code_sha256']
             and args.smoke_report.resolve().parent != args.training_report.resolve().parent
             and report['smoke_report_sha256'] == sha(args.smoke_report),
             'Separate discarded one-update smoke binding differs')
    train_cases = data.generate_cases('train')
    success = overflows = 0
    ordered = []
    loss_scale = 1024.
    retry_entry = None
    for index, row in enumerate(report['history'], 1):
        need(row['attempt'] == index and type(row['overflow']) is bool and success < limit
             and type(row['schedule_entry']) is int and 0 <= row['schedule_entry'] < 1536
             and row['case_id'] == train_cases[row['schedule_entry']]['id']
             and row['prose_window'] == prose['schedule'][success % 448]
             and row['prose_start'] == 512*((success//448) % 4)
             and type(row['answer_targets']) is int and row['answer_targets'] > 0,
             'Recorded TRAIN-only successful-update schedule differs')
        if retry_entry is not None:
            need(row['schedule_entry'] == retry_entry, 'Overflow changed the numeric update schedule')
        for key in ('mk_ce', 'prose_ce', 'prose_kl', 'prose_closure'):
            need(type(row[key]) in (int, float) and math.isfinite(row[key]),
                 'Invalid training scalar: ' + key)
        if row['overflow']:
            overflows += 1
            loss_scale *= .5
            retry_entry = row['schedule_entry']
            need(row['gradient_norm_before_clip'] is None, 'Overflow performed a gradient update')
        else:
            ordered.append(row['schedule_entry'])
            success += 1
            retry_entry = None
            need(type(row['gradient_norm_before_clip']) in (int, float)
                 and math.isfinite(row['gradient_norm_before_clip']) and row['gradient_norm_before_clip'] >= 0,
                 'Invalid successful gradient clip receipt')
        need(row['successful_updates'] == success and row['loss_scale'] == loss_scale,
             'Successful update or GradScaler transition differs')
    need(success == limit and overflows == report['overflows']
         and (len(ordered) == 1 if smoke_only else sorted(ordered) == list(range(1536))),
         'Numeric TRAIN shuffled pass omitted/repeated examples')
    for key in ('initial_student_base_check', 'initial_teacher_base_check',
                'final_student_base_check', 'final_teacher_base_check'):
        weight_check(report[key], expected)
    need([item['successful_updates'] for item in report['checkpoints']] ==
         ([] if smoke_only else [384, 768, 1152, 1536]),
         'Fixed progress-checkpoint sequence differs')
    for checkpoint in report['checkpoints']:
        path = file_in(args.training_report.parent, checkpoint['file'])
        need(checkpoint['file'] == f"checkpoint_{checkpoint['successful_updates']:04d}.pt"
             and checkpoint['successful_updates'] <= checkpoint['attempts'] <= report['attempts']
             and sha(path) == checkpoint['sha256'] and path.stat().st_size == checkpoint['bytes'],
             'Progress checkpoint file differs')
    if not smoke_only:
        need(report['checkpoints'][-1]['attempts'] == report['attempts'], 'Final checkpoint attempts differ')
        need(report['final_checkpoint_export_check'] == dict(complete=True, passed=True, master_tensors=224,
             all_master_casts_equal_export=True, optimizer_exact=True, scaler_exact=True,
             checkpoint_roundtrip_exact=True, final_only=True), 'Final master/export/optimizer/scaler proof differs')
    adapter_file(args.training_report.parent, report['adapter'], smoke=smoke_only)
    restored(report['native_restoration'])
    restored(report['teacher_native_restoration'])
    need(binding['backend_policy'] == report['backend_policy'], 'Training backend binding differs')
    backend_check(report['backend_policy'], report['backend_final_check'])
    return report


def token_windows(directory, receipt):
    path = file_in(directory, receipt['file'])
    raw = path.read_bytes()
    need(receipt == dict(file=path.name, bytes=len(raw), sha256=sha(path))
         and len(raw) == TEST_TOKENS * 8 and sha(path) == TEST_TOKEN_SHA,
         'Full saved official test token stream differs')
    need(all(0 <= item[0] < 256000 for item in struct.iter_unpack('<q', raw)),
         'Token ID outside original vocabulary')
    windows = [(start, min(start + 2049, TEST_TOKENS) - start - 1,
                hashlib.sha256(raw[start*8:min(start+2049, TEST_TOKENS)*8]).hexdigest())
               for start in range(0, TEST_TOKENS - 1, 2048)]
    need(len(windows) == 147 and sum(n for _, n, _ in windows) == 300963
         and windows[-1][1] == 1955, 'Independent target coverage differs')
    return windows


def ppl_check(ppl, windows):
    need(ppl['execution'] == 'prefill' and ppl['cache_dtype'] == 'SSD scan internal precision'
         and ppl['cache_bytes'] is None and ppl['logits_chunk_tokens'] == 64
         and ppl['target_tokens'] == 300963 and len(ppl['windows']) == 147,
         'Native-prefill numerical path/full test population differs')
    total = 0.
    for row, (start, count, token_sha) in zip(ppl['windows'], windows):
        need((row['start'], row['input_tokens'], row['target_tokens'], row['token_sha256_int64le'])
             == (start, count, count, token_sha), 'Window token coverage differs')
        need(type(row['nll']) in (float, int) and math.isfinite(row['nll']) and row['nll'] >= 0,
             'Invalid finite window NLL')
        close(row['ppl'], math.exp(row['nll'] / count), 'Window PPL arithmetic differs')
        total += row['nll']
    close(ppl['nll'], total, 'Pooled NLL does not equal all window NLLs')
    close(ppl['ppl'], math.exp(total / 300963), 'Pooled PPL arithmetic differs')
    return ppl['ppl']


def mk_check(mk, cases):
    need(mk['split'] == 'confirm' and mk['execution'] == 'native prefill then recurrent decoding'
         and mk['cache_dtype'] == 'torch.float16', 'Native MK execution differs')
    rows = mk['rows']
    need(len(rows) == 768 and len({row['id'] for row in rows}) == 768,
         'Full unique CONFIRM MK population differs')
    expected = {case['id']: case for case in cases}
    need(set(expected) == {row['id'] for row in rows}, 'Frozen CONFIRM MK IDs differ')
    for row in rows:
        case = expected[row['id']]
        need(all(row[key] == case[key] for key in ('condition', 'N', 'template', 'query_position', 'answer')),
             'Frozen MK condition/answer/cell differs')
        if 'prompt' in row:
            need(row['prompt'] == case['prompt'], 'Frozen MK prompt changed')
        generated = row['generated_ids']
        need(isinstance(generated, list) and len(generated) <= 12
             and all(type(token) is int and 0 <= token < 256000 for token in generated),
             'MK generation budget or vocabulary differs')
        match = re.search(r'(?<!\d)\d{6}(?!\d)', row['output'])
        prediction = match.group() if match else None
        need(row['prediction'] == prediction and type(row['correct']) is bool
             and row['correct'] == (prediction == row['answer'])
             and is_digest(row['prompt_token_sha256_int64le'])
             and row['cache_bytes'] == CACHE_BYTES, 'MK exact-match/cache accounting differs')
        cache_check(row['actual_cache'], detailed=False)
    for condition in ('normal', 'target_removed'):
        group = [row for row in rows if row['condition'] == condition]
        correct = sum(row['correct'] for row in group)
        need(len(group) == 384 and mk['summary'][condition] ==
             dict(count=384, correct=correct, accuracy=correct/384), 'MK summary arithmetic differs')
    expected_cells = []
    for size in (16, 64):
        for template in (0, 1, 2):
            for condition in ('normal', 'target_removed'):
                selected = [row for row in rows if (row['N'], row['template'], row['condition'])
                            == (size, template, condition)]
                need(len(selected) == 64, 'CONFIRM cell count differs')
                expected_cells.append(dict(N=size, template=template, condition=condition,
                                           count=64, correct=sum(row['correct'] for row in selected)))
    need(mk['cells'] == expected_cells, 'N16/N64 MK cell summaries differ')
    by_n = []
    for size in (16, 64):
        for condition in ('normal', 'target_removed'):
            selected = [row for row in rows if (row['N'], row['condition']) == (size, condition)]
            correct = sum(row['correct'] for row in selected)
            need(len(selected) == 192, 'Incomplete CONFIRM N16/N64 population')
            by_n.append(dict(N=size, condition=condition, correct=correct,
                             count=192, accuracy=correct/192))
    need(mk['summary_by_N'] == by_n, 'N16/N64 aggregate MK summary arithmetic differs')
    return {row['id']: row for row in rows}


def cache_check(receipt, *, detailed):
    expected = dict(dtype='torch.float16', layers=56, tensors=112, conv_bytes=CONV_BYTES,
                    ssm_bytes=SSM_BYTES, total_bytes=CACHE_BYTES, additional_adapter_recurrent_bytes=0,
                    request_owned_distinct_tensor_storage=True)
    need(all(receipt.get(key) == value for key, value in expected.items()),
         'Actual FP16 request cache dtype/storage/ownership differs')
    if not detailed:
        return
    tensors = receipt['tensor_receipts']
    need(set(tensors) == {str(i) for i in range(56)}, 'FP16 cache layer inventory differs')
    totals = {'conv': 0, 'ssm': 0}
    for index in range(56):
        layer = tensors[str(index)]
        need(set(layer) == {'conv', 'ssm'}, 'Unexpected native cache tensor')
        for kind, shape in (('conv', [1, 10240, 4]), ('ssm', [1, 128, 64, 128])):
            row = layer[kind]
            need(row['shape'] == shape and row['dtype'] == 'torch.float16'
                 and row['bytes'] == math.prod(shape)*2 and is_digest(row['sha256']),
                 'Actual per-layer native cache shape/dtype/bytes/hash differs')
            totals[kind] += row['bytes']
    need(totals == dict(conv=CONV_BYTES, ssm=SSM_BYTES), 'Actual cache tensor totals differ')


def native_restoration(receipt):
    need(receipt['complete'] is True and receipt['forward_methods_and_hooks_exact'] is True
         and type(receipt['modules']) is int and receipt['modules'] > 0,
         'Native forwards/hooks were not exactly restored')


def identity_check(receipt):
    need(receipt['complete'] is True and receipt['tensors'] == 507
         and receipt['parameters'] == 8236999680 and receipt['identity_version_gradients_unchanged'] is True,
         'Frozen base identity/version/gradients changed')


def eval_weight_check(receipt, expected):
    need(receipt['complete'] is True and receipt['actual_content_checked'] is True
         and receipt['tensors'] == 507 and receipt['actual_tensor_sha256'] == expected,
         'Actual evaluation weight hashes differ from repaired decoded507 ledger')


def adapter_content(receipt, adapter, loaded):
    expected = dict(complete=True, tensor_sha256=adapter['tensor_sha256'] if loaded else {},
                    payload_bytes=ADAPTER_BYTES if loaded else 0,
                    parameters=1154104 if loaded else 0, tensor_count=224 if loaded else 0)
    need(receipt == expected, 'Recorded actual deployed adapter content/capacity differs')


def paired_check(comparison, control, candidate):
    import numpy as np
    left = {row['id']: row for row in control['mk']['rows'] if row['condition'] == 'normal'}
    right = {row['id']: row for row in candidate['mk']['rows'] if row['condition'] == 'normal'}
    need(set(left) == set(right) and len(left) == 384, 'Paired normal MK population differs')
    for key in left:
        need(all(left[key][field] == right[key][field]
                 for field in ('answer', 'condition', 'prompt_token_sha256_int64le')),
             'Paired MK prompt/answer changed')
    changes = np.asarray([int(right[key]['correct']) - int(left[key]['correct']) for key in sorted(left)])
    rng = np.random.default_rng(20260928)
    samples = changes[rng.integers(0, 384, size=(10000, 384))].mean(axis=1)
    bounds = np.quantile(samples, [.025, .975])
    expected = dict(control_arm='baseline', candidate_arm='resurface',
        control_ppl=control['ppl']['ppl'], candidate_ppl=candidate['ppl']['ppl'],
        control_normal_mk=control['mk']['summary']['normal'], candidate_normal_mk=candidate['mk']['summary']['normal'],
        normal_mk_correct_delta=int(changes.sum()), paired_improvements=int((changes > 0).sum()),
        paired_regressions=int((changes < 0).sum()), paired_unchanged=int((changes == 0).sum()),
        bootstrap_draws=10000, bootstrap_seed=20260928,
        control_target_removed_mk=control['mk']['summary']['target_removed'],
        candidate_target_removed_mk=candidate['mk']['summary']['target_removed'])
    need(all(comparison.get(key) == value for key, value in expected.items()), 'Paired MK comparison differs')
    close(comparison['ppl_relative_change'], candidate['ppl']['ppl']/control['ppl']['ppl'] - 1,
          'Relative PPL arithmetic differs')
    close(comparison['normal_mk_accuracy_delta'], float(changes.mean()), 'MK accuracy delta differs')
    need(len(comparison['normal_mk_paired_bootstrap_95ci']) == 2, 'Paired CI population differs')
    for actual, expected_bound in zip(comparison['normal_mk_paired_bootstrap_95ci'], bounds):
        close(actual, float(expected_bound), 'Independently reproduced paired bootstrap differs')
    return str(np.__version__)


def eval_check(args, training, expected):
    directory = args.comparison.parent
    report = read(args.comparison)
    need(report['format'] == FORMAT and report['complete'] is True and report['mode'] == 'full'
         and report['protocol_sha256'] == PROTOCOL_SHA and report['runner_sha256'] ==
         sha(ROOT / 'scripts/run_fp4_repaired_s16_resurface_v1.py')
         and report['training_report_sha256'] == sha(args.training_report)
         and report['training_binding'] == training['binding'] and report['final_adapter'] == training['adapter']
         and report['weight_format'] == 'fp4_g16_e4m3' and report['state_mode'] == 'fp16_native_ssd'
         and report['statequant_controller_installed'] is False
         and report['candidate_selection'] == 'Final successful update1536 only; no quality selection'
         and report['test_used_for_training_or_selection'] is False
         and report['project_test_and_confirm_previously_exposed'] is True and report['publication_performed'] is False,
         'Frozen native-S16 final-candidate/protocol/train-only scope differs')
    need(args.training_audit is not None, 'Combined audit requires --training-audit')
    prior = read(args.training_audit)
    need(prior['format'] == AUDIT_FORMAT and prior['scope'] == 'training_only' and prior['passed'] is True
         and prior['complete'] is True and prior['successful_updates'] == 1536
         and prior['protocol_sha256'] == PROTOCOL_SHA and prior['auditor_sha256'] == sha(__file__)
         and prior['training_report_sha256'] == sha(args.training_report)
         and report['training_audit_sha256'] == sha(args.training_audit), 'Training-only audit binding differs')
    code_check(report['code_sha256'])
    dataset = report['dataset']
    fixed_dataset = dict(dataset='Salesforce/wikitext', configuration='wikitext-2-raw-v1', split='test',
        revision_argument=REVISION, document_join='two newline characters', text_sha256=TEST_TEXT_SHA,
        token_stream_sha256_int64le=TEST_TOKEN_SHA, total_tokens=TEST_TOKENS,
        tokenizer_sha256=TOKENIZER_SHA, automatic_special_tokens=False)
    need(all(dataset.get(key) == value for key, value in fixed_dataset.items()), 'Pinned official test dataset differs')
    need(report['windowing'] == dict(execution='prefill', state_reset_each_window=True, final_partial_included=True,
        state_requantized_each_token=False, persistent_ppl_cache_bytes=None, window_count=147,
        target_tokens=300963, targets_per_full_window=2048, logits_chunk_tokens=64), 'Native test-window protocol differs')
    windows = token_windows(directory, report['tokens_file'])
    cases = data.generate_cases('confirm')
    need(hashlib.sha256(data.canonical_bytes(cases)).hexdigest() == CONFIRM_SHA,
         'Independent canonical CONFIRM generation changed')
    confirm = report['confirm_cases']
    path = file_in(directory, confirm['file'])
    need(confirm['sha256'] == sha(path) and read(path) == cases and confirm['canonical_sha256'] == CONFIRM_SHA
         and confirm['normal_count'] == confirm['target_removed_count'] == 384, 'Saved CONFIRM cases differ')
    for key, original in (('packed_manifest', args.packed_manifest), ('conversion_ledger', args.conversion_ledger)):
        entry = report[key]
        copied = file_in(directory, entry['file'])
        need(sha(copied) == entry['sha256'] == sha(original), 'Copied packed metadata differs: ' + key)
    provenance = report['base_provenance']
    fixed_provenance = dict(repair_selection_sha256=REPAIR_SELECTION_SHA,
        repair_test_comparison_sha256=REPAIR_COMPARISON_SHA, repair_training_core_sha256=REPAIR_CORE_SHA,
        repaired_packed_manifest_sha256=PACKED_MANIFEST_SHA, parent_packed_manifest_sha256=PARENT_MANIFEST_SHA,
        original_conversion_ledger_sha256=LEDGER_SHA, repair_selected_successful_updates=448,
        repaired_checkpoint_format=REPAIRED_FORMAT)
    need(all(provenance.get(key) == value for key, value in fixed_provenance.items()),
         'Evaluation repaired-base provenance differs')
    for key, name, digest in (('selection', 'repair_selection.json', REPAIR_SELECTION_SHA),
                             ('test_comparison', 'repair_test_comparison.json', REPAIR_COMPARISON_SHA),
                             ('training_core', 'repair_training_core.json', REPAIR_CORE_SHA),
                             ('original_manifest', 'parent_weight_manifest.json', PARENT_MANIFEST_SHA)):
        entry = provenance[key]
        need(entry == dict(file=name, sha256=digest) and sha(file_in(directory, name)) == digest,
             'Copied repaired-base provenance differs: ' + key)
    entry = report['weight_ledger']
    path = file_in(directory, entry['file'])
    need(sha(path) == entry['sha256'], 'Recorded decoded ledger file changed')
    ledger = read(path)
    need(ledger['tensor_sha256'] == expected and ledger['conversion_receipt_sha256'] == LEDGER_SHA
         and ledger['repaired_manifest_sha256'] == PACKED_MANIFEST_SHA,
         'Decoded evaluation ledger differs from repaired base')
    package_check(report['package_receipt'], read(args.packed_manifest), expected)
    eval_weight_check(report['initial_weight_check'], expected)
    eval_weight_check(report['final_weight_check'], expected)
    adapter_content(report['exported_adapter_content'], training['adapter'], True)
    policy = report['backend_before']
    need(policy == report['backend_after'] and policy['packages'] ==
         dict(PACKAGES, numpy='1.26.4', datasets='4.8.5', sentencepiece='0.2.1')
         and policy['cuda_runtime'] == '12.8' and policy['num_threads'] == 8
         and policy['tf32_matmul'] is False and policy['tf32_cudnn'] is False
         and policy['float32_matmul_precision'] == 'highest'
         and report['backend_policy'] == training['backend_policy'], 'Paired native backend differs')
    backend_check(report['backend_policy'], report['backend_policy_final_check'])
    rows = {}
    need(set(report['reports']) == set(ARMS), 'Missing full evaluation arm')
    for arm in ARMS:
        entry = report['reports'][arm]
        path = file_in(directory, entry['file'])
        need(path.name == arm + '.json' and sha(path) == entry['sha256'], 'Bound arm file differs')
        row = read(path)
        loaded = arm == 'resurface'
        need(row['format'] == FORMAT and row['complete'] is True and row['arm'] == arm
             and row['adapter_loaded'] is loaded and row['state_mode'] == 'fp16_native_ssd'
             and row['statequant_controller_installed'] is False
             and row['adapter_sha256'] == (training['adapter']['sha256'] if loaded else None)
             and row['dataset'] == dataset, 'Arm model/adapter/input identity differs')
        ppl_check(row['ppl'], windows)
        mk_check(row['mk'], cases)
        close(entry['ppl'], row['ppl']['ppl'], 'Arm PPL receipt differs')
        need(entry['normal_mk_correct'] == row['mk']['summary']['normal']['correct'], 'Arm MK receipt differs')
        identity_check(row['base_identity'])
        eval_weight_check(row['actual_weight_check'], expected)
        adapter_content(row['adapter_content'], training['adapter'], loaded)
        if loaded:
            adapter_content(row['adapter_content_after'], training['adapter'], True)
            need(row['frozen_base_check']['identity_version_gradients_unchanged'] is True,
                 'Adapted frozen-base identity changed')
        native_restoration(row['native_restoration'])
        backend_check(report['backend_policy'], row['backend_policy_check'])
        probe = row['cache_probe']
        need(probe['prompt_tokens'] == 128 and is_digest(probe['prefill_hidden_sha256'])
             and is_digest(probe['step_hidden_sha256']) and type(probe['first_generated_id']) is int
             and 0 <= probe['first_generated_id'] < 256000,
             'Native prefill/recurrent hidden probe receipt differs')
        cache_check(probe['prefill_cache'], detailed=True)
        cache_check(probe['step_cache'], detailed=True)
        need(row['repeated_reset_probe'] == dict(complete=True, hidden_and_cache_exact=True),
             'Within-arm repeated reset/cache probe differs')
        rows[arm] = row
    baseline, candidate, removed = (rows[arm] for arm in ARMS)
    need(baseline['ppl']['windows'] == removed['ppl']['windows']
         and all(baseline['ppl'][key] == removed['ppl'][key] for key in ('ppl', 'nll', 'target_tokens')),
         'Adapter removal changed any full-test score')
    for left, right in zip(baseline['mk']['rows'], removed['mk']['rows']):
        need(all(left[key] == right[key] for key in ('id', 'prompt', 'prompt_token_sha256_int64le',
             'generated_ids', 'output', 'prediction', 'correct', 'cache_bytes', 'actual_cache')),
             'Adapter removal changed a generated sequence/MK/cache: ' + left['id'])
    need(baseline['cache_probe'] == removed['cache_probe'], 'Adapter removal changed hidden/cache content')
    expected_restore = dict(complete=True, per_window_nll_exact=True, aggregate_ppl_exact=True,
        all_generated_ids_exact=True, all_predictions_exact=True, reset_hidden_and_cache_exact=True,
        ppl_windows_repeated=147, ppl_target_tokens_repeated=300963, mk_cases_repeated=768)
    need(report['restoration'] == expected_restore, 'Full adapter-removal restoration receipt differs')
    numpy_version = paired_check(report['comparison'], baseline, candidate)
    pair = report['comparison']
    checks = dict(adapted_ppl_no_worse=pair['candidate_ppl'] <= pair['control_ppl'],
                  normal_mk_increases=pair['normal_mk_correct_delta'] > 0,
                  paired95_lower_positive=pair['normal_mk_paired_bootstrap_95ci'][0] > 0,
                  all_baseline_nll_and_sequences_restored_exact=True)
    need(report['quality_checks'] == checks and report['quality_gate_pass'] is all(checks.values()),
         'Frozen quality decision differs from audited results')
    native_restoration(report['final_native_restoration'])
    identity_check(report['final_base_identity'])
    need(report['memory'] == dict(encoded_weight_payload_bytes=WEIGHT_PAYLOAD_BYTES,
        actual_weight_shard_bytes=WEIGHT_FILE_BYTES, expanded_fp16_weight_bytes=WEIGHT_BYTES,
        native_mk_ssm_bytes=SSM_BYTES, native_mk_conv_bytes=CONV_BYTES, native_mk_cache_bytes=CACHE_BYTES,
        persistent_ppl_cache_bytes=None, adapter_fp16_payload_bytes=ADAPTER_BYTES,
        adapter_file_bytes=training['adapter']['bytes'], additional_adapter_recurrent_bytes=0),
        'Encoded storage/resident FP16/native cache/adapter accounting differs')
    return dict(ppl={arm: rows[arm]['ppl']['ppl'] for arm in ARMS}, comparison=pair,
                quality_checks=checks, quality_gate_pass=all(checks.values()),
                numpy_version=numpy_version, arm_sha256={arm: report['reports'][arm]['sha256'] for arm in ARMS})


def contract_fixtures(manifest, ledger, expected):
    """Small positive and mutation fixtures, never tensor/logit quality tests."""
    cases = []
    def passes(name, fn):
        fn(); cases.append(dict(name=name, passed=True))
    def rejects(name, fn):
        try:
            fn()
        except (ValueError, KeyError):
            cases.append(dict(name=name, passed=True, rejected=True)); return
        raise ValueError('Contract fixture accepted invalid input: ' + name)
    receipt = dict(complete=True, passed=True, actual_content_checked=True, tensors=507,
                   parameters=8236999680, decoded_tensor_sha256=expected,
                   decoded_ledger_sha256=ledger_digest(expected), weight_payload_bytes=WEIGHT_BYTES)
    passes('new_repaired_507_receipt', lambda: weight_check(receipt, expected))
    stale = deepcopy(receipt)
    stale['decoded_tensor_sha256'] = {name: row['decoded_sha256'] for name,row in ledger['tensors'].items()}
    stale['decoded_ledger_sha256'] = ledger_digest(stale['decoded_tensor_sha256'])
    rejects('old393_values_cannot_bind_repaired_base', lambda: weight_check(stale, expected))
    changed = deepcopy(receipt); changed['decoded_tensor_sha256'][next(iter(expected))] = '0'*64
    rejects('single_decoded_tensor_mutation', lambda: weight_check(changed, expected))
    package = dict(format=REPAIRED_FORMAT, complete=True, base_only=True, source_checkpoint_required=False,
        serialization='safetensors_packed_checkpoint', packed_checkpoint_manifest_sha256=PACKED_MANIFEST_SHA,
        parent_manifest_sha256=PARENT_MANIFEST_SHA, conversion_receipt_sha256=LEDGER_SHA,
        logical_weight_payload_bytes=WEIGHT_PAYLOAD_BYTES, shard_file_bytes=WEIGHT_FILE_BYTES,
        decoded_resident_weight_bytes=WEIGHT_BYTES, tensor_count=507, decoded_tensor_sha256=expected,
        training_binding=manifest['training_binding'], packed_resident_kernel=False)
    passes('repaired_loader_schema', lambda: package_check(package, manifest, expected))
    oldpackage = deepcopy(package); oldpackage['format'] = 'MAMBA2_FP4_G16_PACKED_CHECKPOINT_V1'
    rejects('original_loader_schema_rejected', lambda: package_check(oldpackage, manifest, expected))
    oldcore = deepcopy(package); oldcore['training_binding']['successful_updates'] = 896
    rejects('wrong_selected_base_core_rejected', lambda: package_check(oldcore, manifest, expected))
    windows = [(start, min(start+2049, TEST_TOKENS)-start-1, 'a'*64)
               for start in range(0, TEST_TOKENS-1, 2048)]
    nll = math.log(7.)
    rows = [dict(start=start, input_tokens=count, target_tokens=count, token_sha256_int64le=digest,
                 nll=count*nll, ppl=7.) for start,count,digest in windows]
    ppl = dict(execution='prefill', cache_dtype='SSD scan internal precision', cache_bytes=None,
               logits_chunk_tokens=64, target_tokens=300963, windows=rows,
               nll=sum(row['nll'] for row in rows), ppl=7.)
    passes('147_windows_partial_tail_pooled_arithmetic', lambda: ppl_check(ppl, windows))
    missing = deepcopy(ppl); missing['windows'].pop()
    rejects('partial_tail_omission', lambda: ppl_check(missing, windows))
    fake = deepcopy(ppl); fake['ppl'] = 6.9
    rejects('misstated_pooled_ppl', lambda: ppl_check(fake, windows))
    cache = dict(dtype='torch.float16', layers=56, tensors=112, conv_bytes=CONV_BYTES,
        ssm_bytes=SSM_BYTES, total_bytes=CACHE_BYTES, additional_adapter_recurrent_bytes=0,
        request_owned_distinct_tensor_storage=True)
    passes('native_fp16_cache_accounting', lambda: cache_check(cache, detailed=False))
    extra = dict(cache, additional_adapter_recurrent_bytes=2)
    rejects('hidden_adapter_recurrent_storage', lambda: cache_check(extra, detailed=False))
    wrong = dict(cache, dtype='torch.int8')
    rejects('quantized_state_schema_rejected', lambda: cache_check(wrong, detailed=False))
    with tempfile.TemporaryDirectory(prefix='fp4_repaired_cpu_fixture_') as folder:
        directory = Path(folder); path = directory/'discarded_smoke_adapter_fp16.pt'; path.write_bytes(b'fixture')
        adapter = dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size,
            payload_bytes=ADAPTER_BYTES, parameters=1154104, discarded=True, gate_mode='soft',
            roundtrip_bitwise_equal=True, tensor_sha256={f'layer{i}.{field}':'b'*64 for i in range(56)
                for field in ('V_read','g_read','router_w','router_b')})
        passes('discarded_smoke_file_hash_contract', lambda: adapter_file(directory, adapter, smoke=True))
        rejects('smoke_cannot_be_final_candidate', lambda: adapter_file(directory, adapter, smoke=False))
        path.write_bytes(b'changed')
        rejects('adapter_file_corruption', lambda: adapter_file(directory, adapter, smoke=True))
    return dict(format='FP4_REPAIRED_S16_RESURFACE_AUDITOR_FIXTURES_V1', complete=True, passed=True,
                scope='contract_fixtures', cases=cases, case_count=len(cases),
                actual_small_repaired_metadata_verified=True, cuda_initialized=False,
                torch_imported='torch' in sys.modules, source_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-report', 'smoke-report', 'packed-manifest', 'conversion-ledger',
                 'repair-selection', 'repair-test-comparison', 'repair-training-core'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--expected-protocol-sha256', default=PROTOCOL_SHA)
    parser.add_argument('--smoke-only', action='store_true', help='Audit one-update discarded smoke; no final checkpoint required')
    parser.add_argument('--self-test', action='store_true', help='Small contract fixtures with pinned repaired-base metadata, no Torch/CUDA')
    parser.add_argument('--comparison', type=Path, help='Complete three-arm evaluation; omit for training-only audit')
    parser.add_argument('--training-audit', type=Path, help='Completed training-only audit; required with comparison')
    args = parser.parse_args()
    need(not args.out.exists() and not args.out.is_symlink(), 'Fresh audit output required')
    for field in ('packed_manifest', 'conversion_ledger', 'repair_selection', 'repair_test_comparison'):
        need(getattr(args, field) is not None, 'Required audit metadata: --' + field.replace('_','-'))
    if args.repair_training_core is None:
        args.repair_training_core = args.packed_manifest.parent/'training_report.json'
    manifest, ledger, expected = repaired_metadata(args)
    if args.self_test:
        need(not args.smoke_only and args.comparison is None and args.training_report is None,
             'Contract fixtures cannot be combined with a training/evaluation audit')
        result = contract_fixtures(manifest, ledger, expected)
    else:
        need(not (args.smoke_only and args.comparison), 'Smoke is discarded and cannot be a full evaluation candidate')
        if args.smoke_only and args.training_report is None:
            args.training_report = args.smoke_report
        need(args.training_report is not None, 'Training report required')
        if not args.smoke_only:
            need(args.smoke_report is not None, 'Formal final audit requires separate --smoke-report')
        training = training_check(args, expected)
        scope = 'smoke_only' if args.smoke_only else ('combined' if args.comparison else 'training_only')
        result = dict(format=AUDIT_FORMAT, scope=scope, complete=True, passed=True,
            training_report_sha256=sha(args.training_report),
            smoke_report_sha256=sha(args.smoke_report) if args.smoke_report else None,
            successful_updates=training['successful_updates'], protocol_sha256=PROTOCOL_SHA,
            source_sha256=sha(__file__), auditor_sha256=sha(__file__),
            packed_manifest_sha256=PACKED_MANIFEST_SHA, conversion_ledger_sha256=LEDGER_SHA,
            parent_packed_manifest_sha256=PARENT_MANIFEST_SHA, repair_selection_sha256=REPAIR_SELECTION_SHA,
            repair_test_comparison_sha256=REPAIR_COMPARISON_SHA, repair_training_core_sha256=REPAIR_CORE_SHA,
            decoded_ledger_sha256=ledger_digest(expected), repair_selected_successful_updates=448,
            recorded_all_507_weight_bindings_verified=True, final_only_export_binding_verified=not args.smoke_only,
            discarded_smoke_adapter=args.smoke_only, recorded_native_128_512_parity_verified=True,
            cuda_initialized=False, torch_imported='torch' in sys.modules,
            audit_method='Independent small-report/file/hash/TRAIN-schedule/arithmetic audit. GPU-recorded tensor/parity/cache receipts are bound to the repaired507 ledger. No tensor deserialization, logits/hidden replay, or large packed-weight decoding; recorded Torch random permutation is not independently regenerated.')
        if args.comparison:
            result.update(input_report_sha256=sha(args.comparison), full_ppl_windows=147, full_ppl_targets=300963,
                          full_mk_cases_per_arm=768, normal_mk_cases_per_arm=384, target_removed_mk_cases_per_arm=384,
                          native_fp16_mk_cache_bytes=CACHE_BYTES, persistent_native_prefill_ppl_cache_bytes=None,
                          adapter_fp16_payload_bytes=ADAPTER_BYTES, **eval_check(args, training, expected))
    need(result['torch_imported'] is False, 'CPU report auditor must not import torch')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
