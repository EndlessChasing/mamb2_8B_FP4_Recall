#!/usr/bin/env python3
"""Independent CPU report audit of FP4 G16 base-only repair V1.

Checks recorded training/packing integrity, validation-only selection and saved
token/NLL arithmetic. It never imports Torch, initializes CUDA, loads large
weights, or treats a negative quality outcome as an integrity failure. GPU
tensor receipts are bound and checked, not independently recomputed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = ROOT / 'docs/FP4_BASE_REPAIR_V1_PROTOCOL.md'
PROTOCOL_SHA = 'bbac4faecdbe775c4409234d4fbb649b2047b935afa9b870eb47c44ffa039655'
TRAINER_SHA = 'cbb794e7170b03ac34346f843ef433427908dcd05db21bcdd03936d3cb2acbd8'
AUDIT_FORMAT = 'FP4_BASE_REPAIR_CPU_AUDIT_V1'
PARENT_MANIFEST_SHA = '4e7fb60f5d03c61e63324a6c25847014a26f9aa0c8d5a8c922d0f367e514fe07'
PARENT_LEDGER_SHA = '55de4189a7e04c9564a48b0fbb5f37428d3d1bb5b6274831d878b41ef4686c9b'
SOURCE_SHA = '47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb'
TOKENIZER_SHA = '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09'
PROSE_MANIFEST_SHA = 'facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'
PROSE_FILE_SHA = 'e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233'
SCHEDULE_SHA = '6c0e280e623dc5e78ce478bac006b9844d73a730b18f03a34b70649f7a2eae07'
SCHEDULE = ROOT / 'artifacts/fp4_base_repair_v1/training_schedule.json'
REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
TEST_TOKEN_SHA = '5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608'
TEST_TEXT_SHA = '696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83'
VALIDATION_TOKEN_SHA = '5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027'
CHECKPOINT_STEPS = (224, 448, 672, 896)
PARAMETERS = 8_236_999_680
RETAINED_PARAMETERS = 3_580_928
RETAINED_BYTES = 7_161_856
PAYLOAD_BYTES = 4_638_460_360
SHARD_BYTES = 4_638_539_864
DECODED_BYTES = 16_473_999_360
PACKAGES = {'torch': '2.11.0+cu128', 'mamba-ssm': '2.3.2.post1', 'triton': '3.6.0'}
FAMILY_COUNTS = dict(block_and_final_norm=57, gated_norm=56, conv_weight=56,
                     conv_bias=56, A_log=56, dt_bias=56, D=56)
HP = dict(successful_updates=896, epochs=2, windows_per_epoch=448,
          window_tokens=2048, targets_per_update=2047, seed=2026093001,
          optimizer='AdamW', learning_rate=1e-4, betas=[.9, .999], epsilon=1e-8,
          weight_decay=0., gradient_clip_norm=1., ce_coefficient=1.,
          teacher_to_student_kl_coefficient=.1, temperature=1., logits_chunk_tokens=64,
          cosine_multiplier_start=1., cosine_multiplier_end=.1,
          initial_loss_scale=1024., maximum_overflow_retries=8,
          candidate_successful_updates=list(CHECKPOINT_STEPS), selected_tensors=393,
          selected_parameters=RETAINED_PARAMETERS, selected_fp16_bytes=RETAINED_BYTES)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_keys,
                      parse_constant=nonfinite)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(actual, expected, label):
    need(number(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label)


def file_in(directory, relative):
    need(isinstance(relative, str) and not Path(relative).is_absolute(), 'Relative artifact path required')
    directory = Path(directory).resolve()
    path = directory / relative
    need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory),
         'Artifact file missing or escaped root: ' + relative)
    return path


def bound_file(directory, receipt):
    path = file_in(directory, receipt['file'])
    need(digest(receipt['sha256']) and sha(path) == receipt['sha256']
         and ('bytes' not in receipt or type(receipt['bytes']) is int
              and path.stat().st_size == receipt['bytes']), 'Artifact hash/size differs: ' + str(path))
    return path


def ledger_digest(values):
    return hashlib.sha256((json.dumps(values, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest()


def code_check(code):
    need(isinstance(code, dict) and bool(code), 'Frozen source inventory required')
    for relative, expected in code.items():
        need(digest(expected) and sha(file_in(ROOT, relative)) == expected,
             'Frozen local source differs: ' + relative)


def schedule_check(path=SCHEDULE):
    need(sha(path) == SCHEDULE_SHA, 'Pinned two-pass CPU randperm schedule differs')
    value = read(path)
    order = value['order']
    need(value['format'] == 'FP4_BASE_REPAIR_TRAIN_SCHEDULE_V1' and value['seed'] == 2026093001
         and value['windows'] == 448 and value['epochs'] == 2 and len(order) == 896
         and all(type(i) is int for i in order)
         and sorted(order[:448]) == list(range(448))
         and sorted(order[448:]) == list(range(448)),
         'Schedule does not contain both exact complete TRAIN passes')
    return order


def family(name):
    if name.endswith('.mixer.norm.weight'):
        return 'gated_norm'
    if name.endswith('.norm.weight') or name == 'backbone.norm_f.weight':
        return 'block_and_final_norm'
    if name.endswith('.mixer.conv1d.weight'):
        return 'conv_weight'
    if name.endswith('.mixer.conv1d.bias'):
        return 'conv_bias'
    for kind in ('A_log', 'dt_bias', 'D'):
        if name.endswith('.mixer.' + kind):
            return kind
    raise ValueError('Unexpected retained base parameter: ' + name)


def retained_inventory(ledger):
    need(ledger['complete'] is True and ledger['format_name'] == 'fp4_g16_e4m3'
         and ledger['tensor_count'] == 507 and ledger['parameter_count'] == PARAMETERS
         and ledger['fp4_tensor_count'] == 114 and ledger['fp16_tensor_count'] == 393,
         'Original FP4 G16 representation/population differs')
    entries = ledger['tensors']
    need(len(entries) == 507 and sum(v['numel'] for v in entries.values()) == PARAMETERS,
         'Ledger tensor/parameter count differs')
    retained = {n: v for n, v in entries.items() if v['kind'] == 'fp16'}
    need(len(retained) == 393 and sum(v['numel'] for v in retained.values()) == RETAINED_PARAMETERS
         and sum(v['payload_bytes'] for v in retained.values()) == RETAINED_BYTES
         and Counter(family(n) for n in retained) == FAMILY_COUNTS,
         'Existing retained base parameter families/capacity differs')
    for name, entry in entries.items():
        need(entry['kind'] in ('fp16', 'fp4_g16_e4m3') and digest(entry['decoded_sha256'])
             and entry['numel'] == math.prod(entry['shape']), 'Invalid tensor ledger: ' + name)
    return retained


def packed_metadata(parent_manifest, parent_ledger, candidate_manifest):
    retained = retained_inventory(parent_ledger)
    other = candidate_manifest['decoded_tensor_sha256']
    need(set(other) == set(parent_ledger['tensors']), 'Candidate introduced/omitted a base tensor')
    for name, original in parent_ledger['tensors'].items():
        current = other[name]
        if name not in retained:
            need(current == original['decoded_sha256'], 'A frozen FP4 decoded ledger changed: ' + name)
        else:
            need(digest(current), 'Retained FP16 hash malformed: ' + name)
    need(candidate_manifest['complete'] is True and candidate_manifest['tensor_count'] == 507
         and candidate_manifest['parameter_count'] == PARAMETERS
         and candidate_manifest['format'] == 'MAMBA2_FP4_G16_REPAIRED_CHECKPOINT_V1'
         and candidate_manifest['parent_manifest_sha256'] == PARENT_MANIFEST_SHA
         and candidate_manifest['conversion_receipt_sha256'] == PARENT_LEDGER_SHA
         and candidate_manifest['source_checkpoint_sha256'] == SOURCE_SHA
         and candidate_manifest['base_only'] is True and candidate_manifest['resurface_adapter'] is False
         and candidate_manifest['layout'] == parent_manifest['layout'],
         'Incomplete/mismatched same-geometry candidate base ledger')
    old, new = parent_manifest['files'], candidate_manifest['files']
    need(len(old) == len(new) == 119 and set(old) == set(new), 'Packed shard inventory differs')
    retained_files = ['weights-retained-fp16.safetensors']
    need(retained_files[0] in old, 'One retained FP16 shard required')
    for name in old:
        need(type(new[name]['bytes']) is int and new[name]['bytes'] == old[name]['bytes']
             and digest(new[name]['sha256']), 'Shard byte capacity differs: ' + name)
        if name not in retained_files:
            need(new[name] == old[name], 'One of the 118 frozen packed FP4 shards changed: ' + name)
    for manifest in (parent_manifest, candidate_manifest):
        need(manifest['complete'] is True and manifest['logical_weight_payload_bytes'] == PAYLOAD_BYTES
             and manifest['shard_file_bytes'] == SHARD_BYTES and sum(v['bytes'] for v in manifest['files'].values()) == SHARD_BYTES
             and manifest['decoded_resident_weight_bytes'] == DECODED_BYTES
             and manifest['packed_resident_kernel'] is False,
             'Logical/file/decoded capacity or packed-kernel scope differs')
    return dict(tensors=507, frozen_fp4_tensors=114, frozen_fp4_shards=118,
                retained_fp16_tensors=393, retained_fp16_payload_bytes=RETAINED_BYTES,
                logical_weight_payload_bytes=PAYLOAD_BYTES, shard_file_bytes=SHARD_BYTES,
                decoded_resident_weight_bytes=DECODED_BYTES)


def candidate_check(directory, *, verify_weights=False):
    # Both helpers are stdlib-only; Torch lives solely inside export/GPU load.
    from mamba2_recall.fp4_repaired_checkpoint import (read_repaired_metadata,
                                                     verify_repaired_checkpoint)
    check = verify_repaired_checkpoint if verify_weights else read_repaired_metadata
    manifest, ledger = check(directory, code_root=ROOT)
    parent_path = Path(directory) / 'parent_weight_manifest.json'
    ledger_path = Path(directory) / 'conversion_receipt.json'
    need(sha(parent_path) == PARENT_MANIFEST_SHA and sha(ledger_path) == PARENT_LEDGER_SHA,
         'Embedded original weight metadata differs')
    receipt = packed_metadata(read(parent_path), ledger, manifest)
    receipt.update(candidate_manifest_sha256=sha(Path(directory) / 'weight_manifest.json'),
                   independent_full_shards_read=bool(verify_weights),
                   scope=('Full shard bytes/encoded regions independently verified on CPU.' if verify_weights else
                          'Small embedded metadata/training-core bindings verified; shard identities are recorded metadata, not independently reread.'))
    need('torch' not in sys.modules, 'Packing metadata audit imported Torch')
    return manifest, ledger, receipt


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
         and all(v is None for v in policy['determinism_environment'].values()),
         'Frozen native numerical backend differs')
    need(final['singleton_config_unchanged'] is True and final['selected_config'] == selected
         and final['best_config'] in (None, selected), 'Native backend changed during measurement')


def token_windows(path, split):
    need(split in ('validation', 'test'), 'Only fixed validation or official test is allowed')
    raw = Path(path).read_bytes()
    total, token_sha, windows, targets, tail = ((264765, VALIDATION_TOKEN_SHA, 130, 264764, 572)
                                              if split == 'validation' else
                                              (300964, TEST_TOKEN_SHA, 147, 300963, 1955))
    need(len(raw) == total * 8 and hashlib.sha256(raw).hexdigest() == token_sha,
         'Saved official ' + split + ' token stream differs')
    need(all(0 <= row[0] < 256000 for row in struct.iter_unpack('<q', raw)), 'Token outside original vocabulary')
    plan = [(start, min(start + 2049, total) - start - 1,
             hashlib.sha256(raw[start*8:min(start+2049, total)*8]).hexdigest())
            for start in range(0, total - 1, 2048)]
    need(len(plan) == windows and sum(v[1] for v in plan) == targets and plan[-1][1] == tail,
         'Saved stream target coverage differs')
    return plan


def ppl_check(ppl, plan):
    targets = sum(count for _, count, _ in plan)
    need(ppl['execution'] == 'prefill' and ppl['cache_dtype'] == 'SSD scan internal precision'
         and ppl['cache_bytes'] is None and ppl['logits_chunk_tokens'] == 64
         and ppl['target_tokens'] == targets and len(ppl['windows']) == len(plan),
         'Native-prefill numerical path or target population differs')
    nll = 0.
    for row, (start, count, token_sha) in zip(ppl['windows'], plan):
        need((row['start'], row['input_tokens'], row['target_tokens'], row['token_sha256_int64le'])
             == (start, count, count, token_sha), 'Window token/target coverage differs')
        need(number(row['nll']) and row['nll'] >= 0., 'Invalid per-window NLL')
        close(row['ppl'], math.exp(row['nll'] / count), 'Per-window PPL arithmetic differs')
        nll += row['nll']
    close(ppl['nll'], nll, 'Pooled NLL differs from complete window sum')
    close(ppl['ppl'], math.exp(nll / targets), 'Pooled PPL arithmetic differs')
    return dict(nll=nll, ppl=ppl['ppl'], target_tokens=targets, windows=len(plan))


def selected_step(rows):
    need([row['successful_updates'] for row in rows] == list(CHECKPOINT_STEPS),
         'Validation candidate population/order differs')
    for row in rows:
        need(number(row['nll']) and row['nll'] >= 0 and row['split'] == 'validation',
             'Selection requires full validation NLL only')
        need(row['target_tokens'] == 264764 and row['windows'] == 130,
             'Selection used partial validation or mismatched population')
    return min(rows, key=lambda row: (row['nll'], row['successful_updates']))['successful_updates']


def history_check(history, order, successful_expected, attempts_expected, overflows_expected):
    need(type(successful_expected) is int and 0 < successful_expected <= 896
         and type(attempts_expected) is int and attempts_expected == len(history)
         and type(overflows_expected) is int and 0 <= overflows_expected <= 8
         and attempts_expected == successful_expected + overflows_expected,
         'Successful/attempt/overflow accounting differs')
    successful = overflows = 0
    scale = 1024.
    for attempt, row in enumerate(history, 1):
        need(successful < successful_expected and row['attempt'] == attempt
             and row['successful_updates_before'] == successful
             and type(row['overflow']) is bool and row['window_index'] == order[successful]
             and row['epoch'] == successful // 448 and row['position_in_epoch'] == successful % 448,
             'TRAIN schedule/order or same-window overflow retry differs')
        expected_lr = 1e-4 * (.1 + .9 * .5 * (1 + math.cos(math.pi * successful / 895)))
        need(number(row['learning_rate']) and math.isclose(row['learning_rate'], expected_lr,
             rel_tol=1e-12, abs_tol=1e-15), 'Successful-update cosine learning rate differs')
        need(row['targets'] == row['forward_positions'] == 2047 and row['chunks'] == 32
             and row['chunk_selected_targets'] == [64] * 31 + [63]
             and row['ce_coefficient'] == 1. and row['kl_coefficient'] == .1
             and row['temperature'] == 1. and row['backbone_backward_calls'] == 1
             and row['scaled_hidden_gradient_elements'] == 2047 * 4096
             and row['gate_and_hidden_backward_joint'] is False
             and all(row[key] == 0. for key in ('closure', 'opening', 'weighted_closure', 'weighted_opening')),
             'Full-target CE/KL objective or hidden backbone backward differs')
        need(number(row['ce']) and row['ce'] >= 0 and number(row['teacher_to_student_kl']),
             'Invalid recorded target CE/teacher KL')
        close(row['loss'], row['ce'] + .1 * row['teacher_to_student_kl'], 'TRAIN objective arithmetic differs')
        close(row['loss_scale_before'], scale, 'GradScaler initial/preceding scale differs')
        gradient = row['gradients']
        need(gradient['tensors'] == 393 and gradient['parameters'] == RETAINED_PARAMETERS
             and type(gradient['finite']) is bool and type(gradient['nonzero']) is int
             and 0 <= gradient['nonzero'] <= RETAINED_PARAMETERS,
             'Selected393 gradient coverage differs')
        if row['overflow']:
            overflows += 1
            scale *= .5
            need(row['gradient_norm_before_clip'] is None
                 and row['overflow_optimizer_and_masters_unchanged'] is True
                 and (gradient['finite'] is False or row['scaled_hidden_gradient_finite'] is False),
                 'Overflow retry performed an update or lacks nonfinite gradient proof')
        else:
            successful += 1
            need(gradient['finite'] is True and gradient['nonzero'] > 0
                 and row['scaled_hidden_gradient_finite'] is True
                 and row['scaled_hidden_gradient_nonzero'] > 0
                 and number(row['gradient_norm_before_clip']) and row['gradient_norm_before_clip'] >= 0
                 and row['overflow_optimizer_and_masters_unchanged'] is None,
                 'Successful update lacks finite nonzero gradients/clip proof')
        close(row['loss_scale_after'], scale, 'GradScaler transition differs')
        need(row['successful_updates_after'] == successful, 'Successful-update counter differs')
    need(successful == successful_expected and overflows == overflows_expected,
         'Final successful-update or overflow population differs')
    return dict(successful_updates=successful, attempts=len(history), overflow_retries=overflows,
                successful_target_exposures=successful*2047, same_window_overflow_retries_verified=True)


def parity_check(receipts):
    need(len(receipts) == 2 and [r['input_tokens'] for r in receipts] == [128, 512],
         'Native/export parity token lengths differ')
    for row in receipts:
        need(all(row[key] is True for key in ('complete', 'passed', 'bitwise_equal',
             'hidden_bitwise_equal', 'last_eight_logits_bitwise_equal', 'original_parameter_objects_restored'))
             and digest(row['hidden_sha256']) and digest(row['logits_sha256']),
             'Native/export/restoration GPU parity proof missing')


def native_check(receipt):
    need(receipt['complete'] is True and receipt['forward_methods_and_hooks_exact'] is True
         and type(receipt['modules']) is int and receipt['modules'] > 0,
         'Original native methods/hooks were not restored')


def base_check(receipt, *, actual_content):
    need(receipt['complete'] is True and receipt['tensors'] == 507 and receipt['parameters'] == PARAMETERS
         and receipt['identity_version_gradients_unchanged'] is True
         and receipt['actual_content_checked'] is actual_content,
         'Original507 decoded base identity/content check differs')


def stability_check(receipt):
    need(receipt['complete'] is True and receipt['all_masters_and_fp16_casts_finite'] is True
         and receipt['native_A_finite'] is True and receipt['dt_bias_softplus_finite'] is True
         and receipt['new_clamps_or_reparameterization'] is False,
         'Base dynamics/master stability check differs')
    need(number(receipt['native_A']['minimum']) and number(receipt['native_A']['maximum'])
         and receipt['native_A']['maximum'] < 0 and receipt['native_A']['zeros'] == 0
         and number(receipt['softplus_dt_bias_only']['minimum'])
         and receipt['softplus_dt_bias_only']['minimum'] > 0
         and receipt['softplus_dt_bias_only']['zeros'] == 0,
         'Native negative decay or positive dt-bias diagnostic differs')


def training_binding(binding):
    need(binding['protocol_sha256'] == sha(PROTOCOL)
         and binding['parent_manifest_sha256'] == PARENT_MANIFEST_SHA
         and binding['parent_conversion_receipt_sha256'] == PARENT_LEDGER_SHA
         and binding['source_checkpoint_sha256'] == SOURCE_SHA and binding['tokenizer_sha256'] == TOKENIZER_SHA
         and binding['schedule_sha256'] == SCHEDULE_SHA
         and binding['prose_manifest_sha256'] == PROSE_MANIFEST_SHA
         and binding['prose_tokens_sha256'] == PROSE_FILE_SHA
         and binding['hyperparameters'] == HP and binding['training_split'] == 'train'
         and binding['state_mode'] == 'fp16_native_ssd' and binding['no_resurface_installed'] is True,
         'Frozen TRAIN-only retained393 repair binding differs')
    required_sources = {'scripts/train_fp4_base_repair_v1.py', 'mamba2_recall/fp4_base_repair.py',
        'mamba2_recall/fp4_repaired_checkpoint.py', 'mamba2_recall/fp4_checkpoint.py',
        'mamba2_recall/fp4.py', 'mamba2_recall/runtime.py', 'mamba2_recall/calibration.py',
        'mamba2_recall/evaluation.py', 'mamba2_recall/resurface_loss.py', 'scripts/evaluate_resurface_more.py'}
    need(required_sources.issubset(binding['code_sha256'])
         and binding['code_sha256']['scripts/train_fp4_base_repair_v1.py'] == TRAINER_SHA,
         'Mandatory frozen trainer/native math/packing source closure differs')
    code_check(binding['code_sha256'])
    prose_path = ROOT / 'docs/prose_train_manifest.json'
    prose = read(prose_path)
    need(sha(prose_path) == PROSE_MANIFEST_SHA and prose['complete'] is True
         and prose['split'] == prose['dataset']['split'] == 'train'
         and prose['evaluation_data_used'] is False and prose['heldout_used_for_fitting'] is False
         and prose['training_window_count'] == 448 and prose['seqlen'] == 2048,
         'Pinned prose TRAIN provenance differs')


def training_report_check(report, ledger, order, *, smoke=False):
    need(report['format'] == 'FP4_BASE_REPAIR_TRAIN_V1' and report['complete'] is True
         and report['mode'] == ('smoke' if smoke else 'formal')
         and report['no_resurface_installed'] is True and report['no_statequant_controller_installed'] is True
         and report['test_used_for_training_or_selection'] is False
         and report['teacher_student_independent_storage'] is True,
         'Incomplete/wrong-mode retained-only TRAIN report or hidden adapter/state controller')
    training_binding(report['binding'])
    counting = history_check(report['history'], order, 1 if smoke else 896,
                             report['attempts'], report['overflows'])
    need(report['successful_updates'] == counting['successful_updates'], 'Report successful-update count differs')
    expected_source = {n: item['source_fp16_sha256'] for n, item in ledger['tensors'].items()}
    need(report['teacher_initial_tensor_sha256'] == report['teacher_final_tensor_sha256'] == expected_source,
         'Separate original FP16 teacher507 hashes differ from source ledger')
    expected_selected = {n: dict(shape=item['shape'], numel=item['numel'], decoded_sha256=item['decoded_sha256'])
                         for n, item in retained_inventory(ledger).items()}
    need(report['selected_inventory'] == expected_selected and report['selected_parameters'] == RETAINED_PARAMETERS
         and report['retained_fp16_payload_bytes'] == RETAINED_BYTES,
         'Only existing retained393 selection/byte capacity differs')
    for key in ('initial_base_content', 'final_base_content'):
        base_check(report[key], actual_content=True)
    parity_check(report['initial_native_parity'])
    parity_check(report['final_native_export_parity'])
    native_check(report['final_native_restoration'])
    native_check(report['teacher_native_restoration'])
    stability_check(report['final_stability'])
    need(report['backend_policy'] == report['binding']['backend_policy'], 'Training/backend binding differs')
    backend_check(report['binding']['backend_policy'], report['backend_policy_final_check'])
    if smoke:
        gradient = report['checkpoint_gradient_smoke']
        need(gradient['complete'] is True and gradient['passed'] is True and gradient['tensors'] == 393
             and gradient['rtol'] == .005 and gradient['atol'] == .00003
             and number(gradient['maximum_absolute_difference'])
             and gradient['hidden_bitwise_equal'] is True and gradient['perturbed_masters'] is True
             and gradient['optimizer_steps_performed'] == 0,
             'Separate perturbed checkpoint-gradient smoke proof differs')
        reload = report['actual_packed_reload_parity']
        need(reload == dict(complete=True, passed=True, input_tokens=128,
             hidden_and_last_eight_logits_bitwise_equal=True)
             and digest(report['smoke_candidate_manifest_sha256']) and report['candidates'] == [],
             'Discarded smoke packed/native readback proof differs')
    else:
        need([r['successful_updates'] for r in report['candidates']] == list(CHECKPOINT_STEPS),
             'Four immutable formal candidate checkpoints required')
    return counting


def evidence_path(recorded, args):
    """Remap frozen recorded GPU path strings without changing their binding."""
    need(isinstance(recorded, str), 'Recorded artifact path must be a string')
    path = Path(recorded)
    if path.is_absolute():
        try:
            relative = path.relative_to(args.recorded_root)
        except ValueError:
            try:
                relative = path.relative_to(args.evidence_root.resolve())
            except ValueError as exc:
                raise ValueError('Recorded artifact escaped declared evidence root: ' + recorded) from exc
    else:
        relative = path
    actual = args.evidence_root / relative
    need(actual.resolve().is_relative_to(args.evidence_root.resolve()) and actual.exists()
         and not actual.is_symlink(), 'Missing/remapped artifact outside evidence root: ' + recorded)
    return actual


def dataset_check(value, split):
    total = 264765 if split == 'validation' else 300964
    token_sha = VALIDATION_TOKEN_SHA if split == 'validation' else TEST_TOKEN_SHA
    need(value['dataset'] == 'Salesforce/wikitext' and value['configuration'] == 'wikitext-2-raw-v1'
         and value['split'] == split and value['revision_argument'] == REVISION
         and value['document_join'] == 'two newline characters'
         and value['token_stream_sha256_int64le'] == token_sha
         and value['total_tokens'] == total and value['tokenizer_sha256'] == TOKENIZER_SHA
         and value['automatic_special_tokens'] is False,
         'Pinned official ' + split + ' data/tokenizer identity differs')
    if split == 'test':
        need(value['text_sha256'] == TEST_TEXT_SHA and value['dataset_fingerprint'] == 'a46124b21ac53738',
             'Original official test text/fingerprint differs')


def core_check(directory, manifest, report, candidate, args):
    core_path = directory / 'training_report.json'
    core = read(core_path)
    step = candidate['successful_updates']
    bound = manifest['training_binding']
    need(bound == dict(protocol_sha256=sha(PROTOCOL), training_report_sha256=sha(core_path),
                       successful_updates=step, discarded=False)
         and core['format'] == 'FP4_BASE_REPAIR_CHECKPOINT_CORE_V1' and core['complete'] is True
         and core['mode'] == 'formal' and core['discarded'] is False
         and core['successful_updates'] == step and core['binding'] == report['binding']
         and core['no_resurface_installed'] is True and type(core['attempts']) is int
         and core['history'] == report['history'][:core['attempts']],
         'Immutable candidate core is not a prefix of this complete formal training run')
    history_check(core['history'], schedule_check(args.schedule), step, core['attempts'],
                  sum(row['overflow'] for row in core['history']))
    selected = retained_inventory(args.parent_ledger_value)
    need(core['retained_tensor_sha256'] == {n: manifest['decoded_tensor_sha256'][n] for n in selected}
         and digest(core['retained_export_sha256']), 'Core393 export differs from selected decoded ledger')
    need(core['checkpoint_readback'] == dict(complete=True, passed=True, retained_tensors=393,
         fp16_export_equals_saved_master_cast=True, masters_exact=True, optimizer_exact=True,
         scaler_exact=True, counters_exact=True), 'Saved master/FP16 export/optimizer/scaler readback proof differs')
    parity_check(core['native_export_parity'])
    base_check(core['parent_base_frozen'], actual_content=False)
    stability_check(core['stability'])
    # Binaries may remain only on the GPU. Metadata-only mode checks their
    # recorded binding and readback proof without claiming to read them here.
    need(digest(core['optimizer_checkpoint']['sha256'])
         and type(core['optimizer_checkpoint']['bytes']) is int and core['optimizer_checkpoint']['bytes'] > 0,
         'Optimizer checkpoint binding malformed')
    if args.verify_training_binaries:
        bound_file(directory.parent, core['optimizer_checkpoint'])
        need(sha(file_in(directory.parent, core['retained_export_file'])) == core['retained_export_sha256'],
             'Actual retained FP16 export archive differs')
    return core


def selection_check(args, report):
    selection = read(args.selection)
    need(selection['format'] == 'FP4_BASE_REPAIR_SELECTION_V1' and selection['complete'] is True
         and selection['protocol_sha256'] == sha(PROTOCOL)
         and selection['training_report_sha256'] == sha(args.training_report)
         and evidence_path(selection['training_report_path'], args).resolve() == args.training_report.resolve()
         and selection['successful_updates'] == 896 and selection['test_used_for_selection'] is False
         and selection['criterion'] == 'minimum_full_validation_pooled_nll'
         and selection['tie_breaker'] == 'earlier_successful_update'
         and selection['candidate_pool'] == report['candidates'],
         'Validation-only selection is not bound to complete896 formal training')
    validation_tokens = args.validation_tokens or args.training_report.parent / 'validation_tokens.int64le'
    plan = token_windows(validation_tokens, 'validation')
    dataset_check(report['validation_dataset'], 'validation')
    ppl_check(report['baseline_validation'], plan)
    rows, checkpoints = [], {}
    for candidate in selection['candidate_pool']:
        step = candidate['successful_updates']
        path = evidence_path(candidate['validation_report_path'], args)
        need(sha(path) == candidate['validation_report_sha256'], 'Validation report SHA differs')
        validation = read(path)
        need(validation['format'] == 'FP4_BASE_REPAIR_VALIDATION_V1' and validation['complete'] is True
             and validation['successful_updates'] == step
             and validation['candidate_manifest_sha256'] == candidate['candidate_manifest_sha256']
             and validation['no_resurface_installed'] is True
             and validation['no_statequant_controller_installed'] is True
             and validation['backend_policy'] == report['backend_policy'],
             'Validation candidate execution/checkpoint binding differs')
        native_check(validation['native_restoration'])
        dataset_check(validation['dataset'], 'validation')
        need(validation['dataset'] == report['validation_dataset'], 'Four candidates used different validation data')
        score = ppl_check(validation['ppl'], plan)
        close(candidate['validation_nll'], score['nll'], 'Selection pool NLL differs from full validation')
        need(candidate['validation_target_tokens'] == score['target_tokens'], 'Selection target count differs')
        directory = evidence_path(candidate['candidate_dir'], args)
        manifest, ledger, packing = candidate_check(directory, verify_weights=args.verify_weights)
        need(packing['candidate_manifest_sha256'] == candidate['candidate_manifest_sha256']
             and ledger == args.parent_ledger_value, 'Candidate checkpoint or original conversion ledger differs')
        core_check(directory, manifest, report, candidate, args)
        rows.append(dict(successful_updates=step, split='validation', **score))
        checkpoints[step] = dict(directory=directory, manifest=manifest, packing=packing)
    chosen = selected_step(rows)
    winner = next(c for c in selection['candidate_pool'] if c['successful_updates'] == chosen)
    need(selection['chosen_successful_updates'] == chosen
         and selection['chosen_candidate_dir'] == winner['candidate_dir']
         and selection['chosen_candidate_manifest_sha256'] == winner['candidate_manifest_sha256'],
         'Chosen checkpoint does not minimize complete validation NLL with earlier-step exact tie break')
    return selection, rows, checkpoints[chosen]


def actual_weight_check(receipt, expected):
    need(receipt == dict(complete=True, tensors=507, actual_content_checked=True,
         actual_tensor_sha256=expected, resident_weight_bytes=DECODED_BYTES),
         'Recorded loaded507 FP16 weight contents/residency differ')


def comparison_check(args, training, selection, winner):
    directory = args.comparison.parent
    comparison = read(args.comparison)
    need(comparison['format'] == 'FP4_BASE_REPAIR_PPL_EVAL_V1' and comparison['complete'] is True
         and comparison['mode'] == 'full' and comparison['split'] == 'test'
         and comparison['protocol_sha256'] == sha(PROTOCOL)
         and comparison['runner_sha256'] == sha(ROOT / 'scripts/run_fp4_base_repair_ppl_v1.py')
         and comparison['parent_manifest_sha256'] == PARENT_MANIFEST_SHA
         and comparison['conversion_receipt_sha256'] == PARENT_LEDGER_SHA
         and comparison['candidate_manifest_sha256'] == selection['chosen_candidate_manifest_sha256']
         and comparison['candidate_successful_updates'] == selection['chosen_successful_updates']
         and comparison['selection_receipt_sha256'] == sha(args.selection)
         and comparison['test_used_for_training_or_selection'] is False
         and comparison['project_splits_previously_exposed'] is True
         and comparison['adapter_loaded'] is False and comparison['statequant_controller_installed'] is False
         and comparison['publication_performed'] is False and comparison['integrity_pass'] is True,
         'Final test is not a fixed validation winner or changed execution/scope')
    code_check(comparison['code_sha256'])
    need(comparison['backend_policy'] == training['backend_policy'], 'Final test/backend training mismatch')
    backend_check(comparison['backend_policy'], comparison['backend_policy_final_check'])
    dataset_check(comparison['dataset'], 'test')
    tokens = bound_file(directory, comparison['tokens_file'])
    plan = token_windows(tokens, 'test')
    need(comparison['windowing'] == dict(execution='prefill', state_reset_each_window=True,
         state_requantized_each_token=False, persistent_ppl_cache_bytes=None, window_count=147,
         target_tokens=300963, final_partial_included=True, targets_per_full_window=2048, logits_chunk_tokens=64),
         'Final full native-prefill/reset protocol differs')
    copied = comparison['candidate_metadata']
    copied_dir = directory / copied['directory']
    need(copied_dir.resolve().is_relative_to(directory.resolve()) and not copied_dir.is_symlink()
         and set(copied['files']) == {'weight_manifest.json', 'parent_weight_manifest.json',
             'conversion_receipt.json', 'training_report.json', 'SHA256SUMS'},
         'Copied sealed candidate metadata inventory differs')
    for name, receipt in copied['files'].items():
        bound_file(copied_dir, dict(file=name, **receipt))
    copied_manifest, ledger, _ = candidate_check(copied_dir)
    need(copied_manifest == winner['manifest']
         and sha(copied_dir / 'training_report.json') == comparison['candidate_training_report_sha256'],
         'Final test evaluated a different selected training export')
    selected = comparison['selection']
    need(selected['receipt_sha256'] == sha(args.selection)
         and selected['training_report_sha256'] == sha(args.training_report)
         and selected['chosen_successful_updates'] == selection['chosen_successful_updates']
         and selected['candidate_fixed_before_test'] is True and selected['validation_only'] is True,
         'Final test selection evidence differs')
    need(sha(file_in(directory, selected['receipt_file'])) == sha(args.selection)
         and sha(file_in(directory, selected['training_report_file'])) == sha(args.training_report),
         'Copied complete training/selection files differ')
    evidence = selected['candidate_evidence_files']
    expected_evidence = {f'selection_evidence/step{step:04d}/{name}'
                         for step in CHECKPOINT_STEPS for name in ('validation_report.json', 'weight_manifest.json')}
    need(set(evidence) == expected_evidence, 'Final test lacks all four fixed validation candidates')
    for candidate in selection['candidate_pool']:
        for name, expected in (('validation_report.json', candidate['validation_report_sha256']),
                               ('weight_manifest.json', candidate['candidate_manifest_sha256'])):
            relative = f"selection_evidence/step{candidate['successful_updates']:04d}/{name}"
            path = bound_file(directory, dict(file=relative, **evidence[relative]))
            need(sha(path) == expected, 'Final test copied a different validation candidate')
    expected = dict(baseline={n: v['decoded_sha256'] for n, v in ledger['tensors'].items()},
                    repaired=copied_manifest['decoded_tensor_sha256'],
                    source_fp16={n: v['source_fp16_sha256'] for n, v in ledger['tensors'].items()})
    need(set(comparison['reports']) == set(expected), 'Final test requires baseline/repaired/source FP16 arms')
    rows, scores = {}, {}
    for arm, hashes in expected.items():
        receipt = comparison['reports'][arm]
        path = bound_file(directory, receipt)
        row = read(path)
        need(row['format'] == 'FP4_BASE_REPAIR_PPL_EVAL_V1' and row['complete'] is True
             and row['arm'] == arm and row['dataset'] == comparison['dataset']
             and row['adapter_loaded'] is False and row['statequant_controller_installed'] is False
             and row['state_mode'] == 'fp16_native_ssd', 'Final test arm scope/identity differs')
        for key in ('actual_weight_check_before', 'actual_weight_check_after'):
            actual_weight_check(row[key], hashes)
        native_check(row['native_methods_check'])
        identity = row['base_identity']
        need(identity == dict(complete=True, tensors=507, parameters=PARAMETERS,
             identity_version_gradients_unchanged=True), 'Arm native507 identities/gradients changed')
        backend_check(comparison['backend_policy'], row['backend_policy_check'])
        scores[arm] = ppl_check(row['ppl'], plan)
        for key in ('ppl', 'nll'):
            close(receipt[key], scores[arm][key], 'Comparison arm receipt scalar differs: ' + arm + '/' + key)
        need(receipt['target_tokens'] == 300963, 'Comparison arm target count differs')
        rows[arm] = row['ppl']
    result = comparison['comparison']
    baseline, repaired, source = (scores[name]['ppl'] for name in ('baseline', 'repaired', 'source_fp16'))
    for key, value in dict(baseline_ppl=baseline, repaired_ppl=repaired, source_fp16_ppl=source,
                          ppl_delta=repaired-baseline, ppl_relative_change=repaired/baseline-1).items():
        close(result[key], value, 'Final paired comparison arithmetic differs: ' + key)
    need(result['windows_with_lower_repaired_nll'] == sum(r['nll'] < b['nll'] for r, b in
         zip(rows['repaired']['windows'], rows['baseline']['windows']))
         and result['absolute_test_ppl_target'] == 7.2
         and result['repaired_full_test_ppl_strictly_below_7_2'] is (repaired < 7.2),
         'Fixed strict target or matched improved-window accounting differs')
    need(comparison['memory'] == dict(logical_weight_payload_bytes=PAYLOAD_BYTES,
         actual_weight_shard_bytes=SHARD_BYTES, expanded_fp16_weight_bytes=DECODED_BYTES,
         retained_fp16_payload_bytes=RETAINED_BYTES, persistent_ppl_cache_bytes=None,
         adapter_payload_bytes=0, promoted_weights=0), 'Inference capacity/cache/no-adapter scope differs')
    need(comparison['integrity_checks'] == dict(all507_weight_contents_exact_before_after=True,
         original114_fp4_matrices_unchanged=True, same119_weight_shard_geometry_bytes=True,
         retained393_fp16_payload_bytes_unchanged=True, same_full_paired_population=True,
         native_methods_and_parameters_unchanged=True, no_resurface_or_statequant_installed=True),
         'Recorded final integrity checklist incomplete')
    return dict(ppl={arm: score['ppl'] for arm, score in scores.items()}, windows=147, targets=300963,
                strictly_below_7_2=repaired < 7.2, ppl_relative_change=repaired/baseline-1,
                comparison_sha256=sha(args.comparison), chosen_successful_updates=selection['chosen_successful_updates'])


def self_test():
    rows = [dict(successful_updates=step, nll=500000. - step, split='validation',
                 target_tokens=264764, windows=130) for step in CHECKPOINT_STEPS]
    need(selected_step(rows) == 896, 'Minimum validation NLL fixture failed')
    rows[2]['nll'] = rows[3]['nll']
    need(selected_step(rows) == 672, 'Earlier-step exact tie fixture failed')
    rejected = 0
    for field, value in (('split', 'test'), ('target_tokens', 300963), ('windows', 16), ('nll', float('nan'))):
        altered = [dict(row) for row in rows]
        altered[0][field] = value
        try:
            selected_step(altered)
        except ValueError:
            rejected += 1
    need(rejected == 4, 'Selection boundary fixture accepted test/partial/nonfinite data')
    plan = [(0, 2, '0' * 64)]
    ppl = dict(execution='prefill', cache_dtype='SSD scan internal precision', cache_bytes=None,
               logits_chunk_tokens=64, target_tokens=2, nll=4., ppl=math.exp(2.), windows=[
               dict(start=0, input_tokens=2, target_tokens=2, token_sha256_int64le='0'*64,
                    nll=4., ppl=math.exp(2.))])
    ppl_check(ppl, plan)
    ppl['nll'] = 3.
    try:
        ppl_check(ppl, plan)
    except ValueError:
        pass
    else:
        raise ValueError('Pooled NLL tamper fixture accepted')
    def update(attempt, overflow, scale):
        return dict(attempt=attempt, successful_updates_before=0, successful_updates_after=0 if overflow else 1,
            window_index=7, epoch=0, position_in_epoch=0, learning_rate=1e-4, targets=2047,
            forward_positions=2047, chunks=32, chunk_selected_targets=[64]*31+[63],
            ce_coefficient=1., kl_coefficient=.1, temperature=1., backbone_backward_calls=1,
            gate_and_hidden_backward_joint=False, closure=0., opening=0., weighted_closure=0.,
            weighted_opening=0., ce=2., teacher_to_student_kl=.25, loss=2.025,
            loss_scale_before=scale, loss_scale_after=scale*.5 if overflow else scale,
            gradients=dict(tensors=393, parameters=RETAINED_PARAMETERS, finite=not overflow, nonzero=1),
            overflow=overflow, gradient_norm_before_clip=None if overflow else .25,
            overflow_optimizer_and_masters_unchanged=True if overflow else None,
            scaled_hidden_gradient_finite=not overflow, scaled_hidden_gradient_nonzero=1,
            scaled_hidden_gradient_elements=2047*4096)
    history = [update(1, True, 1024.), update(2, False, 512.)]
    history_check(history, [7], 1, 2, 1)
    rejected_history = 0
    for field, value in (('window_index', 8), ('learning_rate', 2e-4), ('loss', 1.),
                          ('targets', 10), ('successful_updates_after', 2)):
        altered = [dict(row) for row in history]
        altered[-1][field] = value
        try:
            history_check(altered, [7], 1, 2, 1)
        except ValueError:
            rejected_history += 1
    need(rejected_history == 5, 'TRAIN schedule/objective/counter boundary fixture accepted')
    return dict(complete=True, passed=True, selection_minimum_and_tie=True,
                test_partial_nonfinite_selection_rejected=True, pooled_nll_tamper_rejected=True,
                overflow_same_window_accounting=True, wrong_schedule_lr_loss_targets_counter_rejected=True,
                torch_imported='torch' in sys.modules, cuda_initialized=False)


def audit(args):
    need(sha(PROTOCOL) == PROTOCOL_SHA, 'Frozen V1 protocol changed')
    need(sha(args.packed_manifest) == PARENT_MANIFEST_SHA and sha(args.conversion_ledger) == PARENT_LEDGER_SHA,
         'Pinned original packed manifest/conversion ledger differ')
    args.parent_ledger_value = read(args.conversion_ledger)
    retained_inventory(args.parent_ledger_value)
    order = schedule_check(args.schedule)
    smoke = read(args.smoke_report)
    smoke_counting = training_report_check(smoke, args.parent_ledger_value, order, smoke=True)
    result = dict(format=AUDIT_FORMAT, complete=True, integrity_pass=True, protocol_sha256=PROTOCOL_SHA,
        auditor_sha256=sha(__file__), schedule_sha256=SCHEDULE_SHA,
        packed_manifest_sha256=PARENT_MANIFEST_SHA, conversion_ledger_sha256=PARENT_LEDGER_SHA,
        smoke_report_sha256=sha(args.smoke_report), smoke=smoke_counting,
        mode='smoke' if args.smoke_only else 'training', training=None, validation_selection=None,
        test=None, weights_independently_reread=bool(args.verify_weights and args.selection),
        optimizer_binaries_independently_reread=bool(args.verify_training_binaries and args.selection),
        torch_imported=False, cuda_initialized=False,
        scope='CPU saved-report/token/hash/selection/NLL arithmetic audit; recorded GPU native parity and tensor receipts are not GPU inference replay. '
              'Full packed shard reads occur only with --verify-weights; optimizer archive reads occur only with --verify-training-binaries.')
    if not args.smoke_only:
        report = read(args.training_report)
        need(report['binding'] == smoke['binding'] and report['smoke_report_sha256'] == sha(args.smoke_report)
             and args.training_report.resolve().parent != args.smoke_report.resolve().parent,
             'Formal run did not use a separate, discarded same-binding smoke')
        result['training'] = training_report_check(report, args.parent_ledger_value, order)
        result['training_report_sha256'] = sha(args.training_report)
        if args.selection:
            selection, scores, winner = selection_check(args, report)
            result['mode'] = 'selection'
            result['validation_selection'] = dict(selection_sha256=sha(args.selection),
                candidates=scores, chosen_successful_updates=selection['chosen_successful_updates'],
                chosen_candidate_manifest_sha256=selection['chosen_candidate_manifest_sha256'],
                validation_only=True, checkpoint_metadata=winner['packing'])
            if args.comparison:
                result['mode'] = 'combined'
                result['test'] = comparison_check(args, report, selection, winner)
    need('torch' not in sys.modules, 'Independent CPU auditor unexpectedly imported Torch')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--training-report', type=Path)
    parser.add_argument('--smoke-report', type=Path)
    parser.add_argument('--packed-manifest', type=Path, default=ROOT/'artifacts/fp4_g16_s16_resurface_v1/evaluation/weight_manifest.json')
    parser.add_argument('--conversion-ledger', type=Path, default=ROOT/'artifacts/fp4_g16_s16_resurface_v1/evaluation/conversion_receipt.json')
    parser.add_argument('--schedule', type=Path, default=SCHEDULE)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--comparison', type=Path)
    parser.add_argument('--validation-tokens', type=Path)
    parser.add_argument('--evidence-root', type=Path)
    parser.add_argument('--recorded-root', type=Path, default=Path('/home/horde/mamb2_8B_W4A16_Recall_StateQuant/artifacts/fp4_base_repair_v1'))
    parser.add_argument('--verify-weights', action='store_true', help='Independently stream all119 shards/regions for each candidate; absent this flag audit small recorded metadata only')
    parser.add_argument('--verify-training-binaries', action='store_true', help='Hash original saved optimizer/export archives; they may remain only on GPU')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        need(args.smoke_report is not None and args.out is not None, '--smoke-report and fresh --out required')
        need(args.smoke_only or args.training_report is not None, '--training-report required for formal audit')
        need(not args.smoke_only or not any((args.training_report, args.selection, args.comparison)),
             '--smoke-only cannot consume formal selection/test evidence')
        need(args.comparison is None or args.selection is not None, '--comparison requires fixed --selection')
        need(args.selection is not None or not (args.verify_weights or args.verify_training_binaries),
             'Full candidate shard/archive verification requires --selection')
        need(not args.out.exists(), 'Preserve existing CPU audit output')
        anchor = args.training_report or args.smoke_report
        if args.evidence_root is None:
            args.evidence_root = anchor.parent.parent
        result = audit(args)
    need(result['torch_imported'] is False, 'CPU auditor unexpectedly imported Torch')
    if args.out:
        need(not args.out.exists(), 'Preserve existing CPU audit output')
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
