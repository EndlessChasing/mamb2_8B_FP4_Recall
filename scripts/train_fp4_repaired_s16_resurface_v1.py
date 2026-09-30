#!/usr/bin/env python3
"""Fresh final-only Resurface on the selected repaired FP4 G16 base.

Both bases decode the same audited checkpoint. No state controller or surrogate
backward is installed: student training and teacher use the original native SSD
prefill. Native decoding retains its usual FP16 inference cache.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall import fp4_checkpoint as packed, fp4_repaired_checkpoint as repaired

FORMAT = 'FP4_REPAIRED_S16_RESURFACE_TRAIN_V1'
CHECKPOINT_FORMAT = 'FP4_REPAIRED_S16_RESURFACE_CHECKPOINT_V1'
PREFLIGHT_FORMAT = 'FP4_REPAIRED_S16_RESURFACE_TRAIN_PREFLIGHT_V1'
PROTOCOL_SHA = '9592a0dddcaa5c7d000422e5ac902b49df362781e4a4d9f668d99cb849cf8a7f'
STEPS = 1536
SEED = 2026092803
TRAIN_SHA = '451b8703c21120667ef0ea272c21a10d4cd4561779a7b9663773ae47981600c0'
PROSE_MANIFEST_SHA = 'facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'
PROSE_TOKENS_SHA = 'e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233'
PACKED_MANIFEST_SHA = '2d2208b690c2ec0def90578d8590cc122bb1e4542b0d49ae4417a48f72b0b73c'
REPAIR_SELECTION_SHA = '2f1ff559f3e907b3803bcc6d7036d39dab88c11dbb5e69f1de57e2dff5ca452c'
REPAIR_TEST_COMPARISON_SHA = 'c3a38223055d53603a64302b0acdbd8e236baa402d67e6927a126848ab21522a'
REPAIR_TRAINING_CORE_SHA = 'd5fb50f2e045b6202a5f38b353ceab7ae47b21c7cc88e8cb398b46c342632c6e'
TOKENIZER_SHA = '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09'
TOKENIZER_FILENAME = 'mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model'
OBJECTIVE = dict(mk_ce=1., mk_kl=0., mk_closure=0., prose_ce=.5,
                 prose_kl=.5, prose_closure=3., temperature=1., chunk_tokens=64)


def load_training_modules():
    """Bind all lazy loader/training imports before CUDA work and source hashes."""
    global torch, fp4, data, loss, native, runtime
    global check_optimizer, equal_tree, expected_scaler_after
    global pin_replay_backend, check_replay_backend
    import torch
    from mamba2_recall import fp4
    from mamba2_recall import resurface_data as data, resurface_loss as loss
    from mamba2_recall import resurface_native as native, runtime
    from train_resurface_more import check_optimizer, equal_tree, expected_scaler_after
    from evaluate_resurface_more import pin_replay_backend, check_replay_backend


def need(ok, message):
    if not ok:
        raise ValueError(message)


def put(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def hashes(protocol):
    # Include actual inherited imports as well as the native training modules.
    files = {Path(__file__).resolve(), protocol.resolve()}
    for name in ('mamba2_recall/fp4_repaired_checkpoint.py', 'mamba2_recall/fp4_checkpoint.py',
                 'mamba2_recall/fp4.py', 'mamba2_recall/runtime.py',
                 'mamba2_recall/resurface_data.py', 'mamba2_recall/resurface_loss.py',
                 'mamba2_recall/resurface_native.py', 'scripts/train_resurface_more.py',
                 'scripts/evaluate_resurface_more.py'):
        files.add(ROOT / name)
    need(all(p.is_relative_to(ROOT) and p.is_file() for p in files),
         'Training script and protocol must be real files inside this repository')
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            path = Path(path).resolve()
            # Torch can assign a synthetic in-repo __file__ to generated
            # modules. Such labels are not on-disk source and cannot be hashed.
            if path.suffix == '.py' and path.is_relative_to(ROOT) and path.is_file():
                files.add(path)
    return {str(p.relative_to(ROOT)): packed.sha_file(p) for p in sorted(files)}


def ledger_digest(expected):
    raw = (json.dumps(expected, sort_keys=True, separators=(',', ':')) + '\n').encode()
    return hashlib.sha256(raw).hexdigest()


def repaired_metadata(args):
    """Read only immutable small repaired-base/provenance receipts; no Torch."""
    protocol_sha = packed.sha_file(args.protocol)
    need(protocol_sha == args.expected_protocol_sha256 == PROTOCOL_SHA,
         'Frozen repaired-base experiment protocol differs')
    manifest, ledger = repaired.read_repaired_metadata(args.packed_dir)
    need(packed.sha_file(args.packed_dir / 'weight_manifest.json') == PACKED_MANIFEST_SHA
         and manifest['format'] == repaired.FORMAT and manifest['complete'] is True
         and manifest['parent_manifest_sha256'] == repaired.PARENT_MANIFEST_SHA
         and manifest['conversion_receipt_sha256'] == packed.LEDGER_SHA
         and manifest['format_name'] == 'fp4_g16_e4m3'
         and manifest['source_checkpoint_sha256'] == packed.SOURCE_SHA
         and manifest['training_binding']['training_report_sha256'] == REPAIR_TRAINING_CORE_SHA
         and manifest['training_binding']['successful_updates'] == 448
         and manifest['training_binding']['discarded'] is False,
         'Expected the exact validation-selected repaired step448 checkpoint')
    core_path = args.packed_dir / 'training_report.json'
    need(packed.sha_file(core_path) == REPAIR_TRAINING_CORE_SHA,
         'Selected repair training core changed')
    core = packed.read_json(core_path)
    need(core['format'] == 'FP4_BASE_REPAIR_CHECKPOINT_CORE_V1'
         and core['complete'] is True and core['mode'] == 'formal'
         and core['discarded'] is False and core['successful_updates'] == 448
         and core['no_resurface_installed'] is True,
         'Expected the immutable no-adapter base-repair core')
    need(packed.sha_file(args.repair_selection) == REPAIR_SELECTION_SHA,
         'Completed base-repair selection receipt changed')
    selection = packed.read_json(args.repair_selection)
    need(selection['format'] == 'FP4_BASE_REPAIR_SELECTION_V1'
         and selection['complete'] is True and selection['successful_updates'] == 896
         and selection['test_used_for_selection'] is False
         and selection['criterion'] == 'minimum_full_validation_pooled_nll'
         and selection['tie_breaker'] == 'earlier_successful_update'
         and selection['chosen_successful_updates'] == 448
         and selection['chosen_candidate_manifest_sha256'] == PACKED_MANIFEST_SHA
         and [r['successful_updates'] for r in selection['candidate_pool']] == [224, 448, 672, 896],
         'Base repair was not selected from the fixed full-validation pool')
    need(packed.sha_file(args.repair_test_comparison) == REPAIR_TEST_COMPARISON_SHA,
         'Completed base-repair full-test comparison changed')
    previous = packed.read_json(args.repair_test_comparison)
    need(previous['format'] == 'FP4_BASE_REPAIR_PPL_EVAL_V1'
         and previous['complete'] is True and previous['mode'] == 'full'
         and previous['split'] == 'test' and previous['integrity_pass'] is True
         and previous['candidate_manifest_sha256'] == PACKED_MANIFEST_SHA
         and previous['candidate_successful_updates'] == 448
         and previous['candidate_training_report_sha256'] == REPAIR_TRAINING_CORE_SHA
         and previous['selection_receipt_sha256'] == REPAIR_SELECTION_SHA
         and previous['adapter_loaded'] is False
         and previous['statequant_controller_installed'] is False,
         'Previous full test did not bind this exact unadapted repaired base')
    # The original conversion ledger is only historical quantization provenance.
    # Its old retained-FP16 hashes are NOT the expected current base values.
    expected = dict(manifest['decoded_tensor_sha256'])
    need(len(expected) == 507 and set(expected) == set(ledger['tensors']),
         'Repaired decoded ledger must cover all 507 tensors')
    need(sum(expected[n] != row['decoded_sha256'] for n, row in ledger['tensors'].items()
             if row['kind'] == 'fp16') == 393,
         'Selected repair must contain its updated393 retained FP16 values')
    provenance = dict(repaired_checkpoint_format=repaired.FORMAT,
        repair_selected_successful_updates=448,
        parent_packed_manifest_sha256=repaired.PARENT_MANIFEST_SHA,
        repair_selection_sha256=REPAIR_SELECTION_SHA,
        repair_test_comparison_sha256=REPAIR_TEST_COMPARISON_SHA,
        repair_training_core_sha256=REPAIR_TRAINING_CORE_SHA)
    return manifest, expected, provenance


def metadata_preflight(args):
    """Immutable input-file checks only, without reading large weight shards."""
    manifest, expected, provenance = repaired_metadata(args)
    tokenizer_path = args.source_dir / TOKENIZER_FILENAME if args.source_dir.is_dir() else args.source_dir
    need(packed.sha_file(tokenizer_path) == TOKENIZER_SHA, 'Pinned original tokenizer differs')
    need(packed.sha_file(args.data_root / 'train/manifest.json') == TRAIN_SHA
         and packed.sha_file(args.prose_manifest) == PROSE_MANIFEST_SHA
         and packed.sha_file(args.prose_tokens) == PROSE_TOKENS_SHA,
         'Pinned numeric/prose TRAIN input files differ')
    prose = packed.read_json(args.prose_manifest)
    need(prose['complete'] is True and prose['training_tokens_file_sha256'] == PROSE_TOKENS_SHA
         and sorted(prose['schedule']) == list(range(448)), 'Prose manifest/schedule differs')
    return dict(format=PREFLIGHT_FORMAT, complete=True, passed=True,
        protocol_sha256=PROTOCOL_SHA, packed_manifest_sha256=PACKED_MANIFEST_SHA,
        conversion_ledger_sha256=packed.LEDGER_SHA,
        decoded_ledger_sha256=ledger_digest(expected),
        decoded_tensor_sha256=expected, base_provenance=provenance,
        code_sha256=hashes(args.protocol), tokenizer_sha256=TOKENIZER_SHA,
        numeric_train_manifest_sha256=TRAIN_SHA,
        prose_manifest_sha256=PROSE_MANIFEST_SHA, prose_tokens_sha256=PROSE_TOKENS_SHA,
        logical_weight_payload_bytes=manifest['logical_weight_payload_bytes'],
        shard_file_bytes=manifest['shard_file_bytes'],
        independent_weight_shards_read=False,
        scope='Small metadata/core/source bindings and immutable input files; '
              'no model load, tensor deserialization, CUDA or physical weight-shard verification',
        torch_imported='torch' in sys.modules, cuda_initialized=False)


def inputs(args):
    _, expected, provenance = repaired_metadata(args)
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    _, _, examples = data.load_training(args.data_root, TRAIN_SHA, tokenizer)
    need(packed.sha_file(args.data_root / 'train/manifest.json') == TRAIN_SHA
         and len(examples) == STEPS, 'Pinned numeric TRAIN differs')
    need(packed.sha_file(args.prose_manifest) == PROSE_MANIFEST_SHA
         and packed.sha_file(args.prose_tokens) == PROSE_TOKENS_SHA,
         'Pinned prose TRAIN files differ')
    prose = packed.read_json(args.prose_manifest)
    windows = torch.load(args.prose_tokens, map_location='cpu', weights_only=True)
    need(prose['complete'] is True and tuple(windows.shape) == (448, 2048)
         and windows.dtype == torch.int64
         and prose['training_tokens_file_sha256'] == PROSE_TOKENS_SHA
         and sorted(prose['schedule']) == list(range(448))
         and runtime.token_digest(windows.flatten().numpy()) ==
         prose['training_tokens_sha256_int64le'], 'Pinned prose TRAIN population differs')
    binding = dict(format=FORMAT, protocol_sha256=PROTOCOL_SHA,
        source_checkpoint_sha256=packed.SOURCE_SHA, tokenizer_sha256=tokenizer.sha256,
        packed_manifest_sha256=packed.sha_file(args.packed_dir / 'weight_manifest.json'),
        conversion_ledger_sha256=packed.LEDGER_SHA, decoded_ledger_sha256=ledger_digest(expected),
        weight_format='fp4_g16_e4m3', state_mode='fp16_native_ssd',
        student_execution='original native SSD prefill and backward; no state-controller replacement',
        teacher='separate same repaired FP4 G16 base, unadapted, native SSD FP16 state',
        decode_state='native FP16 inference cache; no per-token artificial rounding during prefill',
        numeric_train_manifest_sha256=TRAIN_SHA, prose_manifest_sha256=PROSE_MANIFEST_SHA,
        prose_tokens_sha256=PROSE_TOKENS_SHA,
        prose_tokens_int64le_sha256=prose['training_tokens_sha256_int64le'],
        initial_adapter='fresh V=0,g=1,w=0,b=-4; no pretrained adapter',
        adapter=native.FORMAT, gate_mode='soft', objective=OBJECTIVE, seed=SEED,
        planned_successful_updates=STEPS, successful_updates=STEPS,
        checkpoint_selection='final successful update only; no quality selection')
    binding.update(provenance)
    return examples, windows, prose['schedule'], expected, binding


def check_model(model, expected, manifest_sha):
    receipt = model._fp4_repaired_receipt
    need(receipt['format'] == repaired.FORMAT and receipt['complete'] is True
         and receipt['base_only'] is True and receipt['source_checkpoint_required'] is False
         and receipt['logical_weight_payload_bytes'] == packed.PAYLOAD_BYTES
         and receipt['shard_file_bytes'] == repaired.SHARD_BYTES
         and receipt['decoded_resident_weight_bytes'] == packed.WEIGHT_BYTES
         and receipt['serialization'] == 'safetensors_packed_checkpoint'
         and receipt['packed_checkpoint_manifest_sha256'] == manifest_sha == PACKED_MANIFEST_SHA
         and receipt['parent_manifest_sha256'] == repaired.PARENT_MANIFEST_SHA
         and receipt['conversion_receipt_sha256'] == packed.LEDGER_SHA
         and receipt['training_binding']['training_report_sha256'] == REPAIR_TRAINING_CORE_SHA
         and receipt['training_binding']['successful_updates'] == 448
         and receipt['decoded_tensor_sha256'] == expected and receipt['packed_resident_kernel'] is False,
         'Repaired packed loader provenance differs')
    params = dict(model.named_parameters())
    need(set(params) == set(expected) and len(params) == 507
         and sum(p.numel() for p in params.values()) == 8236999680,
         'Decoded parameter inventory differs')
    actual = {}
    for name, value in params.items():
        need(value.dtype == torch.float16 and not value.is_meta
             and not value.requires_grad and value.grad is None,
             'Decoded base dtype/allocation/gradient differs: ' + name)
        actual[name] = native.tensor_hash(value)
        need(actual[name] == expected[name], 'Decoded base bytes differ: ' + name)
    return dict(complete=True, passed=True, actual_content_checked=True,
        tensors=507, parameters=8236999680, decoded_tensor_sha256=actual,
        decoded_ledger_sha256=ledger_digest(actual), weight_payload_bytes=packed.WEIGHT_BYTES)


def snapshot(model):
    return [(mx, mx.forward, dict(mx._forward_pre_hooks), dict(mx._forward_hooks))
            for layer in model.backbone.layers for mx in (layer.mixer, layer.mixer.norm)]


def restored(before):
    need(all(mx.forward == forward and dict(mx._forward_pre_hooks) == pre
             and dict(mx._forward_hooks) == post for mx, forward, pre, post in before),
         'Original native mixer forwards/hooks were not restored')
    return dict(complete=True, native_mixer_forwards_restored=True, native_hooks_restored=True)


def lr_factor(j):
    return .1 + .9 * (1 + math.cos(math.pi * j / (STEPS - 1))) / 2


def schedule():
    return torch.randperm(STEPS, generator=torch.Generator(device='cpu').manual_seed(SEED)).tolist()


def pair_for(j, ordered, examples, windows, prose_order):
    row = examples[ordered[j]]
    full = torch.tensor(row['full_ids'], dtype=torch.long, device='cuda')[None]
    mask = torch.zeros_like(full[:, :-1], dtype=torch.bool)
    mask[:, row['ce_hidden_positions']] = True
    need(int(mask.sum()) == row['answer_target_count'], 'Numeric answer mask differs')
    prose_index = prose_order[j % 448]
    start = 512 * ((j // 448) % 4)
    prose = windows[prose_index, start:start + 512].cuda()[None]
    need(tuple(prose.shape) == (1, 512), 'Prose segment geometry differs')
    return dict(id=row['id'], schedule_entry=ordered[j], prose_window=prose_index,
        prose_start=start, mk_ids=full[:, :-1], mk_targets=full[:, 1:], answer_mask=mask,
        prose_ids=prose[:, :-1], prose_targets=prose[:, 1:])


def optimizer_for(bank):
    mix = [p for n, p in bank.masters.items() if n.endswith(('.V_read', '.g_read'))]
    router = [p for n, p in bank.masters.items() if n.endswith(('.router_w', '.router_b'))]
    return torch.optim.AdamW([dict(params=mix, lr=1e-4, base_lr=1e-4),
                             dict(params=router, lr=3e-4, base_lr=3e-4)],
                            betas=(.9, .999), eps=1e-8, weight_decay=0.)


def attempt(bank, teacher, pair, optimizer, scaler, j):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr'] * lr_factor(j)
    hidden, gates = bank.forward_hidden(pair['mk_ids'], use_checkpoint=True)
    mk = loss.staged_loss_backward(hidden, pair['mk_targets'], bank.model.lm_head,
        gates=gates, scaler=scaler, answer_mask=pair['answer_mask'],
        ce_weight=OBJECTIVE['mk_ce'], kl_weight=0., closure_weight=0., chunk_tokens=64)
    del hidden, gates
    with torch.no_grad():
        teacher_hidden = teacher.backbone(pair['prose_ids'])
    hidden, gates = bank.forward_hidden(pair['prose_ids'], use_checkpoint=True)
    prose = loss.staged_loss_backward(hidden, pair['prose_targets'], bank.model.lm_head,
        gates=gates, scaler=scaler, teacher_hidden=teacher_hidden, teacher_head=teacher.lm_head,
        ce_weight=OBJECTIVE['prose_ce'], kl_weight=OBJECTIVE['prose_kl'],
        closure_weight=OBJECTIVE['prose_closure'], chunk_tokens=64)
    del hidden, gates, teacher_hidden
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in bank.parameters()]
    need(all(g is not None for g in gradients), 'Missing adapter gradient')
    overflow = (not all(bool(torch.isfinite(g).all()) for g in gradients)
                or not mk['scaled_hidden_gradient_finite']
                or not prose['scaled_hidden_gradient_finite'])
    if overflow:
        scaler.update(new_scale=scaler.get_scale() * .5)
        magnitude = None
    else:
        magnitude = float(torch.nn.utils.clip_grad_norm_(bank.parameters(), 1., error_if_nonfinite=True))
        scaler.step(optimizer); scaler.update()
        need(all(bool(torch.isfinite(p).all()) and bool(torch.isfinite(p.half()).all())
                 for p in bank.parameters()), 'Nonfinite adapter master or FP16 cast')
    bank.assert_base_frozen()
    return dict(mk_ce=mk['ce'], prose_ce=prose['ce'], prose_kl=prose['teacher_to_student_kl'],
        prose_closure=prose['closure'], overflow=overflow,
        gradient_norm_before_clip=magnitude, loss_scale=scaler.get_scale())


def save_final(path, bank, optimizer, scaler, binding, successful, attempts):
    need(not path.exists() and not path.with_suffix('.pt.tmp').exists(), 'Checkpoint already exists')
    value = dict(format=CHECKPOINT_FORMAT, binding=binding, successful_updates=successful,
        attempts=attempts, masters=bank.state_dict(), optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict())
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('xb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    actual = torch.load(path, map_location='cpu', weights_only=True)
    equal_tree(actual, value, 'final checkpoint roundtrip')
    return dict(file=path.name, sha256=packed.sha_file(path), bytes=path.stat().st_size,
                successful_updates=successful, attempts=attempts), actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('packed-dir', 'source-dir', 'data-root', 'prose-manifest', 'prose-tokens', 'out-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--repair-selection', type=Path, required=True)
    parser.add_argument('--repair-test-comparison', type=Path, required=True)
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'docs/FP4_REPAIRED_S16_RESURFACE_V1_PROTOCOL.md')
    parser.add_argument('--expected-protocol-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-report', type=Path)
    args = parser.parse_args()
    need(not args.out_dir.exists(), 'Fresh output directory required')
    args.out_dir.mkdir(parents=True)
    if args.preflight_only:
        receipt = dict(format=PREFLIGHT_FORMAT, complete=False, passed=False,
                       torch_imported='torch' in sys.modules, cuda_initialized=False)
        try:
            need('torch' not in sys.modules, 'Metadata-only preflight must not import Torch')
            receipt = metadata_preflight(args)
            print(json.dumps(dict(format=PREFLIGHT_FORMAT, passed=True,
                  packed_manifest_sha256=receipt['packed_manifest_sha256'],
                  torch_imported=False, cuda_initialized=False)), flush=True)
        except BaseException as error:
            receipt.update(error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            put(args.out_dir / 'preflight.json', receipt)
        return
    need(args.smoke == (args.smoke_report is None), 'Formal requires a separate discarded smoke')
    report = dict(format=FORMAT, complete=False, mode='smoke' if args.smoke else 'formal',
        started_unix=time.time(), code_sha256=hashes(args.protocol), history=[], checkpoints=[],
        successful_updates=0, attempts=0, overflows=0)
    bank = None
    try:
        load_training_modules()
        torch.set_num_threads(8)
        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        examples, windows, prose_order, expected, binding = inputs(args)
        report['code_sha256'] = hashes(args.protocol)
        need('mamba2_recall/fp4_repaired_checkpoint.py' in report['code_sha256']
             and 'mamba2_recall/fp4.py' in report['code_sha256']
             and not torch.cuda.is_initialized(),
             'Loader closure must be bound before backend hardware query or model work')
        # The isolated failed-smoke diagnosis identified the backend import's
        # quack RMSNorm configuration query (get_device_capability) as the first
        # CUDA initialization. It loads no model and computes no model output.
        # Bind local sources before that query, then require the closure to stay
        # exact. The pinned policy remains applied before either model is loaded.
        policy = pin_replay_backend()
        need(hashes(args.protocol) == report['code_sha256'],
             'Local source closure changed during backend hardware-query setup')
        report.update(source_binding_before_backend_hardware_query=True,
            source_binding_before_first_model_load_and_compute=True,
            source_closure_unchanged_after_backend_setup=True,
            cuda_initialized_before_backend_setup=False,
            cuda_initialized_after_backend_setup=torch.cuda.is_initialized(),
            backend_hardware_query_initializes_cuda_context=torch.cuda.is_initialized())
        binding.update(code_sha256=report['code_sha256'], backend_policy=policy)
        report.update(binding=binding, backend_policy=policy)
        if args.smoke_report is not None:
            smoke = packed.read_json(args.smoke_report)
            need(smoke['format'] == FORMAT and smoke['complete'] is True
                 and smoke['mode'] == 'smoke' and smoke['successful_updates'] == 1
                 and smoke['binding'] == binding and smoke['code_sha256'] == report['code_sha256']
                 and smoke['adapter']['discarded'] is True
                 and args.smoke_report.parent.resolve() != args.out_dir.resolve(),
                 'Independent discarded smoke differs')
            report['smoke_report_sha256'] = packed.sha_file(args.smoke_report)
        put(args.out_dir / 'report.json', report)
        torch.cuda.manual_seed_all(SEED)
        print('[setup] decoding two independent repaired FP4 G16 bases for native SSD', flush=True)
        # Each independent strict reload performs full CPU shard/region verification.
        # Neither path can substitute the original conversion ledger's FP16 values.
        student = repaired.load_repaired_model(args.packed_dir, device='cuda', chunk_rows=512)
        teacher = repaired.load_repaired_model(args.packed_dir, device='cuda', chunk_rows=512)
        need(not {p.data_ptr() for p in student.parameters()} &
             {p.data_ptr() for p in teacher.parameters()}, 'Student and teacher share storage')
        manifest_sha = binding['packed_manifest_sha256']
        report.update(initial_student_base_check=check_model(student, expected, manifest_sha),
                      initial_teacher_base_check=check_model(teacher, expected, manifest_sha))
        student_snapshot, teacher_snapshot = snapshot(student), snapshot(teacher)
        teacher_identity = {n: (id(p), p.data_ptr(), p._version) for n, p in teacher.named_parameters()}
        reference = {}
        with torch.no_grad():
            for length in (128, 512):
                ids = windows[0, :length].cuda()[None]
                hidden = student.backbone(ids)
                need(bool(torch.isfinite(hidden).all()) and torch.equal(hidden, teacher.backbone(ids)),
                     'Independent native teacher/student initial forward differs')
                reference[length] = (ids, hidden.detach().clone())
        bank = native.ResurfaceNative(student, 'soft', expected_base_hashes=expected)
        need(len(bank.masters) == 224 and sum(p.numel() for p in bank.parameters()) == 1154104,
             'Fresh adapter geometry differs')
        for name, value in bank.masters.items():
            initial = dict(V_read=0., g_read=1., router_w=0., router_b=-4.)[name.rsplit('.', 1)[-1]]
            need(bool((value == initial).all()), 'Adapter is not freshly initialized')
        optimizer = optimizer_for(bank)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024., growth_factor=2.,
                                      backoff_factor=.5, growth_interval=2000)
        need(not optimizer.state and all(p.grad is None for p in bank.parameters()), 'Optimizer is not fresh')
        with torch.no_grad():
            for length, (ids, hidden) in reference.items():
                need(torch.equal(bank.forward_hidden(ids, use_checkpoint=False)[0], hidden),
                     f'Initial native training parity failed: {length}')
        report.update(fresh_initialization=True, initial_native_parity_128_512=True,
                      teacher_student_independent_storage=True, no_state_controller_installed=True)
        print('[setup] native SSD fresh initialization and 128/512 parity passed', flush=True)
        ordered = schedule(); limit = 1 if args.smoke else STEPS
        successful = attempts = overflows = 0
        while successful < limit:
            pair = pair_for(successful, ordered, examples, windows, prose_order)
            before = copy.deepcopy(scaler.state_dict()); started = time.monotonic()
            result = attempt(bank, teacher, pair, optimizer, scaler, successful)
            equal_tree(scaler.state_dict(), expected_scaler_after(before, result['overflow']), 'GradScaler transition')
            need(all((id(p), p.data_ptr(), p._version) == teacher_identity[n] and p.grad is None
                     and not p.requires_grad for n, p in teacher.named_parameters()), 'Teacher changed')
            restored(teacher_snapshot)
            attempts += 1; overflows += int(result['overflow']); successful += int(not result['overflow'])
            need(overflows <= 8, 'Overflow retry budget exceeded')
            report['history'].append(dict(attempt=attempts, successful_updates=successful,
                case_id=pair['id'], schedule_entry=pair['schedule_entry'],
                prose_window=pair['prose_window'], prose_start=pair['prose_start'],
                answer_targets=int(pair['answer_mask'].sum()), seconds=time.monotonic() - started, **result))
            report.update(successful_updates=successful, attempts=attempts, overflows=overflows)
            if not args.smoke and 0 < successful < STEPS and successful % 384 == 0 and not result['overflow']:
                progress, _ = save_final(args.out_dir / f'checkpoint_{successful:04d}.pt', bank,
                                          optimizer, scaler, binding, successful, attempts)
                report['checkpoints'].append(progress)
            if attempts % 16 == 0 or successful == limit or result['overflow']:
                put(args.out_dir / 'report.json', report)
                print(json.dumps(dict(updates=successful, attempts=attempts, mk_ce=result['mk_ce'],
                                      prose_ce=result['prose_ce'], overflow=result['overflow'])), flush=True)
        need(bank.assert_base_frozen(check_values=True)['actual_content_checked'] is True, 'Student changed')
        check_optimizer(optimizer.state_dict(), bank.state_dict(), successful, lr_factor(successful - 1))
        final = None
        if not args.smoke:
            checkpoint, final = save_final(args.out_dir / 'checkpoint_1536.pt', bank, optimizer,
                                           scaler, binding, successful, attempts)
            report['checkpoints'].append(checkpoint)
        adapter_path = args.out_dir / ('discarded_smoke_adapter_fp16.pt' if args.smoke else 'adapter_fp16.pt')
        exported = bank.export_fp16(adapter_path, binding=binding)
        payload = native.read_fp16(adapter_path, expected_binding=binding)
        need(len(payload['tensors']) == 224 and exported['parameters'] == 1154104
             and exported['payload_bytes'] == 2308208 and payload['gate_mode'] == 'soft'
             and any(bool(torch.count_nonzero(v)) for n, v in payload['tensors'].items() if n.endswith('.V_read')),
             'FP16 adapter export is wrong or identity')
        report['adapter'] = dict(exported, discarded=bool(args.smoke))
        if final is not None:
            need(all(torch.equal(final['masters'][n].half(), value)
                     for n, value in payload['tensors'].items()), 'Final masters/export casts differ')
            equal_tree(final['optimizer'], optimizer.state_dict(), 'Final optimizer roundtrip')
            equal_tree(final['scaler'], scaler.state_dict(), 'Final scaler roundtrip')
            report['final_checkpoint_export_check'] = dict(complete=True, passed=True,
                master_tensors=224, all_master_casts_equal_export=True, optimizer_exact=True,
                scaler_exact=True, checkpoint_roundtrip_exact=True, final_only=True)
        with torch.no_grad():
            trained = {length: bank.forward_hidden(ids, use_checkpoint=False)[0].detach().clone()
                       for length, (ids, _) in reference.items()}
        bank.close(); bank = None
        restored(student_snapshot)
        with torch.no_grad(), native.install_fp16(student, adapter_path, expected_binding=binding,
                                                  expected_base_hashes=expected):
            for length, (ids, _) in reference.items():
                need(torch.equal(student.backbone(ids), trained[length]),
                     f'Exported native deployment/training parity failed: {length}')
        report['native_restoration'] = restored(student_snapshot)
        with torch.no_grad():
            for length, (ids, hidden) in reference.items():
                need(torch.equal(student.backbone(ids), hidden), f'Adapter removal changed base output: {length}')
        report.update(export_native_parity_128_512=True, removal_native_parity_128_512=True,
            parity_128_512_initial_and_export=True,
            final_student_base_check=check_model(student, expected, manifest_sha),
            final_teacher_base_check=check_model(teacher, expected, manifest_sha),
            teacher_native_restoration=restored(teacher_snapshot),
            backend_final_check=check_replay_backend(policy), gpu_memory=runtime.gpu_memory_receipt())
        need(hashes(args.protocol) == report['code_sha256'], 'Executed local code changed during training')
        report.update(complete=True, finished_unix=time.time())
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc(), finished_unix=time.time())
        raise
    finally:
        if bank is not None:
            bank.close()
        put(args.out_dir / 'report.json', report)


if __name__ == '__main__':
    main()
