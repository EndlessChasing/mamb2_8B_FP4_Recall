#!/usr/bin/env python3
"""Build or verify the complete repaired FP4 G16/native-FP16 Resurface bundle.

Standard library only. No GPU work or external publication. Original repaired
weight/provenance files and frozen measured sources are copied byte for byte.
GitHub uses one runtime archive plus the 119 original safetensors assets.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall.fp4_repaired_checkpoint import (
    read_repaired_metadata, verify_repaired_checkpoint, sha_file, need)

FORMAT = 'MAMBA2_FP4_G16_REPAIRED_S16_RESURFACE_RELEASE_V1'
NAME = 'mamba2-8b-fp4g16-repaired-s16-resurface-v1'
TAG = 'v0.1.0-fp4g16-repaired-s16-resurface'
GITHUB_REPO = 'EndlessChasing/mamb2_8B_FP4_Recall'
HF_REPO = 'EndlessChasing/Mamb2_8B_FP4_Recall'
PROTOCOL_SHA = '9592a0dddcaa5c7d000422e5ac902b49df362781e4a4d9f668d99cb849cf8a7f'
PACKED_SHA = '2d2208b690c2ec0def90578d8590cc122bb1e4542b0d49ae4417a48f72b0b73c'
TOKENIZER_SHA = '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09'
TOKENIZER_FILENAME = 'mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model'
ADAPTER_SHA = 'dca3cda5a9ab8a665abd1a355c6e5b294acee023d2a1a5a8e20f4e84c9674580'
ARMS = ('baseline', 'resurface', 'adapter_removed')
AUDIT_FORMAT = 'FP4_REPAIRED_S16_RESURFACE_CPU_AUDIT_V1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def relative(name):
    need(isinstance(name, str) and bool(name) and '\\' not in name and ':' not in name,
         'Unsafe relative publication path')
    path = Path(name)
    need(not path.is_absolute() and all(x not in ('', '.', '..') for x in path.parts)
         and str(path) == name, 'Unsafe relative publication path')
    return path


def regular(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
         'Expected regular publication file: ' + str(path))
    return path


def copy(source, target):
    source = regular(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def local_imports(name):
    tree = ast.parse((ROOT / name).read_text())
    found = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = Path(name).parent
                for _ in range(node.level - 1):
                    base = base.parent
                module = base / ((node.module or '').replace('.', '/'))
                names.append(str(module).replace('/', '.'))
                names.extend(str(module / alias.name).replace('/', '.') for alias in node.names)
            else:
                names.append(node.module or '')
                names.extend((node.module + '.' if node.module else '') + alias.name for alias in node.names)
        for module in names:
            if not module or '*' in module:
                continue
            stem = module.replace('.', '/')
            for child in (stem + '.py', stem + '/__init__.py', 'scripts/' + stem + '.py'):
                if (ROOT / child).is_file():
                    found.add(child)
    return found


def source_closure(seed):
    result, queue = set(seed), list(seed)
    while queue:
        name = queue.pop()
        regular(ROOT / relative(name))
        if name.endswith('.py'):
            for child in local_imports(name) - result:
                result.add(child)
                queue.append(child)
    return result


def close(actual, expected, label):
    need(type(actual) in (float, int) and math.isfinite(actual)
         and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label)


def quality(directory):
    """Bind the original completed audits and independently check result arithmetic.

    The published bundle omits optimizer/checkpoint archives. These passed
    audits document the original research files; this function does not claim
    to reread those omitted archives or replay GPU logits.
    """
    directory = Path(directory)
    evidence = directory / 'evidence'
    comp, train, ta, gpu, mac = (read(evidence / name) for name in
        ('comparison.json', 'training_report.json', 'training_audit.json', 'audit_gpu.json', 'audit_mac.json'))
    need(comp['format'] == 'FP4_REPAIRED_S16_RESURFACE_EVAL_V1' and comp['complete'] is True
         and comp['mode'] == 'full' and comp['protocol_sha256'] == PROTOCOL_SHA
         and comp['state_mode'] == 'fp16_native_ssd' and comp['statequant_controller_installed'] is False
         and comp['candidate_selection'] == 'Final successful update1536 only; no quality selection',
         'Complete fixed-candidate native-FP16 evaluation required')
    need(train['format'] == 'FP4_REPAIRED_S16_RESURFACE_TRAIN_V1' and train['complete'] is True
         and train['mode'] == 'formal' and train['successful_updates'] == 1536
         and [r['successful_updates'] for r in train['checkpoints']] == [384, 768, 1152, 1536]
         and train['final_checkpoint_export_check'] == dict(complete=True, passed=True,
             master_tensors=224, all_master_casts_equal_export=True, optimizer_exact=True,
             scaler_exact=True, checkpoint_roundtrip_exact=True, final_only=True),
         'Complete final-only training/export required')
    tsha, csha = sha_file(evidence / 'training_report.json'), sha_file(evidence / 'comparison.json')
    need(comp['training_report_sha256'] == tsha and comp['training_binding'] == train['binding']
         and comp['final_adapter'] == train['adapter']
         and comp['training_audit_sha256'] == sha_file(evidence / 'training_audit.json')
         and ta['format'] == AUDIT_FORMAT and ta['scope'] == 'training_only'
         and ta['complete'] is True and ta['passed'] is True and ta['training_report_sha256'] == tsha,
         'Training/audit/candidate binding differs')
    for audit in (gpu, mac):
        need(audit['format'] == AUDIT_FORMAT and audit['scope'] == 'combined'
             and audit['complete'] is True and audit['passed'] is True
             and audit['input_report_sha256'] == csha and audit['training_report_sha256'] == tsha
             and audit['torch_imported'] is False and audit['cuda_initialized'] is False,
             'Bound completed GPU-host and independent Mac CPU audits required')
    need(train['binding']['packed_manifest_sha256'] == PACKED_SHA
         and train['binding']['state_mode'] == 'fp16_native_ssd'
         and train['binding']['repair_selected_successful_updates'] == 448,
         'Different repaired base or state mode')
    smoke = read(evidence / 'smoke_report.json')
    need(smoke['complete'] is True and smoke['mode'] == 'smoke'
         and smoke['successful_updates'] == 1 and smoke['adapter']['discarded'] is True
         and smoke['binding'] == train['binding']
         and train['smoke_report_sha256'] == sha_file(evidence / 'smoke_report.json'),
         'Completed independent discarded smoke differs')
    for values in (comp['code_sha256'], train['code_sha256']):
        for name, digest in values.items():
            need(sha_file(regular(directory / relative(name))) == digest, 'Frozen source changed: ' + name)
    adapter = train['adapter']
    need(adapter['file'] == 'adapter_fp16.pt' and adapter['discarded'] is False
         and adapter['sha256'] == ADAPTER_SHA == sha_file(directory / 'adapter_fp16.pt')
         and adapter['bytes'] == (directory / 'adapter_fp16.pt').stat().st_size == 2381119
         and adapter['parameters'] == 1154104 and adapter['payload_bytes'] == 2308208
         and len(adapter['tensor_sha256']) == 224 and adapter['roundtrip_bitwise_equal'] is True
         and adapter['gate_mode'] == 'soft', 'Final adapter identity/capacity differs')
    rows = {}
    need(set(comp['reports']) == set(ARMS), 'Complete three-arm reports required')
    for arm in ARMS:
        entry = comp['reports'][arm]
        need(entry['file'] == arm + '.json', 'Unsafe or incorrect arm filename')
        path = evidence / entry['file']
        need(sha_file(path) == entry['sha256'], 'Bound arm bytes changed')
        row = read(path)
        need(row['complete'] is True and row['arm'] == arm
             and row['state_mode'] == 'fp16_native_ssd' and row['statequant_controller_installed'] is False
             and row['adapter_loaded'] is (arm == 'resurface'), 'Arm execution/adapter differs')
        ppl = row['ppl']
        need(len(ppl['windows']) == 147 and ppl['target_tokens'] == 300963
             and ppl['execution'] == 'prefill' and ppl['cache_bytes'] is None,
             'Full native-prefill test coverage required')
        counts = 0
        for index, window in enumerate(ppl['windows']):
            count = 2048 if index < 146 else 1955
            need(window['start'] == 2048 * index and window['target_tokens'] == count
                 and window['input_tokens'] == count, 'Window target coverage differs')
            close(window['ppl'], math.exp(window['nll'] / count), 'Window PPL arithmetic differs')
            counts += count
        nll = math.fsum(w['nll'] for w in ppl['windows'])
        need(counts == 300963, 'Target coverage differs')
        close(ppl['nll'], nll, 'Pooled NLL differs')
        close(ppl['ppl'], math.exp(nll / counts), 'Pooled PPL differs')
        close(entry['ppl'], ppl['ppl'], 'Arm PPL binding differs')
        need(len(row['mk']['rows']) == 768 and len({r['id'] for r in row['mk']['rows']}) == 768,
             'Full unique MK population required')
        for condition in ('normal', 'target_removed'):
            subset = [r for r in row['mk']['rows'] if r['condition'] == condition]
            correct = sum(r['correct'] for r in subset)
            need(len(subset) == 384 and row['mk']['summary'][condition] ==
                 dict(correct=correct, count=384, accuracy=correct / 384), 'MK summary differs')
        rows[arm] = row
    left, right, removed = (rows[a] for a in ARMS)
    need(left['ppl']['windows'] == removed['ppl']['windows']
         and all(left['ppl'][k] == removed['ppl'][k] for k in ('ppl', 'nll', 'target_tokens'))
         and left['cache_probe'] == removed['cache_probe'], 'Adapter-removal PPL/cache restoration differs')
    for old, restored in zip(left['mk']['rows'], removed['mk']['rows']):
        need(all(old[k] == restored[k] for k in ('id', 'prompt', 'prompt_token_sha256_int64le',
             'generated_ids', 'output', 'prediction', 'correct', 'cache_bytes', 'actual_cache')),
             'Adapter-removal generation/cache differs')
    pair = comp['comparison']
    need(pair['control_ppl'] == left['ppl']['ppl'] and pair['candidate_ppl'] == right['ppl']['ppl'],
         'Paired PPL differs')
    delta = right['mk']['summary']['normal']['correct'] - left['mk']['summary']['normal']['correct']
    need(pair['normal_mk_correct_delta'] == delta, 'Paired MK gain differs')
    checks = dict(adapted_ppl_no_worse=pair['candidate_ppl'] <= pair['control_ppl'],
        normal_mk_increases=delta > 0, paired95_lower_positive=pair['normal_mk_paired_bootstrap_95ci'][0] > 0,
        all_baseline_nll_and_sequences_restored_exact=comp['restoration']['complete'] is True)
    need(comp['quality_checks'] == checks and all(checks.values()) and comp['quality_gate_pass'] is True,
         'All four frozen publication quality conditions must pass')
    for audit in (gpu, mac):
        need(audit['quality_checks'] == checks and audit['quality_gate_pass'] is True
             and audit['comparison'] == pair
             and audit['arm_sha256'] == {a: comp['reports'][a]['sha256'] for a in ARMS},
             'Audited quality results differ')
    return comp, train


def verify_bundle(directory):
    directory = Path(directory).resolve()
    need(directory.is_dir() and not any(p.is_symlink() for p in directory.rglob('*')),
         'Regular complete bundle required')
    manifest = read(directory / 'manifest.json')
    need(manifest['format'] == FORMAT and manifest['complete'] is True
         and manifest['github_repo'] == GITHUB_REPO and manifest['hugging_face_repo'] == HF_REPO
         and manifest['release_tag'] == TAG and manifest['state_mode'] == 'fp16_native_ssd'
         and manifest['statequant_controller'] is False and manifest['packed_resident_kernel'] is False,
         'Different release format/repository/state mode')
    files = manifest['files']
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    need(actual == set(files) | {'manifest.json', 'SHA256SUMS'}, 'Unexpected/missing bundle files')
    hashes = {}
    for name, expected in files.items():
        path = regular(directory / relative(name))
        hashes[name] = sha_file(path)
        need(expected == dict(bytes=path.stat().st_size, sha256=hashes[name]), 'Bundle file differs: ' + name)
    hashes['manifest.json'] = sha_file(directory / 'manifest.json')
    need((directory / 'SHA256SUMS').read_text() == ''.join(
         f'{hashes[name]}  {name}\n' for name in sorted(hashes)), 'Bundle checksum list differs')
    weights, _ = verify_repaired_checkpoint(directory / 'weights', code_root=directory)
    need(sha_file(directory / 'weights/weight_manifest.json') == PACKED_SHA, 'Selected base changed')
    comp, train = quality(directory)
    need(manifest['adapter_binding'] == train['binding'] and manifest['backend_policy'] == comp['backend_policy']
         and manifest['base_provenance'] == comp['base_provenance']
         and manifest['candidate_ppl'] == comp['comparison']['candidate_ppl']
         and manifest['baseline_ppl'] == comp['comparison']['control_ppl']
         and manifest['packed_weight_payload_bytes'] == weights['logical_weight_payload_bytes']
         and manifest['actual_weight_shard_bytes'] == weights['shard_file_bytes']
         and manifest['decoded_resident_weight_bytes'] == weights['decoded_resident_weight_bytes']
         and manifest['native_mk_cache_bytes'] == 122028032
         and manifest['adapter_fp16_payload_bytes'] == train['adapter']['payload_bytes'],
         'Release quality/accounting/configuration binding differs')
    return manifest


def prepare(args):
    """Resolve the allowlist and fixed input identities without reading large shards."""
    comp, train = read(args.comparison), read(args.training_report)
    weights, ledger = read_repaired_metadata(args.packed_checkpoint)
    need(sha_file(args.packed_checkpoint / 'weight_manifest.json') == PACKED_SHA,
         'Exact selected repaired base required')
    core = read(args.packed_checkpoint / 'training_report.json')
    seed = set(comp['code_sha256']) | set(train['code_sha256']) | set(weights['code_sha256'])
    seed |= set(ledger['code_sha256']) | set(core['binding']['code_sha256'])
    seed.update(('mamba2_recall/__init__.py', 'scripts/release_fp4_repaired_s16_resurface_v1.py',
        'scripts/infer_fp4_repaired_s16_resurface_v1.py', 'scripts/audit_fp4_repaired_s16_resurface_v1.py',
        'scripts/verify_github_fp4_repaired_s16_resurface_v1.py',
        'scripts/verify_hf_fp4_repaired_s16_release_v1.py',
        'scripts/audit_fp4_base_repair_v1.py', 'docs/RESURFACE_MORE_BACKEND_REPLAY.md',
        'docs/FP4_BASE_REPAIR_V1_PROTOCOL.md',
        'docs/prose_train_manifest.json', 'LICENSE', 'reference/w4/WEIGHTS_LICENSE.txt',
        'reference/statequant/LICENSE', 'reference/statequant/selective_state_update_pairnib.py'))
    sources = source_closure(seed)
    payload = {'evidence/comparison.json': args.comparison,
        'evidence/training_report.json': args.training_report, 'evidence/training_audit.json': args.training_audit,
        'evidence/audit_gpu.json': args.audit, 'evidence/audit_mac.json': args.mac_audit,
        'evidence/audit.json': args.mac_audit,
        'evidence/smoke_report.json': args.smoke_report, 'evidence/smoke_audit.json': args.smoke_audit,
        'evidence/repair_selection.json': args.repair_selection,
        'evidence/repair_test_comparison.json': args.repair_test_comparison,
        'evidence/repair_training_core.json': args.packed_checkpoint / 'training_report.json',
        'adapter_fp16.pt': args.adapter, 'README.md': args.readme, 'docs/RESULTS.md': args.results,
        'docs/MINIMAL_REPRODUCTION.md': args.reproduction, 'LICENSES.md': args.licenses,
        'THIRD_PARTY_NOTICES.md': args.notices, 'docs/THIRD_PARTY_NOTICES.md': args.notices,
        'licenses/GPL-3.0.txt': ROOT / 'LICENSE',
        'licenses/Apache-2.0.txt': ROOT / 'reference/w4/WEIGHTS_LICENSE.txt',
        'licenses/StateQuant-Apache-2.0.txt': ROOT / 'reference/statequant/LICENSE',
        'tokenizer/' + TOKENIZER_FILENAME: args.tokenizer, 'source_config.json': args.source_config}
    for arm in ARMS:
        need(comp['reports'][arm]['file'] == arm + '.json', 'Incorrect arm evidence filename')
        payload['evidence/' + arm + '.json'] = args.comparison.parent / (arm + '.json')
    for source in payload.values():
        regular(source)
    regular(args.canonical_results)
    need(sha_file(ROOT / 'docs/FP4_BASE_REPAIR_V1_PROTOCOL.md') == core['binding']['protocol_sha256'],
         'Frozen base-repair protocol differs from selected core')
    runtime_tree = ast.parse((ROOT / 'mamba2_recall/runtime.py').read_text())
    configs = [ast.literal_eval(node.value) for node in runtime_tree.body
               if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
               and target.id == 'MODEL_CONFIG' for target in node.targets)]
    need(len(configs) == 1 and read(args.source_config) == configs[0],
         'source_config.json must equal the frozen native runtime MODEL_CONFIG')
    need(sha_file(args.tokenizer) == TOKENIZER_SHA and sha_file(args.adapter) == ADAPTER_SHA,
         'Tokenizer or final adapter changed')
    for source in (args.readme, args.results, args.notices, args.reproduction, args.licenses):
        text = source.read_text()
        need('{{' not in text and 'DRAFT' not in text and 'PENDING' not in text,
             'Presentation file contains unresolved placeholders')
    for values in (comp['code_sha256'], train['code_sha256'], weights['code_sha256'],
                   ledger['code_sha256'], core['binding']['code_sha256']):
        for name, digest in values.items():
            need(name in sources and sha_file(ROOT / name) == digest, 'Frozen source closure differs: ' + name)
    need('reference/w4/WEIGHTS_NOTICE.md' not in sources, 'Stale unrelated INT4 adapter notice in closure')
    return comp, train, weights, sources, payload


def build(args):
    output = Path(args.build)
    need(not output.exists() and not output.is_symlink(), 'Fresh build directory required')
    comp, train, weights, sources, payload = prepare(args)
    # Before any large file copy, require both independently audited decisions.
    for path in (args.audit, args.mac_audit):
        audit = read(path)
        need(audit['complete'] is True and audit['passed'] is True and audit['quality_gate_pass'] is True
             and audit['input_report_sha256'] == sha_file(args.comparison)
             and audit['training_report_sha256'] == sha_file(args.training_report),
             'Bound completed passing quality audits required before build')
    need(comp['complete'] is True and comp['quality_gate_pass'] is True
         and all(comp['quality_checks'].values()), 'Full frozen PPL/MK gate must pass')
    verify_repaired_checkpoint(args.packed_checkpoint)
    output.mkdir(parents=True)
    bundle = output / NAME
    bundle.mkdir()
    for name in sorted(sources):
        copy(ROOT / name, bundle / name)
    for name, source in payload.items():
        copy(source, bundle / name)
    # Copy all 124 base files verbatim; do not rebuild its manifest or checksums.
    shutil.copytree(args.packed_checkpoint, bundle / 'weights', copy_function=shutil.copyfile)
    comp, train = quality(bundle)
    files = {str(p.relative_to(bundle)): dict(bytes=p.stat().st_size, sha256=sha_file(p))
             for p in sorted(bundle.rglob('*')) if p.is_file()}
    manifest = dict(format=FORMAT, complete=True, github_repo=GITHUB_REPO, hugging_face_repo=HF_REPO,
        release_tag=TAG, candidate_ppl=comp['comparison']['candidate_ppl'],
        baseline_ppl=comp['comparison']['control_ppl'], adapter_binding=train['binding'],
        backend_policy=comp['backend_policy'], base_provenance=comp['base_provenance'],
        repaired_weight_manifest_sha256=PACKED_SHA, tokenizer_sha256=TOKENIZER_SHA,
        state_mode='fp16_native_ssd', statequant_controller=False,
        packed_weight_payload_bytes=weights['logical_weight_payload_bytes'],
        actual_weight_shard_bytes=weights['shard_file_bytes'],
        decoded_resident_weight_bytes=weights['decoded_resident_weight_bytes'], packed_resident_kernel=False,
        native_mk_cache_bytes=122028032, persistent_ppl_cache_bytes=None,
        additional_adapter_recurrent_bytes=0, adapter_fp16_payload_bytes=2308208,
        adapter_file_bytes=train['adapter']['bytes'], source_closure_sha256={n: sha_file(bundle / n) for n in sorted(sources)},
        canonical_results_sha256=sha_file(args.canonical_results),
        quality_checks=comp['quality_checks'], measured_quality_scope=comp['quality_scope'], files=files,
        scope='Complete real packed repaired weights, native FP16 compute/cache runtime and final Resurface adapter; no StateQuant or packed-resident FP4 kernel claim')
    write(bundle / 'manifest.json', manifest)
    names = sorted(set(files) | {'manifest.json'})
    (bundle / 'SHA256SUMS').write_text(''.join(f'{sha_file(bundle / n)}  {n}\n' for n in names))
    verify_bundle(bundle)
    archive = output / (NAME + '-runtime.tar.gz')
    shards = {'weights/' + n for n in weights['files']}
    with archive.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for name in sorted((set(names) | {'SHA256SUMS'}) - shards):
                path = bundle / name
                info = tar.gettarinfo(str(path), arcname=NAME + '/' + name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
    assets = [dict(path=archive.name, bytes=archive.stat().st_size, sha256=sha_file(archive))]
    assets += [dict(path=NAME + '/weights/' + n, **weights['files'][n]) for n in sorted(weights['files'])]
    need(len(assets) == 120 and all(a['bytes'] < 2 * 1024**3 for a in assets), 'GitHub asset capacity differs')
    receipt = dict(format=FORMAT, complete=True, verified=True, publication_performed=False,
        github_repo=GITHUB_REPO, hugging_face_repo=HF_REPO, release_tag=TAG,
        candidate_ppl=manifest['candidate_ppl'], baseline_ppl=manifest['baseline_ppl'],
        bundle_manifest_sha256=sha_file(bundle / 'manifest.json'), github_assets=assets,
        file_count=len(files) + 2, total_file_bytes=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file()),
        source_closure_files=len(sources), source_closure_sha256=manifest['source_closure_sha256'],
        state_config_included=False, independent_full_weight_files_verified=True, cuda_initialized=False,
        torch_imported='torch' in sys.modules)
    need(receipt['torch_imported'] is False, 'Builder must remain standard-library CPU only')
    write(output / 'build_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build', type=Path)
    mode.add_argument('--verify', type=Path)
    mode.add_argument('--preflight', type=Path, help='Write a small allowlist/input receipt; no weight shard read')
    for key in ('comparison', 'audit', 'mac-audit', 'training-report', 'training-audit', 'smoke-report',
                'smoke-audit', 'adapter', 'packed-checkpoint', 'repair-selection', 'repair-test-comparison',
                'tokenizer', 'source-config', 'readme', 'results', 'canonical-results', 'notices',
                'reproduction', 'licenses'):
        parser.add_argument('--' + key, type=Path)
    args = parser.parse_args()
    if args.verify:
        value = verify_bundle(args.verify)
        result = dict(verified=True, candidate_ppl=value['candidate_ppl'], baseline_ppl=value['baseline_ppl'],
                      independent_full_file_hashes=True, cuda_initialized=False, torch_imported='torch' in sys.modules)
    else:
        required = [k for k, v in vars(args).items() if k not in ('build', 'verify', 'preflight') and v is None]
        if required:
            parser.error('Required: ' + ', '.join('--' + k.replace('_', '-') for k in required))
        if args.preflight:
            need(not args.preflight.exists(), 'Fresh preflight receipt required')
            _, _, weights, sources, payload = prepare(args)
            result = dict(format=FORMAT, complete=True, metadata_preflight=True,
                github_repo=GITHUB_REPO, hugging_face_repo=HF_REPO, release_tag=TAG,
                source_files={n: dict(bytes=(ROOT/n).stat().st_size, sha256=sha_file(ROOT/n)) for n in sorted(sources)},
                payload_files={n: dict(bytes=p.stat().st_size, sha256=sha_file(p)) for n,p in sorted(payload.items())},
                weights=weights['files'], independent_weight_shards_read=False,
                torch_imported='torch' in sys.modules, cuda_initialized=False)
            args.preflight.parent.mkdir(parents=True, exist_ok=True)
            write(args.preflight, result)
        else:
            result = build(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
