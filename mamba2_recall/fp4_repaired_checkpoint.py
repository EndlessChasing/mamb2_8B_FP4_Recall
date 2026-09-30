"""Same-size FP4 G16 checkpoint with repaired original retained FP16 tensors.

All 118 original FP4 shard files stay byte-identical. Only the existing
retained-FP16 shard changes; its header, tensor geometry and file size stay
identical. Metadata and the immutable training core report are additional
provenance files, accounted separately from the 119 weight shard files.
Verification uses only the standard library. Reload expands weights to FP16.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct

from . import fp4_checkpoint as parent

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'MAMBA2_FP4_G16_REPAIRED_CHECKPOINT_V1'
PARENT_MANIFEST_SHA = '4e7fb60f5d03c61e63324a6c25847014a26f9aa0c8d5a8c922d0f367e514fe07'
RETAINED_FILE = 'weights-retained-fp16.safetensors'
RETAINED_COUNT = 393
RETAINED_BYTES = 7_161_856
SHARD_COUNT = 119
SHARD_BYTES = 4_638_539_864

need = parent.need
sha_file = parent.sha_file


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _source_hashes():
    return dict(parent.source_code_hashes(), **{
        'mamba2_recall/fp4_repaired_checkpoint.py': sha_file(__file__)})


def _check_sources(hashes, code_root):
    for name, digest in hashes.items():
        parts = Path(name).parts
        need(isinstance(name, str) and parts and not Path(name).is_absolute()
             and all(p not in ('', '.', '..') for p in parts)
             and '\\' not in name and ':' not in name and _digest(digest),
             'Unsafe source binding')
        path = Path(code_root).joinpath(*parts)
        need(path.is_file() and not any(p.is_symlink() for p in [path, *path.parents]),
             'Expected regular bound source: ' + name)
        need(sha_file(path) == digest, 'Bound source changed: ' + name)


def _parent_metadata(directory, code_root=ROOT):
    """Verify the embedded original manifest and its strict pinned ledger."""
    directory = Path(directory)
    need(sha_file(directory / 'parent_weight_manifest.json') == PARENT_MANIFEST_SHA,
         'Different original FP4 packed manifest')
    original = parent.read_json(directory / 'parent_weight_manifest.json')
    ledger = parent.read_ledger(directory / 'conversion_receipt.json')
    plan = parent.make_plan(ledger, original['shard_limit_bytes'])
    need(original['format'] == parent.FORMAT and original['complete'] is True
         and original['layout'] == plan and original['conversion_receipt_sha256'] == parent.LEDGER_SHA
         and original['logical_weight_payload_bytes'] == parent.PAYLOAD_BYTES
         and original['decoded_resident_weight_bytes'] == parent.WEIGHT_BYTES
         and original['shard_file_bytes'] == SHARD_BYTES and len(original['files']) == SHARD_COUNT,
         'Original packed geometry differs')
    _check_sources(original['code_sha256'], code_root)
    _check_sources(ledger['code_sha256'], code_root)
    return original, ledger, plan


def _finite_fp16_region(path, start, size):
    """Check IEEE binary16 exponent bits without importing torch or NumPy."""
    need(size % 2 == 0, 'Unaligned FP16 payload')
    with Path(path).open('rb') as stream:
        stream.seek(start)
        left = size
        while left:
            raw = stream.read(min(left, 8 * 1024 * 1024))
            need(raw and len(raw) % 2 == 0, 'Truncated FP16 payload')
            need(all(bits & 0x7c00 != 0x7c00 for (bits,) in struct.iter_unpack('<H', raw)),
                 'Nonfinite repaired FP16 tensor')
            left -= len(raw)


def _check_core(core, binding, expected, retained_names, code_root):
    need(core['format'] == 'FP4_BASE_REPAIR_CHECKPOINT_CORE_V1' and core['complete'] is True
         and core['mode'] in ('formal', 'smoke') and core['no_resurface_installed'] is True,
         'Different/incomplete retained-only training core')
    need((core['mode'] == 'formal' and core['successful_updates'] in (224,448,672,896))
         or (core['mode'] == 'smoke' and core['successful_updates'] == 1 and core['discarded'] is True),
         'Different predeclared training candidate')
    b = core['binding']
    need(b['protocol_sha256'] == binding['protocol_sha256']
         and b['parent_manifest_sha256'] == PARENT_MANIFEST_SHA
         and b['parent_conversion_receipt_sha256'] == parent.LEDGER_SHA
         and b['source_checkpoint_sha256'] == parent.SOURCE_SHA,
         'Training core used a different base/protocol')
    need(core['retained_tensor_sha256'] == {n: expected[n] for n in sorted(retained_names)},
         'Repaired tensors differ from immutable training export393 ledger')
    _check_sources(b['code_sha256'], code_root)


def read_repaired_metadata(directory, code_root=ROOT):
    """Verify small metadata/core/source bindings; do not read weight shards.

    This supports a report audit on copied small receipts. It does not
    independently verify the actual shard files, their headers or tensor data.
    Use verify_repaired_checkpoint for full byte verification.
    """
    directory = Path(directory)
    need(directory.is_dir() and not directory.is_symlink(), 'Regular checkpoint directory required')
    manifest = parent.read_json(directory / 'weight_manifest.json')
    original, ledger, plan = _parent_metadata(directory, code_root)
    need(manifest['format'] == FORMAT and manifest['complete'] is True
         and manifest['parent_manifest_sha256'] == PARENT_MANIFEST_SHA
         and manifest['conversion_receipt_sha256'] == parent.LEDGER_SHA
         and manifest['source_checkpoint_sha256'] == parent.SOURCE_SHA
         and manifest['format_name'] == 'fp4_g16_e4m3' and manifest['layout'] == plan
         and manifest['logical_weight_payload_bytes'] == parent.PAYLOAD_BYTES
         and manifest['decoded_resident_weight_bytes'] == parent.WEIGHT_BYTES
         and manifest['shard_file_bytes'] == SHARD_BYTES
         and manifest['retained_fp16_payload_bytes'] == RETAINED_BYTES
         and manifest['retained_fp16_tensor_count'] == RETAINED_COUNT
         and manifest['fp4_tensor_count'] == 114 and manifest['tensor_count'] == 507
         and manifest['parameter_count'] == 8_236_999_680
         and manifest['packed_resident_kernel'] is False and manifest['custom_runtime'] is True
         and manifest['base_only'] is True and manifest['resurface_adapter'] is False,
         'Repair manifest geometry/identity differs')
    need(set(manifest['code_sha256']) == set(_source_hashes()), 'Incomplete repair runtime closure')
    _check_sources(manifest['code_sha256'], code_root)
    binding = manifest['training_binding']
    need(_digest(binding['protocol_sha256']) and _digest(binding['training_report_sha256'])
         and 'training_report_path' not in binding,
         'Invalid/nonportable training core binding')
    need(sha_file(directory / 'training_report.json') == binding['training_report_sha256'],
         'Immutable training core report changed')
    core = parent.read_json(directory / 'training_report.json')
    expected = manifest['decoded_tensor_sha256']
    need(set(expected) == set(ledger['tensors']) and all(_digest(x) for x in expected.values()),
         'Incomplete decoded507 hash ledger')
    specs = parent._file_specs(plan)
    need(set(manifest['files']) == set(specs) and len(specs) == SHARD_COUNT,
         'Changed weight file inventory')
    retained = [name for name, item in plan.items() if item['kind'] == 'fp16']
    need(len(retained) == RETAINED_COUNT
         and sum(math.prod(plan[n]['shape']) * 2 for n in retained) == RETAINED_BYTES,
         'Retained tensor class/geometry changed')
    _check_core(core, binding, expected, retained, code_root)
    for filename, spec in specs.items():
        row = manifest['files'][filename]
        raw, _, payload = parent._header(spec)
        need(set(row) == {'bytes', 'sha256'} and _digest(row['sha256'])
             and row['bytes'] == original['files'][filename]['bytes'] == 8 + len(raw) + payload,
             'Declared weight file geometry changed: ' + filename)
        if filename != RETAINED_FILE:
            need(row == original['files'][filename], 'Declared FP4 shard identity changed')
    need(all(expected[n] == item['decoded_sha256'] for n, item in ledger['tensors'].items()
             if item['kind'] != 'fp16'), 'Declared FP4 decoded hash changed')
    need(sum(x['bytes'] for x in manifest['files'].values()) == SHARD_BYTES,
         'Declared total weight file bytes changed')
    return manifest, ledger


def verify_repaired_checkpoint(directory, code_root=ROOT):
    """Hash every file and encoded region, enforcing unchanged FP4 storage."""
    directory = Path(directory)
    manifest, ledger = read_repaired_metadata(directory, code_root)
    original = parent.read_json(directory / 'parent_weight_manifest.json')
    plan, expected = manifest['layout'], manifest['decoded_tensor_sha256']
    specs = parent._file_specs(plan)
    headers, hashes = {}, {}
    for filename, spec in sorted(specs.items()):
        path = directory / filename
        headers[filename] = parent.inspect_shard(path, spec)
        hashes[filename] = sha_file(path)
        actual = {'bytes': path.stat().st_size, 'sha256': hashes[filename]}
        need(manifest['files'][filename] == actual, 'Shard hash/size differs: ' + filename)
        need(actual['bytes'] == original['files'][filename]['bytes'], 'Weight file grew: ' + filename)
        if filename != RETAINED_FILE:
            need(actual == original['files'][filename], 'FP4 codes/scales/global file changed: ' + filename)
    for name, item in plan.items():
        old = ledger['tensors'][name]
        if item['kind'] == 'fp16':
            base, header = headers[RETAINED_FILE]
            begin, end = header[name]['data_offsets']
            digest = hashlib.sha256()
            parent._hash_region(directory / RETAINED_FILE, base + begin, end - begin, digest)
            _finite_fp16_region(directory / RETAINED_FILE, base + begin, end - begin)
            need(digest.hexdigest() == expected[name], 'Repaired F16 tensor differs: ' + name)
        else:
            need(expected[name] == old['decoded_sha256'], 'FP4 decoded tensor identity changed: ' + name)
            for key, hash_key in (('codes', 'codes_sha256'), ('block_scale_bytes', 'scales_sha256')):
                digest = hashlib.sha256()
                for part in item['parts']:
                    base, header = headers[part['file']]
                    begin, end = header[key]['data_offsets']
                    parent._hash_region(directory / part['file'], base + begin, end - begin, digest)
                need(digest.hexdigest() == old[hash_key], 'FP4 encoded region changed: ' + name + '/' + key)
            base, header = headers[item['global_scale_file']]
            begin, end = header['global_scale']['data_offsets']
            digest = hashlib.sha256()
            parent._hash_region(directory / item['global_scale_file'], base + begin, end - begin, digest)
            need(digest.hexdigest() == old['global_scale_sha256'], 'FP4 global scale changed: ' + name)
    need(sum((directory / name).stat().st_size for name in specs) == SHARD_BYTES,
         'Total weight shard bytes changed')
    inventory = set(specs) | {'conversion_receipt.json', 'parent_weight_manifest.json',
                              'training_report.json', 'weight_manifest.json', 'SHA256SUMS'}
    need(not any(p.is_symlink() for p in directory.rglob('*')), 'Checkpoint contains symlink')
    need({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} == inventory,
         'Unexpected files in repaired checkpoint')
    for name in inventory - set(specs) - {'SHA256SUMS'}:
        hashes[name] = sha_file(directory / name)
    need((directory / 'SHA256SUMS').read_text() == ''.join(
        f'{hashes[name]}  {name}\n' for name in sorted(inventory - {'SHA256SUMS'})),
        'Repaired checksum list differs')
    return manifest, ledger


def export_repaired_checkpoint(parent_dir, retained_dict, out_dir, training_binding):
    """Export the original 119-shard geometry, replacing only retained FP16."""
    parent_dir, out_dir = Path(parent_dir), Path(out_dir)
    need(sha_file(parent_dir / 'weight_manifest.json') == PARENT_MANIFEST_SHA,
         'Pinned original parent checkpoint required')
    original, ledger = parent.verify_checkpoint(parent_dir)
    binding = copy.deepcopy(training_binding)
    report_path = Path(binding.pop('training_report_path'))
    need(_digest(binding['protocol_sha256']) and _digest(binding['training_report_sha256'])
         and report_path.is_file() and not report_path.is_symlink()
         and sha_file(report_path) == binding['training_report_sha256'],
         'Immutable completed training core report required')
    need(parent.read_json(report_path).get('complete') is True, 'Incomplete training core')
    json.dumps(binding, sort_keys=True, allow_nan=False)
    plan = original['layout']
    names = {name for name, item in plan.items() if item['kind'] == 'fp16'}
    need(set(retained_dict) == names and len(names) == RETAINED_COUNT,
         'Repair must replace exactly the original393 retained tensors')
    import torch
    from . import fp4
    values = {}
    for name in sorted(names):
        value = retained_dict[name]
        need(isinstance(value, torch.Tensor) and value.dtype == torch.float16 and not value.is_meta
             and list(value.shape) == plan[name]['shape'] and bool(torch.isfinite(value).all()),
             'Wrong/nonfinite repaired FP16 tensor: ' + name)
        values[name] = value.detach().cpu().contiguous()
    need(sum(x.numel() * x.element_size() for x in values.values()) == RETAINED_BYTES,
         'Retained payload bytes changed')
    decoded = {name: row['decoded_sha256'] for name, row in ledger['tensors'].items()}
    decoded.update({name: hashlib.sha256(fp4._bytes(value)).hexdigest() for name,value in values.items()})
    _check_core(parent.read_json(report_path), binding, decoded, names, ROOT)
    need(not out_dir.exists() and not out_dir.is_symlink(), 'Fresh repair checkpoint directory required')
    out_dir.mkdir(parents=True)
    for name in sorted(original['files']):
        if name != RETAINED_FILE:
            try:
                os.link(parent_dir / name, out_dir / name)
            except OSError:
                shutil.copyfile(parent_dir / name, out_dir / name)
    specs = parent._file_specs(plan)
    base, header = parent._create_shard(out_dir / RETAINED_FILE, specs[RETAINED_FILE])
    with (out_dir / RETAINED_FILE).open('r+b') as stream:
        for name, value in sorted(values.items()):
            raw = fp4._bytes(value)
            begin, end = header[name]['data_offsets']
            need(len(raw) == end - begin, 'Repaired tensor serialized size changed')
            stream.seek(base + begin); stream.write(raw)
            decoded[name] = hashlib.sha256(raw).hexdigest()
    for source, destination in ((parent_dir / 'weight_manifest.json', 'parent_weight_manifest.json'),
                                (parent_dir / 'conversion_receipt.json', 'conversion_receipt.json'),
                                (report_path, 'training_report.json')):
        shutil.copyfile(source, out_dir / destination)
    need(sha_file(out_dir / 'training_report.json') == binding['training_report_sha256'],
         'Training core changed during export')
    files = {name: {'sha256': sha_file(out_dir / name), 'bytes': (out_dir / name).stat().st_size}
             for name in sorted(specs)}
    manifest = dict(original, format=FORMAT, parent_manifest_sha256=PARENT_MANIFEST_SHA,
                    files=files, training_binding=binding, decoded_tensor_sha256=decoded,
                    code_sha256=_source_hashes(), tensor_count=507, fp4_tensor_count=114,
                    parameter_count=8_236_999_680, retained_fp16_tensor_count=RETAINED_COUNT,
                    retained_fp16_payload_bytes=RETAINED_BYTES, base_only=True, resurface_adapter=False,
                    scope='Same119 weight shard names/headers/bytes; only original393 retained FP16 payloads replaced. Metadata and training core report are separate.')
    parent.write_json(out_dir / 'weight_manifest.json', manifest)
    filenames = sorted([*specs, 'conversion_receipt.json', 'parent_weight_manifest.json',
                        'training_report.json', 'weight_manifest.json'])
    (out_dir / 'SHA256SUMS').write_text(''.join(f'{sha_file(out_dir / name)}  {name}\n' for name in filenames))
    verify_repaired_checkpoint(out_dir)
    return manifest


def load_repaired_model(directory, *, device='cuda', chunk_rows=512, progress=None):
    """Strict source-free reload, using the unchanged FP4 decode helpers."""
    directory = Path(directory)
    manifest, ledger = verify_repaired_checkpoint(directory)
    need(chunk_rows == 512, 'Recorded decode chunk geometry required')
    import torch
    from torch import nn
    from . import fp4, runtime
    model = runtime.make_model(device='meta', dtype=torch.float16)
    plan = manifest['layout']
    need({n: list(x.shape) for n, x in model.state_dict().items()} ==
         {n: item['shape'] for n, item in plan.items()}, 'Native model geometry differs')
    headers = {name: parent.inspect_shard(directory / name, spec)
               for name, spec in parent._file_specs(plan).items()}
    for index, (name, item) in enumerate(sorted(plan.items()), 1):
        destination = torch.empty(item['shape'], device=device, dtype=torch.float16)
        digest = hashlib.sha256()
        if item['kind'] == 'fp16':
            base, header = headers[item['file']]
            begin, end = header[name]['data_offsets']
            raw = parent._read_region(directory / item['file'], base + begin, end - begin)
            value = torch.frombuffer(raw, dtype=torch.float16).reshape(item['shape'])
            destination.copy_(value); digest.update(raw)
            del raw, value
        else:
            columns = item['shape'][1]
            step = fp4._rows(columns, chunk_rows)
            base, header = headers[item['global_scale_file']]
            raw = parent._read_region(directory / item['global_scale_file'], base + header['global_scale']['data_offsets'][0], 4)
            global_scale = torch.frombuffer(raw, dtype=torch.float32).clone().to(device=device).reshape(())
            for part in item['parts']:
                base, header = headers[part['file']]
                for start in range(part['row_start'], part['row_stop'], step):
                    stop = min(start + step, part['row_stop'])
                    rows, local = stop - start, start - part['row_start']
                    codes_raw = parent._read_region(directory / part['file'], base + header['codes']['data_offsets'][0] + local * (columns // 2), rows * (columns // 2))
                    scales_raw = parent._read_region(directory / part['file'], base + header['block_scale_bytes']['data_offsets'][0] + local * (columns // 16), rows * (columns // 16))
                    codes = torch.frombuffer(codes_raw, dtype=torch.uint8).clone().reshape(rows, columns // 2).to(device=device)
                    scales = torch.frombuffer(scales_raw, dtype=torch.uint8).clone().view(torch.float8_e4m3fn).reshape(rows, columns // 16).to(device=device)
                    decoded = fp4.decode_packet({'format': 'fp4_g16_e4m3', 'shape': [rows, columns],
                        'group_size': 16, 'packed': codes, 'scales': scales, 'global_scale': global_scale})
                    destination[start:stop].copy_(decoded); digest.update(fp4._bytes(decoded))
                    del codes_raw, scales_raw, codes, scales, decoded
        need(digest.hexdigest() == manifest['decoded_tensor_sha256'][name], 'Repaired decoded tensor differs: ' + name)
        module_name, local_name = name.rsplit('.', 1)
        module = model.get_submodule(module_name)
        if local_name in module._parameters:
            setattr(module, local_name, nn.Parameter(destination, requires_grad=False))
        elif local_name in module._buffers:
            module._buffers[local_name] = destination
        else:
            raise ValueError('Unknown native tensor: ' + name)
        if progress:
            progress({'tensor': name, 'index': index, 'total': 507, 'decoded_sha256': digest.hexdigest()})
    need(not any(x.is_meta for x in model.state_dict().values()), 'Incomplete repaired checkpoint reload')
    need(model.backbone.embedding.weight.data_ptr() != model.lm_head.weight.data_ptr(), 'Unexpected tied embeddings')
    model._fp4_repaired_receipt = dict(format=FORMAT, complete=True, base_only=True,
        source_checkpoint_required=False, serialization='safetensors_packed_checkpoint',
        packed_checkpoint_manifest_sha256=sha_file(directory / 'weight_manifest.json'),
        parent_manifest_sha256=PARENT_MANIFEST_SHA, conversion_receipt_sha256=parent.LEDGER_SHA,
        logical_weight_payload_bytes=parent.PAYLOAD_BYTES, shard_file_bytes=SHARD_BYTES,
        decoded_resident_weight_bytes=parent.WEIGHT_BYTES, tensor_count=507,
        decoded_tensor_sha256=manifest['decoded_tensor_sha256'],
        training_binding=manifest['training_binding'], packed_resident_kernel=False)
    return model.eval().requires_grad_(False)
