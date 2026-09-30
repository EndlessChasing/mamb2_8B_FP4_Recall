#!/usr/bin/env python3
"""Create a fresh, allowlisted source directory for the repaired FP4 GitHub repo.

Local filesystem only. Does not initialize Git, commit, create a remote, or upload.
Never copies the research checkout history, weights, checkpoints, logs, or data.
The complete model remains in the separately verified release assets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
from release_fp4_repaired_s16_resurface_v1 import (
    GITHUB_REPO, HF_REPO, NAME, TAG, need, quality, relative, sha_file, verify_bundle,
)
from verify_github_fp4_repaired_s16_resurface_v1 import expected_assets

FORMAT = 'MAMBA2_FP4_G16_REPAIRED_S16_GITHUB_SOURCE_V1'
ROOT_FILES = {'README.md', 'THIRD_PARTY_NOTICES.md', 'LICENSE', 'source_config.json'}
SUPPLEMENTAL_TOOLS = ('scripts/verify_github_fp4_repaired_s16_resurface_v1.py',
    'scripts/verify_hf_fp4_repaired_s16_release_v1.py',
    'scripts/remote_upload_fp4_repaired_s16_release_v1.py',
    'scripts/bridge_remote_fp4_repaired_s16_upload_v1.py',
    'scripts/stage_hf_fp4_repaired_s16_resurface_v1.py',
    'scripts/prepare_github_fp4_repaired_s16_source_v1.py')
IGNORE = b'__pycache__/\n*.pyc\n*.egg-info/\n.venv/\nartifacts/\nreports/\nlogs/\n*.safetensors\n*.pt\n*.model\n*.log\n.DS_Store\n'


def source_allowed(name):
    path = relative(name)
    if name in ROOT_FILES:
        return True
    if path.parts[0] in ('mamba2_recall', 'scripts'):
        return path.suffix == '.py'
    if path.parts[0] == 'docs':
        return path.suffix in ('.md', '.json')
    if path.parts[0] == 'evidence':
        return path.suffix in ('.json', '.md')
    if path.parts[0] == 'licenses':
        return path.suffix in ('.txt', '.md')
    if path.parts[0] == 'weights':
        return len(path.parts) == 2 and path.suffix == '.json'
    if path.parts[0] == 'reference':
        return path.suffix == '.py' or path.name in ('LICENSE', 'WEIGHTS_LICENSE.txt', 'WEIGHTS_NOTICE.md')
    return False


def prepare(args):
    bundle = args.bundle.resolve()
    need(not any(value.is_symlink() for value in (args.out, args.receipt, args.pathspec)),
         'Symlink source output paths are forbidden')
    args.out, args.receipt, args.pathspec = (value.resolve() for value in (args.out, args.receipt, args.pathspec))
    built = json.loads(args.build_receipt.read_text())
    assets = expected_assets(built)
    need(sha_file(bundle / 'manifest.json') == built['bundle_manifest_sha256'], 'Frozen build manifest differs')
    manifest = verify_bundle(bundle)
    comp, train = quality(bundle)
    need(manifest['complete'] is True and manifest['github_repo'] == GITHUB_REPO
         and manifest['release_tag'] == TAG and manifest['candidate_ppl'] == built['candidate_ppl']
         and manifest['candidate_ppl'] == comp['comparison']['candidate_ppl']
         and manifest['baseline_ppl'] == comp['comparison']['control_ppl']
         and manifest['adapter_binding'] == train['binding'], 'Passing exact source recipe required')
    need(not args.out.exists() and not args.out.is_symlink()
         and not args.receipt.exists() and not args.receipt.is_symlink()
         and not args.pathspec.exists() and not args.pathspec.is_symlink(), 'Fresh source, receipt and pathspec paths required')
    need(args.out not in args.receipt.parents and args.out not in args.pathspec.parents
         and args.out != args.receipt and args.out != args.pathspec and args.receipt != args.pathspec,
         'Source receipt/pathspec must be outside the new source checkout')
    need(bundle not in args.out.parents and bundle != args.out,
         'New source checkout must not change the frozen bundle')
    for path in (args.receipt, args.pathspec):
        need(bundle != path and bundle not in path.parents,
             'Output receipt/pathspec must not enter the frozen bundle')
    names = sorted(name for name in manifest['files'] if source_allowed(name))
    need(ROOT_FILES <= set(names) and 'scripts/infer_fp4_repaired_s16_resurface_v1.py' in names,
         'Missing public README, notices, license or runtime source')
    selected = {}
    for name in names:
        path = bundle / relative(name)
        need(path.is_file() and not path.is_symlink(), 'Regular allowlisted source required')
        item = {'sha256': sha_file(path), 'bytes': path.stat().st_size}
        need(item == manifest['files'][name], 'Allowlisted source differs: ' + name)
        need(path.stat().st_size <= 128 * 1024**2, 'Source file exceeds the small-payload allowance: ' + name)
        if path.suffix == '.md':
            data = path.read_bytes()
            need(b'DRAFT' not in data and re.search(rb'\{\{[^{}\n]{1,120}\}\}', data) is None,
                 'Unresolved draft source documentation: ' + name)
        selected[name] = item
    args.out.mkdir(parents=True)
    for name in names:
        target = args.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / name, target)
    presentation = dict(replaced=False, original_readme_sha256=selected['README.md']['sha256'])
    if args.readme is not None:
        data = args.readme.read_bytes()
        need(args.readme.is_file() and not args.readme.is_symlink() and len(data) <= 1024**2
             and GITHUB_REPO.encode() in data and b'DRAFT' not in data
             and re.search(rb'\{\{[^{}\n]{1,120}\}\}', data) is None,
             'Reviewed GitHub presentation must name the exact repo without draft placeholders')
        (args.out / 'README.md').write_bytes(data)
        selected['README.md'] = dict(sha256=sha_file(args.out/'README.md'), bytes=len(data))
        presentation.update(replaced=True, reviewed_readme_sha256=selected['README.md']['sha256'],
            measured_bundle_bytes_changed=False, scope='GitHub root presentation only; scientific release README remains unchanged')
    supplemental = {}
    tool_root = args.tool_root.resolve()
    for name in SUPPLEMENTAL_TOOLS:
        original = tool_root / relative(name)
        need(original.is_file() and not original.is_symlink() and original.stat().st_size <= 1024**2,
             'Explicit supplemental publication tool missing: ' + name)
        item = dict(sha256=sha_file(original), bytes=original.stat().st_size)
        if name in selected:
            need(item == selected[name], 'Supplemental tool differs from its frozen bundle copy: ' + name)
            supplemental[name] = dict(item, source='already_in_verified_bundle')
        else:
            target = args.out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            selected[name] = item
            names.append(name)
            supplemental[name] = dict(item, source='explicit_publication_tool_outside_scientific_bundle')
    (args.out / '.gitignore').write_bytes(IGNORE)
    selected['.gitignore'] = dict(sha256=sha_file(args.out/'.gitignore'), bytes=len(IGNORE))
    names.append('.gitignore')
    provenance = dict(format=FORMAT, complete=True, repository=GITHUB_REPO, hugging_face_repo=HF_REPO,
        release_tag=TAG, bundle_name=NAME, bundle_manifest_sha256=built['bundle_manifest_sha256'],
        candidate_ppl=manifest['candidate_ppl'], baseline_ppl=manifest['baseline_ppl'],
        source_checkpoint_sha256=train['binding']['source_checkpoint_sha256'],
        tokenizer_sha256=manifest['tokenizer_sha256'],
        source_files=selected, release_assets=assets,
        readme_presentation=presentation, supplemental_tools=supplemental,
        scope='Allowlisted source bytes and public asset provenance from the audited complete bundle; '
              'no research Git history, raw logs, optimizer/checkpoint archives, tokenized corpus, adapter/tokenizer binaries or weight shards included in source commit. '
              'PUBLICATION_PROVENANCE.json cannot include its own SHA256; all other source files are bound here and every file is bound by the outside preparation receipt.')
    (args.out / 'PUBLICATION_PROVENANCE.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    names.append('PUBLICATION_PROVENANCE.json')
    args.pathspec.parent.mkdir(parents=True, exist_ok=True)
    with args.pathspec.open('xb') as stream:
        stream.write(b''.join(name.encode() + b'\0' for name in sorted(names)))
    final = {name: {'sha256': sha_file(args.out / name), 'bytes': (args.out / name).stat().st_size}
             for name in sorted(names)}
    actual = {str(path.relative_to(args.out)) for path in args.out.rglob('*') if path.is_file()}
    need(actual == set(final) and not any(path.is_symlink() for path in args.out.rglob('*')),
         'Prepared source contains an unassigned file or symlink')
    receipt = dict(format=FORMAT, complete=True, verified=True, publication_performed=False,
        repository=GITHUB_REPO, tag=TAG, bundle_manifest_sha256=built['bundle_manifest_sha256'],
        source_directory=str(args.out.resolve()), files=final,
        readme_presentation=presentation, supplemental_tools=supplemental,
        full_bundle_verification_passed=True, pathspec_file_count=len(final),
        pathspec_sha256=sha_file(args.pathspec), scope=provenance['scope'])
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'build-receipt', 'out', 'receipt', 'pathspec'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--readme', type=Path, help='Reviewed GitHub root README replacement; bundle bytes are preserved')
    parser.add_argument('--tool-root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Repository root for the six explicit supplemental publication tools')
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2))


if __name__ == '__main__':
    main()
