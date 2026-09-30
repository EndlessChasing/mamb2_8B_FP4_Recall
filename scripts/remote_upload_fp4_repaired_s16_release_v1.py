#!/usr/bin/env python3
"""Upload assigned artifacts on the GPU host after the audited publication gate.

Requires a credential envelope on stdin, normally supplied by the SSH bridge.
Never logs or stores credentials. Does not create GitHub repositories/releases,
publish drafts, create HF repositories explicitly, delete assets, or force updates.
HF uses the original installed hf CLI against a preflighted existing main branch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import time
from urllib.parse import quote, urlencode, urlparse

from release_fp4_repaired_s16_resurface_v1 import GITHUB_REPO, HF_REPO, NAME, TAG, need, quality, relative, sha_file
from verify_github_fp4_repaired_s16_resurface_v1 import expected_assets, COMMIT, SHA, RUNTIME

FORMAT = 'MAMBA2_FP4_REPAIRED_S16_REMOTE_UPLOAD_V1'
SECRET_FORMAT = 'MAMBA2_FP4_REPAIRED_S16_SSH_CREDENTIAL_V1'
TOKEN = re.compile(r'[A-Za-z0-9_.-]{16,4096}')
USER_AGENT = 'Mamba2-FP4-repaired-S16-assigned-artifact-upload/1'


class Redactor:
    def __init__(self, *secrets):
        self.secrets = [value for value in secrets if value]

    def __call__(self, text):
        text = str(text)
        for secret in self.secrets:
            text = text.replace(secret, '[REDACTED]')
        text = re.sub(r'(?i)Bearer\s+[^\s"\x27,;]+', 'Bearer [REDACTED]', text)
        return re.sub(r'(?:github_pat_|gh[pousr]_|hf_)[A-Za-z0-9_]{8,}', '[REDACTED]', text)


def emit(value, redact, *, error=False):
    text = json.dumps(value, sort_keys=True, allow_nan=False)
    print(redact(text), file=sys.stderr if error else sys.stdout, flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def envelope(args, built):
    need(args.confirm_audited_quality is True and not sys.stdin.isatty(),
         'Root-confirmed audited quality and non-interactive SSH stdin are required')
    raw = sys.stdin.buffer.read(16385)
    need(0 < len(raw) <= 16384, 'Invalid credential envelope size')
    value = json.loads(raw)
    need(value['format'] == SECRET_FORMAT and value['root_confirmed_quality'] is True
         and value['service'] == args.service and value['github_repo'] == GITHUB_REPO
         and value['hugging_face_repo'] == HF_REPO and value['tag'] == TAG
         and value['commit'] == args.expected_commit
         and value['bundle_manifest_sha256'] == built['bundle_manifest_sha256']
         and isinstance(value['credential'], str) and TOKEN.fullmatch(value['credential']),
         'Credential envelope does not bind the exact approved artifact upload')
    return value['credential']


class HTTPS:
    def __init__(self, token, timeout=60):
        self.token, self.timeout = token, timeout

    def request(self, host, path, *, method='GET', file=None, size=None):
        need(host in ('api.github.com', 'uploads.github.com', 'huggingface.co')
             and path.startswith('/') and '\r' not in path and '\n' not in path,
             'Unexpected upload API destination')
        need(method == 'GET' or (method == 'POST' and host == 'uploads.github.com' and file is not None),
             'Only API reads and assigned GitHub binary uploads are allowed')
        if host == 'api.github.com':
            root = '/repos/' + GITHUB_REPO
            need(path == root or path.startswith(root + '/'), 'Credential API read left the assigned GitHub repo')
        elif host == 'huggingface.co':
            need(path == '/api/models/' + HF_REPO + '/revision/main?blobs=true',
                 'Credential API read left the assigned Hugging Face main')
        else:
            need(path.startswith('/repos/' + GITHUB_REPO + '/releases/') and '/assets?' in path,
                 'Credential upload left the assigned GitHub assets endpoint')
        headers = {'User-Agent': USER_AGENT, 'Accept': 'application/json' if host == 'huggingface.co' else 'application/vnd.github+json',
                   'Authorization': 'Bearer ' + self.token, 'Accept-Encoding': 'identity'}
        if host.endswith('github.com'):
            headers['X-GitHub-Api-Version'] = '2026-03-10'
        if file is not None:
            headers.update({'Content-Type': 'application/octet-stream', 'Content-Length': str(size)})
        connection = http.client.HTTPSConnection(host, timeout=self.timeout, context=ssl.create_default_context())
        try:
            connection.putrequest(method, path, skip_accept_encoding=True)
            for key, value in headers.items():
                connection.putheader(key, value)
            connection.endheaders()
            if file is not None:
                sent = 0
                with Path(file).open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        sent += len(chunk)
                        need(sent <= size, 'Local asset grew during upload')
                        connection.send(chunk)
                need(sent == size, 'Local asset shrank during upload')
            response = connection.getresponse()
            body = response.read(16 * 1024**2 + 1)
            need(len(body) <= 16 * 1024**2, 'Oversized API response')
            need(response.status in (200, 201), 'Upload API HTTP status ' + str(response.status))
            return json.loads(body)
        finally:
            connection.close()

    def github_json(self, path):
        root = '/repos/' + GITHUB_REPO
        need(path == root or path.startswith(root + '/'), 'Only the assigned GitHub repo is allowed')
        return self.request('api.github.com', path)


def peel_tag(api):
    root = '/repos/' + GITHUB_REPO
    ref = api.github_json(root + '/git/ref/tags/' + quote(TAG, safe=''))
    need(ref['ref'] == 'refs/tags/' + TAG, 'Draft release tag differs')
    value = ref['object']
    seen = set()
    while value['type'] == 'tag':
        need(COMMIT.fullmatch(value['sha']) and value['sha'] not in seen and len(seen) < 8,
             'Invalid annotated GitHub tag')
        seen.add(value['sha'])
        row = api.github_json(root + '/git/tags/' + value['sha'])
        need(row['sha'] == value['sha'], 'Annotated tag identity differs')
        value = row['object']
    need(value['type'] == 'commit' and COMMIT.fullmatch(value['sha']), 'Tag must resolve to a commit')
    return value['sha']


def draft(api, expected_commit, release_id=None):
    repo = api.github_json('/repos/' + GITHUB_REPO)
    need(repo['full_name'] == GITHUB_REPO and repo['private'] is False, 'Exact public repo required')
    # The by-tag REST endpoint is documented for published releases. Discover
    # the authorized draft through the release list once, then pin its id.
    if release_id is None:
        matches = []
        for page in range(1, 11):
            items = api.github_json('/repos/' + GITHUB_REPO + '/releases?per_page=100&page=' + str(page))
            need(isinstance(items, list), 'Release list must be a list')
            matches.extend(item for item in items if item['tag_name'] == TAG)
            if len(items) < 100:
                break
        need(len(matches) == 1 and matches[0]['draft'] is True, 'One exact existing authorized draft required')
        release_id = matches[0]['id']
    need(type(release_id) is int and release_id > 0, 'Exact draft release id required')
    row = api.github_json('/repos/' + GITHUB_REPO + '/releases/' + str(release_id))
    need(row['tag_name'] == TAG and row['draft'] is True and row['prerelease'] is False
         and type(row['id']) is int and row['id'] > 0
         and (release_id is None or release_id == row['id']), 'Only the exact existing draft release is allowed')
    need(peel_tag(api) == expected_commit, 'Draft tag does not point to the approved source commit')
    prefix = 'https://uploads.github.com/repos/' + GITHUB_REPO + '/releases/' + str(row['id']) + '/assets'
    need(row['upload_url'] == prefix + '{?name,label}', 'Unexpected release upload endpoint')
    return row


def remote_assets(api, release_id, expected):
    rows = {}
    for page in range(1, 12):
        items = api.github_json('/repos/' + GITHUB_REPO + '/releases/' + str(release_id)
                                + '/assets?per_page=100&page=' + str(page))
        need(isinstance(items, list), 'Asset page must be a list')
        for item in items:
            name = item['name']
            need(name in expected and name not in rows, 'Unexpected or duplicate draft asset: ' + str(name))
            want = expected[name]
            need(item['state'] == 'uploaded' and item['size'] == want['bytes']
                 and item.get('digest') == 'sha256:' + want['sha256'],
                 'Existing asset must match state, bytes and SHA-256; no clobber: ' + name)
            rows[name] = item
        if len(items) < 100:
            return rows
    raise ValueError('Asset list exceeds the allowed release inventory')


def common_local(args):
    need(COMMIT.fullmatch(args.expected_commit), 'Exact source commit required')
    need(not args.receipt.exists() and not args.receipt.is_symlink(), 'Fresh upload receipt required')
    built = read(args.build_receipt)
    expected = expected_assets(built)
    need(sha_file(args.bundle / 'manifest.json') == built['bundle_manifest_sha256'], 'Frozen bundle manifest differs')
    comp, _ = quality(args.bundle)
    need(comp['comparison']['candidate_ppl'] == built['candidate_ppl']
         and comp['comparison']['control_ppl'] == built['baseline_ppl'], 'Audited full candidate differs')
    return built, expected


def github_upload(args, built, expected, token, redact):
    # Check every local payload before making any upload. Streaming hashes do
    # not materialize the checkpoint in RAM or download it onto the Mac.
    paths = {}
    for item in built['github_assets']:
        path = args.build_receipt.parent / relative(item['path'])
        need(path.is_file() and not path.is_symlink() and path.stat().st_size == item['bytes']
             and sha_file(path) == item['sha256'], 'Local assigned GitHub asset differs: ' + path.name)
        paths[path.name] = path
    api = HTTPS(token)
    release = draft(api, args.expected_commit)
    present = remote_assets(api, release['id'], expected)
    uploaded, resumed = [], sorted(present)
    for index, name in enumerate(sorted(expected), 1):
        if name in present:
            emit({'asset': name, 'action': 'resume_exact_existing', 'progress': [index, 120]}, redact)
            continue
        # Never continue writing if another process has already published or
        # replaced the draft during this worker's session.
        draft(api, args.expected_commit, release['id'])
        upload_path = urlparse(release['upload_url'].split('{', 1)[0]).path + '?' + urlencode({'name': name})
        success = False
        for attempt in range(3):
            try:
                item = api.request('uploads.github.com', upload_path, method='POST',
                                   file=paths[name], size=expected[name]['bytes'])
                need(item['name'] == name and item['state'] == 'uploaded'
                     and item['size'] == expected[name]['bytes']
                     and item.get('digest') == 'sha256:' + expected[name]['sha256'],
                     'Uploaded asset digest or bytes differ')
                success = True
            except Exception:
                # A response can be lost after a successful write. Re-list and
                # resume only an exact uploaded digest/size; never delete a
                # partial "starter" asset, retry a conflicting name, or clobber.
                time.sleep(attempt + 1)
                found = remote_assets(api, release['id'], expected)
                if name in found:
                    success = True
                elif attempt == 2:
                    raise
                else:
                    draft(api, args.expected_commit, release['id'])
            if success:
                break
        need(success, 'Assigned asset upload was not confirmed')
        uploaded.append(name)
        emit({'asset': name, 'action': 'uploaded', 'bytes': expected[name]['bytes'],
              'progress': [index, 120]}, redact)
    final = remote_assets(api, release['id'], expected)
    need(set(final) == set(expected), 'Complete uploaded draft inventory differs')
    draft(api, args.expected_commit, release['id'])
    return dict(format=FORMAT, service='github', complete=True, verified=True,
                repository=GITHUB_REPO, tag=TAG, commit=args.expected_commit, release_id=release['id'],
                draft=True, release_publication_performed=False, asset_upload_performed=bool(uploaded),
                uploaded_assets=uploaded, resumed_exact_assets=resumed, assets=expected,
                bundle_manifest_sha256=built['bundle_manifest_sha256'], build_receipt_sha256=sha_file(args.build_receipt),
                root_confirmed_audited_quality=True, candidate_ppl=built['candidate_ppl'],
                credentials_persisted=False, token_cli_argument_used=False,
                scope='Assigned exact binary assets uploaded directly on GPU host to existing draft; '
                      'matching existing assets resumed; no deletion, clobber, repository creation or draft publication.')


def stage_files(args, built):
    from verify_hf_fp4_repaired_s16_release_v1 import expected_files
    manifest = read(args.stage / 'publish_manifest.json')
    gh = read(args.github_receipt)
    expected = expected_assets(built)
    need(gh['complete'] is True and gh['verified'] is True and gh['anonymous_access_verified'] is True
         and gh['repository'] == GITHUB_REPO and gh['tag'] == TAG and gh['commit'] == args.expected_commit
         and gh['bundle_manifest_sha256'] == built['bundle_manifest_sha256'] and gh['assets'] == expected,
         'Prior passed anonymous public GitHub receipt required')
    need(manifest['complete'] is True and manifest['ready_for_upload'] is True
         and manifest['repository'] == HF_REPO and manifest['github_commit'] == args.expected_commit
         and manifest['github_receipt_sha256'] == sha_file(args.github_receipt)
         and manifest['bundle_manifest_sha256'] == built['bundle_manifest_sha256'], 'Bound HF stage manifest required')
    checked_manifest, _, _ = expected_files(args.stage / 'publish_manifest.json', args.stage / 'SHA256SUMS')
    need(checked_manifest == manifest, 'HF full-stage file contract differs')
    files = dict(manifest['files'])
    files['publish_manifest.json'] = {'sha256': sha_file(args.stage / 'publish_manifest.json'),
                                     'bytes': (args.stage / 'publish_manifest.json').stat().st_size}
    hashes = {name: item['sha256'] for name, item in files.items()}
    need((args.stage / 'SHA256SUMS').read_text() == ''.join(
        digest + '  ' + name + '\n' for name, digest in sorted(hashes.items())), 'HF stage checksum list differs')
    files['SHA256SUMS'] = {'sha256': sha_file(args.stage / 'SHA256SUMS'),
                          'bytes': (args.stage / 'SHA256SUMS').stat().st_size}
    actual = set()
    for path in args.stage.rglob('*'):
        need(not path.is_symlink(), 'No HF stage symlinks')
        if path.is_file():
            actual.add(str(path.relative_to(args.stage)))
    need(actual == set(files), 'HF stage contains unassigned or missing files')
    for name, item in files.items():
        path = args.stage / relative(name)
        need(path.stat().st_size == item['bytes'] and sha_file(path) == item['sha256'], 'HF stage file differs: ' + name)
    need(sha_file(args.stage / 'release/manifest.json') == built['bundle_manifest_sha256'], 'Staged bundle manifest differs')
    quality(args.stage / 'release')
    return files


def hf_info(api):
    row = api.request('huggingface.co', '/api/models/' + HF_REPO + '/revision/main?blobs=true')
    need(row['id'] == HF_REPO and row['private'] is False and row.get('gated') is False
         and COMMIT.fullmatch(row['sha']), 'Exact existing public ungated HF model/main branch required')
    return row


def git_blob_sha(path):
    path = Path(path)
    digest = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0')
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def hf_upload_command(executable, stage):
    """Accept the installed official CLI's documented flag spelling."""
    env = {key:value for key,value in os.environ.items() if key not in
           ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN')}
    result = subprocess.run([str(executable), 'upload', '--help'], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=30)
    need(result.returncode == 0, 'Installed official hf upload help unavailable; output withheld')
    help_text = result.stdout.decode(errors='replace')
    type_option = '--repo-type' if '--repo-type' in help_text else '--type'
    need(type_option in help_text, 'Installed hf upload has no documented model type option')
    command = [str(executable), 'upload', HF_REPO, str(stage), '.', type_option, 'model',
               '--revision', 'main', '--commit-message', 'Release audited repaired FP4 G16 native-S16 Resurface model']
    if '--quiet' in help_text:
        command.append('--quiet')
    elif '--format' in help_text:
        command += ['--format', 'quiet']
    return command


def hf_inventory(info, stage, files):
    rows = {}
    for item in info['siblings']:
        name = item['rfilename']
        relative(name)
        need(name not in rows and name in files, 'Unexpected or duplicate HF file: ' + name)
        want = files[name]
        need(item.get('size') == want['bytes'], 'HF uploaded file size differs: ' + name)
        if item.get('lfs'):
            need(item['lfs'].get('sha256') == want['sha256'], 'HF LFS SHA-256 differs: ' + name)
        else:
            need(item.get('blobId') == git_blob_sha(stage / name), 'HF Git blob SHA-1 differs: ' + name)
        rows[name] = {'bytes': want['bytes'], 'sha256': want['sha256'],
                      'hash_source': 'hf_lfs_sha256_metadata' if item.get('lfs') else 'hf_git_blob_sha1_metadata'}
    need(set(rows) == set(files), 'HF uploaded inventory is incomplete')
    return rows


def hf_upload(args, built, token, redact):
    files = stage_files(args, built)
    need(COMMIT.fullmatch(args.expected_hf_head), 'Exact existing HF main head required')
    api = HTTPS(token)
    before = hf_info(api)
    need(before['sha'] == args.expected_hf_head, 'HF main changed from the root-preflighted head')
    names = {row['rfilename'] for row in before['siblings']}
    if names <= {'.gitattributes'}:
        already_complete = False
    else:
        # A retry can resume the exact complete payload; another existing model
        # must never be overwritten just because the repository name matches.
        hf_inventory(before, args.stage, files)
        already_complete = True
    if not already_complete:
        need(args.hf_cli.is_file() and os.access(args.hf_cli, os.X_OK), 'Installed remote hf executable required')
        env = {key: value for key, value in os.environ.items() if key not in
               ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN')
               and key.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
        env.update(HF_TOKEN=token, HF_TOKEN_PATH='/dev/null', HF_HUB_DISABLE_TELEMETRY='1',
                   HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_VERBOSITY='error',
                   HF_HUB_DISABLE_XET='1', HF_HUB_ENABLE_HF_TRANSFER='0',
                   HF_HUB_DISABLE_IMPLICIT_TOKEN='0', HF_DEBUG='0')
        command = hf_upload_command(args.hf_cli, args.stage)
        need(token not in command, 'Token must not be passed as a CLI argument')
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        # Progress while waiting comes from this worker, not raw vendor logs.
        while True:
            try:
                stdout, stderr = process.communicate(timeout=30)
                break
            except subprocess.TimeoutExpired:
                emit({'service': 'hf', 'status': 'uploading_assigned_stage', 'repository': HF_REPO}, redact)
        env.pop('HF_TOKEN', None)
        if process.returncode:
            message = redact((stderr or stdout).decode(errors='replace'))[-2000:]
            raise ValueError('Official hf upload failed, exit ' + str(process.returncode) + ': ' + message)
        emit({'service': 'hf', 'status': 'official_cli_completed', 'repository': HF_REPO,
              'output': redact(stdout.decode(errors='replace'))[-1000:]}, redact)
    after = hf_info(api)
    verified = hf_inventory(after, args.stage, files)
    return dict(format=FORMAT, service='hf', complete=True, verified=True, repository=HF_REPO,
                commit=after['sha'], preflight_main_commit=before['sha'], github_commit=args.expected_commit,
                github_receipt_sha256=sha_file(args.github_receipt), assets=verified,
                bundle_manifest_sha256=built['bundle_manifest_sha256'], candidate_ppl=built['candidate_ppl'],
                upload_performed=not already_complete, resumed_exact_payload=already_complete,
                credentials_persisted=False, token_cli_argument_used=False, root_confirmed_audited_quality=True,
                scope='Official remote hf upload to root-created, preflighted existing public model/main; '
                      'all assigned file sizes and HF LFS SHA-256/Git blob SHA-1 metadata matched locally hashed stage. '
                      'No independent complete public model download or GPU quality rerun performed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('service', choices=('github', 'hf'))
    for key in ('bundle', 'build-receipt', 'receipt'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--confirm-audited-quality', action='store_true', required=True)
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--github-receipt', type=Path)
    parser.add_argument('--expected-hf-head')
    parser.add_argument('--hf-cli', type=Path, default=Path('/home/horde/.venvs/lodram/bin/hf'))
    args = parser.parse_args()
    redact = Redactor()
    try:
        built, expected = common_local(args)
        token = envelope(args, built)
        redact = Redactor(token)
        if args.service == 'hf':
            need(args.stage is not None and args.github_receipt is not None and args.expected_hf_head is not None,
                 'HF requires exact stage, public GitHub receipt and root-preflighted main head')
            result = hf_upload(args, built, token, redact)
        else:
            result = github_upload(args, built, expected, token, redact)
        result['finished_at'] = datetime.now(timezone.utc).isoformat()
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        # Only a structured artifact receipt is persisted; never vendor logs,
        # credentials, request headers, subprocess environment or stdin payload.
        with args.receipt.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        emit({'complete': True, 'verified': True, 'service': args.service,
              'receipt': str(args.receipt), 'repository': result['repository']}, redact)
    except Exception as error:
        emit({'complete': False, 'verified': False, 'service': args.service,
              'error_type': type(error).__name__, 'error': redact(str(error))[:2500]}, redact, error=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
