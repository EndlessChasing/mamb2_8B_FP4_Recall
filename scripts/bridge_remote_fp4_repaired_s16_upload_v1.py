#!/usr/bin/env python3
"""Prepare or invoke the GPU upload worker through encrypted SSH stdin.

Default is a plan: no SSH, auth token lookup or publication. --execute together
with --confirm-audited-quality is required to read a local credential in memory
and send it to the approved worker. Never writes a token, secret CLI argument,
request body or environment to disk or logs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from remote_upload_fp4_repaired_s16_release_v1 import Redactor, SECRET_FORMAT, TOKEN
from release_fp4_repaired_s16_resurface_v1 import GITHUB_REPO, HF_REPO, TAG, need, sha_file
from verify_github_fp4_repaired_s16_resurface_v1 import expected_assets, COMMIT

ALLOWED_PROXY_COMMAND = 'ssh -o HostName=192.168.68.62 -o HostKeyAlias=gwqty2p1v2.local -W %h:%p gwqty2p1v2'
REMOTE_WORKER = '/home/horde/mamb2_8B_W4A16_Recall_StateQuant/scripts/remote_upload_fp4_repaired_s16_release_v1.py'


def credential(service, gh_cli, hf_cli):
    command = ([gh_cli, 'auth', 'token', '--hostname', 'github.com'] if service == 'github'
               else [hf_cli, 'auth', 'token', '--format', 'quiet'])
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Do not echo auth-command stderr/stdout on failures, even before a token
    # has been successfully parsed and added to the redactor.
    need(result.returncode == 0, 'Local authentication helper failed; credential output withheld')
    token = result.stdout.decode().strip()
    need(TOKEN.fullmatch(token), 'Authentication helper did not return a single valid token; output withheld')
    return token


def command(args):
    need(args.host == 'horde-gpu' and args.remote_worker == REMOTE_WORKER,
         'Only the current verified GPU alias and assigned upload worker are allowed')
    need(COMMIT.fullmatch(args.expected_commit), 'Exact source commit required')
    worker = [args.remote_python, args.remote_worker, args.service,
              '--bundle', args.remote_bundle, '--build-receipt', args.remote_build_receipt,
              '--receipt', args.remote_receipt, '--expected-commit', args.expected_commit,
              '--confirm-audited-quality']
    if args.service == 'hf':
        need(args.remote_stage and args.remote_github_receipt and COMMIT.fullmatch(args.expected_hf_head or ''),
             'Exact remote HF stage, public GitHub receipt and existing HF main head required')
        worker += ['--stage', args.remote_stage, '--github-receipt', args.remote_github_receipt,
                   '--expected-hf-head', args.expected_hf_head, '--hf-cli', args.remote_hf_cli]
    # OpenSSH executes the remote command through a shell, so quote each path
    # and argument. Credentials are absent from this command and every argv.
    options = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes']
    proxy = getattr(args, 'proxy_command', None)
    if proxy is not None:
        need(proxy == ALLOWED_PROXY_COMMAND, 'Only the root-verified LAN ProxyCommand is accepted')
        options += ['-o', 'ProxyCommand=' + proxy]
    return options + [args.host, shlex.join(worker)]


def verify_worker_sources(args, argv):
    """No credential lookup until the assigned remote Python sources match."""
    names = ('remote_upload_fp4_repaired_s16_release_v1.py',
             'verify_github_fp4_repaired_s16_resurface_v1.py',
             'verify_hf_fp4_repaired_s16_release_v1.py',
             'release_fp4_repaired_s16_resurface_v1.py')
    local = Path(__file__).resolve().parent
    remote = Path(REMOTE_WORKER).parent
    expected = {str(remote/name): sha_file(local/name) for name in names}
    precheck = list(argv)
    precheck[-1] = 'sha256sum -- ' + shlex.join(sorted(expected))
    env = {key:value for key,value in os.environ.items() if key not in
           ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN')}
    response = subprocess.run(precheck, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, env=env, timeout=60)
    need(response.returncode == 0, 'Assigned remote-source hash precheck failed; no credential read')
    actual = {}
    for line in response.stdout.decode().splitlines():
        digest, name = line.split(maxsplit=1)
        need(name not in actual, 'Duplicate remote source hash')
        actual[name] = digest
    need(actual == expected, 'Assigned remote worker/dependency SHA256 differs; no credential read')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('service', choices=('github', 'hf'))
    parser.add_argument('--local-build-receipt', type=Path, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--proxy-command', help='Exact root-verified GWQTY2P1V2 LAN SSH ProxyCommand')
    for key in ('remote-bundle', 'remote-build-receipt', 'remote-receipt'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--remote-python', default='/home/horde/.venvs/lodram/bin/python')
    parser.add_argument('--remote-worker', default=REMOTE_WORKER)
    parser.add_argument('--remote-stage')
    parser.add_argument('--remote-github-receipt')
    parser.add_argument('--expected-hf-head')
    parser.add_argument('--remote-hf-cli', default='/home/horde/.venvs/lodram/bin/hf')
    parser.add_argument('--local-gh-cli', default='gh')
    parser.add_argument('--local-hf-cli', default='hf')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm-audited-quality', action='store_true')
    args = parser.parse_args()
    redact = Redactor()
    try:
        built = json.loads(args.local_build_receipt.read_text())
        expected_assets(built)
        argv = command(args)
        if not args.execute:
            print(json.dumps({'plan_only': True, 'credential_read': False, 'ssh_invoked': False,
                              'service': args.service, 'command_without_credentials': argv,
                              'local_build_receipt_sha256': sha_file(args.local_build_receipt),
                              'bundle_manifest_sha256': built['bundle_manifest_sha256']}, indent=2))
            return 0
        need(args.confirm_audited_quality is True,
             '--execute requires explicit root confirmation of passed full PPL/MK and independent audit')
        verify_worker_sources(args, argv)
        token = credential(args.service, args.local_gh_cli, args.local_hf_cli)
        redact = Redactor(token)
        payload = {'format': SECRET_FORMAT, 'root_confirmed_quality': True, 'service': args.service,
                   'github_repo': GITHUB_REPO, 'hugging_face_repo': HF_REPO, 'tag': TAG,
                   'commit': args.expected_commit, 'bundle_manifest_sha256': built['bundle_manifest_sha256'],
                   'credential': token}
        env = {key: value for key, value in os.environ.items() if key not in
               ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN')}
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=env)
        try:
            process.stdin.write(json.dumps(payload, separators=(',', ':')).encode())
            process.stdin.close()
            for line in iter(process.stdout.readline, b''):
                sys.stdout.write(redact(line.decode(errors='replace')))
                sys.stdout.flush()
            status = process.wait()
        finally:
            payload.clear()
        if status:
            print(json.dumps({'complete': False, 'service': args.service, 'worker_exit_code': status}), file=sys.stderr)
        return status
    except Exception as error:
        print(json.dumps({'complete': False, 'error_type': type(error).__name__,
                          'error': redact(str(error))[:1500]}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
