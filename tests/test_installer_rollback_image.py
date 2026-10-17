"""Execute installer preflight with isolated commands and no real Docker."""
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(os.name == 'nt' or getattr(os, 'geteuid', lambda: 1)() != 0,
    reason='Linux installer preflight runs as root in the deployment test image')
@pytest.mark.parametrize('missing,recovery,accepted', [
    (True, '', False), (True, 'xiaomachi-bot:local', False),
    (True, 'sha256:'+'b'*64, True), (False, '', True)])
def test_missing_image_requires_verified_digest_without_using_candidate(tmp_path, missing, recovery, accepted):
    source=(Path(__file__).parents[1]/'infra/wsl/scripts/install_linux_runtime.sh').read_text(encoding='utf-8')
    script=tmp_path/'preflight.sh'; script.write_text(source[:source.index('memory_audit_timer_was_enabled=')])
    commands=tmp_path/'bin'; commands.mkdir()
    docker=commands/'docker'
    docker.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
if args[:1]==['inspect']:
 print('sha256:'+'a'*64)
elif args[:2]==['image','inspect']:
 raise SystemExit(0 if args[2]=='sha256:'+'b'*64 or os.environ['TEST_MISSING']=='0' else 1)
elif args[:2]==['image','tag']:
 Path(os.environ['TEST_CALLS']).write_text(json.dumps(args[2:]))
else:
 raise SystemExit('unexpected docker action')
'''); docker.chmod(0o755)
    systemctl=commands/'systemctl'; systemctl.write_text('#!/bin/sh\nexit 0\n'); systemctl.chmod(0o755)
    calls=tmp_path/'calls.json'
    environment={**os.environ,'PATH':str(commands)+os.pathsep+os.environ['PATH'],
        'TEST_MISSING':str(int(missing)),'TEST_CALLS':str(calls),
        'XIAOMACHI_INSTALL_ROOT':str(tmp_path/'install'), 'XIAOMACHI_ROLLBACK_IMAGE_ID':recovery}
    result=subprocess.run(['bash',str(script)],env=environment,capture_output=True,text=True)
    assert (result.returncode==0) is accepted, result.stderr
    if accepted:
        values=json.loads(calls.read_text())
        assert values[0]=='sha256:'+('b' if missing else 'a')*64
        assert values[1].startswith('xiaomachi-bot:rollback-')
    else:
        assert not calls.exists()
