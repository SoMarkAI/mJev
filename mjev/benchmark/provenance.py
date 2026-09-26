"""Portable run evidence; no credentials, hostname or private absolute paths."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def revision(value):
    if not re.fullmatch(r'[0-9a-f]{40}', value):
        raise ValueError('Use an immutable 40-character model commit, not main/tag')
    return value


def capture(settings, model_revision, model_dir=None):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    # Hash actual source, including uncommitted/new source files. Commit alone is insufficient.
    files = sorted(p for folder in ('mjev', 'runtime', 'benchmarks', 'scripts', 'configs', 'tests')
                   for p in (root / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.sh', '.json'))
    files += [p for p in (root/'pyproject.toml', root/'sitecustomize.py', root/'demo.py', root/'demo_hf.py') if p.exists()]
    sources = {str(p.relative_to(root)): sha256(p) for p in files}
    dependencies = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    model_files = {}
    if model_dir:
        model_files = {p.name: sha256(p) for p in sorted(Path(model_dir).iterdir())
                       if p.is_file() and p.suffix in ('.json', '.jinja')}
    status = git('status', '--porcelain')
    return {'code_commit': git('rev-parse', 'HEAD'), 'working_tree_dirty': None if status is None else bool(status),
            'source_sha256': sources,
            'source_tree_sha256': hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            'model_revision': revision(model_revision) if model_revision is not None else None,
            'model_revision_status': 'declared_local_unverified' if model_revision else 'unverified_not_provided',
            'model_config_sha256': model_files,
            'dependencies': dict(sorted(dependencies.items())),
            'python': platform.python_version(), 'platform': platform.system(), 'architecture': platform.machine(),
            'settings': settings,
            'environment': {key: os.environ.get(key) for key in ('VLLM_BATCH_INVARIANT', 'MJEV_ENABLE', 'MJEV_ENABLE_PATCHES', 'VLLM_USE_V2_MODEL_RUNNER', 'FORCE_QWENVL_VIDEO_READER')}}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def evaluation_arguments(parser, required=True):
    parser.add_argument('--numerics', choices=('native','stable'), required=required,
                        help='Explicit profile; old results cannot be inferred from current defaults')
    parser.add_argument('--model-revision', type=revision, required=required,
                        help='Official immutable model commit; local contents are separately marked unverified')


def check_numerics(backend, numerics):
    if backend=='vllm':
        expected='1' if numerics=='stable' else '0'
        if os.environ.get('VLLM_BATCH_INVARIANT')!=expected:
            raise ValueError(f'Start vLLM with VLLM_BATCH_INVARIANT={expected} for numerics={numerics}')
