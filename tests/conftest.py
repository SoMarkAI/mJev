"""Explicit suites: no recursive/function-test omissions, GPU is opt-in."""
from pathlib import Path
import pytest


def pytest_addoption(parser):
    parser.addoption('--suite', choices=('all','base','hf','runtime','vllm','gpu'), default='all',
                     help='all includes every CPU suite; GPU requires --suite gpu')


def suite_for(path):
    path=Path(path)
    if 'gpu' in path.parts:return 'gpu'
    if 'hf' in path.parts:return 'hf'
    if 'runtime' in path.parts:return 'runtime'
    if path.name in ('test_interleaved.py','test_prefix_bypass.py'):return 'vllm'
    return 'base'


def pytest_ignore_collect(collection_path, config):
    if not collection_path.is_file() or not collection_path.name.startswith('test_'):
        return False
    wanted=config.getoption('--suite'); actual=suite_for(collection_path)
    return actual=='gpu' if wanted=='all' else actual!=wanted


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(getattr(pytest.mark,suite_for(item.path)))
