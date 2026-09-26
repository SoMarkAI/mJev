"""Experimental runtime must not continue after partial patch installation."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import mjev_vllm_patch as patch


@pytest.fixture
def installers(monkeypatch):
    monkeypatch.setattr(patch,'_INSTALLED',False)
    monkeypatch.setattr(patch,'_INSTALL_ERROR',None)
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(__version__='0.25.1'))
    calls=[]
    for name in ['_install_topology_a_scheduler','_install_masked_scheduler','_install_masked_runner','_install_legacy_v1_runner']:
        monkeypatch.setattr(patch,name,lambda name=name:calls.append(name))
    monkeypatch.setattr(patch,'_hook_status',lambda:{'all_hooks':True})
    return calls


def test_install_version_and_readiness(monkeypatch,installers):
    patch.install();patch.install()
    assert len(installers)==4 and patch.require_installed()=={'all_hooks':True}
    monkeypatch.setattr(patch,'_hook_status',lambda:{'missing_hook':False})
    with pytest.raises(RuntimeError,match='not ready'):patch.require_installed()


def test_wrong_version_fails_before_mutation(monkeypatch,installers):
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(__version__='0.99.0'))
    with pytest.raises(RuntimeError,match='requires vLLM'):patch.install()
    assert not installers and not patch._INSTALLED


def test_partial_failure_cannot_be_retried(monkeypatch,installers):
    def fail():raise ValueError('injected failure')
    monkeypatch.setattr(patch,'_install_masked_runner',fail)
    with pytest.raises(ValueError,match='injected'):patch.install()
    assert not patch._INSTALLED
    with pytest.raises(RuntimeError,match='restart'):patch.install()


@pytest.mark.parametrize('enabled',['0','1'])
def test_sitecustomize_aborts_before_application(tmp_path,enabled):
    root=Path(__file__).resolve().parents[2]
    (tmp_path/'sitecustomize.py').write_text((root/'runtime/sitecustomize.py').read_text())
    (tmp_path/'mjev_vllm_patch.py').write_text("def install():\n    raise RuntimeError('injected failure')\n")
    env={**os.environ,'PYTHONPATH':str(tmp_path),'MJEV_ENABLE_PATCHES':enabled,'MJEV_ENABLE':'0'}
    result=subprocess.run([sys.executable,'-c',"print('BODY_RAN')"],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert (result.returncode==0)==(enabled=='0')
    assert ('BODY_RAN' in result.stdout)==(enabled=='0')
    if enabled=='1':assert 'refusing to start' in result.stderr
