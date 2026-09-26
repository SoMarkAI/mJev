"""Installation errors must not masquerade as successful/partial hook setup."""
import pytest
from mjev import patch


def test_install_marks_success_only_once(monkeypatch):
    calls=[]
    monkeypatch.setattr(patch,'_INSTALLED',False)
    monkeypatch.setattr(patch,'_INSTALL_ERROR',None)
    monkeypatch.setattr(patch,'_install_hooks',lambda:calls.append(True))
    patch.install();patch.install()
    assert patch._INSTALLED and calls==[True]


def test_failed_install_is_not_retried_or_reported_ready(monkeypatch):
    calls=[]
    def fail():
        calls.append(True)
        assert not patch._INSTALLED
        raise ValueError('partial installation')
    monkeypatch.setattr(patch,'_INSTALLED',False)
    monkeypatch.setattr(patch,'_INSTALL_ERROR',None)
    monkeypatch.setattr(patch,'_install_hooks',fail)
    with pytest.raises(ValueError,match='partial installation'):patch.install()
    assert not patch._INSTALLED
    with pytest.raises(RuntimeError,match='restart the process') as exc:patch.install()
    assert isinstance(exc.value.__cause__,ValueError) and calls==[True]


@pytest.mark.parametrize('enabled,should_run',[('1',False),('0',True)])
def test_sitecustomize_failure_cannot_continue_startup(tmp_path,enabled,should_run):
    import os,subprocess,sys
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'sitecustomize.py'
    (tmp_path/'sitecustomize.py').write_text(source.read_text())
    package=tmp_path/'mjev';package.mkdir()
    (package/'__init__.py').write_text('')
    (package/'patch.py').write_text("def install():\n    raise RuntimeError('injected hook failure')\n")
    env={**os.environ,'PYTHONPATH':str(tmp_path),'MJEV_ENABLE':enabled}
    run=subprocess.run([sys.executable,'-c',"print('APPLICATION_BODY_RAN')"],
        cwd=tmp_path,env=env,capture_output=True,text=True)
    assert ('APPLICATION_BODY_RAN' in run.stdout)==should_run
    assert (run.returncode==0)==should_run
    if not should_run:assert 'refusing to start an unpatched process' in run.stderr
