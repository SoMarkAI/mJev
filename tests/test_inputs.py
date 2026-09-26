"""Public defaults and preflight must match real scoring contracts."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from mjev.inputs import normalize_questions, normalize_task

ROOT=Path(__file__).resolve().parents[1]


def test_causal_default_across_main_interfaces():
    for file in ('demo.py','demo_hf.py','mjev/engine.py','mjev/hf.py','mjev/protocol.py','mjev/prompt.py'):
        tree=ast.parse((ROOT/file).read_text())
        defaults=[]
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                args=node.args
                pairs=list(zip(args.args[-len(args.defaults):],args.defaults)) if args.defaults else []
                pairs+=list(zip(args.kwonlyargs,args.kw_defaults))
                defaults.extend(value.value for arg,value in pairs if arg.arg=='mode' and isinstance(value,ast.Constant))
            if isinstance(node,ast.Call) and any(isinstance(arg,ast.Constant) and arg.value=='--mode' for arg in node.args):
                defaults.extend(k.value.value for k in node.keywords if k.arg=='default')
        assert defaults and set(defaults)=={'causal'}, (file,defaults)


@pytest.mark.parametrize('questions',[[],None,{},[{'question':'Q','candidates':{'B':'blue','A':'red'}}],
                                      [{'question':'Q','candidates':['same','same']}],
                                      [{'question':'','candidates':['a','b']}]])
@pytest.mark.parametrize('cli',['demo.py','demo_hf.py'])
@pytest.mark.parametrize('check_only',[False,True])
def test_bad_inputs_fail_before_model_or_runtime_load(tmp_path,questions,cli,check_only):
    path=tmp_path/'input.json'
    path.write_text(json.dumps({'image':'missing.png','questions':questions}))
    command=[sys.executable,str(ROOT/cli),'--model','missing-model','--input',str(path)]
    if check_only:command.append('--check-only')
    result=subprocess.run(command,cwd=ROOT,env={**os.environ,'MJEV_ENABLE':'0','MJEV_ENABLE_PATCHES':'0'},capture_output=True,text=True)
    assert result.returncode!=0
    assert 'ValueError:' in result.stderr
    assert 'ModuleNotFoundError' not in result.stderr and 'from_pretrained' not in result.stderr


def test_normalization_is_shared_and_non_mutating(tmp_path):
    source={'question':'Q','candidates':{'A':'red','B':'blue'}}
    task=normalize_task({'image':'a.png','questions':[source]},tmp_path)
    assert task['questions']==normalize_questions([source])
    assert task['questions'][0]['candidates']==['red','blue']
    assert isinstance(source['candidates'],dict)
    from mjev.hf import HFMJevEngine
    engine=HFMJevEngine.__new__(HFMJevEngine)
    with pytest.raises(ValueError,match='nonempty list'):engine.score_many('unused',[])
    with pytest.raises(ValueError,match='ordered'):engine.score_many('unused',[{'question':'Q','candidates':{'B':'x','A':'y'}}])
