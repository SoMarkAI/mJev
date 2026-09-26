"""Report integrity checks using explicit synthetic unit-test fixtures."""
import importlib.util,json,sys
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('paired_report',Path(__file__).parents[1]/'benchmarks/integrated/report_comparison.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def fixture_files(root):
    groups=[{'dataset':d,'reference_type':ref,'modality':mod,'questions':[{'id':d,'question':'unit test','candidates':['yes','no'],'label':'A'}]} for d,ref,mod in [('infinity','synthetic_proxy','image'),('unit_audio','original_dataset','audio')]]
    (root/'manifest.json').write_text(json.dumps({'groups':groups,'modes':['causal','isolated'],'timing_boundary':'unit test only'}))
    for backend in ('hf','vllm'):
        out=root/backend;out.mkdir();rows=[]
        for g in groups:
            for mode in ['causal','isolated']:
                rows.append({'id':g['dataset'],'dataset':g['dataset'],'reference_type':g['reference_type'],'modality':g['modality'],'reference_label':'A','mode':mode,'seconds':.5,'result':{'prompt_token_ids_sha256':'unit-hash','candidates':[{'label':'A','text':'yes','raw_logit':1.},{'label':'B','text':'no','raw_logit':0.}],'decision':{'label':'A','tied_labels':['A']}}})
        (out/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(out/'rejected.json').write_text('[]');(out/'run_timing.json').write_text('{}');(out/'_COMPLETE').touch()

def test_separates_generated_reference_agreement(tmp_path,monkeypatch):
    fixture_files(tmp_path);monkeypatch.setattr(sys,'argv',['report','--out',str(tmp_path)])
    module.main();r=json.loads((tmp_path/'report.json').read_text())
    assert r['configuration_evidence']['hf']['status']=='unverified_historical_configuration'
    assert r['configuration_evidence']['hf']['numerics'] is None
    assert r['paired_questions']==2
    proxy=r['reports']['hf']['dataset:infinity|causal']['metrics']
    assert 'accuracy' not in proxy and proxy['generated_reference_agreement']==1
    assert r['reports']['vllm']['dataset:unit_audio|causal']['metrics']['accuracy']==1

@pytest.mark.parametrize('corruption',['missing','tokens'])
def test_refuses_unpaired_or_different_inputs(tmp_path,monkeypatch,corruption):
    fixture_files(tmp_path);p=tmp_path/'vllm/raw.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
    if corruption=='missing':rows.pop()
    else:rows[0]['result']['prompt_token_ids_sha256']='changed'
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows));monkeypatch.setattr(sys,'argv',['report','--out',str(tmp_path)])
    with pytest.raises((RuntimeError,ValueError)):module.main()
    assert not (tmp_path/'_SUCCESS').exists()

@pytest.mark.parametrize('field,value', [('reference_label','B'),('dataset','wrong'),
                                        ('modality','video'),('reference_type','original_dataset')])
def test_rejects_metadata_disagreeing_with_frozen_manifest(tmp_path, monkeypatch, field, value):
    fixture_files(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['report','--out',str(tmp_path)])
    module.main()
    assert (tmp_path/'_SUCCESS').exists()
    path=tmp_path/'hf/raw.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field]=value
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='frozen manifest'):
        module.main()
    assert not (tmp_path/'_SUCCESS').exists()


def test_failed_rerun_revokes_success_and_supports_one_mode(tmp_path, monkeypatch):
    fixture_files(tmp_path)
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    manifest['modes']=['causal']
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    for backend in ('hf','vllm'):
        p=tmp_path/backend/'raw.jsonl'
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['mode']=='causal'))
    monkeypatch.setattr(sys,'argv',['report','--out',str(tmp_path)])
    module.main()
    result=json.loads((tmp_path/'report.json').read_text())
    assert result['paired_questions']==2
    assert all(p['hf_only_correct']==p['vllm_only_correct']==0 for p in result['paired'].values())
    (tmp_path/'vllm/raw.jsonl').write_text('')
    with pytest.raises(ValueError,match='coverage'):module.main()
    assert not (tmp_path/'_SUCCESS').exists()
