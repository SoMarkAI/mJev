"""Contracts for public reproduction; fake scores below are NOT model results."""
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from mjev.benchmark import public_mini as mini
from mjev.benchmark.provenance import capture, check_numerics, revision, sha256, write
from mjev.benchmark.prepare import write_jsonl

ROOT=Path(__file__).resolve().parents[1]


def question(ident, vid='abcdefghijk', duration=10):
    return {'question_id':ident,'video_url':'https://www.youtube.com/watch?v='+vid,
            'video_duration':duration,'question':'Original question?',
            'options':{'A':'first','B':'second'},'correct_option_letter':'B',
            'question_type':['test'],'start_time':'0:00:01','end_time':'0:00:02'}


def test_selection_group_complete_and_order_independent():
    rows=[question('q2'),question('q1'),question('long','lmnopqrstuv',50),question('missing','zyxwvutsrqp')]
    names={'test/abcdefghijk.mp4','test/lmnopqrstuv.mp4'}
    selected,excluded=mini.select_groups(rows,names,1,15,37)
    assert selected==mini.select_groups(list(reversed(rows)),names,1,15,37)[0]
    assert [q['question_id'] for q in selected[0][1]]==['q1','q2']
    assert {x['reason'] for x in excluded}=={'annotation_duration_filter','missing_from_pinned_index'}
    changed=copy.deepcopy(rows);changed[0]['correct_option_letter']='A'
    assert [g[0] for g in mini.select_groups(changed,names,1,15,37)[0]]==[g[0] for g in selected]


@pytest.mark.parametrize('field',['numerics','model_revision','projection','video_options'])
def test_configuration_requires_explicit_settings(tmp_path,field):
    c=json.loads((ROOT/'configs/public_mini_hf.json').read_text());del c[field]
    path=tmp_path/'config.json';write(path,c)
    with pytest.raises(ValueError,match='explicit'):mini.config_load(path)


def test_reject_floating_revision_and_vllm_mismatch(monkeypatch):
    with pytest.raises(ValueError):revision('main')
    monkeypatch.setenv('VLLM_BATCH_INVARIANT','0')
    with pytest.raises(ValueError):check_numerics('vllm','stable')


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    def fake_download(url,target):
        target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
        if target.name=='annotations.json':write(target,[question('q1'),question('q2')])
        elif target.name=='media-index.json':write(target,{'siblings':[{'rfilename':'test/abcdefghijk.mp4'}]})
        else:target.write_bytes(b'unit fixture, not a real video')
        return {'url':url,'bytes':target.stat().st_size,'sha256':sha256(target)}
    monkeypatch.setattr(mini,'download',fake_download)
    root=tmp_path/'data';mini.prepare(root,1,15,37)
    return root


def test_preparation_preserves_original_and_detects_tamper(bundle):
    dataset=mini.verify(bundle)
    assert len(dataset)==2
    for row in dataset:
        assert row['question']==row['original']['question']
        assert row['candidates']==row['original']['options']
        assert row['label']==row['original']['correct_option_letter']
        payload=dataset.model_input(row)
        assert not {'label','original','metadata'} & payload.keys()
    (bundle/'media/abcdefghijk.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Media changed'):mini.verify(bundle)


def test_prepare_failure_has_no_success_marker(tmp_path,monkeypatch):
    def fail(*a):raise RuntimeError('network failure')
    monkeypatch.setattr(mini,'download',fail)
    with pytest.raises(RuntimeError):mini.prepare(tmp_path/'failed')
    assert (tmp_path/'failed/ERROR.json').exists()
    assert not (tmp_path/'failed/_DATA_READY').exists()


def test_inference_contract_and_report_with_fake_engine(bundle,tmp_path,monkeypatch):
    import huggingface_hub
    import mjev.hf
    model=tmp_path/'model';model.mkdir();write(model/'config.json',{'fixture':True})
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**kw:str(model))
    seen={}
    class FakeEngine:
        def __init__(self,path,**kwargs):seen['init']=kwargs
        def score_many(self,path,questions,**kwargs):
            seen['score']=kwargs
            assert all(set(q)=={'id','question','candidates'} for q in questions)
            return [{'candidates':[{'label':'A','text':'first','raw_logit':0.},
                                   {'label':'B','text':'second','raw_logit':0.}]} for q in questions]
    monkeypatch.setattr(mjev.hf,'HFMJevEngine',FakeEngine)
    monkeypatch.setenv('MJEV_ENABLE','0');monkeypatch.setenv('MJEV_ENABLE_PATCHES','0')
    out=tmp_path/'run'
    asyncio.run(mini.infer(bundle,ROOT/'configs/public_mini_hf.json',out))
    assert seen['init']['numerics']=='stable' and seen['score']['projection']=='full'
    assert seen['score']['batch_size']==1 and seen['score']['use_prefix_cache'] is False
    mini.report(out)
    result=json.loads((out/'report.json').read_text())
    assert result['evaluation']['metrics']['count']==2
    # Source archives and minimal runtime images may not contain Git.
    # Record that absence explicitly; actual source hashes remain mandatory.
    assert result['run']['code_commit']==capture({},None)['code_commit']
    assert result['run']['source_sha256']
    assert result['run']['dependencies']['torch']
    assert (out/'_SUCCESS').exists()
    # Reject missing coverage instead of reporting successful partial results.
    predictions=(out/'predictions.jsonl').read_text().splitlines()
    (out/'predictions.jsonl').write_text(predictions[0]+'\n')
    with pytest.raises(ValueError):mini.report(out)
    assert not (out/'_SUCCESS').exists()


def test_failed_inference_cannot_report(tmp_path):
    (tmp_path/'_INFERENCE_COMPLETE').write_text('complete')
    (tmp_path/'ERROR.json').write_text('{}')
    with pytest.raises(ValueError,match='failed'):mini.report(tmp_path)
