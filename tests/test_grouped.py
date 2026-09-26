"""Public grouped runner contracts, no fabricated pretrained results."""
import asyncio
import json
from pathlib import Path
import pytest
from mjev.benchmark.grouped import config_load, run
from mjev.benchmark.prepare import write_jsonl

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('backend',['hf','vllm'])
def test_vl_config_pins_model_and_execution(backend):
    c=config_load(ROOT/f'configs/qwen3_vl_4b_{backend}.json')
    assert c['model_revision']=='ebb281ec70b05090aa6165b016eac8ec08e71b17'
    assert c['numerics']=='stable' and c['question_batch_size']==3


def test_grouped_rejects_audio_before_downloading_or_creating_output(tmp_path):
    (tmp_path/'audio.wav').write_bytes(b'not decoded')
    row=dict(id='audio:1',dataset='test',source={},task='test',modality='audio',media_path='audio.wav',
        media={'type':'audio','path':'audio.wav'},question='Question?',candidates={'A':'yes','B':'no'},label='A')
    write_jsonl(tmp_path/'manifest.jsonl',[row])
    with pytest.raises(ValueError,match='never silently discarded'):
        asyncio.run(run(tmp_path/'manifest.jsonl',ROOT/'configs/qwen3_vl_4b_hf.json',tmp_path/'out'))
    assert not (tmp_path/'out').exists()


def test_grouped_requires_all_settings(tmp_path):
    c=json.loads((ROOT/'configs/qwen3_vl_4b_hf.json').read_text());del c['numerics']
    (tmp_path/'bad.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='explicit'):config_load(tmp_path/'bad.json')


@pytest.mark.parametrize('fail_second',[False,True])
def test_completed_groups_survive_later_failure(tmp_path,monkeypatch,fail_second):
    import huggingface_hub
    import mjev.hf
    import mjev.benchmark.grouped as grouped
    monkeypatch.setenv('MJEV_ENABLE','0');monkeypatch.setenv('MJEV_ENABLE_PATCHES','0')
    rows=[]
    for i in range(2):
        media=f'{i}.png';(tmp_path/media).write_bytes(b'unit fixture, not decoded')
        rows.append(dict(id=str(i),dataset='unit',source={},task='unit',modality='image',media_path=media,
            media={'type':'image','path':media},question='Question?',candidates={'A':'yes','B':'no'},label='A'))
    manifest=tmp_path/'manifest.jsonl';write_jsonl(manifest,rows)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**k:str(tmp_path))
    monkeypatch.setattr(grouped,'capture',lambda *a,**k:{})
    class FakeEngine:
        def __init__(self,*a,**k):pass
        def score_many(self,path,questions,**kwargs):
            assert 'label' not in questions[0]
            if fail_second and questions[0]['id']=='1':raise RuntimeError('second group failed')
            return [{'candidates':[{'label':'A','text':'yes','raw_logit':1.0},{'label':'B','text':'no','raw_logit':0.0}]}]
    monkeypatch.setattr(mjev.hf,'HFMJevEngine',FakeEngine)
    out=tmp_path/'out'
    if fail_second:
        with pytest.raises(RuntimeError,match='second group'):asyncio.run(run(manifest,ROOT/'configs/qwen3_vl_4b_hf.json',out))
    else:asyncio.run(run(manifest,ROOT/'configs/qwen3_vl_4b_hf.json',out))
    assert json.loads((out/'groups/000000.json').read_text())['status']=='complete'
    assert json.loads((out/'progress.json').read_text())['completed_groups']==(1 if fail_second else 2)
    assert len((out/'predictions.jsonl').read_text().splitlines())==(1 if fail_second else 2)
    assert (out/'_SUCCESS').exists()!=fail_second
    assert (out/'ERROR.json').exists()==fail_second


def test_media_audit_revokes_success_before_manifest_validation(tmp_path):
    from mjev.benchmark.audit import audit
    (tmp_path/'_SUCCESS').touch()
    (tmp_path/'selection.json').write_text('invalid JSON')
    with pytest.raises(ValueError):audit(tmp_path)
    assert not (tmp_path/'_SUCCESS').exists()
