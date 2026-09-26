"""The cache probe must execute the media sampling settings it records."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import validate_model
from mjev.benchmark.prepare import write_jsonl


@pytest.mark.parametrize('modality',['video','audio_video'])
def test_probe_passes_sampling_settings_to_scoring(tmp_path,monkeypatch,modality):
    import huggingface_hub
    import mjev.hf
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/qwen3_vl_4b_hf.json').read_text())
    config['model_id']='Qwen/Qwen3-Omni-30B-A3B-Instruct'
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'clip.mp4').write_bytes(b'unit fixture')
    row=dict(id='q',dataset='unit',source={},task='unit',modality=modality,media_path='clip.mp4',
             media={'type':modality,'path':'clip.mp4'},question='Q?',candidates={'A':'yes','B':'no'},label='A')
    write_jsonl(tmp_path/'manifest.jsonl',[row])
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**k:str(tmp_path))
    monkeypatch.setattr(validate_model,'capture',lambda *a,**k:{})
    calls=[]
    class FakeEngine:
        def __init__(self,*a,**k):pass
        def score_many(self,path,questions,**kwargs):
            calls.append(kwargs)
            assert kwargs['video_options']==config['video_options'] and kwargs['modality']==modality
            return [dict(candidates=[dict(label='A',text=q['candidates'][0],raw_logit=0.,probability=.5),
                                     dict(label='B',text=q['candidates'][1],raw_logit=0.,probability=.5)],
                         decision={'label':'A'},probability_sum=1.,num_cached_tokens=8) for q in questions]
    monkeypatch.setattr(mjev.hf,'HFMJevEngine',FakeEngine)
    args=SimpleNamespace(config=tmp_path/'config.json',manifest=tmp_path/'manifest.jsonl',out=tmp_path/'out',repeats=2,atol=.0001)
    asyncio.run(validate_model.run(args))
    assert len(calls)==22 and (args.out/'_SUCCESS').exists()
