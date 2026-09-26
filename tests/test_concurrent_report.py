import json,sys,copy
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks/integrated'))
from report_concurrent_cache import summarize

def fixture(root):
    qs=[{'id':str(i),'candidates':['yes','no']} for i in range(3)]
    m=dict(groups=[dict(media_path='media',questions=qs)],modes=['causal'],repeats=1,
           logit_atol=.1,probability_atol=.02,scope='Synthetic unit fixture',timing_boundary='test')
    (root/'manifest.json').write_text(json.dumps(m))
    value=dict(candidates=[dict(text='yes',raw_logit=0.,probability=.5),dict(text='no',raw_logit=0.,probability=.5)],
               decision=dict(label='A'),prompt_token_ids_sha256='hash',probability_sum=1.,num_cached_tokens=0)
    variants={}
    for name in ['serial_no_cache','parallel_no_cache','parallel_warm_cache']:
        values=[copy.deepcopy(value) for _ in qs]
        if name.endswith('warm_cache'):
            for v in values:v['num_cached_tokens']=16
        variants[name]=dict(values=values,group_seconds=.3,prime_seconds=.1,cold_total_seconds=.4,
                            max_scheduled_questions=3,forward_frames=[dict(batch_size=3)])
    row=dict(media_path='media',mode='causal',repeat=0,question_ids=['0','1','2'],
             prompt_hashes=['hash']*3,variants=variants,comparisons={})
    check=dict(same_answer=True,max_abs_logit_error=0.,max_abs_probability_error=0.,passed=True)
    for key in ['parallel_no_cache','parallel_warm_cache','cache_effect_at_parallel']:
        row['comparisons'][key]=[copy.deepcopy(check) for _ in qs]
    for b in ('vllm','hf'):
        d=root/b;d.mkdir();(d/'_COMPLETE').touch();(d/'raw.jsonl').write_text(json.dumps(row)+'\n')
    return row

def test_valid_small_report(tmp_path):
    fixture(tmp_path);r=summarize(tmp_path)
    assert r['configuration_evidence']['vllm']['status']=='unverified_historical_configuration'
    assert r['configuration_evidence']['vllm']['model_revision'] is None
    assert r['questions']==3
    assert r['backends']['hf']['strategies']['causal|parallel_warm_cache']['cache_hit_questions']==3

def test_refuses_fabricated_pass(tmp_path):
    row=fixture(tmp_path);row['variants']['parallel_warm_cache']['values'][0]['candidates'][0]['raw_logit']=1.
    (tmp_path/'hf/raw.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='Stored comparison'):summarize(tmp_path)

def test_refuses_missing_rows_despite_complete_marker(tmp_path):
    fixture(tmp_path);(tmp_path/'hf/raw.jsonl').write_text('')
    with pytest.raises(ValueError,match='Coverage mismatch'):summarize(tmp_path)


def test_failed_report_rerun_revokes_completion(tmp_path,monkeypatch):
    import report_concurrent_cache
    fixture(tmp_path)
    monkeypatch.setattr(sys,'argv',['report','--out',str(tmp_path)])
    report_concurrent_cache.main()
    assert (tmp_path/'_REPORT_COMPLETE').exists()
    (tmp_path/'hf/raw.jsonl').write_text('')
    with pytest.raises(ValueError,match='Coverage mismatch'):report_concurrent_cache.main()
    assert not (tmp_path/'_REPORT_COMPLETE').exists()
