"""Tiny native-model witnesses, not full-weight performance claims."""
import sys
from pathlib import Path
from copy import deepcopy
import pytest,torch
from test_forward import model
from test_cache import setup,logits
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks/integrated'))
from hf_batch_probe import score_batch

@pytest.mark.parametrize('kind',['text','image','audio','video'])
@pytest.mark.parametrize('mode',['causal','isolated'])
def test_real_batch_padding_cache_and_branch_independence(model,kind,mode):
    engine,b,s,cache=setup(model,kind)
    b2=deepcopy(b);s2=deepcopy(s)
    b2['input_ids']=torch.cat([b2['input_ids'],torch.tensor([[12,13]])],dim=1)
    b2['attention_mask']=torch.ones_like(b2['input_ids']);s2['prompt_tokens']+=2
    end=s2['spans'][-1][1];s2['spans'].append([end,end+1])
    s2['label_ids'].append(12);s2['labels'].append('C');s2['candidate_texts'].append('unknown')
    prepared=[(b,s),(b2,s2)]
    baseline=[engine.score_prepared(x,y,mode=mode,projection='full') for x,y in prepared]
    for c in (None,cache):
        results,frames=score_batch(engine,prepared,mode=mode,cache=c)
        assert len(frames)==1 and frames[0]['batch_size']==2
        for actual,expected in zip(results,baseline):
            torch.testing.assert_close(logits(actual),logits(expected),atol=3e-6,rtol=1e-5)
        reversed_results,_=score_batch(engine,list(reversed(prepared)),mode=mode,cache=c)
        for actual,expected in zip(results,reversed(reversed_results)):
            torch.testing.assert_close(logits(actual),logits(expected),atol=3e-6,rtol=1e-5)
    assert cache.past_key_values.get_seq_length()==cache.prefix_length

@pytest.mark.parametrize('mode',['causal','isolated'])
def test_batch_preserves_fractional_native_positions(model,monkeypatch,mode):
    # The full Omni audio-video processor produces fractional temporal MRoPE.
    # Inject that native API contract into the tiny model (no media accuracy claim).
    native=model.get_rope_index
    def fractional(*args,**kwargs):
        positions,delta=native(*args,**kwargs)
        positions=positions.float()
        positions[0,:,2]=positions[0,:,2]+0.375
        return positions,delta
    monkeypatch.setattr(model,'get_rope_index',fractional)
    engine,b,s,cache=setup(model,'text')
    captured=[]
    h=model.model.register_forward_pre_hook(lambda m,a,k:captured.append(k['position_ids'].clone()),with_kwargs=True)
    try:
        baseline=engine.score_prepared(b,s,mode=mode,projection='full')
        results,_=score_batch(engine,[(b,s),(b,s)],mode=mode)
        expected=captured[0]
        torch.testing.assert_close(captured[1],expected.repeat(1,2,1),atol=0,rtol=0)
        for result in results:torch.testing.assert_close(logits(result),logits(baseline),atol=3e-6,rtol=1e-5)
        cached,_=score_batch(engine,[(b,s),(b,s)],mode=mode,cache=cache)
        for result in cached:torch.testing.assert_close(logits(result),logits(baseline),atol=3e-6,rtol=1e-5)
    finally:h.remove()
