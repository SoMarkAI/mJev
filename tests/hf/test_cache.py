"""Native tiny Thinker cache parity and isolation; no pretrained performance claim."""
from copy import deepcopy
from threading import RLock
import pytest
import torch
from test_forward import model
from mjev.hf import HFMJevEngine,HFContext


def setup(model,kind='text'):
    engine=HFMJevEngine.__new__(HFMJevEngine);engine.model=model;engine.lock=RLock()
    extras={}
    if kind in ('image','video'):
        prefix=[1,100,105 if kind=='image' else 106,101,2]
        extras={'pixel_values' if kind=='image' else 'pixel_values_videos':torch.randn(4,3*2*16*16),
                'image_grid_thw' if kind=='image' else 'video_grid_thw':torch.tensor([[1,2,2]])}
        if kind=='video':extras['video_second_per_grid']=torch.tensor([1.])
    elif kind=='audio':
        features=torch.randn(1,128,100);mask=torch.ones(1,100,dtype=torch.long)
        with torch.inference_mode():n=model.get_audio_features(features,mask,return_dict=True).last_hidden_state.shape[0]
        prefix=[1,102]+[104]*n+[103,2]
        extras={'input_features':features,'feature_attention_mask':mask}
    else:prefix=[1,2,3]
    n=len(prefix);ids=torch.tensor([prefix+[4,5,6,7,8,9,10]])
    batch={'input_ids':ids,'attention_mask':torch.ones_like(ids),**extras}
    spec={'prefix_length':n,'spans':[[n+1,n+3],[n+3,n+5]],'label_ids':[10,11],
          'labels':['A','B'],'candidate_texts':['yes','no'],'prompt_tokens':ids.shape[1]}
    past,positions=engine._prefill(batch,n)
    cache=HFContext(engine,None,kind,'',None,None,deepcopy(batch),past,positions,n)
    return engine,batch,spec,cache


def logits(result):return torch.tensor([r['raw_logit'] for r in result['candidates']])


@pytest.mark.parametrize('kind',['text','image','video','audio'])
@pytest.mark.parametrize('mode',['causal','isolated'])
def test_cached_multimodal_parity_and_no_reencoding(model,kind,mode):
    engine,batch,spec,cache=setup(model,kind)
    baseline=engine.score_prepared(batch,spec,mode=mode)
    def fail(*a,**k):raise AssertionError('Media encoder called after prefill')
    model.get_audio_features=fail;model.get_image_features=fail;model.get_video_features=fail
    result=engine.score_prepared(batch,spec,mode=mode,cache=cache)
    torch.testing.assert_close(logits(baseline),logits(result),atol=2e-6,rtol=1e-5)
    assert result['num_cached_tokens']==cache.prefix_length
    assert abs(result['probability_sum']-1)<1e-6


@pytest.mark.parametrize('mode',['causal','isolated'])
def test_order_branch_isolation_and_native_rope_state(model,mode):
    engine,batch,spec,cache=setup(model,'video')
    before=[(l.keys.clone(),l.values.clone()) for l in cache.past_key_values.layers]
    other=deepcopy(batch);other['input_ids'][0,cache.prefix_length:]+=13
    first=engine.score_prepared(batch,spec,mode=mode,cache=cache)
    second=engine.score_prepared(other,spec,mode=mode,cache=cache)
    model.rope_deltas=torch.tensor([[9876]])
    second_again=engine.score_prepared(other,spec,mode=mode,cache=cache)
    first_again=engine.score_prepared(batch,spec,mode=mode,cache=cache)
    torch.testing.assert_close(logits(first),logits(first_again),atol=0,rtol=0)
    torch.testing.assert_close(logits(second),logits(second_again),atol=0,rtol=0)
    for l,(k,v) in zip(cache.past_key_values.layers,before):
        assert torch.equal(l.keys,k) and torch.equal(l.values,v)
    assert cache.past_key_values.get_seq_length()==cache.prefix_length


def test_cached_mask_intervention_and_causal_control(model):
    engine,batch,spec,cache=setup(model)
    changed=deepcopy(batch);a,b=spec['spans'][0];changed['input_ids'][0,a:b]+=20
    states=[]
    h=model.model.register_forward_hook(lambda m,a,o:states.append(o.last_hidden_state.clone()))
    try:
        for mode in ('isolated','causal'):
            for data in (batch,changed):engine.score_prepared(data,spec,mode=mode,cache=cache)
    finally:h.remove()
    a,b=spec['spans'][1];a-=cache.prefix_length;b-=cache.prefix_length
    torch.testing.assert_close(states[0][:,a:b],states[1][:,a:b],atol=1e-6,rtol=1e-5)
    assert (states[2][:,a:b]-states[3][:,a:b]).abs().max()>1e-4
    assert not torch.equal(states[0][:,-1],states[1][:,-1])


def test_reject_stale_media_prefix_owner(model):
    engine,batch,spec,cache=setup(model,'video')
    bad=deepcopy(batch);bad['pixel_values_videos'][0,0]+=1
    with pytest.raises(ValueError,match='Media metadata'):engine.score_prepared(bad,spec,cache=cache)
    bad=deepcopy(batch);bad['input_ids'][0,0]+=1
    with pytest.raises(ValueError,match='Token prefix'):engine.score_prepared(bad,spec,cache=cache)
    cache.owner=object()
    with pytest.raises(ValueError,match='another engine'):engine.score_prepared(batch,spec,cache=cache)


def test_public_interfaces_reuse_one_prefill(model,monkeypatch):
    engine,batch,spec,_=setup(model)
    calls=[]
    def prepare(media,question,candidates,**kwargs):
        data=deepcopy(batch);data['input_ids'][0,spec['prefix_length']]=12 if question=='q2' else 4
        return data,deepcopy(spec),None
    engine.prepare=prepare
    h=model.model.register_forward_pre_hook(lambda m,a,k:calls.append(k['inputs_embeds'].shape[1]),with_kwargs=True)
    try:
        context=engine.prepare_context('unused')
        out=engine.score_questions(context,[{'question':'q1','candidates':['yes','no']},{'question':'q2','candidates':['yes','no']}],mode='isolated')
    finally:h.remove()
    assert calls==[spec['prefix_length'],7,7]
    assert len(out)==2 and all(r['num_cached_tokens']==spec['prefix_length'] for r in out)


@pytest.mark.parametrize('mode',['causal','isolated'])
def test_variable_suffix_and_candidate_permutation_parity(model,mode):
    engine,batch,spec,cache=setup(model)
    other=deepcopy(batch);n=cache.prefix_length
    other['input_ids']=torch.cat((batch['input_ids'][:,:n],torch.tensor([[20,21,22]]),batch['input_ids'][:,n:]),dim=1)
    other['attention_mask']=torch.ones_like(other['input_ids'])
    longer=deepcopy(spec);longer['spans']=[[a+3,b+3] for a,b in spec['spans']];longer['prompt_tokens']+=3
    # Swap candidate text-token blocks; compare with the same permutation uncached.
    a,b=longer['spans'][0];c,d=longer['spans'][1]
    other['input_ids'][:,a:b],other['input_ids'][:,c:d]=other['input_ids'][:,c:d].clone(),other['input_ids'][:,a:b].clone()
    cached=engine.score_prepared(other,longer,mode=mode,cache=cache)
    uncached=engine.score_prepared(other,longer,mode=mode)
    torch.testing.assert_close(logits(cached),logits(uncached),atol=2e-6,rtol=1e-5)
