from copy import deepcopy
import pytest,torch
from test_forward import model
from test_cache import setup,logits
from mjev.hf import HFContext
from mjev.hf_batch import score_batch

@pytest.mark.parametrize('kind',['text','image','audio','video'])
@pytest.mark.parametrize('mode',['causal','isolated'])
def test_stable_weights_batch_and_cache(model,kind,mode):
    engine,b,s,_=setup(model,kind)
    model.to(dtype=torch.bfloat16);engine._numerics_mode='stable'
    before={n:p.clone() for n,p in model.named_parameters()}
    past,pos=engine._prefill(b,s['prefix_length'])
    cache=HFContext(engine,None,kind,'',None,None,deepcopy(b),past,pos,s['prefix_length'])
    b2=deepcopy(b);b2['input_ids']=torch.cat([b2['input_ids'],torch.tensor([[11,12]])],1)
    b2['attention_mask']=torch.ones_like(b2['input_ids']);s2=deepcopy(s);s2['prompt_tokens']+=2
    ps=[(b,s),(b2,s2)]
    ref=[engine.score_prepared(x,y,mode=mode,projection='full') for x,y in ps]
    for c in [None,cache]:
        values,frames=score_batch(engine,ps,mode=mode,cache=c)
        assert frames[0]['batch_size']==2
        for a,z in zip(values,ref):torch.testing.assert_close(logits(a),logits(z),atol=3e-5,rtol=1e-5)
    for n,p in model.named_parameters():assert torch.equal(p,before[n]) and p.dtype==torch.bfloat16
    assert cache.past_key_values.get_seq_length()==s['prefix_length']


def test_stable_context_restores_after_failure(model):
    from mjev.hf_numerics import numerical_context
    old=torch.nn.functional.linear;old_experts=model.config.text_config._experts_implementation
    oldtf=torch.backends.cuda.matmul.allow_tf32
    with pytest.raises(RuntimeError,match='deliberate'):
        with numerical_context(model,'stable'):raise RuntimeError('deliberate')
    assert torch.nn.functional.linear is old
    assert model.config.text_config._experts_implementation==old_experts
    assert torch.backends.cuda.matmul.allow_tf32==oldtf
    assert not model.model._forward_pre_hooks


def test_public_api_actual_batches_and_ids(model,monkeypatch):
    engine,b,s,_=setup(model);engine._numerics_mode='stable'
    def prepare(media,question,candidates,**kw):return deepcopy(b),deepcopy(s),None
    monkeypatch.setattr(engine,'prepare',prepare)
    frames=[]
    h=model.model.register_forward_pre_hook(lambda m,a,k:frames.append(k['inputs_embeds'].shape[0]),with_kwargs=True)
    try:
        qs=[dict(id=str(i),question='q',candidates=['yes','no']) for i in range(5)]
        result=engine.score_many('unused',qs,batch_size=3)
        assert frames==[3,2] and [r['id'] for r in result]==list(map(str,range(5)))
        frames.clear();result=engine.score_many('unused',qs,batch_size=3,use_prefix_cache=True)
        assert frames==[1,3,2] and all(r['num_cached_tokens']==s['prefix_length'] for r in result)
    finally:h.remove()


@pytest.mark.parametrize('mode',['causal','isolated'])
def test_selected_batch_honors_ragged_label_sets(model,mode):
    engine,b,s,_=setup(model)
    model.to(dtype=torch.bfloat16);engine._numerics_mode='stable'
    t=deepcopy(s);n=s['prefix_length']
    t.update(label_ids=[12,13,14],labels=['A','B','C'],candidate_texts=['x','y','z'],
             spans=[[n+1,n+2],[n+2,n+3],[n+3,n+5]])
    ps=[(b,s),(deepcopy(b),t)]
    ref=[engine.score_prepared(x,y,mode=mode,projection='selected') for x,y in ps]
    past,pos=engine._prefill(b,n)
    cache=HFContext(engine,None,'text','',None,None,deepcopy(b),past,pos,n)
    for c in (None,cache):
        values,_=score_batch(engine,ps,mode=mode,projection='selected',cache=c)
        assert [len(v['candidates']) for v in values]==[2,3]
        for a,z in zip(values,ref):
            assert a['projection']=='selected' and a['backend']=='transformers'
            torch.testing.assert_close(logits(a),logits(z),atol=3e-5,rtol=1e-5)
