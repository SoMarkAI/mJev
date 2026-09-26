"""Real tiny Thinker forwards, random weights; no pretrained accuracy claims."""
import torch,pytest
from transformers import Qwen3OmniMoeThinkerConfig,Qwen3OmniMoeThinkerForConditionalGeneration
from mjev.hf import isolated_mask,scoring_hooks

@pytest.fixture
def model():
    torch.manual_seed(19)
    c=Qwen3OmniMoeThinkerConfig(text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,moe_intermediate_size=32,num_experts=2,num_experts_per_tok=1,head_dim=8,rope_parameters={'rope_type':'default','rope_theta':10000.,'mrope_section':[1,1,2]}),audio_config=dict(encoder_layers=1,encoder_attention_heads=2,encoder_ffn_dim=32,d_model=16,output_dim=32,downsample_hidden_size=8),vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=32,deepstack_visual_indexes=[]))
    c.vision_start_token_id=100;c.vision_end_token_id=101;c.audio_start_token_id=102;c.audio_end_token_id=103;c.audio_token_id=104;c.image_token_id=105;c.video_token_id=106
    c._attn_implementation='sdpa'
    m=Qwen3OmniMoeThinkerForConditionalGeneration(c).eval()
    m.generate=lambda *a,**k: (_ for _ in ()).throw(AssertionError('generate forbidden'))
    return m

def run(m,ids,mode,projection='selected'):
    with torch.inference_mode(),scoring_hooks(m,[[3,5],[5,7]],[10,11],mode,projection):
        return m(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits

def test_selected_matches_full_last(model):
    ids=torch.tensor([[1,2,3,4,5,6,7,8,9]])
    for mode in ['causal','isolated']:
        torch.testing.assert_close(run(model,ids,mode),run(model,ids,mode,'full'),atol=1e-6,rtol=1e-5)
    assert not model.model._forward_pre_hooks

def test_real_hidden_state_isolation_and_causal_control(model):
    ids=torch.tensor([[1,2,3,4,5,6,7,8,9]]);changed=ids.clone();changed[0,3:5]=torch.tensor([21,22]);states=[]
    handle=model.model.register_forward_hook(lambda m,a,o:states.append(o.last_hidden_state.detach().clone()))
    try:
        run(model,ids,'isolated');run(model,changed,'isolated')
        torch.testing.assert_close(states[0][:,5:7],states[1][:,5:7],atol=1e-6,rtol=1e-5)
        assert not torch.equal(states[0][:,-1],states[1][:,-1])
        run(model,ids,'causal');run(model,changed,'causal')
        assert (states[2][:,5:7]-states[3][:,5:7]).abs().max()>1e-4
    finally:handle.remove()

def test_mask_and_cleanup(model):
    mask=isolated_mask(9,[[3,5],[5,7]],'cpu',torch.float32)
    assert torch.isneginf(mask[0,0,6,3]) and mask[0,0,8,3]==0
    original=model.lm_head.forward
    with pytest.raises(RuntimeError):
        with scoring_hooks(model,[[3,5],[5,7]],[10,11],'isolated'):raise RuntimeError('test')
    assert model.lm_head.forward==original and not model.model._forward_pre_hooks

@pytest.mark.parametrize('kind',['image','video','audio'])
def test_native_multimodal_forward(model,kind):
    kwargs={}
    if kind in ('image','video'):
        token=105 if kind=='image' else 106
        ids=torch.tensor([[1,100,token,101,2,3,4,5,6,7,8]])
        grid=torch.tensor([[1,2,2]])
        pixels=torch.randn(4,3*2*16*16)
        if kind=='image':kwargs=dict(pixel_values=pixels,image_grid_thw=grid)
        else:kwargs=dict(pixel_values_videos=pixels,video_grid_thw=grid,video_second_per_grid=torch.tensor([1.]))
        start=5
    else:
        features=torch.randn(1,128,100);mask=torch.ones(1,100,dtype=torch.long)
        with torch.inference_mode():n=model.get_audio_features(features,mask,return_dict=True).last_hidden_state.shape[0]
        ids=torch.tensor([[1,102]+[104]*n+[103,2,3,4,5,6,7,8]])
        kwargs=dict(input_features=features,feature_attention_mask=mask);start=4+n
    spans=[[start,start+2],[start+2,start+4]]
    with torch.inference_mode(),scoring_hooks(model,spans,[10,11],'isolated'):
        output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,**kwargs)
    assert output.logits.shape==(1,1,2) and torch.isfinite(output.logits).all()

def test_engine_output_probability_and_unchanged_weights(model):
    from threading import RLock
    from mjev.hf import HFMJevEngine
    engine=HFMJevEngine.__new__(HFMJevEngine);engine.model=model;engine.lock=RLock()
    ids=torch.tensor([[1,2,3,4,5,6,7,8,9]])
    batch={'input_ids':ids,'attention_mask':torch.ones_like(ids)}
    spec={'spans':[[3,5],[5,7]],'label_ids':[10,11],'labels':['A','B'],'candidate_texts':['yes','no'],'prompt_tokens':9}
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=engine.score_prepared(batch,spec,mode='isolated')
    assert abs(result['probability_sum']-1)<1e-6
    assert result['decision']['label'] in ['A','B']
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])

def test_official_nested_checkpoint_prefix_loads_thinker_only(model,tmp_path):
    from transformers import Qwen3OmniMoeConfig
    from safetensors.torch import save_file
    Qwen3OmniMoeConfig(thinker_config=model.config.to_dict()).save_pretrained(tmp_path)
    # Emulate the real checkpoint's unfused per-expert layout, not the runtime's
    # packed tensors; subclass loading can otherwise falsely pass this test.
    weights={}
    for k,v in model.state_dict().items():
        if k.endswith('.experts.gate_up_proj'):
            prefix=k.removesuffix('gate_up_proj')
            for i,pair in enumerate(v):
                gate,up=pair.chunk(2,dim=0)
                weights[f'thinker.{prefix}{i}.gate_proj.weight']=gate.contiguous()
                weights[f'thinker.{prefix}{i}.up_proj.weight']=up.contiguous()
        elif k.endswith('.experts.down_proj'):
            prefix=k.removesuffix('down_proj')
            for i,down in enumerate(v):weights[f'thinker.{prefix}{i}.down_proj.weight']=down.contiguous()
        else:weights['thinker.'+k]=v.contiguous()
    save_file(weights,str(tmp_path/'model.safetensors'))
    loaded=Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(tmp_path,attn_implementation='sdpa',local_files_only=True)
    assert not hasattr(loaded,'talker')
    from types import SimpleNamespace
    from mjev.hf import HFMJevEngine
    engine=HFMJevEngine(str(tmp_path),device_map='cpu',dtype='float32',processor=SimpleNamespace(tokenizer=None))
    assert 'Qwen3OmniMoeThinkerTextDecoderLayer' in engine.model._no_split_modules
    for k,v in model.state_dict().items():assert torch.equal(v,engine.model.state_dict()[k])
    for k,v in model.state_dict().items():assert torch.equal(v,loaded.state_dict()[k])
    assert not engine.loading_info['missing_keys']
    del weights['thinker.model.embed_tokens.weight']
    save_file(weights,str(tmp_path/'model.safetensors'))
    with pytest.raises(RuntimeError,match='Incomplete Thinker checkpoint'):
        HFMJevEngine(str(tmp_path),device_map='cpu',dtype='float32',processor=SimpleNamespace(tokenizer=None))

def test_bfloat16_audio_engine_casts_features_not_ids(model):
    from threading import RLock
    from mjev.hf import HFMJevEngine
    model=model.to(torch.bfloat16)
    features=torch.randn(1,128,100);mask=torch.ones(1,100,dtype=torch.long)
    with torch.inference_mode():n=model.get_audio_features(features.bfloat16(),mask,return_dict=True).last_hidden_state.shape[0]
    ids=torch.tensor([[1,102]+[104]*n+[103,2,3,4,5,6,7,8]])
    engine=HFMJevEngine.__new__(HFMJevEngine);engine.model=model;engine.lock=RLock();start=4+n
    spec={'spans':[[start,start+2],[start+2,start+4]],'label_ids':[10,11],'labels':['A','B'],'candidate_texts':['yes','no'],'prompt_tokens':ids.shape[1]}
    r=engine.score_prepared({'input_ids':ids,'attention_mask':torch.ones_like(ids),'input_features':features,'feature_attention_mask':mask},spec,mode='isolated')
    assert abs(r['probability_sum']-1)<1e-6
