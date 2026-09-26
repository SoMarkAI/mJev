"""Regression for already-expanded native audio/video placeholders."""
from types import SimpleNamespace as N
import pytest,torch
pytest.importorskip("vllm", reason="requires optional pinned vLLM environment")
from mjev.model import MJevProcessor

def processor():
    p=object.__new__(MJevProcessor)
    p.info=N(get_hf_config=lambda:N(vision_start_token_id=100,vision_end_token_id=101,video_token_id=106,audio_token_id=104))
    p._validate_mm_kwargs=lambda *args:None
    return p

def test_preserves_native_interleaving_and_disjoint_feature_masks():
    p=processor();ids=[5,100,102,106,104,106,104,103,101,6]
    items=N(get_all_counts=lambda:{'audio':1,'video':1})
    kwargs={'video':[{'use_audio_in_video':N(data=torch.tensor(True))}]}
    out,regions=p._maybe_apply_prompt_updates(items,ids,kwargs,{},True)
    assert out is ids
    v,a=regions['video'][0],regions['audio'][0]
    assert v.start_idx==a.start_idx==2
    assert v.tokens==a.tokens==ids[2:8]
    assert v.is_embed.tolist()==[False,True,False,True,False,False]
    assert a.is_embed.tolist()==[False,False,True,False,True,False]
    assert not (v.is_embed&a.is_embed).any()

@pytest.mark.parametrize('ids',[[5,100,102,106,103,101,6],[5,100,106,101,100,104,101,6]])
def test_rejects_missing_audio_or_ambiguous_media(ids):
    p=processor();items=N(get_all_counts=lambda:{'audio':1,'video':1})
    with pytest.raises(ValueError):p._maybe_apply_prompt_updates(items,ids,{'video':[{'use_audio_in_video':N(data=True)}]}, {},True)
