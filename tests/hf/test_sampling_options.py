"""Sampling options reach the actual Omni media-processor call boundary."""
import sys
from types import SimpleNamespace
import torch
from mjev.hf import HFMJevEngine
from mjev.prompt import PromptBuilder
import pytest


@pytest.mark.parametrize('modality',['video','audio_video'])
def test_sampling_and_audio_flag_reach_processor(monkeypatch,modality):
    calls=[]
    def process_mm_info(messages,**kwargs):
        calls.append((messages,kwargs))
        return None,None,None,{}
    monkeypatch.setitem(sys.modules,'qwen_omni_utils',SimpleNamespace(process_mm_info=process_mm_info))
    class Tokenizer:
        all_special_tokens=[]
        def encode(self,text,**kwargs):return list(map(ord,text))
        def __call__(self,text,**kwargs):
            return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}
    class Processor:
        tokenizer=Tokenizer()
        image_processor=SimpleNamespace(patch_size=14)
        def apply_chat_template(self,messages,**kwargs):return messages[0]['content'][-1]['text']
        def __call__(self,text,**kwargs):
            assert kwargs['use_audio_in_video']==(modality=='audio_video')
            ids=torch.tensor([self.tokenizer.encode(text)])
            return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
    engine=HFMJevEngine.__new__(HFMJevEngine)
    engine.model=SimpleNamespace(config=SimpleNamespace(model_type='qwen3_omni_moe'))
    engine.builder=PromptBuilder.__new__(PromptBuilder)
    engine.builder.processor=Processor();engine.builder.tokenizer=engine.builder.processor.tokenizer
    engine.max_input_tokens=4000
    options={'fps':2,'max_frames':8,'min_pixels':3136,'max_pixels':50176}
    engine.prepare('clip.mp4','What happened?',['yes','no'],modality=modality,video_options=options)
    media=calls[0][0][0]['content'][0]
    assert all(media[k]==v for k,v in options.items())
    assert calls[0][1]['use_audio_in_video']==(modality=='audio_video')
