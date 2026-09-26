"""Transformers-only Omni Thinker / Qwen3-VL scoring. No generation, vLLM, or weight updates."""
from contextlib import contextmanager
from threading import RLock
from unittest.mock import patch
from copy import deepcopy
from dataclasses import dataclass
import torch
import warnings
from .prompt import PromptBuilder
from .decision import choose
from .inputs import normalize_questions, normalize_question
from .families import family, detect, text_model, multimodal_model, positions, check_modality, VL

_LOADING_LOCK = RLock()


def isolated_mask(length, spans, device, dtype, query_start=0):
    if not spans or spans[-1][1]>=length:
        raise ValueError('Candidate spans must leave a decision suffix')
    prev=spans[0][0]
    for start,end in spans:
        if start!=prev or not 0<=start<end<=length:raise ValueError('Invalid candidate spans')
        prev=end
    if not 0<=query_start<=spans[0][0]:raise ValueError('Invalid cached query offset')
    keys=torch.arange(length,device=device)
    queries=torch.arange(query_start,length,device=device)
    visible=keys[None,:]<=queries[:,None]
    for start,end in spans:
        visible[start-query_start:end-query_start] &= (keys<spans[0][0]) | ((keys>=start)&(keys<end))
    mask=torch.zeros((length-query_start,length),device=device,dtype=dtype)
    return mask.masked_fill(~visible,float('-inf'))[None,None]


@contextmanager
def scoring_hooks(model, spans, label_ids, mode, projection='selected', prefix_length=0):
    """Install after native MRoPE construction; remove even if forward fails."""
    seen=[]
    def before_text(module,args,kwargs):
        if kwargs.get('position_ids') is None:raise RuntimeError('Native MRoPE positions missing')
        embeddings=kwargs['inputs_embeds']
        if embeddings.shape[0]!=1:raise ValueError('One question per HF forward is required')
        if mode=='isolated':
            kwargs['attention_mask']=isolated_mask(embeddings.shape[1]+prefix_length,spans,embeddings.device,embeddings.dtype,query_start=prefix_length)
        seen.append(True)
        return args,kwargs
    handle=text_model(model).register_forward_pre_hook(before_text,with_kwargs=True)
    head=model.lm_head;original=head.forward
    def readout(hidden):
        last=hidden[:,-1:,:]
        ids=torch.tensor(label_ids,device=head.weight.device)
        if projection=='full':return original(last).index_select(-1,ids)
        weight=head.weight.index_select(0,ids)
        bias=head.bias.index_select(0,ids) if getattr(head,'bias',None) is not None else None
        return torch.nn.functional.linear(last.to(weight.device).float(),weight.float(),None if bias is None else bias.float())
    try:
        with patch.object(head,'forward',readout):
            yield
        if len(seen)!=1:raise RuntimeError('Thinker text-model interception did not execute exactly once')
    finally:handle.remove()


@dataclass
class HFContext:
    """Engine-local immutable-by-convention prefix; release references to free KV memory.

    Treat fields as private implementation details. Each scoring call clones KV.
    """
    owner: object
    media: object
    modality: str
    context: str
    video_options: object
    decoded: object
    batch: dict
    past_key_values: object
    position_ids: torch.Tensor
    prefix_length: int


class HFMJevEngine:
    def __init__(self, model_path, *, device_map='auto', dtype='bfloat16', max_input_tokens=4000,
                 model=None, processor=None, max_memory=None, numerics='stable'):
        if numerics not in ('native','stable'):raise ValueError('Unknown HF numerical mode')
        self._numerics_mode=numerics
        import transformers
        if transformers.__version__!='5.13.1':raise RuntimeError('HF backend currently validated against Transformers 5.13.1')
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3VLForConditionalGeneration
        kind=family(model.config) if model is not None else detect(model_path)
        self.builder=PromptBuilder(model_path) if processor is None else PromptBuilder.__new__(PromptBuilder)
        if processor is not None:self.builder.processor=processor;self.builder.tokenizer=processor.tokenizer
        self.loading_info=None
        if model is not None:
            self.model=model
        else:
            if max_memory is None and numerics=='stable' and device_map=='auto' and torch.cuda.is_available():
                # Auto placement otherwise budgets only parameter storage and
                # can leave no room for FP32 activations/projection temporaries.
                max_memory={i:max(0,torch.cuda.mem_get_info(i)[0]-4*1024**3)
                            for i in range(torch.cuda.device_count())}
            # Keep the exact native class: weight converters are registered by class name.
            # A subclass silently bypasses conversion of the original unfused MoE weights.
            cls=Qwen3VLForConditionalGeneration if kind==VL else Qwen3OmniMoeThinkerForConditionalGeneration
            no_split=list(cls._no_split_modules)+(['Qwen3VLTextDecoderLayer'] if kind==VL else ['Qwen3OmniMoeThinkerTextDecoderLayer'])
            with _LOADING_LOCK,patch.object(cls,'_no_split_modules',no_split):
                self.model,info=cls.from_pretrained(model_path,dtype=getattr(torch,dtype),
                    device_map=device_map,max_memory=max_memory,attn_implementation='sdpa',
                    local_files_only=True,output_loading_info=True)
            unexpected=[k for k in info.get('unexpected_keys',[]) if kind==VL or not k.startswith(('talker.','code2wav.'))]
            if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs') or unexpected:
                raise RuntimeError(f'Incomplete Thinker checkpoint load: missing={info.get("missing_keys")}, '
                                   f'mismatched={info.get("mismatched_keys")}, unexpected={unexpected}, errors={info.get("error_msgs")}')
            self.loading_info={k:list(v) if isinstance(v,set) else v for k,v in info.items()}
            self.model._no_split_modules=no_split
        self.model.eval();self.lock=RLock();self.max_input_tokens=max_input_tokens
        if self.model.config.text_config._attn_implementation not in ('sdpa','eager'):
            raise ValueError('HF custom masks require SDPA or eager attention')

    @property
    def numerics(self):
        return getattr(self,'_numerics_mode','native')

    def numerical_context(self):
        from .hf_numerics import numerical_context
        return numerical_context(self.model,self.numerics)

    def prepare(self, media, question, candidates, *, modality='image', context='Inspect the supplied media carefully.', video_options=None, decoded=None):
        candidates = normalize_question({'question': question, 'candidates': candidates})['candidates']
        kind=family(self.model.config) if hasattr(self,'model') else detect(self.builder.model_path)
        check_modality(kind,modality)
        spec=self.builder.build(media,context,question,candidates,modality=modality,video_options=video_options)
        if kind==VL:
            from .vl_media import prepare_media
            if decoded is None:decoded=prepare_media(self.builder.processor,media,modality,video_options)
            batch=self.builder.processor(text=spec['text'],return_tensors='pt',padding=False,truncation=False,**decoded)
        else:
            from qwen_omni_utils import process_mm_info
            use_audio=modality=='audio_video'
            if decoded is None:
                decoded=process_mm_info(spec['messages'],use_audio_in_video=use_audio,return_video_kwargs=True,
                                        image_patch_size=self.builder.processor.image_processor.patch_size)
            audios,images,videos,kwargs=decoded;kwargs=dict(kwargs)
            if isinstance(kwargs.get('fps'),list):
                rates=kwargs.pop('fps')
                if len(rates)>1:raise ValueError('Only one video is supported per request')
                if rates:kwargs['fps']=float(rates[0])
            # qwen-omni-utils already sampled and resized the video using video_options.
            # Keep those explicit dimensions rather than resizing it a second time.
            if videos is not None:kwargs['videos_kwargs']={'do_resize':False}
            if images is not None:kwargs['images_kwargs']={'do_resize':False}
            batch=self.builder.processor(text=spec['text'],audio=audios,images=images,videos=videos,
                use_audio_in_video=use_audio,return_tensors='pt',padding=False,truncation=False,**kwargs)
        ids=batch['input_ids'][0].tolist();delta=len(ids)-len(spec['raw_ids']);start=spec['spans'][0][0]
        if ids[start+delta:]!=spec['raw_ids'][start:]:raise ValueError('Processor changed the candidate/decision suffix')
        if len(ids)>self.max_input_tokens:raise ValueError(f'Full input {len(ids)} exceeds limit {self.max_input_tokens}; no truncation')
        spec['spans']=[[a+delta,b+delta] for a,b in spec['spans']];spec['prompt_tokens']=len(ids)
        spec['prefix_length']+=delta
        spec['candidate_texts']=candidates
        if kind!=VL:batch['use_audio_in_video']=use_audio
        return batch,spec,decoded

    def _move(self, batch):
        device=self.model.get_input_embeddings().weight.device
        moved={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        for name in ('input_features','pixel_values','pixel_values_videos'):
            if name in moved:
                owner=multimodal_model(self.model)
                tower=owner.audio_tower if name=='input_features' else owner.visual
                moved[name]=moved[name].to(dtype=next(tower.parameters()).dtype)
        return moved

    def _prefill(self, batch, prefix_length):
        if not 0 < prefix_length < batch['input_ids'].shape[1]:
            raise ValueError('Invalid common prefix length')
        if not torch.all(batch['attention_mask']==1):
            raise ValueError('Prefix caching requires one unpadded sequence')
        prefix=dict(batch)
        for name in ('input_ids','attention_mask','mm_token_type_ids'):
            if name not in prefix:continue
            prefix[name]=prefix[name][:,:prefix_length]
        positions=[]
        def capture(module,args,kwargs):
            positions.append(kwargs['position_ids'].detach().clone())
        with self.lock,self.numerical_context(),torch.inference_mode():
            handle=text_model(self.model).register_forward_pre_hook(capture,with_kwargs=True)
            try:
                # No vocabulary projection is needed while constructing the prefix.
                with patch.object(self.model.lm_head,'forward',lambda h:h[:,-1:,:0]):
                    output=self.model(**self._move(prefix),use_cache=True,return_dict=True)
            finally:
                handle.remove()
            if len(positions)!=1 or output.past_key_values.get_seq_length()!=prefix_length:
                raise RuntimeError('Prefix cache was not constructed correctly')
        return output.past_key_values,positions[0]

    def prepare_context(self,media,*,modality='image',context='Inspect the supplied media carefully.',video_options=None):
        """Encode media/public context once. No question tokens enter the cached prefix."""
        if self.numerics=='native':
            warnings.warn('Native BF16 scoring is not batch/cache invariant; use stable numerics when comparing cached and uncached scores.', RuntimeWarning, stacklevel=2)
        batch,spec,decoded=self.prepare(media,'Describe the media.' ,['Yes','No'],
            modality=modality,context=context,video_options=video_options)
        past,positions=self._prefill(batch,spec['prefix_length'])
        return HFContext(self,media,modality,context,deepcopy(video_options),decoded,
                         batch,past,positions,spec['prefix_length'])

    def _cached_inputs(self,batch,spec,cache):
        if cache.owner is not self:raise ValueError('Context belongs to another engine')
        n=cache.prefix_length
        if spec['prefix_length']!=n or not torch.equal(batch['input_ids'][:,:n],cache.batch['input_ids'][:,:n]):
            raise ValueError('Token prefix changed; rebuild context')
        # Token IDs alone do not identify media, sampling settings or temporal metadata.
        if 'mm_token_type_ids' in batch:
            types=batch['mm_token_type_ids']
            if not torch.equal(types[:,:n],cache.batch['mm_token_type_ids'][:,:n]) or torch.any(types[:,n:]!=0):
                raise ValueError('Media token types changed outside the shared prefix')
        excluded={'input_ids','attention_mask','mm_token_type_ids'}
        if set(batch)-excluded != set(cache.batch)-excluded:raise ValueError('Media metadata changed')
        for key in set(batch)-excluded:
            a,b=batch[key],cache.batch[key]
            same=torch.equal(a,b) if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) else a==b
            if not same:raise ValueError(f'Media metadata changed: {key}')
        if not torch.all(batch['attention_mask']==1):raise ValueError('Padding is unsupported for context reuse')
        if cache.past_key_values.get_seq_length()!=n:raise ValueError('Context KV was modified')
        # Native full-input MRoPE is inexpensive and avoids mutable model.rope_deltas.
        device=self.model.get_input_embeddings().weight.device
        def tensor(name):
            v=batch.get(name)
            return v.to(device) if isinstance(v,torch.Tensor) else v
        native_positions=positions(self.model,batch)
        if not torch.equal(native_positions[:,:,:n],cache.position_ids):raise ValueError('Native prefix positions changed')
        return dict(input_ids=tensor('input_ids')[:,n:],attention_mask=tensor('attention_mask'),
                    position_ids=native_positions[:,:,n:],past_key_values=deepcopy(cache.past_key_values))

    def score_prepared(self,batch,spec,*,mode='causal',projection='selected',cache=None):
        if mode not in ('causal','isolated'):raise ValueError('mode must be causal or isolated')
        if projection not in ('selected','full'):raise ValueError('Unknown projection')
        with self.lock,self.numerical_context(),torch.inference_mode():
            moved=self._move(batch) if cache is None else self._cached_inputs(batch,spec,cache)
            prefix_length=0 if cache is None else cache.prefix_length
            with scoring_hooks(self.model,spec['spans'],spec['label_ids'],mode,projection,prefix_length):
                output=self.model(**moved,use_cache=cache is not None,return_dict=True)
            logits=output.logits[0,-1].float()
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite label logits')
            probabilities=torch.softmax(logits,dim=-1)
            rows=[{'label':l,'text':t,'token_id':i,'raw_logit':float(v),'probability':float(p)} for l,t,i,v,p in
                  zip(spec['labels'],spec['candidate_texts'],spec['label_ids'],logits,probabilities)]
        return {'backend':'transformers','model_family':family(self.model.config),'mode':mode,'projection':projection,'numerics':self.numerics,'candidates':rows,
                'decision':choose(rows),'probability_sum':sum(r['probability'] for r in rows),
                'prompt_tokens':spec['prompt_tokens'],'num_cached_tokens':prefix_length,
                'cache_note':'Independent prefix KV clone; numerical profile: '+self.numerics if cache is not None else 'No KV reuse.'}

    def _score_question_batches(self,prepared,questions,*,mode,projection,batch_size,cache=None):
        if projection not in ('selected','full'):raise ValueError('Unknown projection')
        if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
            raise ValueError('batch_size must be a positive integer')
        from .hf_batch import score_batch
        results=[]
        for start in range(0,len(prepared),batch_size):
            chunk=prepared[start:start+batch_size]
            if batch_size==1:
                values=[self.score_prepared(*chunk[0],mode=mode,projection=projection,cache=cache)]
            else:values,_=score_batch(self,chunk,mode=mode,cache=cache,projection=projection)
            results.extend({'id':q.get('id'),**v} for q,v in zip(questions[start:start+batch_size],values))
        return results

    def score_many(self,media,questions,*,modality='image',context='Inspect the supplied media carefully.',mode='causal',video_options=None,projection='selected',use_prefix_cache=False,batch_size=1):
        questions = normalize_questions(questions)
        if use_prefix_cache:
            cache=self.prepare_context(media,modality=modality,context=context,video_options=video_options)
            return self.score_questions(cache,questions,mode=mode,projection=projection,batch_size=batch_size)
        decoded=None;prepared=[]
        for q in questions:
            candidates=q['candidates']
            batch,spec,decoded=self.prepare(media,q['question'],candidates,modality=modality,context=context,video_options=video_options,decoded=decoded)
            spec['candidate_texts']=candidates
            prepared.append((batch,spec))
        return self._score_question_batches(prepared,questions,mode=mode,projection=projection,batch_size=batch_size)

    def score_questions(self,cache,questions,*,mode='causal',projection='selected',batch_size=1):
        """Independent questions, optional real batches, immutable shared prefill."""
        questions = normalize_questions(questions)
        if cache.owner is not self:raise ValueError('Context belongs to another engine')
        prepared=[]
        for q in questions:
            candidates=q['candidates']
            batch,spec,_=self.prepare(cache.media,q['question'],candidates,modality=cache.modality,
                context=cache.context,video_options=cache.video_options,decoded=cache.decoded)
            prepared.append((batch,spec))
        return self._score_question_batches(prepared,questions,mode=mode,projection=projection,batch_size=batch_size,cache=cache)
