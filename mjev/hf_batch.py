"""Real batched forward for independently scored questions sharing media.

Each row is an independent question about the same media. Right padding is
masked, original per-question MRoPE positions are retained, and the last *real*
token is scored. Shared prefix KV is cloned and repeated across batch rows;
this saves prefix compute but is not zero-copy Tree-KV. No thread concurrency.
"""
from copy import deepcopy
from unittest.mock import patch
import torch
from mjev.hf import isolated_mask
from mjev.decision import choose
from .families import text_model, family


def native_positions(engine, batch):
    from .families import positions
    return positions(engine.model,batch)


def score_batch(engine, prepared, *, mode='causal', cache=None, projection='full'):
    if not prepared:raise ValueError('Empty batch')
    if projection not in ('selected','full'):raise ValueError('Unknown projection')
    if mode not in ('causal','isolated'):raise ValueError(mode)
    with engine.lock,engine.numerical_context(),torch.inference_mode():
        batches,specs=zip(*prepared)
        prefix=cache.prefix_length if cache is not None else 0
        if cache is not None:
            # Retain all existing prefix/media/position validation. Release each
            # temporary clone immediately; the actual batch has one KV clone.
            positions=[]
            for batch,spec in prepared:
                validated=engine._cached_inputs(batch,spec,cache)
                positions.append(validated['position_ids'])
                del validated
            past=deepcopy(cache.past_key_values)
            past.batch_repeat_interleave(len(prepared))
        else:
            positions=[native_positions(engine,b) for b in batches]
            past=None
        lengths=[b['input_ids'].shape[1]-prefix for b in batches]
        width=max(lengths);device=engine.model.get_input_embeddings().weight.device
        ids=torch.zeros((len(batches),width),dtype=torch.long,device=device)
        valid=torch.zeros((len(batches),prefix+width),dtype=torch.long,device=device)
        # Native video MRoPE can contain fractional temporal coordinates.
        # Preserve its dtype; casting to long silently changes the model input.
        if any(p.dtype!=positions[0].dtype for p in positions):
            raise ValueError('Mixed position dtypes are unsupported')
        pos=torch.zeros((3,len(batches),width),dtype=positions[0].dtype,device=device)
        for i,(b,n,p) in enumerate(zip(batches,lengths,positions)):
            ids[i,:n]=b['input_ids'][0,prefix:].to(device)
            valid[i,:prefix+n]=1
            pos[:,i,:n]=p[:,0,:]
        packed=dict(input_ids=ids,attention_mask=valid,position_ids=pos)
        if past is not None:packed['past_key_values']=past
        else:
            if 'mm_token_type_ids' in batches[0]:
                types=torch.zeros_like(ids)
                for i,(b,n) in enumerate(zip(batches,lengths)):types[i,:n]=b['mm_token_type_ids'][0].to(device)
                packed['mm_token_type_ids']=types
            keys=set(batches[0])-{'input_ids','attention_mask','mm_token_type_ids'}
            if any(set(b)-{'input_ids','attention_mask','mm_token_type_ids'}!=keys for b in batches):
                raise ValueError('Mixed media structure is unsupported')
            for key in keys:
                vals=[b[key] for b in batches]
                if isinstance(vals[0],torch.Tensor):packed[key]=torch.cat(vals,dim=0)
                else:
                    if any(v!=vals[0] for v in vals):raise ValueError('Mixed media metadata')
                    packed[key]=vals[0]
            packed=engine._move(packed)
        frames=[]
        def before(module,args,kwargs):
            embeddings=kwargs['inputs_embeds']
            frames.append({'batch_size':embeddings.shape[0],
                           'query_tokens':embeddings.shape[1],
                           'valid_query_lengths':lengths,'prefix_tokens':prefix})
            if mode=='isolated':
                mask=torch.full((len(batches),1,width,prefix+width),float('-inf'),
                                dtype=embeddings.dtype,device=embeddings.device)
                for i,(n,s) in enumerate(zip(lengths,specs)):
                    mask[i:i+1,:,:n,:prefix+n]=isolated_mask(prefix+n,s['spans'],
                        embeddings.device,embeddings.dtype,query_start=prefix)
                    # Unused padding rows must not introduce NaNs.
                    mask[i,0,n:,0]=0
                kwargs['attention_mask']=mask
            return args,kwargs
        head=engine.model.lm_head;original=head.forward
        def readout(hidden):
            row=torch.arange(len(lengths),device=hidden.device)
            last=torch.tensor(lengths,device=hidden.device)-1
            last_hidden=hidden[row,last].unsqueeze(1)
            if projection=='full':return original(last_hidden)
            width=max(len(s['label_ids']) for s in specs);values=[]
            for h,s in zip(last_hidden,specs):
                ids=torch.tensor(s['label_ids'],device=head.weight.device)
                weight=head.weight.index_select(0,ids)
                bias=head.bias.index_select(0,ids) if getattr(head,'bias',None) is not None else None
                value=torch.nn.functional.linear(h.to(weight.device).float(),weight.float(),
                    None if bias is None else bias.float())
                values.append(torch.nn.functional.pad(value,(0,width-len(s['label_ids']))))
            return torch.stack(values)

        handle=text_model(engine.model).register_forward_pre_hook(before,with_kwargs=True)
        try:
            with patch.object(head,'forward',readout):
                logits=engine.model(**packed,use_cache=past is not None,
                                    return_dict=True).logits[:,0].float()
        finally:handle.remove()
        if len(frames)!=1 or frames[0]['batch_size']!=len(prepared):
            raise RuntimeError('Expected one real batched model forward')
        results=[]
        for row,spec in zip(logits,specs):
            selected=row[spec['label_ids']] if projection=='full' else row[:len(spec['label_ids'])]
            if not torch.isfinite(selected).all():raise ValueError('Nonfinite logits')
            probabilities=torch.softmax(selected,dim=-1)
            candidates=[dict(label=l,text=t,token_id=tid,raw_logit=float(v),probability=float(p))
                for l,t,tid,v,p in zip(spec['labels'],spec['candidate_texts'],
                                     spec['label_ids'],selected,probabilities)]
            results.append(dict(backend='transformers',model_family=family(engine.model.config),projection=projection,mode=mode,candidates=candidates,decision=choose(candidates),
                probability_sum=sum(c['probability'] for c in candidates),
                prompt_tokens=spec['prompt_tokens'],num_cached_tokens=prefix,
                implementation='hf_real_batch',batch_size=len(prepared),numerics=engine.numerics))
        if cache is not None and cache.past_key_values.get_seq_length()!=prefix:
            raise RuntimeError('Base KV cache mutated')
        return results,frames
