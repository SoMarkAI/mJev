"""Full-weight shared-media cache/batch probe. No accuracy or speedup assumptions."""
import argparse
import asyncio
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time
import torch
from mjev.benchmark.dataset import MJevDataset
from mjev.benchmark.grouped import config_load
from mjev.benchmark.provenance import capture, sha256, write


def compare(reference, actual):
    if not reference or len(reference)!=len(actual):raise ValueError('Missing questions')
    max_logit=max_probability=0.;agreements=0
    for a,b in zip(reference,actual):
        if [(x['label'],x['text']) for x in a['candidates']]!=[(x['label'],x['text']) for x in b['candidates']]:raise ValueError('Candidate mapping changed')
        for x,y in zip(a['candidates'],b['candidates']):
            if not all(math.isfinite(z[k]) for z in (x,y) for k in ('raw_logit','probability')):raise ValueError('Non-finite score')
            max_logit=max(max_logit,abs(x['raw_logit']-y['raw_logit']))
            max_probability=max(max_probability,abs(x['probability']-y['probability']))
        if not math.isfinite(b['probability_sum']) or abs(b['probability_sum']-1)>1e-5:raise ValueError('Probability sum failed')
        agreements+=a['decision']['label']==b['decision']['label']
    return dict(max_abs_logit_delta=max_logit,max_abs_probability_delta=max_probability,
                decision_agreement=agreements,questions=len(reference))


async def run(args):
    c=config_load(args.config);data=MJevDataset(args.manifest)
    from mjev.families import check_modality,VL,OMNI
    kind=VL if 'Qwen3-VL' in c['model_id'] else OMNI
    if not len(data):raise ValueError('Empty dataset')
    for row in data:check_modality(kind,row['modality'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    os.environ['VLLM_BATCH_INVARIANT']='1' if c['numerics']=='stable' else '0'
    os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
    from huggingface_hub import snapshot_download
    model=snapshot_download(c['model_id'],revision=c['model_revision'])
    torch.manual_seed(c['seed']);engine=None
    def sync():
        for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
    try:
        evidence=capture(c,c['model_revision'],model);evidence['model_revision_status']='Hub snapshot at requested commit'
        evidence['manifest_sha256']=sha256(args.manifest)
        evidence['gpu_names']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        write(out/'run.json',evidence)
        if c['backend']=='hf':
            from mjev.hf import HFMJevEngine
            engine=HFMJevEngine(model,numerics=c['numerics'],dtype=c['dtype'],max_input_tokens=c['max_input_tokens'])
        else:
            os.environ['MJEV_ENABLE']='1'
            from mjev.patch import install
            install()
            from mjev.engine import MJevEngine
            engine=MJevEngine(model,enable_av=True,tensor_parallel_size=c['tensor_parallel_size'],max_model_len=c['max_input_tokens'])
        groups=defaultdict(list)
        for r in data:groups[(r['modality'],r['media_path'])].append(r)
        probes=[]
        for (modality,media),rows in groups.items():
            qs=[{'id':r['id'],'question':r['question'],'candidates':list(r['candidates'].values())} for r in rows]
            path=str(data.media_path(rows[0]));options=c['video_options'] if modality in ('video','audio_video') else None
            for mode in ['causal','isolated']:
                if c['backend']=='hf':
                    async def score(cache,batch,questions=qs):
                        return engine.score_many(path,questions,modality=modality,context=c['context'],mode=mode,
                            video_options=options,projection=c['projection'],use_prefix_cache=cache,batch_size=batch)
                else:
                    prepared=[engine.protocol.build(path,c['context'],q['question'],q['candidates'],mode,
                        modality=modality,video_options=options) for q in qs]
                    async def score(cache,batch,questions=qs):
                        values=[]
                        import copy
                        for start in range(0,len(questions),batch):
                            calls=[]
                            for q in questions[start:start+batch]:
                                p=copy.deepcopy(prepared[qs.index(q)]) if q in qs else engine.protocol.build(path,c['context'],q['question'],q['candidates'],mode,modality=modality,video_options=options)
                                p[1].skip_reading_prefix_cache=not cache
                                calls.append(engine.score_prepared(p,q['question'],q['candidates'],mode=mode))
                            values.extend(await asyncio.gather(*calls))
                        return values
                # Warm up once; each HF cached measurement includes its own prefill.
                await score(False,1)
                baseline=None
                for name,cache,batch in [('serial',False,1),('batch',False,len(qs)),('cache_serial',True,1),('cache_batch',True,len(qs))]:
                    seconds=[];repeat_comparisons=[]
                    for _ in range(args.repeats):
                        sync();start=time.perf_counter();results=await score(cache,batch);sync();seconds.append(time.perf_counter()-start)
                        if baseline is None:baseline=results
                        delta=compare(baseline,results)
                        if cache and not any(r['num_cached_tokens']>0 for r in results):raise ValueError('Cache reuse was requested but no tokens were reused')
                        if delta['max_abs_logit_delta']>args.atol or delta['decision_agreement']!=len(qs):raise ValueError(f'Cache/batch parity failed: {delta}')
                        repeat_comparisons.append(delta)
                    probes.append(dict(modality=modality,mode=mode,execution=name,seconds=seconds,
                        comparison=delta,repeat_comparisons=repeat_comparisons,results=results))
                    write(out/'probes.partial.json',probes)
                reverse=await score(True,len(qs),list(reversed(qs)))
                parity=compare(baseline,list(reversed(reverse)))
                if parity['max_abs_logit_delta']>args.atol or parity['decision_agreement']!=len(qs):raise ValueError(f'Question-order parity failed: {parity}')
                probes.append(dict(modality=modality,mode=mode,execution='question_order_reversed',comparison=parity,results=reverse))
                permuted=[{**q,'candidates':list(reversed(q['candidates']))} for q in qs]
                permuted_results=await score(True,len(qs),permuted)
                compare(permuted_results,permuted_results)  # Validate finite scores and normalization too.
                for q,r in zip(permuted,permuted_results):
                    if [x['text'] for x in r['candidates']]!=q['candidates'] or abs(r['probability_sum']-1)>1e-5:raise ValueError('Candidate permutation mapping failed')
                probes.append(dict(modality=modality,mode=mode,execution='candidate_order_reversed',results=permuted_results,
                    note='Mapping and normalization checked; answer invariance is not expected or asserted'))
        write(out/'report.json',dict(run=evidence,atol=args.atol,probes=probes,
            scope='Mechanism/timing probe on supplied data; synthetic data is not benchmark accuracy',
            timing_boundary='HF includes decode/process and prefill per call; vLLM prepared inputs exclude media decoding and warm cache can persist. Backend times are not directly comparable. One warmup per mode; model load excluded.'))
        (out/'_SUCCESS').write_text('Cache/batch numerical checks passed.\n')
    except Exception as exc:
        write(out/'ERROR.json',{'error':str(exc)});raise
    finally:
        if engine is not None and c['backend']=='vllm':engine.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=2);p.add_argument('--atol',type=float,default=1e-4)
    a=p.parse_args()
    if a.repeats<1 or a.atol<=0:p.error('repeats and atol must be positive')
    asyncio.run(run(a))

if __name__=='__main__':main()
